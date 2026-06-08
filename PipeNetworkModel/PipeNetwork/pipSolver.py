# networkSolver.py
import numpy as np
import warnings
from typing import Dict, Tuple
from PipeNetworkModel.PipeNetwork.networkModel import NetworkModel
from PipeNetworkModel.Components.networkNode import NodeType
from typing import Callable

class SingleNetworkNewtonSolver:
    """
    对单个 NetworkModel 做牛顿迭代求解的基础类。
    这里只实现框架，具体的压降公式、雅可比需要补充。
    """

    def __init__(self, network: NetworkModel):
        self.network = network
        self.verbose: bool = False
        self.last_x: np.ndarray | None = None
        self.last_converged: bool = False
        self.last_residual_norm: float | None = None
        self.residual_history: list[dict[str, float | int | str]] = []
        self.last_delta_p: float | None = None
        # 建立未知量索引：节点压力、边流量
        self.node_index = {name: i for i, name in enumerate(self.network.networkNodesDict.keys())}
        self.edge_index = {name: i for i, name in enumerate(self.network.networkConnDict.keys())}
        self.n_nodes = len(self.node_index)
        self.n_edges = len(self.edge_index)
        # 为每口井设置地层压力 pres 和产能系数 k
        self.source_pq_params: dict[str, dict[str, float]] = {}
        self.networkNodesDict_not_include_sink = {name: node for name, node in self.network.networkNodesDict.items() if node.type != NodeType.SINK}
        self.node_index_not_include_sink = {name: i for i, name in enumerate(self.networkNodesDict_not_include_sink.keys())}
        self.n_nodes_not_include_sink = len(self.node_index_not_include_sink)
        self.n_unknowns = self.n_nodes_not_include_sink + self.n_edges  # 注意：如果有汇节点，最后一个节点的压力不作为未知量了
        # 举例参数：你可以后面按井单独调
        default_pres = 7e6  # 地层压力
        default_k = 1e-11  # 产能系数
        default_p_min = 4e6  # 井口压力下界
        default_p_max = 6e6  # 井口压力上界
        default_q_max = 1e3  # 最大产量（绝对值）

        for node_name, node in self.network.networkNodesDict.items():
            if node.type == NodeType.SOURCE:
                self.source_pq_params[node_name] = {
                    "pres": default_pres,
                    "k": default_k,
                    "p_min": default_p_min,
                    "p_max": default_p_max,
                    "q_min": 0.0,  # 不允许负产
                    "q_max": default_q_max,  # 产量上限
                }

    def _source_injection(self, node_name: str, p_i: float) -> float:
        """
        源井注入：q(p) = k * (pres^2 - p^2)，再做 (p,q) 夹紧：
        - p 限制在 [p_min, p_max]
        - q 限制在 [q_min, q_max]，特别是 q_min=0，保证源井不会成为“吸流口”
        """
        params = self.source_pq_params.get(node_name)
        if params is None:
            return 0.0

        pres = params["pres"]
        k = params["k"]
        p_min = params.get("p_min", -float("inf"))
        p_max = params.get("p_max", float("inf"))
        q_min = params.get("q_min", 0.0)
        q_max = params.get("q_max", float("inf"))

        # 1) 先把 p 限制在物理范围内
        p_clamped = min(max(p_i, p_min), p_max)

        # 2) 按二次形式计算 q
        #    这里仍然用井产能形式：q = k * (pres^2 - p^2)
        q_raw = k * (pres ** 2 - p_clamped ** 2)

        # 3) 再把 q 限制在 [q_min, q_max]
        q_limited = min(max(q_raw, q_min), q_max)

        return q_limited

    def _pack_x(self, p: Dict[str, float], q: Dict[str, float]) -> np.ndarray:
        x = np.zeros(self.n_unknowns)
        # 节点压力
        for name, idx in self.node_index_not_include_sink.items():
            x[idx] = p.get(name, 0.0)
        # 边流量
        for name, idx in self.edge_index.items():
            x[self.n_nodes_not_include_sink + idx] = q.get(name, 0.0)
        return x

    def _unpack_x(self, x: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        p = {}
        q = {}
        for name, idx in self.node_index_not_include_sink.items():
            p[name] = float(x[idx])
        for name, idx in self.edge_index.items():
            q[name] = float(x[self.n_nodes_not_include_sink + idx])
        return p, q

    def _safe_build_residual(self, x, boundary_conditions):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                F = self._build_residual(x, boundary_conditions)

            if not np.all(np.isfinite(F)):
                F = np.full(self.n_unknowns, 1e20)

            return F

        except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError, RuntimeWarning) as e:
            if self.verbose:
                print(f"[Inner safe residual] invalid trial rejected: {e}")
            return np.full(self.n_unknowns, 1e20)

    def _build_residual(self, x: np.ndarray,
                        boundary_conditions: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        构造 F(x)，包含：
        1) 每个节点的质量平衡
        2) 每条管线的压降关系
        3) 边界条件（指定 p 或 指定注入流量）
        boundary_conditions: {
            "pressure": {node_name: p_value},
            "injection": {node_name: q_inj},  # 源/汇注入(+)/抽取(-)
        }
        """

        p, q = self._unpack_x(x)
        F = np.zeros(self.n_unknowns)

        pBoundary = boundary_conditions.get("pressure", {})
        inj_dict = boundary_conditions.get("injection", {})
        flowBoundary = boundary_conditions.get("flow", {})  # 如果以后需要指定某些管段流量，单独用 flow

        # 节点压力边界只进入 p
        p_full = p.copy()
        p_full.update(pBoundary)

        # 管段流量边界只允许进入 q，不能把 injection 混进来
        q_full = q.copy()
        for conn_name, q_fix in flowBoundary.items():
            if conn_name in self.edge_index:
                q_full[conn_name] = q_fix
            else:
                raise KeyError(f"flow boundary '{conn_name}' is not a connection name")

        # 检查所有 sink 节点是否都有压力边界
        missing_sink = [
            name for name, node in self.network.networkNodesDict.items()
            if node.type == NodeType.SINK and name not in p_full
        ]
        if missing_sink:
            raise KeyError(
                f"Missing pressure boundary for sink node(s): {missing_sink}. "
                f"Please provide boundary_conditions['pressure'] for all sink nodes."
            )

        try:
            update_flag = self.network.updateFluidParam(p_full, q_full)
        except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError) as exc:
            self.last_update_error = exc
            update_flag = False
        # update_flag = True
        if update_flag:
            # 提前取好外部注入字典

            inj_dict = boundary_conditions.get("injection", {})

            # 1. 节点质量守恒：对每个节点 i: sum_in(q) - sum_out(q) + q_inj = 0
            # 约定：q_inj > 0 表示外界向该节点注入网络；q_inj < 0 表示该节点从网络向外抽取
            for node_name, node in self.networkNodesDict_not_include_sink.items():
                idx = self.node_index_not_include_sink[node_name]

                flow_in = 0.0
                flow_out = 0.0

                # 统计进入 / 流出该节点的所有管段流量
                for conn_name, conn in self.network.networkConnDict.items():
                    # 使用冻结参考流向
                    n_from, n_to = conn.flowDirection
                    q_e = q_full[conn_name]

                    if n_to == node_name:
                        flow_in += q_e
                    if n_from == node_name:
                        flow_out += q_e

                # 外部给定的注入/抽取（比如出口负荷），约定：+ 注入，- 抽取
                q_inj = inj_dict.get(node_name, 0.0)

                # 源节点：叠加 p-q 曲线注入
                if node.type == NodeType.SOURCE:
                    q_inj += self._source_injection(node_name, p_full[node_name])

                F[idx] = flow_in - flow_out + q_inj

            # 2. 管段压降关系
            #    F_edge = p_from - p_to - Δp(q, conn) = 0
            #    这里给一个简单的 R*q*|q| 形式示意，具体你可以改。
            for conn_name, conn in self.network.networkConnDict.items():
                e_idx = self.edge_index[conn_name]
                # 对应 residual 的位置是在 F 的后半部分
                F_idx = self.n_nodes_not_include_sink + e_idx

                n_from, n_to = conn.flowDirection
                p_from = p_full[n_from]
                # if n_to == 'sink1':
                #     d = dict(boundary_conditions["pressure"].items())
                #     p_to = d[n_to]
                # else:
                #     p_to = p[n_to]
                p_to = p_full[n_to]
                # q_e = q[conn_name]


                # # # TODO: 这里可以根据 length / 直径 / 流体性质构建更真实的 Δp 公式

                delta_p_model = conn.flowlineSim.getFlowlinePressureDrop()

                F[F_idx] = p_from - p_to - delta_p_model

                # R = getattr(conn, "resistance", 1e-0)  # 举例：如果没设置，就给个很小的阻力
                # delta_p_model = R * q_e * abs(q_e)
                # F[F_idx] = p_from - p_to - delta_p_model

                # delta_p_model = p_from - p_to - 5
                # F[F_idx] = delta_p_model
            self.last_delta_p = delta_p_model
            # print(f"last Δp model value: {self.last_delta_p:.3e}", f"last p_from: {p_from:.3e}", f"last p_to: {p_to:.3e}")
            for node_name, p_fix in pBoundary.items():
                if node_name in self.node_index_not_include_sink:
                    idx = self.node_index_not_include_sink[node_name]
                    # 注意：这里必须用原始未知量 p，而不是被覆盖后的 p_full
                    F[idx] = p[node_name] - p_fix

            # # 3. 边界条件：指定压力（用强制方程）
            # if "pressure" in boundary_conditions:
            #     for node_name, p_fix in boundary_conditions["pressure"].items():
            #         idx = self.node_index[node_name]
            #         # 用 p_i - p_fix = 0 替换原有质量平衡方程
            #         F[idx] = p[node_name] - p_fix
        else:
            F[:] = 1e20
        return F

    def _project_pressure_bc_to_x(self, x, boundary_conditions):
        p_bc = boundary_conditions.get("pressure", {})
        for node_name, p_fix in p_bc.items():
            if node_name in self.node_index_not_include_sink:
                idx = self.node_index_not_include_sink[node_name]
                x[idx] = p_fix
        return x

    def _debug_residual_blocks(self, F):
        node_F = F[:self.n_nodes_not_include_sink]
        edge_F = F[self.n_nodes_not_include_sink:]

        print(
            f"[debug residual] "
            f"node_norm={np.linalg.norm(node_F):.3e}, "
            f"edge_norm={np.linalg.norm(edge_F):.3e}, "
            f"node_max={np.max(np.abs(node_F)) if node_F.size else 0:.3e}, "
            f"edge_max={np.max(np.abs(edge_F)) if edge_F.size else 0:.3e}"
        )

    def _project_x(self, x, boundary_conditions):
        x = x.copy()

        p_min = 1e5
        p_max = 5e7
        q_min_abs = 1e-8
        q_max_abs = 1e3

        # 1. 压力变量限制
        for name, idx in self.node_index_not_include_sink.items():
            x[idx] = np.clip(x[idx], p_min, p_max)

        # 2. 压力边界强制回写
        p_bc = boundary_conditions.get("pressure", {})
        for node_name, p_fix in p_bc.items():
            if node_name in self.node_index_not_include_sink:
                idx = self.node_index_not_include_sink[node_name]
                x[idx] = p_fix

        # 3. 流量变量限制
        for conn_name, eidx in self.edge_index.items():
            idx = self.n_nodes_not_include_sink + eidx

            # 当前冻结流向版本：q 只允许沿 flowDirection 正向流动
            x[idx] = np.clip(x[idx], q_min_abs, q_max_abs)

        return x

    def _x_bounds(self):
        lower = np.zeros(self.n_unknowns, dtype=float)
        upper = np.zeros(self.n_unknowns, dtype=float)

        lower[:self.n_nodes_not_include_sink] = 1e5
        upper[:self.n_nodes_not_include_sink] = 5e7
        lower[self.n_nodes_not_include_sink:] = 1e-8
        upper[self.n_nodes_not_include_sink:] = 1e3
        return lower, upper

    def _least_squares_refine(self, x, boundary_conditions, tol, max_iter):
        try:
            from scipy.optimize import least_squares
        except ImportError:
            return x, np.linalg.norm(self._safe_build_residual(x, boundary_conditions), ord=2)

        lower, upper = self._x_bounds()
        x0 = np.minimum(np.maximum(self._project_x(x, boundary_conditions), lower), upper)

        def residual(x_vec):
            x_vec = self._project_x(x_vec, boundary_conditions)
            return self._safe_build_residual(x_vec, boundary_conditions)

        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            xtol=tol,
            ftol=tol,
            gtol=tol,
            max_nfev=max(max_iter * max(self.n_unknowns, 1), 100),
        )
        x_best = self._project_x(result.x, boundary_conditions)
        return x_best, np.linalg.norm(self._safe_build_residual(x_best, boundary_conditions), ord=2)

    def _finite_difference_jacobian(self, x, Fx, boundary_conditions, eps):
        n = x.size
        J = np.zeros((n, n), dtype=float)
        for j in range(n):
            x_pert = x.copy()
            if j < self.n_nodes_not_include_sink:
                h = max(abs(x[j]) * eps, 100.0)
            else:
                h = max(abs(x[j]) * eps, 1e-6)

            x_pert[j] += h
            x_pert = self._project_x(x_pert, boundary_conditions)
            Fj = self._safe_build_residual(x_pert, boundary_conditions)
            J[:, j] = (Fj - Fx) / h
        return J

    def solve(self,
              boundary_conditions: Dict[str, Dict[str, float]],
              x0=None,
              eps: float = 1e-5,
              tol: float = 1e-2,
              max_iter: int = 100,
              damping: float = 0.5) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        使用牛顿法求解单个子网，返回 (p, q)。
        """
        if self.verbose:
            print(f'start solving {self.network.name}')
        self.residual_history = []
        if x0 is None:
            if self.last_x is not None:
                x0 = self.last_x.copy()
            else:
                # 如果有 pressure 边界，就用边界里的值，否则用一个默认 5e6
                default_p0 = 1e6
                p_bc = boundary_conditions.get("pressure", {})
                p0 = {}
                for name in self.network.networkNodesDict.keys():
                    if name in p_bc:
                        p0[name] = p_bc[name]
                    else:
                        p0[name] = default_p0

                q0 = {name: 0.05 for name in self.network.networkConnDict.keys()}
                x0 = self._pack_x(p0, q0)
        else:
            x0 = x0.copy()
        x = x0.copy()
        x = self._project_x(x, boundary_conditions)
        J = None
        jacobian_refresh_interval = 5

        for k in range(max_iter):
            if self.verbose:
                print(f'{self.network.name} iteration {k}')
            Fx = self._safe_build_residual(x, boundary_conditions)
            normF = np.linalg.norm(Fx, 2)
            node_F = Fx[:self.n_nodes_not_include_sink]
            edge_F = Fx[self.n_nodes_not_include_sink:]
            current_method = "newton_fd" if J is None or k % jacobian_refresh_interval == 0 else "broyden"
            self.residual_history.append({
                "iteration": k,
                "method": current_method,
                "residual_norm": float(normF),
                "node_norm": float(np.linalg.norm(node_F, ord=2)),
                "edge_norm": float(np.linalg.norm(edge_F, ord=2)),
            })
            if normF < tol:
                break
            if self.verbose:
                print(f"[Inner {id(self)}][Iter {k}] ||F|| = {normF:.3e}")
                self._debug_residual_blocks(Fx)
            if J is None or k % jacobian_refresh_interval == 0:
                J = self._finite_difference_jacobian(x, Fx, boundary_conditions, eps)
            # 有限差分雅可比
            n = 0
            J = J
            for j in range(n):
                x_pert = x.copy()
                if j < self.n_nodes_not_include_sink:
                    # 压力变量，单位 Pa
                    h = max(abs(x[j]) * eps, 100.0)
                else:
                    # 流量变量，通常量级远小于压力
                    h = max(abs(x[j]) * eps, 1e-6)

                x_pert[j] += h
                x_pert = self._project_x(x_pert, boundary_conditions)
                Fj = self._safe_build_residual(x_pert, boundary_conditions)
                J[:, j] = (Fj - Fx) / h
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]

            alpha = 1.0
            accepted = False

            for _ in range(8):
                x_trial = x + alpha * dx
                x_trial = self._project_x(x_trial, boundary_conditions)
                F_trial = self._safe_build_residual(x_trial, boundary_conditions)
                norm_trial = np.linalg.norm(F_trial, 2)

                if np.isfinite(norm_trial) and norm_trial < normF:
                    step = x_trial - x
                    residual_delta = F_trial - Fx
                    denom = float(np.dot(step, step))
                    if denom > 0.0:
                        J = J + np.outer(residual_delta - J @ step, step) / denom
                    x = x_trial
                    accepted = True
                    break

                alpha *= 0.5

            if not accepted:
                if current_method == "broyden":
                    J = self._finite_difference_jacobian(x, Fx, boundary_conditions, eps)
                    try:
                        dx = np.linalg.solve(J, -Fx)
                    except np.linalg.LinAlgError:
                        dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]

                    alpha = 1.0
                    for _ in range(8):
                        x_trial = x + alpha * dx
                        x_trial = self._project_x(x_trial, boundary_conditions)
                        F_trial = self._safe_build_residual(x_trial, boundary_conditions)
                        norm_trial = np.linalg.norm(F_trial, 2)

                        if np.isfinite(norm_trial) and norm_trial < normF:
                            step = x_trial - x
                            residual_delta = F_trial - Fx
                            denom = float(np.dot(step, step))
                            if denom > 0.0:
                                J = J + np.outer(residual_delta - J @ step, step) / denom
                            x = x_trial
                            accepted = True
                            break

                        alpha *= 0.5

                if accepted:
                    continue
                if self.verbose:
                    print("[Inner] line search failed, keep current x and stop inner iteration")
                break

            # 结束时再检查一次
        final_norm = np.linalg.norm(self._safe_build_residual(x, boundary_conditions), ord=2)
        if final_norm >= 10 * tol:
            x_refined, refined_norm = self._least_squares_refine(x, boundary_conditions, tol, max_iter)
            if np.isfinite(refined_norm) and refined_norm < final_norm:
                x = x_refined
                final_norm = refined_norm
                F_refined = self._safe_build_residual(x, boundary_conditions)
                node_F = F_refined[:self.n_nodes_not_include_sink]
                edge_F = F_refined[self.n_nodes_not_include_sink:]
                self.residual_history.append({
                    "iteration": len(self.residual_history),
                    "method": "least_squares",
                    "residual_norm": float(final_norm),
                    "node_norm": float(np.linalg.norm(node_F, ord=2)),
                    "edge_norm": float(np.linalg.norm(edge_F, ord=2)),
                })
        self.last_residual_norm = float(final_norm)
        self.last_converged = final_norm < 10 * tol

        if not self.last_converged:
            print(f"[Inner {id(self)}][WARN] not converged, ||F|| = {final_norm:.3e}")

        self.last_x = x.copy()
        p, q = self._unpack_x(x)
        if self.verbose:
            print(f'end solving {self.network.name}')
        return p, q

        # ---------- 从解里提取源节点的 p-q ----------
    def extract_source_pq(
            self,
            p: Dict[str, float],
            q: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        根据求解结果算每个源井的：
        - p        : 井口压力
        - q_net    : 节点净流入 = sum_in(q) - sum_out(q)
        - q_out    : 向外产量 = max(0, -q_net)，保证 >=0
        """
        source_pq: Dict[str, Dict[str, float]] = {}

        for node_name, node in self.network.networkNodesDict.items():
            if node.type == NodeType.SOURCE:
                flow_in = 0.0
                flow_out = 0.0
                for conn_name, conn in self.network.networkConnDict.items():
                    n_from, n_to = conn.nodes
                    q_e = q[conn_name]
                    if n_to == node_name:
                        flow_in += q_e
                    if n_from == node_name:
                        flow_out += q_e

                q_net = flow_in - flow_out
                # 从“网络视角”的净流入转成“井向外生产量”
                q_out = max(0.0, -q_net)  # 不允许负产

                source_pq[node_name] = {
                    "p": p[node_name],
                    "q_net": q_net,
                    "q_out": q_out,
                }

        return source_pq

class OriginalNetworkNewtonSolver(SingleNetworkNewtonSolver):
    """
    单层原始管网 Newton 求解器。

    调用对象是未拆分的 original NetworkModel。未知量仍沿用
    SingleNetworkNewtonSolver 的组织方式：
    - 非 sink 节点压力
    - 全部连接边的质量流量

    边界条件建议由 sink 节点压力给定；source 节点流量由节点自身
    getMassFlowRateByPressure(p) 生成，因此不再需要一、二级子网交互。
    """

    def __init__(self, network: NetworkModel):
        super().__init__(network)
        self.last_pressure: Dict[str, float] = {}
        self.last_flowrate: Dict[str, float] = {}

    @staticmethod
    def _as_float(value, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_boundary_type(value) -> str:
        if value is None:
            return ""
        return str(value).strip().replace(" ", "").lower()

    def _source_injection(self, node_name: str, p_i: float) -> float:
        node = self.network.networkNodesDict.get(node_name)
        if node is None or node.type != NodeType.SOURCE:
            return 0.0
        if hasattr(node, "isActive") and not node.isActive:
            return 0.0

        boundary_type = self._normalize_boundary_type(getattr(node, "boundaryType", ""))
        if boundary_type == "massflowrate" and not getattr(node, "isPQCurve", False):
            return max(self._as_float(getattr(node, "massFlow", 0.0), 0.0), 0.0)

        return max(float(node.getMassFlowRateByPressure(p_i)), 0.0)

    def _default_pressure_boundary(self) -> Dict[str, float]:
        pressure_bc: Dict[str, float] = {}
        for node_name, node in self.network.networkNodesDict.items():
            if node.type == NodeType.SINK:
                pressure = self._as_float(node.pressure, 0.0)
                if pressure > 0.0:
                    pressure_bc[node_name] = pressure
        return pressure_bc

    def _default_injection_boundary(self) -> Dict[str, float]:
        injection_bc: Dict[str, float] = {}
        for node_name, node in self.network.networkNodesDict.items():
            if (
                    node.type == NodeType.SINK
                    and self._normalize_boundary_type(getattr(node, "boundaryType", "")) == "massflowrate"
            ):
                injection_bc[node_name] = -self._as_float(node.massFlow, 0.0)
        return injection_bc

    def _build_boundary_conditions(
            self,
            boundary_conditions: Dict[str, Dict[str, float]] | None = None,
            sink_pressures: Dict[str, float] | None = None,
            injection: Dict[str, float] | None = None
    ) -> Dict[str, Dict[str, float]]:
        bc = {"pressure": self._default_pressure_boundary(), "injection": self._default_injection_boundary()}

        if boundary_conditions is not None:
            bc["pressure"].update(boundary_conditions.get("pressure", {}))
            bc["injection"].update(boundary_conditions.get("injection", {}))
        if sink_pressures is not None:
            bc["pressure"].update(sink_pressures)
        if injection is not None:
            bc["injection"].update(injection)

        missing_sinks = [
            name for name, node in self.network.networkNodesDict.items()
            if node.type == NodeType.SINK and name not in bc["pressure"]
        ]
        if missing_sinks:
            raise ValueError(f"原始管网单层求解需要给定所有 sink 节点压力: {missing_sinks}")
        return bc

    def _build_initial_x(self, boundary_conditions: Dict[str, Dict[str, float]]) -> np.ndarray:
        pressure_bc = boundary_conditions.get("pressure", {})
        p0: Dict[str, float] = {}
        for node_name, node in self.network.networkNodesDict.items():
            if node_name in pressure_bc:
                p0[node_name] = self._as_float(pressure_bc[node_name], 1.0e6)
            elif self._as_float(getattr(node, "pressure", None), 0.0) > 0.0:
                p0[node_name] = self._as_float(node.pressure, 1.0e6)
            else:
                p0[node_name] = 1.0e6

        source_rates = [
            self._source_injection(node_name, p0[node_name])
            for node_name, node in self.network.networkNodesDict.items()
            if node.type == NodeType.SOURCE
        ]
        default_q = max(sum(source_rates) / max(self.n_edges, 1), 1.0e-3)
        q0 = {conn_name: default_q for conn_name in self.network.networkConnDict.keys()}
        return self._pack_x(p0, q0)

    def _commit_solution(
            self,
            p: Dict[str, float],
            q: Dict[str, float],
            boundary_conditions: Dict[str, Dict[str, float]]
    ) -> None:
        full_p = dict(p)
        full_p.update(boundary_conditions.get("pressure", {}))

        for node_name, pressure in full_p.items():
            self.network.networkNodesDict[node_name].pressure = pressure
        for conn_name, flowrate in q.items():
            self.network.networkConnDict[conn_name].flowlineSim.flowRate = flowrate

        self.last_pressure = full_p
        self.last_flowrate = dict(q)

    def solve(
            self,
            boundary_conditions: Dict[str, Dict[str, float]] | None = None,
            sink_pressures: Dict[str, float] | None = None,
            injection: Dict[str, float] | None = None,
            x0=None,
            eps: float = 1e-3,
            tol: float = 1e-2,
            max_iter: int = 100,
            damping: float = 0.5
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        bc = self._build_boundary_conditions(boundary_conditions, sink_pressures, injection)
        if x0 is None and self.last_x is None:
            x0 = self._build_initial_x(bc)

        p, q = super().solve(
            bc,
            x0=x0,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            damping=damping
        )
        self._commit_solution(p, q, bc)
        return self.last_pressure, q
