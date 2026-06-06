# networkSolver.py
import numpy as np
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
        self.last_x: np.ndarray | None = None
        self.last_converged: bool = False
        self.last_residual_norm: float | None = None
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
        pBoundary= boundary_conditions.get("pressure",{})
        qBoundary= boundary_conditions.get("injection",{})
        p.update(pBoundary)
        q.update(qBoundary)

        update_flag = self.network.updateFluidParam(p, q)
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
                    n_from, n_to = conn.nodes
                    q_e = q[conn_name]
                    if n_to == node_name:
                        flow_in += q_e
                    if n_from == node_name:
                        flow_out += q_e

                # 外部给定的注入/抽取（比如出口负荷），约定：+ 注入，- 抽取
                q_inj = inj_dict.get(node_name, 0.0)

                # 源节点：叠加 p-q 曲线注入
                if node.type == NodeType.SOURCE:
                    q_inj += self._source_injection(node_name, p[node_name])

                F[idx] = flow_in - flow_out + q_inj

            # 2. 管段压降关系
            #    F_edge = p_from - p_to - Δp(q, conn) = 0
            #    这里给一个简单的 R*q*|q| 形式示意，具体你可以改。
            for conn_name, conn in self.network.networkConnDict.items():
                e_idx = self.edge_index[conn_name]
                # 对应 residual 的位置是在 F 的后半部分
                F_idx = self.n_nodes_not_include_sink + e_idx

                n_from, n_to = conn.nodes
                p_from = p[n_from]
                # if n_to == 'sink1':
                #     d = dict(boundary_conditions["pressure"].items())
                #     p_to = d[n_to]
                # else:
                #     p_to = p[n_to]
                p_to = p[n_to]
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

    def solve(self,
              boundary_conditions: Dict[str, Dict[str, float]],
              x0=None,
              eps: float = 1e-3,
              tol: float = 1e-2,
              max_iter: int = 100,
              damping: float = 0.5) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        使用牛顿法求解单个子网，返回 (p, q)。
        """
        print(f'start solving {self.network.name}')
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

                q0 = {name: 0.006 for name in self.network.networkConnDict.keys()}
                x0 = self._pack_x(p0, q0)
        else:
            x0 = x0.copy()
        x = x0.copy()
        for k in range(max_iter):
            print(f'{self.network.name} iteration {k}')
            Fx = self._build_residual(x, boundary_conditions)
            normF = np.linalg.norm(Fx, 2)
            if normF < tol:
                break
            print(f"[Inner {id(self)}][Iter {k}] ||F|| = {normF:.3e}")
            # 有限差分雅可比
            n = x.size
            J = np.zeros((n, n), dtype=float)
            for j in range(n):
                x_pert = x.copy()
                x_pert[j] += x_pert[j]*eps
                Fj = self._build_residual(x_pert, boundary_conditions)
                J[:, j] = (Fj - Fx) / (x_pert[j]*eps)
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]

            x = x + damping * dx

            # 结束时再检查一次
        final_norm = np.linalg.norm(self._build_residual(x, boundary_conditions), ord=2)
        self.last_residual_norm = float(final_norm)
        self.last_converged = final_norm < 10 * tol

        if not self.last_converged:
            print(f"[Inner {id(self)}][WARN] not converged, ||F|| = {final_norm:.3e}")

        self.last_x = x.copy()
        p, q = self._unpack_x(x)
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
