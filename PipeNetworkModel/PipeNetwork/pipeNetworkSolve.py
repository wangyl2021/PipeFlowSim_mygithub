from PipeNetworkModel.PipeNetwork.networkModel import *
from PipeNetworkModel.Components.networkNode import *
from PipeNetworkModel.Components.networkConnections import *
from typing import Dict, Optional, Set
from collections import defaultdict
import numpy as np
from PipeNetworkModel.PipeNetwork.pipSolver import *




class PipeNetworkSolve:
    def __init__(self,networkModel:NetworkModel):
        self.originalNetworkModel: NetworkModel= networkModel
        self.original_solver: OriginalNetworkNewtonSolver | None = None
        self.subNetworkModelLevel1: dict[str, NetworkModel] = {}
        self.subNetworkModelLevel2: dict[str, NetworkModel] = {}

        # interface_node -> 参与的 level1 子网列表
        self.interface_to_level1: dict[str, list[str]] = defaultdict(list)

        self.solvers: Dict[str, SingleNetworkNewtonSolver] = {}
        # self.level1_solvers: Dict[str, PipeSubNetworkLevel1Solver] = {}
        self.level2_solvers: Dict[str, SingleNetworkNewtonSolver] = {}

        # 新增：接口节点的结构压力向量，在两层迭代中持续更新
        self.interface_p_vec: np.ndarray | None = None

        # 二级接口节点上的“本地注入/负荷”，不是跨层那一部分
        # key: 节点名，value: 本地注入（正为注入，负为外送）
        self.local_injection: Dict[str, float] = {}
        self.level2_local_injection = self.local_injection

    def _normalize_sink_pressures(self, sink_pressures):
        sink_names = [
            name for name, node in self.originalNetworkModel.networkNodesDict.items()
            if node.type == NodeType.SINK
        ]
        if sink_pressures is None or isinstance(sink_pressures, dict):
            return sink_pressures

        pressure = float(sink_pressures)
        return {name: pressure for name in sink_names}

    @staticmethod
    def _compute_node_q_net(
            network: NetworkModel,
            node_name: str,
            q_sol: Dict[str, float],
            *,
            subnet_nodes: Optional[Set[str]] = None,
            mode: str = "iface_edges"  # "iface_edges" 或 "internal_edges"
    ) -> float:
        """
            返回节点在当前子网内部的净流入：
                q_net = sum_in(q) - sum_out(q)
            q_net > 0 表示网络内部流入该节点；
            q_net < 0 `表示该节点向网络内部流出。
        """
        if subnet_nodes is None:
            subnet_nodes = set(network.networkNodesDict.keys())

        flow_in = 0.0
        flow_out = 0.0

        for conn_name, conn in network.networkConnDict.items():
            n_from, n_to = conn.nodes

            # 没有该边的解，直接跳过（避免 KeyError）
            if conn_name not in q_sol:
                continue

            # ---------- 过滤规则 ----------
            if mode == "iface_edges":
                # 只保留：node_name 与某个“子网节点”相连的边
                if node_name == n_from:
                    other = n_to
                elif node_name == n_to:
                    other = n_from
                else:
                    continue

                if other not in subnet_nodes:
                    continue

            elif mode == "internal_edges":
                # 只保留：两端都在子网中的边
                if (n_from not in subnet_nodes) or (n_to not in subnet_nodes):
                    continue
                # 若 node_name 不在这条边上，跳过
                if node_name != n_from and node_name != n_to:
                    continue
            else:
                raise ValueError(f"Unknown mode={mode}")

            # ---------- 累加流入/流出 ----------
            q_e = q_sol[conn_name]
            if n_to == node_name:
                flow_in += q_e
            if n_from == node_name:
                flow_out += q_e

        return flow_in - flow_out

    def splitedNetwork(self):
        """
        将类中保存的原始网络模型分割为一级网络和二级网络
        :return:
        """

        ####### 构建接口映射：接口节点 → 参与的一级子网列表-new
        self.interface_to_level1.clear()
        for sub_name, sub_model in self.subNetworkModelLevel1.items():
            for iface_name in sub_model.convergingNodeSubNetwork.keys():
                # 确保该节点确实出现在 Level2 某个子网里
                if any(iface_name in m.networkNodesDict for m in self.subNetworkModelLevel2.values()):
                    self.interface_to_level1[iface_name].append(sub_name)

        ### solvers
        # for name, model in self.subNetworkModelLevel1.items():
        #     self.level1_solvers[name] = PipeSubNetworkLevel1Solver(model)

        for name, model in self.subNetworkModelLevel2.items():
            self.level2_solvers[name] = SingleNetworkNewtonSolver(model)

    def _vec_to_iface_pressure_dict(
            self,
            interface_nodes: list[str],
            name_to_idx: dict[str, int],
            p_vec: np.ndarray
    ) -> Dict[str, float]:
        return {name: float(p_vec[name_to_idx[name]]) for name in interface_nodes}

    def _iface_pressure_dict_to_vec(
            self,
            interface_nodes: list[str],
            name_to_idx: dict[str, int],
            p_dict: Dict[str, float],
            default_vec: np.ndarray | None = None
    ) -> np.ndarray:
        if default_vec is None:
            p_vec = np.zeros(len(interface_nodes), dtype=float)
        else:
            p_vec = default_vec.copy()

        for name in interface_nodes:
            if name in p_dict:
                p_vec[name_to_idx[name]] = float(p_dict[name])

        return p_vec

    def _eval_F(
            self,
            interface_nodes: list[str],
            name_to_idx: dict[str, int],
            sink_pressures: dict[str, float],
            p_vec=None
    ) -> tuple[np.ndarray, Dict[str, float]]:
        """
                在给定接口压力 p_vec 下：
                1) 求解所有一级子网，得到接口节点从一级流向二级的注入 q_inj_level2；
                2) 再求解所有二级子网，得到二级接口节点净流入 q_net_level2；
                3) 构造接口耦合残差 F；
                4) 从二级解中抽取新的接口压力向量 p_level2_vec，供外层更新使用。

                返回：
                - F: 接口残差向量
                - q_inj_level2: 一级侧注入
                - p_level2_vec: 从二级当前解提取的接口压力向量
                """
        if p_vec is None:
            if self.interface_p_vec is None:
                raise RuntimeError("interface_p_vec is None, 请先初始化")
            p_vec = self.interface_p_vec.copy()

        iface_p = self._vec_to_iface_pressure_dict(interface_nodes, name_to_idx, p_vec)

        # ---------- 1. 计算一级子网在接口节点的注入 q_inj_level2 ----------
        # 初始化接口节点的一级注入量为0
        q_inj_level2: Dict[str, float] = {name: 0.0 for name in interface_nodes}

        # 遍历所有一级子网，计算接口节点的注入量
        for sub_name, sub_model in self.subNetworkModelLevel1.items():
            # 创建一级子网的牛顿求解器
            solver1 = self.level1_solvers[sub_name]
            # 一级子网的压力边界条件

            pressure_bc: Dict[str, float] = {}
            # 一级子网的注入边界条件
            injection_bc: Dict[str, float] = {}

            # 一级子网的注入边界条件
            for iface_name in sub_model.convergingNodeSubNetwork.keys():
                pressure_bc[iface_name] = iface_p[iface_name]

            # 求解一级子网，得到节点压力p1和流量q1
            q_iface = solver1.computeNetworkLevel1(pressure_bc)

            # 计算接口节点的注入量，累加到q_inj_level2中
            for iface_name in sub_model.convergingNodeSubNetwork.keys():
                # 统一约定：
                #   q_couple_level1 > 0 表示：Level-1 → Level-2
                q_inj_level2[iface_name] += q_iface[iface_name]

        # ---------- 2. 计算二级子网在接口节点的“总净流入” q_net_level2 ----------
        # 初始化接口节点的二级净流入为0
        q_net_level2: Dict[str, float] = {name: 0.0 for name in interface_nodes}
        p_level2_dict: Dict[str, float] = {}
        all_level2_converged = True

        # 遍历所有二级子网，计算接口节点的净流入
        for sub_name, sub_model in self.subNetworkModelLevel2.items():
            solver2 = self.level2_solvers[sub_name]

            pressure_bc2: Dict[str, float] = {}
            injection_bc2: Dict[str, float] = {}
            # 对于接口节点，设置注入边界为一级注入量
            for node_name, node in sub_model.networkNodesDict.items():
                if node_name in interface_nodes:
                    injection_bc2[node_name] = q_inj_level2[node_name]
                if node_name in sink_pressures:
                    pressure_bc2[node_name] = sink_pressures[node_name]

            bc2 = {"pressure": pressure_bc2, "injection": injection_bc2}
            p2, q2 = solver2.solve(bc2)

            print(f"[outer] {sub_name}: inner_converged={solver2.last_converged}, "
                  f"inner_residual={solver2.last_residual_norm}")

            if not solver2.last_converged:
                all_level2_converged = False
                print(f"[outer][WARN] level2 子网 {sub_name} 未收敛，"
                      f"residual = {solver2.last_residual_norm:.3e}")

            # 记录本二级子网中接口点的压力
            for iface_name in sub_model.convergingNodeSubNetwork.keys():
                if iface_name in p2:
                    p_level2_dict[iface_name] = p2[iface_name]

            # 统计每个接口节点的净流入
            for iface_name in sub_model.convergingNodeSubNetwork.keys():
                if iface_name in q_net_level2:
                    q_net = self._compute_node_q_net(sub_model, iface_name, q2)
                    q_net_level2[iface_name] += q_net

        # ---------- 3. 如果接口节点上还有“本地负荷/注入”，在这里扣掉 ----------
        # level2_local_injection[name] > 0 代表本地往网里注入，<0 代表本地从网中抽取
        F = np.zeros(len(interface_nodes), dtype=float)
        for name, idx in name_to_idx.items():
            q_local = self.level2_local_injection.get(name, 0.0)
            # 目标：二级节点质量守恒
            # q_net_level2 + q_inj_from_level1 - q_local = 0
            F[idx] = q_net_level2[name] + q_inj_level2[name] - q_local

        # ---------- 5) 从二级当前解提取“新的接口压力向量” ----------
        p_level2_vec = self._iface_pressure_dict_to_vec(
            interface_nodes,
            name_to_idx,
            p_level2_dict,
            default_vec=p_vec
        )

        return F, q_inj_level2, p_level2_vec, all_level2_converged

    def solve_two_levels_iterative(
            self,
            sink_pressures: Dict[str, float] = None,
            tol: float = 1e-2,
            max_outer_iter: int = 200,
            p_init: float = 1e5,

            fd_eps: float = 1e-2,
            damping: float = 0.5,
    ) -> None:
        self.splitedNetwork()
        if not self.subNetworkModelLevel1 or not self.subNetworkModelLevel2:
            raise RuntimeError("请先调用 splitNetwork() 构建一、二级子网")

        if sink_pressures is None:
            sink_pressures = {}

        interface_nodes = sorted(self.interface_to_level1.keys())
        n_if = len(interface_nodes)
        if n_if == 0:
            raise RuntimeError("interface_to_level1 为空，无法做一二层耦合迭代")

        name_to_idx = {name: i for i, name in enumerate(interface_nodes)}

        # 第一次调用时初始化；之后再次调用可以沿用旧的解，作为 warm-start
        if self.interface_p_vec is None or self.interface_p_vec.size != n_if:
            self.interface_p_vec = np.full(n_if, p_init, dtype=float)

        converged = False

        for it in range(max_outer_iter):
            p_old = self.interface_p_vec.copy()  # 只是局部引用，方便下面写
            F, q_inj2, p_level2_vec, ok_level2 = self._eval_F(interface_nodes, name_to_idx, sink_pressures, p_old)
            normF = np.linalg.norm(F, ord=2)
            dp = p_level2_vec - p_old
            # J = np.zeros((n_if, n_if), dtype=float)
            # for j in range(n_if):
            #     p_perturb = p_old.copy()
            #     p_perturb[j] += fd_eps
            #     F_perturb, _, _, _ = self._eval_F(interface_nodes, name_to_idx, sink_pressures, p_perturb)
            #     J[:, j] = (F_perturb - F) / fd_eps
            # try:
            #     dp = np.linalg.solve(J, -F)
            # except np.linalg.LinAlgError:
            #     dp = np.linalg.lstsq(J, -F, rcond=None)[0]

            norm_dp = np.linalg.norm(dp, ord=2)

            print(f"[outer iter {it}] ||F|| = {normF:.6e}, ||dp|| = {norm_dp:.6e}")

            # 用二级算出来的接口压力，对外层接口压力做松弛更新
            self.interface_p_vec = p_old + damping * dp
            # 同时满足：流量耦合残差小 + 接口压力变化小
            if normF < tol and norm_dp < tol and ok_level2:
                converged = True
                print("[outer] 一级-二级协调收敛")
                break

        if not converged:
            print("[outer][WARN] 达到最大外层迭代次数，未完全协调收敛")
            # 用最终接口压力再做一次完整求解，作为最终一致解
        F_final, q_inj2_final, p_final_vec, ok_level2  = self._eval_F(interface_nodes, name_to_idx, sink_pressures, self.interface_p_vec)
        self.interface_p_vec = p_final_vec.copy()

        print("\n==== Coupled solution at interfaces ====")
        for name in interface_nodes:
            idx = name_to_idx[name]
            print(
                f"Interface {name}: p = {self.interface_p_vec[idx]:.3f}, "
                f"residual = {F_final[idx]:.6f}, "
                f"q_inj_from_level1 = {q_inj2_final[name]:.3f}"
            )



    def solve_original_network(
            self,
            sink_pressures: Dict[str, float] = None,
            boundary_conditions: Dict[str, Dict[str, float]] = None,
            injection: Dict[str, float] = None,
            tol: float = 1e-2,
            max_iter: int = 100,
            damping: float = 0.5,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """
        使用单层 Newton 法直接求解 originalNetworkModel。

        参数接口：
        - sink_pressures: 可选，覆盖模型中 sink 节点自带的压力边界
        - boundary_conditions: 可选，格式 {"pressure": {...}, "injection": {...}}
        - injection: 可选，节点外部注入量，正值为注入网络，负值为从网络抽取

        返回：
        - node_pressure: 全网节点压力
        - conn_flowrate: 全网连接边质量流量
        """
        sink_pressures = self._normalize_sink_pressures(sink_pressures)
        self.original_solver = OriginalNetworkNewtonSolver(self.originalNetworkModel)
        node_pressure, conn_flowrate = self.original_solver.solve(
            boundary_conditions=boundary_conditions,
            sink_pressures=sink_pressures,
            injection=injection,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
        )
        if not self.original_solver.last_converged:
            raise RuntimeError(
                "Original network Newton solver did not converge: "
                f"residual={self.original_solver.last_residual_norm}"
            )
        return node_pressure, conn_flowrate

    def solve_direct_network(
            self,
            sink_pressures: Dict[str, float] = None,
            boundary_conditions: Dict[str, Dict[str, float]] = None,
            injection: Dict[str, float] = None,
            tol: float = 1e-2,
            max_iter: int = 100,
            damping: float = 0.5,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """
        Directly solve the original full network with the Newton solver in pipSolver.
        This path does not split the network into level-1/level-2 subnetworks.
        """
        return self.solve_original_network(
            sink_pressures=sink_pressures,
            boundary_conditions=boundary_conditions,
            injection=injection,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
        )

    def solve(
            self,
            mode: str = "direct",
            sink_pressures: Dict[str, float] = None,
            boundary_conditions: Dict[str, Dict[str, float]] = None,
            injection: Dict[str, float] = None,
            tol: float = 1e-2,
            max_iter: int = 100,
            damping: float = 0.5,
    ):
        mode = (mode or "direct").strip().lower()
        sink_pressures = self._normalize_sink_pressures(sink_pressures)

        if mode == "two_levels":
            # 一、二层迭代耦合求解，给出二级终端的压力边界
            sink_p = sink_pressures or {"sink1": 7e5}  # 按需调整
            self.solve_two_levels_iterative(sink_pressures=sink_p)
            return None

        if mode not in ("direct", "single", "original", "newton"):
            raise ValueError(f"Unknown solve mode: {mode}")

        return self.solve_direct_network(
            sink_pressures=sink_pressures,
            boundary_conditions=boundary_conditions,
            injection=injection,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
        )
