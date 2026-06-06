import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import deque



class GraphOperation:
    """
    包含图操作常用函数
    """
    def add_edges_by_name(self,graph, edges_by_name):
        """
        通过节点名称给图添加边
        :param graph: 图实例对象
        :param edges_by_name: 包含节点名称的边集合
        :return:
        """
        name2idx = {v["name"]: v.index for v in graph.vs}
        edges_by_idx = [(name2idx[u], name2idx[v]) for u, v in edges_by_name]
        graph.add_edges(edges_by_idx)
        return graph

    def get_edges_idx_by_names(self,graph, edge_names):
        """
        通过边的节点名称对，获取边在图中的索引
        :param graph: 图实例对象
        :param edge_names: 边的节点名称对
        :return: 边在图中的索引
        """

        edge_idx_list = []
        name2idx = {v["name"]: v.index for v in graph.vs}
        # 遍历所有边名称对，匹配图中边的索引
        for u_name, v_name in edge_names:
            u_idx = name2idx[u_name]
            v_idx = name2idx[v_name]
            # 找到对应边的索引（支持有向边，u→v）
            edge_idx = graph.get_eid(u_idx, v_idx, directed=True, error=False)
            if edge_idx != -1:
                edge_idx_list.append(edge_idx)
        return edge_idx_list

    def getConvergingNodes(self,graph):
        '''
        获取有向图的汇合节点，度大于2
        :param graph:
        :return:
        '''
        # 校验输入图是否为有向图
        if not graph.is_directed():
            raise ValueError("输入图不是有向图，请先构建有向图。")
        # 校验节点是否有name属性
        if "name" not in graph.vs.attributes():
            raise ValueError("原始图节点缺少name属性，无法基于name索引进行子图分割。")

        # -------------------------- 步骤1：计算节点度并筛选初始核心节点（度>2） --------------------------
        # 构建节点name到总度、索引的映射（总度=入度+出度）
        node_info = {
            v["name"]: {
                "degree": graph.degree(v.index, mode="all"),
                "index": v.index
            }
            for v in graph.vs
        }
        # 初始核心节点：总度>2的节点（存储节点name和索引，不可删除）
        initial_core_names = set(name for name, info in node_info.items() if info["degree"] > 2)
        return initial_core_names

    def split_directed_graph_by_connected_core(self,directed_graph, degree_type="total", threshold=2,
                                                 return_edge_indices=False):
        """
        划分1级核心子图，支持返回子图包含的边索引
        :param return_edge_indices: 是否返回1级子图的所有边索引，默认False
        :return: 子图列表 + （可选）1级子图边索引集合
        """
        if not directed_graph.is_directed():
            print("输入图不是有向图，请先构建有向图。")
            return [] if not return_edge_indices else ([], set())

        # 1. 计算所有节点的度，筛选核心节点（名称集合）
        node_names = directed_graph.vs["name"]
        if degree_type == "in":
            node_degrees = {name: directed_graph.indegree(name) for name in node_names}
        elif degree_type == "out":
            node_degrees = {name: directed_graph.outdegree(name) for name in node_names}
        else:
            node_degrees = {name: directed_graph.indegree(name) + directed_graph.outdegree(name) for name in node_names}
        core_names = set([name for name, deg in node_degrees.items() if deg > threshold])
        if not core_names:
            print(f"无节点的{degree_type}度超过{threshold}，不进行划分。")
            return [] if not return_edge_indices else ([], set())

        # 2. 初始化：全局已使用节点集合、子图列表、1级子图边索引集合
        global_used_nodes = set()
        subgraphs = []
        level1_edge_indices = set()

        # 3. 遍历每个核心节点，生成子图
        for core_name in core_names:
            if core_name in global_used_nodes:
                continue

            # 3.1 BFS遍历连通节点（约束：遇其他核心节点断开 + 跳过已用节点）
            visited_node_names = set()
            queue = [core_name]
            visited_node_names.add(core_name)
            global_used_nodes.add(core_name)

            while queue:
                current_name = queue.pop(0)
                in_neighbors_idx = directed_graph.predecessors(current_name)
                out_neighbors_idx = directed_graph.successors(current_name)
                all_neighbors_names = directed_graph.vs[in_neighbors_idx + out_neighbors_idx]["name"]

                for neighbor_name in all_neighbors_names:
                    if neighbor_name in core_names and neighbor_name != core_name:
                        continue
                    if neighbor_name in global_used_nodes:
                        continue
                    if neighbor_name not in visited_node_names:
                        visited_node_names.add(neighbor_name)
                        global_used_nodes.add(neighbor_name)
                        queue.append(neighbor_name)

            # 3.2 提取子图并记录边索引
            subgraph = directed_graph.induced_subgraph(visited_node_names, implementation='create_from_scratch')
            if subgraph.ecount() == 0:
                continue

            # 3.3 记录1级子图的边索引（从原始图中获取）
            if return_edge_indices:
                # 提取子图的边名称对（u_name, v_name）
                subgraph_edge_names = [(subgraph.vs[e.source]["name"], subgraph.vs[e.target]["name"]) for e in
                                       subgraph.es]
                # 转换为原始图的边索引并加入集合
                subgraph_edge_indices = self.get_edges_idx_by_names(directed_graph, subgraph_edge_names)
                for idx in subgraph_edge_indices:
                    level1_edge_indices.add(idx)

            # 3.4 标记子图属性
            subgraph["core_node_name"] = core_name
            subgraph["degree_type"] = degree_type
            subgraph["used_nodes"] = list(visited_node_names)
            subgraph["level"] = 1  # 标记为1级子图
            subgraphs.append(subgraph)

        if return_edge_indices:
            return subgraphs, level1_edge_indices
        return subgraphs

    def split_remaining_edges_to_2nd_subgraphs(self,original_graph, level1_edge_indices):
        """
        基于原始图剩余边（删除1级子图边后），划分2级子图
        :param original_graph: 原始有向图
        :param level1_edge_indices: 1级子图包含的边索引集合
        :return: 2级子图列表
        """
        if not original_graph.is_directed():
            print("输入图不是有向图，无法划分2级子图。")
            return []

        # 1. 计算剩余边的索引（原始边索引 - 1级子图边索引）
        all_edge_indices = set(range(original_graph.ecount()))
        remaining_edge_indices = list(all_edge_indices - level1_edge_indices)
        if not remaining_edge_indices:
            print("删除1级子图边后，无剩余边，不划分2级子图。")
            return []

        # 2. 基于剩余边构建临时图（保留原始节点属性，仅包含剩余边）
        # 复制原始节点（保留名称、pressure等属性）
        temp_graph = ig.Graph(directed=True)
        temp_graph.add_vertices(original_graph.vs["name"])


        # 提取剩余边的（源节点索引，目标节点索引）并添加到临时图
        remaining_edges = [original_graph.es[idx].tuple for idx in remaining_edge_indices]
        temp_graph.add_edges(remaining_edges)


        # 3. 按弱连通性划分2级子图（有向图的弱连通：忽略方向视为无向图的连通）
        connected_components = temp_graph.decompose(mode="weak")
        level2_subgraphs = []

        for idx, component in enumerate(connected_components):
            # 仅保留有边的连通分量作为2级子图
            if component.ecount() == 0:
                continue

            # 标记2级子图属性
            component["level"] = 2  # 标记为2级子图
            component["component_idx"] = idx + 1  # 连通分量序号
            # 记录子图包含的节点名称和边（原始节点名称）
            component["used_nodes"] = component.vs["name"]
            component["used_edges"] = [(component.vs[e.source]["name"], component.vs[e.target]["name"]) for e in
                                       component.es]
            level2_subgraphs.append(component)

        return level2_subgraphs

    def split_directed_graph(self, original_graph):
        """
        将输入图进行分割划分为一级子图和二级子图，先根据核心节点划分一级子图，然后再划分二级子图
        :param original_graph: 原始图
        :return: 一级子图和二级子图
        """
        # 3. 划分1级子图（返回子图及边索引）
        level1_subgraphs, level1_edge_indices = self.split_directed_graph_by_connected_core(
            directed_graph=original_graph,
            degree_type="total",
            threshold=2,
            return_edge_indices=True
        )

        # 5. 划分2级子图（基于剩余边）
        level2_subgraphs = self.split_remaining_edges_to_2nd_subgraphs(
            original_graph=original_graph,
            level1_edge_indices=level1_edge_indices
        )
        return level1_subgraphs, level2_subgraphs

    def split_directed_graph_2to1(self, original_graph,lstFlowInNodes:list[str]):
        """
        节点索引使用name属性，边索引使用节点name对；
        参数：
        ---original_graph: 原始有向图（igraph对象，节点必须含name属性）
        ---lstFlowInNodes：仅流体流入的节点
        return: 一级子图列表和二级子图列表（仅一个）
        """
        # 校验输入图是否为有向图
        if not original_graph.is_directed():
            raise ValueError("输入图不是有向图，请先构建有向图。")
        # 校验节点是否有name属性
        if "name" not in original_graph.vs.attributes():
            raise ValueError("原始图节点缺少name属性，无法基于name索引进行子图分割。")

        # -------------------------- 步骤1：计算节点度并筛选初始核心节点（度>2） --------------------------
        initial_core_names = self.getConvergingNodes(original_graph)

        # if not initial_core_names or len(initial_core_names) == 1:
        if not initial_core_names:
            print("原始图中无总度>2的核心节点,或者只有一个节点的度大于2，不需要进行图分割，返回空列表")
            original_graph["convergingNode"]=initial_core_names
            # return [original_graph],[original_graph]
            return [],[]
        original_graph["convergingNode"] = initial_core_names

        # -------------------------- 步骤2：循环删除度≤1的非核心节点及关联边 --------------------------

        working_graph = original_graph.copy()
        # 构建工作图中节点索引到name的映射（方便后续查询）
        idx_to_name = {v.index: v["name"] for v in working_graph.vs}

        while True:
            # 计算当前工作图中所有节点的总度
            current_degrees = working_graph.degree(mode="all")  # 索引对应节点索引，值为度
            # 筛选当前需要删除的节点（度≤1 + 非核心节点）
            nodes_to_delete = []
            for node_idx in range(working_graph.vcount()):
                node_name = idx_to_name[node_idx]
                node_degree = current_degrees[node_idx]
                # 条件：度≤1 且 不是核心节点 → 需要删除
                if node_degree <= 1 and node_name not in initial_core_names and node_name not in lstFlowInNodes:
                    nodes_to_delete.append(node_idx)

            if not nodes_to_delete:
                break

            working_graph.delete_vertices(nodes_to_delete)
            # 更新 idx_to_name 映射（删除节点后索引会变化，需重新构建）
            idx_to_name = {v.index: v["name"] for v in working_graph.vs}

        # -------------------------- 步骤3：提取2级子图节点（工作图剩余节点的name集合） --------------------------
        level2_node_names = set(working_graph.vs["name"])

        # -------------------------- 步骤4：提取最终2级子图（基于原始图的节点和边） --------------------------
        # 基于节点name从原始图提取子图（保留原始节点和边的所有属性）
        level2_subgraph = original_graph.subgraph(level2_node_names, implementation="copy_and_delete")
        # 为2级子图添加元数据（均基于name）
        level2_subgraph["convergingNode"] = initial_core_names
        level2_subgraph["subGraphName"] = "level2SubNetwork"
        level2_subgraph["level"] = 2
        level2_subgraph["component_idx"] = 0
        level2_subgraph["used_nodes"] = level2_subgraph.vs["name"]  # 节点name列表
        level2_subgraph["used_edges"] = [
            (level2_subgraph.vs[e.source]["name"], level2_subgraph.vs[e.target]["name"])
            for e in level2_subgraph.es
        ]  # 边的name对列表

        # -------------------------- 步骤5：构建剩余图（仅移除2级子图的边，保留所有节点） --------------------------
        # 复制原始图（保留所有节点和边）
        temp_graph = original_graph.copy()
        # 获取2级子图的所有边（基于name对）
        level2_edge_name_pairs = set(level2_subgraph["used_edges"])
        # 收集temp_graph中需要删除的边索引（基于name对匹配）
        edges_to_delete = []
        for e in temp_graph.es:
            src_name = temp_graph.vs[e.source]["name"]
            tgt_name = temp_graph.vs[e.target]["name"]
            if (src_name, tgt_name) in level2_edge_name_pairs:
                edges_to_delete.append(e.index)
        # 移除2级子图的边（保留所有节点）
        if edges_to_delete:
            temp_graph.delete_edges(edges_to_delete)

        # -------------------------- 步骤6：分割1级子图（基于弱连通分量，保留所有节点） --------------------------
        level1_subgraphs = []
        # 按弱连通分量分割（忽略边方向）
        components = temp_graph.components(mode="weak")
        index =0
        for comp_idx, comp_node_ids in enumerate(components):
            # 提取分量子图（包含该连通分量的节点和剩余边）
            comp_subgraph = temp_graph.subgraph(comp_node_ids)
            # 过滤有效子图：边数>0（排除仅含孤立节点的分量）
            if comp_subgraph.ecount() > 0:

                comp_subgraph["level"] = 1
                comp_subgraph["subGraphName"] = "level1SubNetwork"+str(index)
                comp_subgraph["component_idx"] = comp_idx
                comp_subgraph["used_nodes"] = comp_subgraph.vs["name"]  # 节点name列表
                comp_subgraph["used_edges"] = [
                    (comp_subgraph.vs[e.source]["name"], comp_subgraph.vs[e.target]["name"])
                    for e in comp_subgraph.es
                ]  # 边的name对列表
                convergingNodeName=[]
                for nodeName  in comp_subgraph.vs["name"]:
                    if nodeName  in initial_core_names:
                        convergingNodeName.append(nodeName)
                comp_subgraph["convergingNode"] = convergingNodeName
                level1_subgraphs.append(comp_subgraph)
                index +=1

        return level1_subgraphs, [level2_subgraph]


    def plot_single_directed_subgraph(self,subgraph, save_path):
        """显示节点自定义名称，突出核心节点（保留原始属性，适配1/2级子图）"""
        # 字体设置（兼容中文）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        except:
            try:
                plt.rcParams['font.sans-serif'] = ['PingFang SC', 'DejaVu Sans']
            except:
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 布局选择：1级子图用放射状（突出核心），2级子图用Kamada-Kawai（适配普通连通分量）
        if subgraph["level"] == 1:
            layout = subgraph.layout("kk")
            # title_suffix = f"Core:{subgraph['core_node_name']}"
        else:
            layout = subgraph.layout("kk")
            title_suffix = f"Component:{subgraph['component_idx']}"
        layout = np.array(layout)

        fig, ax = plt.subplots(figsize=(10, 7))

        # 1. 绘制有向边（箭头在边中间，保留原始边的flow属性）
        for edge in subgraph.es:
            x1, y1 = layout[edge.source]  # 边的起点（源节点）
            x2, y2 = layout[edge.target]  # 边的终点（目标节点）

            # 计算边的中点（箭头绘制在此处）
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # 计算边的方向向量（单位化，确保箭头方向一致）
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx ** 2 + dy ** 2)  # 边的长度
            unit_dx = dx / length if length != 0 else 0  # x方向单位向量
            unit_dy = dy / length if length != 0 else 0  # y方向单位向量

            # 箭头参数：长度为原边的1/10（可调整），起点在中点
            arrow_length = length / 10
            arrow_start_x = mid_x - unit_dx * arrow_length / 2  # 箭头起点（中点略微左移，使箭头居中）
            arrow_start_y = mid_y - unit_dy * arrow_length / 2
            arrow_end_x = mid_x + unit_dx * arrow_length / 2  # 箭头终点
            arrow_end_y = mid_y + unit_dy * arrow_length / 2

            # 先绘制完整的边（无箭头的实线）
            ax.plot([x1, x2], [y1, y2], color="#666666", linewidth=1.2, zorder=1)

            # 再在边中间绘制箭头（箭头方向与边一致）
            ax.arrow(arrow_start_x, arrow_start_y,
                     arrow_end_x - arrow_start_x,
                     arrow_end_y - arrow_start_y,
                     head_width=0.04, head_length=0.04,
                     fc="#666666", ec="#666666", linewidth=1.2,
                     length_includes_head=True, zorder=1)

            # 可选：在箭头旁边显示flow属性（如果有）
            # if 'flow' in edge.attributes():
            #     flow_val = edge['flow']
            #     ax.text(mid_x + unit_dy * 0.1, mid_y - unit_dx * 0.1,
            #             f"Flow:{flow_val}", fontsize=8, zorder=2)

        # 2. 计算节点度（基于子图属性）
        degree_type = subgraph["degree_type"] if "degree_type" in subgraph.attributes() else "total"
        if degree_type == "in":
            sub_degrees = {v["name"]: subgraph.indegree(v["name"]) for v in subgraph.vs}
            degree_label = "In-Degree"
        elif degree_type == "out":
            sub_degrees = {v["name"]: subgraph.outdegree(v["name"]) for v in subgraph.vs}
            degree_label = "Out-Degree"
        else:
            sub_degrees = {v["name"]: subgraph.indegree(v["name"]) + subgraph.outdegree(v["name"]) for v in subgraph.vs}
            degree_label = "Total-Degree"

        # 3. 绘制节点（显示名称、度值、压力属性，区分1/2级子图）
        convergingNode=subgraph["convergingNode"]
        for i, (x, y) in enumerate(layout):
            node_name = subgraph.vs[i]["name"]
            node_deg = sub_degrees[node_name]

            if subgraph["level"] == 1 and node_name in convergingNode:
                # 1级子图核心节点（红色）
                color = "#E74C3C"
                size = 700
                label = f"{node_name}"
            elif subgraph["level"] == 1:
                # 1级子图普通节点（蓝色）
                color = "#3498DB"
                size = 500
                label = f"{node_name}"
            elif subgraph["level"] == 2 and node_name in convergingNode:
                # 2级子图节点（绿色）
                color = "#2ECC71"
                size = 500
                label = f"{node_name}"
            else:
                # 2级子图节点（绿色）
                color =  "#FFB90F"
                size = 500
                label = f"{node_name}"

            circle = plt.Circle((x, y), size / 10000, color=color, alpha=0.8, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)

        # 图表样式调整
        ax.set_xlim(layout[:, 0].min() - 0.8, layout[:, 0].max() + 0.8)
        ax.set_ylim(layout[:, 1].min() - 0.8, layout[:, 1].max() + 0.8)
        ax.set_aspect("equal")
        ax.axis("off")
        # plt.title(
        #     f"Level {subgraph['level']} Subgraph ({title_suffix}, Nodes:{len(subgraph.vs)}, Edges:{len(subgraph.es)})",
        #     fontsize=14, pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Level {subgraph['level']} Subgraph saved to: {save_path}")

    def plot_original_directed_graph(self,original_graph, save_path, degree_type="total"):
        """绘制完整有向图，显示原始节点属性"""
        # 字体设置
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        except:
            try:
                plt.rcParams['font.sans-serif'] = ['PingFang SC', 'DejaVu Sans']
            except:
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # Kamada-Kawai布局（适合全图展示）
        layout = original_graph.layout("kk")
        layout = np.array(layout)

        fig, ax = plt.subplots(figsize=(12, 8))

        # 1. 绘制有向边（箭头在边中间，保留原始边的flow属性）
        for edge in original_graph.es:
            x1, y1 = layout[edge.source]  # 边的起点（源节点）
            x2, y2 = layout[edge.target]  # 边的终点（目标节点）

            # 计算边的中点（箭头绘制在此处）
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # 计算边的方向向量（单位化，确保箭头方向一致）
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx ** 2 + dy ** 2)  # 边的长度
            unit_dx = dx / length if length != 0 else 0  # x方向单位向量
            unit_dy = dy / length if length != 0 else 0  # y方向单位向量

            # 箭头参数：长度为原边的1/10（可调整），起点在中点
            arrow_length = length / 10
            arrow_start_x = mid_x - unit_dx * arrow_length / 2  # 箭头起点（中点略微左移，使箭头居中）
            arrow_start_y = mid_y - unit_dy * arrow_length / 2
            arrow_end_x = mid_x + unit_dx * arrow_length / 2  # 箭头终点
            arrow_end_y = mid_y + unit_dy * arrow_length / 2

            # 先绘制完整的边（无箭头的实线）
            ax.plot([x1, x2], [y1, y2], color="#666666", linewidth=1.2, zorder=1)

            # 再在边中间绘制箭头（箭头方向与边一致）
            ax.arrow(arrow_start_x, arrow_start_y,
                     arrow_end_x - arrow_start_x,
                     arrow_end_y - arrow_start_y,
                     head_width=0.04, head_length=0.04,
                     fc="#666666", ec="#666666", linewidth=1.2,
                     length_includes_head=True, zorder=1)

            # 可选：在箭头旁边显示flow属性（如果有）
            # if 'flow' in edge.attributes():
            #     flow_val = edge['flow']
            #     ax.text(mid_x + unit_dy * 0.1, mid_y - unit_dx * 0.1,
            #             f"Flow:{flow_val}", fontsize=8, zorder=2)

        # 2. 计算节点度，标记核心节点（用于1级子图筛选的核心节点）
        if degree_type == "in":
            node_degrees = {v["name"]: original_graph.indegree(v["name"]) for v in original_graph.vs}
            degree_label = "In-Degree"
        elif degree_type == "out":
            node_degrees = {v["name"]: original_graph.outdegree(v["name"]) for v in original_graph.vs}
            degree_label = "Out-Degree"
        else:
            node_degrees = {v["name"]: original_graph.indegree(v["name"]) + original_graph.outdegree(v["name"]) for v in
                            original_graph.vs}
            degree_label = "Total-Degree"
        core_names = [name for name, deg in node_degrees.items() if deg > 2]

        # 3. 绘制节点
        for i, (x, y) in enumerate(layout):
            node_name = original_graph.vs[i]["name"]
            node_deg = node_degrees[node_name]

            color = "#E74C3C" if node_name in core_names else "#95A5A6"
            size = 600
            label = f"{node_name}"

            circle = plt.Circle((x, y), size / 10000, color=color, alpha=0.8, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)

        # 图表样式调整
        ax.set_xlim(layout[:, 0].min() - 1.0, layout[:, 0].max() + 1.0)
        ax.set_ylim(layout[:, 1].min() - 1.0, layout[:, 1].max() + 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.title(
            f"Original Graph (Nodes:{original_graph.vcount()}, Edges:{original_graph.ecount()}, Cores:{len(core_names)})",
            fontsize=14, pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Original graph saved to: {save_path}")

    def get_node_neighbors(self,node_list, edge_set):
        """
        从顶点列表和边集合中，确定每个节点的邻居节点

        参数：
            node_list (list): 图的顶点列表（节点名称列表），如 ['S', 'A', 'B', 'C', 'D', 'E']
            edge_set (set/tuple/list): 图的边集合（无向边），每条边为二元组，如 {(S,A), (S,C), (A,B)}

        返回：
            dict: 键为节点，值为该节点的邻居列表（去重、有序）
        """
        # 初始化邻居字典：每个节点对应空列表
        neighbors = {node: [] for node in node_list}

        # 遍历所有边，为节点添加邻居
        for edge in edge_set:
            # 确保每条边是二元组（处理无向边）
            if len(edge) != 2:
                raise ValueError(f"无效的边格式：{edge}，必须是包含两个节点的二元组")
            u, v = edge

            # 检查节点是否在顶点列表中（避免无效节点）
            if u not in node_list or v not in node_list:
                raise ValueError(f"边 {edge} 中的节点不在顶点列表中")

            # 为 u 添加邻居 v（避免重复）
            if v not in neighbors[u]:
                neighbors[u].append(v)
            # 为 v 添加邻居 u（无向边，双向互为邻居）
            if u not in neighbors[v]:
                neighbors[v].append(u)

        # 可选：对邻居列表排序，保证输出有序（便于后续处理）
        for node in neighbors:
            neighbors[node].sort()


        return neighbors

    def topological_sort_optimized(self, undirected_graph, outflow_nodes, inflow_nodes):
        """优化拓扑排序：确保所有节点层级严格递增（源→汇方向），支持所有边保留"""
        # 初始化层级：源节点层级0，其他节点设为无穷大
        level = {node: float('inf') for node in undirected_graph}
        queue = deque()

        #所有源节点到所有汇节点的最短路径，键为源节点，值为路径
        shortestPath = {}
        for source in outflow_nodes:
            shortest_paths_undir = self.bfs_shortest_path_undirected(undirected_graph,source,inflow_nodes)
            if not shortest_paths_undir:
                print(f"警告：源节点 {source} 在无向图中无法到达任何汇节点")
                continue
            shortestPath[source] = list(shortest_paths_undir.values())[0]
            for dest in shortest_paths_undir.values():
                if len(dest) < len(shortestPath[source]):
                    shortestPath[source] = dest

        sorted_items = sorted(shortestPath.items(), key=lambda x: len(x[1]), reverse=True)
        for key, value in sorted_items:
            idx = 0
            for nodeName in value:
                if level[nodeName] == float('inf'):
                    level[nodeName] = idx
                else:
                    if level[nodeName] <= idx:
                        level[nodeName] = idx
                    else:
                        idx = level[nodeName]
                idx += 1


        # 源节点层级设为0，入队
        for source in outflow_nodes:
            if source in undirected_graph:
                level[source] = 0
                queue.append(source)

        # 多轮BFS：确保每个节点的层级是"源到该节点的最短距离"（严格递增）
        while queue:
            current = queue.popleft()
            for neighbor in undirected_graph[current]:
                # 邻居层级 = min(当前层级, 当前节点层级+1)
                if level[neighbor] == float('inf'):
                    level[neighbor] = level[current] + 1
                    queue.append(neighbor)

        # 处理汇节点：确保汇节点层级是全局最大值（避免汇节点被中间节点指向）
        max_level = max(level.values()) if level else 0
        for sink in inflow_nodes:
            if sink in undirected_graph:
                level[sink] = max_level + 1

        # 处理孤立节点（无连接的节点）：设为源节点同级（避免层级异常）
        for node in undirected_graph:
            if level[node] == float('inf'):
                level[node] = 0
                print(f"警告：节点 {node} 是孤立节点（无连接），层级设为0")

        # 按层级排序节点（拓扑序）
        sorted_nodes = sorted(undirected_graph.keys(), key=lambda x: level[x])
        return sorted_nodes, level

    def verify_acyclic(self, directed_graph):
        """验证有向图是否无环（兜底检查）"""
        visited = set()
        rec_stack = set()
        has_cycle = False

        def dfs(node):
            nonlocal has_cycle
            if has_cycle:
                return
            visited.add(node)
            rec_stack.add(node)
            for neighbor in directed_graph[node]:
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    has_cycle = True
                    print(f"警告：检测到环路！涉及节点 {neighbor} -> ... -> {node} -> {neighbor}")
            rec_stack.remove(node)

        for node in directed_graph:
            if node not in visited and not has_cycle:
                dfs(node)

        if not has_cycle:
            print("    验证通过：有向图无环路 ✔️")
        else:
            raise RuntimeError("严重错误：有向图存在环路，且无法通过移除边修正（违反保留所有边约束）")
        return not has_cycle

    def bfs_shortest_path_undirected(self,undirected_graph, start, ends):
        """
        BFS计算无向图中起点到多个终点的最短路径（无权图）
        Args:
            undirected_graph (dict): 无向图 {节点: [邻居列表]}
            start (str): 起点（源节点）
            ends (list): 终点列表（汇节点）
        Returns:
            dict: {终点: 最短路径列表}，如 {'E': ['S', 'C', 'D', 'E']}
        """
        shortest_paths = {}
        # 初始化：每个节点的前驱节点
        predecessors = {node: None for node in undirected_graph.keys()}
        visited = set([start])
        queue = deque([start])

        # BFS遍历
        while queue:
            current = queue.popleft()
            # 找到所有终点的最短路径后可提前终止
            if all(end in shortest_paths for end in ends):
                break
            for neighbor in undirected_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    predecessors[neighbor] = current
                    queue.append(neighbor)
                    # 若当前邻居是终点，回溯生成路径
                    if neighbor in ends and neighbor not in shortest_paths:
                        path = []
                        temp = neighbor
                        while temp is not None:
                            path.append(temp)
                            temp = predecessors[temp]
                        shortest_paths[neighbor] = path[::-1]  # 反转得到 起点→终点 路径
        return shortest_paths

    def bfs_shortest_path_directed(self, directed_graph, start, targets):
        """【有向图】BFS计算起点到多个目标的最短路径"""
        shortest_paths = {}
        visited = {start: None}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            if current in targets:
                # 回溯路径
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = visited[node]
                shortest_paths[current] = path[::-1]
                if len(shortest_paths) == len(targets):
                    break  # 找到所有目标则提前退出

            # 仅遍历有向边的邻居（区别于无向图）
            for neighbor in directed_graph.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
        return shortest_paths

    def adjust_unreachable_source(self, directed_graph, undirected_graph, node_level, source, inflow_nodes):
        """
        调整无法到达汇节点的源节点（解决中间路径方向错误问题）
        逻辑：
        1. 基于无向图找到源→汇的最短路径
        2. 遍历路径中所有边，检查有向方向是否与路径方向一致
        3. 对方向相反且拓扑序允许的边，调整方向（无环）
        4. 保留所有边，确保调整后无环
        """
        print(f"  开始调整源节点 {source} 的可达性（含中间路径）...")

        # 步骤1：基于无向图找到源→汇的最短路径（作为调整参考）
        shortest_paths_undir = self.bfs_shortest_path_undirected(undirected_graph, source, inflow_nodes)
        if not shortest_paths_undir:
            print(f"    无向图中源节点 {source} 本身无法到达任何汇节点，无法调整")
            return False

        # 选最短的一条路径作为调整基准
        sink = min(shortest_paths_undir.keys(), key=lambda x: len(shortest_paths_undir[x]))
        ref_path = shortest_paths_undir[sink]
        print(f"    参考无向最短路径：{' -> '.join(ref_path)}")

        # 步骤2：遍历参考路径中的每一条边，检查并调整方向
        adjusted_edges = []
        for i in range(len(ref_path) - 1):
            u = ref_path[i]  # 路径中的前一个节点
            v = ref_path[i + 1]  # 路径中的后一个节点

            # 检查当前有向边方向：是否是 u→v（路径期望方向）
            current_is_correct = v in directed_graph.get(u, [])
            if current_is_correct:
                continue  # 方向正确，无需调整

            # 检查是否是反向边 v→u
            is_reverse = u in directed_graph.get(v, [])
            if not is_reverse:
                print(f"    路径边 {u}-{v} 未定向（理论上不会发生），按拓扑序定向")
                # 按拓扑序定向
                if node_level[u] < node_level[v]:
                    directed_graph[u].append(v)
                    adjusted_edges.append((u, v))
                else:
                    directed_graph[v].append(u)
                    adjusted_edges.append((v, u))
                continue

            # 步骤3：反向边，检查是否可调整为 u→v（拓扑序允许+无环）
            # 条件1：拓扑序允许（u层级 < v层级）
            if node_level[u] >= node_level[v]:
                print(f"    反向边 {v}→{u} 无法调整（u层级 {node_level[u]} ≥ v层级 {node_level[v]}），跳过")
                continue

            # 条件2：调整后无环
            # 临时移除反向边，添加正向边，检测是否有环
            directed_graph[v].remove(u)  # 移除反向边
            directed_graph[u].append(v)  # 添加正向边

            # 检测环
            if self.verify_acyclic(directed_graph):
                adjusted_edges.append((u, v))
                print(f"    调整路径边方向：{v}→{u} → {u}→{v}（拓扑序允许+无环）")
            else:
                # 有环，恢复反向边
                directed_graph[u].remove(v)
                directed_graph[v].append(u)
                print(f"    调整路径边 {v}→{u} 会形成环，放弃调整")

        # 步骤3：调整后验证可达性
        if adjusted_edges:
            print(f"    共调整 {len(adjusted_edges)} 条路径边：{adjusted_edges}")
            new_paths = self.bfs_shortest_path_directed(directed_graph, source, inflow_nodes)
            if new_paths:
                print(f"    源节点 {source} 调整后可到达汇节点：{list(new_paths.keys())}")
                return True
            else:
                # 调整后仍不可达，尝试恢复调整的边（兜底）
                print(f"    调整后仍不可达，恢复调整的边")
                for u, v in adjusted_edges:
                    directed_graph[u].remove(v)
                    directed_graph[v].append(u)
                return False
        else:
            print(f"    参考路径中无需要调整的边，可达性问题非路径方向导致")
            return False

    def calc_in_out_degree(self,directed_graph):
        """计算入度和出度，入参为字典，key为节点名称，值为节点流出边连接的邻居节点"""
        in_degree = {node: 0 for node in directed_graph.keys()}
        out_degree = {node: len(neighbors) for node, neighbors in directed_graph.items()}
        for start, ends in directed_graph.items():
            for end in ends:
                in_degree[end] += 1
        return in_degree, out_degree


    def fix_dead_end(self, directed_graph, undirected_graph, sources, sinks):
        """修正死胡同节点
        :param directed_graph: 有向图，key为节点名称，values为当前节点流向的节点
        :param undirected_graph: 无向图，key为节点名称，values为邻居节点
        :param sources: 所有的源节点名称
        :param sinks: 所有的汇节点名称
        """
        in_degree, out_degree = self.calc_in_out_degree(directed_graph)
        fixed = False
        for node in undirected_graph.keys():
            if node in sources or node in sinks:
                continue
            if in_degree[node] == 0 or out_degree[node] == 0:
                print(f"  修正死胡同节点 {node} (入度={in_degree[node]}, 出度={out_degree[node]})")
                fixed = True
                # 遍历邻居找未处理边
                for neighbor in undirected_graph[node]:
                    edge_key = (node, neighbor) if node < neighbor else (neighbor, node)
                    has_out = neighbor in directed_graph[node]
                    has_in = node in directed_graph[neighbor]
                    if not has_out and not has_in:
                        if in_degree[node] == 0:
                            directed_graph[neighbor].append(node)
                            print(f"    边 {neighbor} -> {node} (补充{node}流入)")
                        else:
                            directed_graph[node].append(neighbor)
                            print(f"    边 {node} -> {neighbor} (补充{node}流出)")
                        # 重新计算度数
                        in_degree, out_degree = self.calc_in_out_degree(directed_graph)
                        break
        if not fixed:
            print("  无死胡同节点需要修正")

    def fix_dead_end_2(self, directed_graph, undirected_graph, node_level, inflow_nodes):
        """修正死胡同节点（保留所有边+无环）"""
        in_degree, out_degree = self.calc_in_out_degree(directed_graph)
        # 死胡同：非汇节点且出度为0
        dead_ends = [n for n in directed_graph if out_degree[n] == 0 and n not in inflow_nodes]

        for node in dead_ends:
            # 遍历所有未定向的邻居（必须保留边，按拓扑序定向）
            for neighbor in undirected_graph[node]:
                # 检查该边是否已定向
                edge_exists = neighbor in directed_graph[node] or node in directed_graph[neighbor]
                if not edge_exists:
                    # 按拓扑序定向（低层级→高层级）
                    if node_level[node] < node_level[neighbor]:
                        directed_graph[node].append(neighbor)
                        print(f"    修正死胡同 {node}：添加边 {node} -> {neighbor} (拓扑序定向)")
                    else:
                        directed_graph[neighbor].append(node)
                        print(f"    修正死胡同 {node}：添加边 {neighbor} -> {node} (拓扑序定向)")
                    break


if __name__ == "__main__":

    ############################
    g = ig.Graph(directed=True)
    g.add_vertices(["A", "B", "C", "D"])
    # 1. 先添加边（用顶点名称或索引）
    g.add_edges([("B", "A"), ("B", "C"), ("C", "D")])

    ############################

    graphOper=GraphOperation()
    # 目标文件夹路径
    savePath = "./segGraphResult"  # 可替换为你的文件夹路径
    directed_test_graph = ig.Graph(directed=True)
    nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
    directed_test_graph.add_vertices(nodes)
    # 节点属性：压力
    directed_test_graph.vs["pressure"] = [1.2, 1.5, 1.1, 0.9, 1.3, 1.4, 0.8, 1.0, 1.6, 0.7, 1.2, 0.9, 1.1]

    edges = [
        ('B','A'), ('A','E'), ('A','D'), ('A','C'),
        ('B','M'), ('M','G'), ('G','F'), ('H','F'),
        ('F','I'), ('L','K'), ('K','J'), ('J','B')
    ]
    graphOper.add_edges_by_name(directed_test_graph, edges)
    # 边属性：流量（14条边对应14个流量值）
    directed_test_graph.es["flow"] = [2.3, 1.8, 2.1, 1.5, 3.0, 2.7, 1.9, 2.4, 2.0, 1.7, 2.2, 2.5, 2.8, 1.6]

    # 2. 绘制完整原图
    graphOper.plot_original_directed_graph(
        original_graph=directed_test_graph,
        save_path=savePath+"/original_directed_graph.png",
        degree_type="total"
    )


    level1_subgraphs,level2_subgraphs=graphOper.split_directed_graph(directed_test_graph)


    # 4. 可视化1级子图
    if level1_subgraphs:
        for idx, subgraph in enumerate(level1_subgraphs, 1):
            graphOper.plot_single_directed_subgraph(subgraph, savePath+f"/level1_subgraph_{idx}.png")
            print(f"\nLevel 1 Subgraph {idx} - Core:{subgraph['core_node_name']}")
            print(f"Used nodes: {subgraph['used_nodes']}")
            print(f"Edges: {[(subgraph.vs[e.source]['name'], subgraph.vs[e.target]['name']) for e in subgraph.es]}")
    else:
        print("No valid Level 1 Subgraphs.")


    # 6. 可视化2级子图
    if level2_subgraphs:
        for idx, subgraph in enumerate(level2_subgraphs, 1):
            graphOper.plot_single_directed_subgraph(subgraph, savePath+f"/level2_subgraph_{idx}.png")
            print(f"\nLevel 2 Subgraph {idx} - Component:{subgraph['component_idx']}")
            print(f"Used nodes: {subgraph['used_nodes']}")
            print(f"Edges: {subgraph['used_edges']}")
    else:
        print("No valid Level 2 Subgraphs.")