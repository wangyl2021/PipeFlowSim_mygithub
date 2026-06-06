from ..Components.networkNode import *
from ..Components.networkConnections import *
from Fluid.FluidMix import *
import igraph as ig
from .graphPartitionComplete import GraphOperation
from enum import Enum
import numpy as np
from collections import deque


class NetworkLevel(Enum):
    ORIGINAL = 0 #未划分网络
    LEVEL_1 = 1 #一级子网络
    LEVEL_2 = 2 #二级子网络

class NetworkModel:
    """
    管网模型
    """
    def __init__(self,name:str, networkNodes:list[NetworkNode],networkConnections:list[Connection],enNetworkLevel:NetworkLevel=NetworkLevel.ORIGINAL):
        """
        初始化管网模型
        :param networkNodes:管网节点信息
        :param networkConnections: 管网连接信息
        """
        self.name=name
        '''网络名称'''
        self.networkNodesDict:dict[str,NetworkNode]={}
        '''节点信息字典，键为节点名称，值为节点对象'''
        self.networkConnDict:dict[str,Connection]={}
        '''边信息字典，键为边名称，值为边对象'''
        self.networkNodesDict, self.networkConnDict = self.initDict(networkNodes,networkConnections)
        '''节点和边的字典'''
        self.enNetworkLevel = enNetworkLevel
        '''管网划分的等级'''
        self.dicBoundaryCondition:dict[str,BoundaryCondition]=dict()
        '''边界条件字典'''
        self.dicRateConstraint:dict[str,RateConstraint]=dict()
        '''边界约束字典'''
        #初始化管网流体流向,仅原始网络需要，分割后的网络不需要
        self.initFlowByShortestPath()
        self.networkGraph = self.updateGraph()
        '''管网拓扑结构'''
        self.neighborNodesEdges: dict[NetworkNode, dict[str,NetworkNode]]=self.updateNeighborNodesEdges()
        '''存放各节点的流入节点、流出节点、流入边以及流出边列表，{节点实例对象:{'inflowNode','outflowNode','inflowConn','outflowConn'}} '''
        self.nodesCalOrder:list[NetworkNode] = self.getNodesSortByDirection()
        """ 按流向存放从起始节点到终止节点的名称列表,用于迭代计算顺序索引"""
        self.setConvergingNode()
        '''设置汇合节点'''





    def initDict(self,networkNodes:list[NetworkNode],networkConnections:list[Connection]):
        """
        初始化节点和边的字典列表，方便后续通过名称查询
        """
        networkNodesDict={}
        networkConnDict={}
        for node in networkNodes:
            networkNodesDict[node.name]=node
        for conn in networkConnections:
            networkConnDict[conn.name]=conn
        return networkNodesDict,networkConnDict

    def setBoundaryConstraint(self,boundary:list[dict],constraint:list[dict]):
        '''根据边界和约束条件列表设置管网边界约束字典'''
        for dicBoundary in boundary:
            bc= BoundaryCondition(dicBoundary)
            self.dicBoundaryCondition[dicBoundary['name']]=bc

        for dicConstraint in constraint:
            rc = RateConstraint(dicConstraint)
            self.dicRateConstraint[dicConstraint['name']]=rc



    def updateGraph(self):
        """
        通过管网模型初始化图实例对象
        :return: 图实例对象
        """
        graphOper= GraphOperation()
        graphNodes=[]
        graphEdges=[]
        graphEdgesName=[]

        for nodeName,node in self.networkNodesDict.items():
            graphNodes.append(nodeName)

        for connName,conn in self.networkConnDict.items():
            graphEdgesName.append(connName)
            #使用流体流向初始化拓扑图结构
            graphEdges.append(conn.flowDirection)


        directedGraph = ig.Graph(directed=True)

        directedGraph.add_vertices(graphNodes)
        directedGraph.add_edges(graphEdges, attributes={"name": graphEdgesName})


        initial_core_names =graphOper.getConvergingNodes(directedGraph)
        directedGraph["convergingNode"] = initial_core_names

        # print(directedGraph.vs["name"])
        # print(directedGraph.es["name"])
        return directedGraph

    def initFlowDirection(self):
        """根据源节点、末节点等节点特性，初始流体流向，方便后续计算，除了边界节点的其他节点流向应满足同时包含流入边和流出边
        此函数会改变管道对象中的flowDirection流向属性
        """

        outflowNodes=[]
        inflowNodes=[]
        queue = deque()
        activated_nodes = set()
        nodesName=[]


        for nodeName, node in self.networkNodesDict.items():
            nodesName.append(nodeName)
            if node.type == NodeType.SOURCE or node.type == NodeType.WELL:
                outflowNodes.append(nodeName)
            elif node.type == NodeType.SINK :
                inflowNodes.append(nodeName)
            else:
                queue.append(nodeName)

        edges=[]
        for edge in self.networkConnDict.values():
            edges.append(edge.nodes)

        graphOper = GraphOperation()
        nodeNeighbors = graphOper.get_node_neighbors(nodesName,edges)

        # 1. 初始化有向图的存储结构
        #    key: 起始节点, value: 所有流出的邻居节点列表
        directed_graph = {node: [] for node in nodeNeighbors.keys()}

        # 处理源节点 (流体流出)
        for source in outflowNodes:
            if source not in nodeNeighbors:
                print(f"警告: 源节点 {source} 不在管网中。")
                continue
            for neighbor in nodeNeighbors[source]:
                # 将边的方向设为从源节点流出
                directed_graph[source].append(neighbor)
                print(f"  边 {source} -> {neighbor} (源节点流出)")

                # 激活邻居节点
                if neighbor not in activated_nodes:
                    activated_nodes.add(neighbor)
                    queue.append(neighbor)
            activated_nodes.add(source)

        # 处理汇节点 (流体流入)
        for sink in inflowNodes:
            if sink not in nodeNeighbors:
                print(f"警告: 汇节点 {sink} 不在管网中。")
                continue
            for neighbor in nodeNeighbors[sink]:
                # 将边的方向设为流入汇节点
                directed_graph[neighbor].append(sink)
                print(f"  边 {neighbor} -> {sink} (汇节点流入)")

                # 激活邻居节点
                if neighbor not in activated_nodes:
                    activated_nodes.add(neighbor)
                    queue.append(neighbor)
            activated_nodes.add(sink)

        print("\n--- 步骤 2: BFS 递推确定方向 ---")
        # BFS 循环
        while queue:
            current_node = queue.popleft()
            print(f"  处理激活节点: {current_node}")

            for neighbor in nodeNeighbors[current_node]:
                # 检查边是否已存在于有向图中 (任一方向)
                edge_exists = (neighbor in directed_graph[current_node]) or (current_node in directed_graph[neighbor])

                if not edge_exists:
                    # 边不存在，根据规则确定方向：从当前激活节点流向邻居
                    directed_graph[current_node].append(neighbor)
                    print(f"    边 {current_node} -> {neighbor} (BFS 递推)")

                    # 如果邻居未被激活，则激活它并加入队列
                    if neighbor not in activated_nodes:
                        activated_nodes.add(neighbor)
                        queue.append(neighbor)

        print("\n--- 步骤 3: 处理剩余边 (环) ---")
        # 遍历所有原始边，检查是否有遗漏
        # 为了避免重复检查 (A,B 和 B,A)，我们规定只检查 u < v 的边
        all_nodes = list(nodeNeighbors.keys())
        for i in range(len(all_nodes)):
            u = all_nodes[i]
            for j in range(i + 1, len(all_nodes)):
                v = all_nodes[j]
                # 检查 u 和 v 是否是邻居
                if v in nodeNeighbors[u]:
                    # 检查边是否已存在于有向图中 (任一方向)
                    edge_exists = (v in directed_graph[u]) or (u in directed_graph[v])

                    if not edge_exists:
                        # 边未被处理，赋予一个默认方向
                        # 规则：从节点ID较小的指向较大的
                        default_from, default_to = (u, v) if u < v else (v, u)
                        directed_graph[default_from].append(default_to)
                        print(f"    边 {default_from} -> {default_to} (默认假设，处理环)")

        graphOper.fix_dead_end(directed_graph,nodeNeighbors,outflowNodes,inflowNodes)

        #更新边的流向信息
        for startNodeName, endNodesNameLst in directed_graph.items():
            for endNodeName in endNodesNameLst:
                for edge in self.networkConnDict.values():
                    if startNodeName in edge.nodes and endNodeName in edge.nodes:
                        edge.flowDirection = (startNodeName,endNodeName)

        print("\n--- 流向初始化完成 ---")

        return directed_graph

    def initFlowByShortestPath(self):
        """
        根据所有源节点至少能到一个末节点的最短连通路径初始化流体方向
        最终版本：
        1. 保留所有边 + 无环路
        2. 基于有向图验证源→汇最短路径可达性
        3. 调整中间路径边方向解决不可达问题（非仅源节点直连边）
        """

        if self.enNetworkLevel!=NetworkLevel.ORIGINAL:
            print('子网不需要初始化流体流向！')
            return

        outflowNodes = []
        inflowNodes = []
        nodesName = []

        print(f'init {self.name} fluid direction ')
        # 分类节点：源/汇/中间节点
        for nodeName, node in self.networkNodesDict.items():
            nodesName.append(nodeName)
            if node.type == NodeType.SOURCE or node.type == NodeType.WELL:
                outflowNodes.append(nodeName)
            elif node.type == NodeType.SINK:
                inflowNodes.append(nodeName)

        # 提取所有边（无向，必须全部保留）
        edges = []
        for edge in self.networkConnDict.values():
            edges.append(edge.nodes)
        print(f"原始无向边数量：{len(edges)}（将全部保留）")

        # 步骤1：构建无向图邻居字典
        graphOper = GraphOperation()
        undirected_graph = graphOper.get_node_neighbors(nodesName, edges)

        # 步骤2：优化拓扑排序（核心：为所有节点分配严格递增的层级，确保无环）
        print("\n--- 步骤1：优化拓扑排序（分配节点层级） ---")
        sorted_nodes, node_level = graphOper.topological_sort_optimized(
            undirected_graph, outflowNodes, inflowNodes
        )
        # print(f"  拓扑排序结果：{' -> '.join(sorted_nodes)}")
        # print(f"  节点层级分配：{node_level}")

        # 步骤3：初始化有向图（确保所有边都被保留）
        directed_graph = {node: [] for node in nodesName}
        processed_edges = set()  # 存储 (u, v) 且 u < v，统一无向边标识

        # 步骤4：沿无向图最短路径+拓扑序赋值核心流向（保留边）
        print("\n--- 步骤2：沿无向图源→汇最短路径+拓扑序赋值核心流向 ---")
        for source in outflowNodes:
            if source not in undirected_graph:
                print(f"警告：源节点 {source} 不在图中，跳过")
                continue
            # 基于无向图计算最短路径（初始参考）
            shortest_paths_undir = graphOper.bfs_shortest_path_undirected(undirected_graph, source, inflowNodes)
            if not shortest_paths_undir:
                print(f"警告：源节点 {source} 在无向图中无法到达任何汇节点")
                continue
            # 遍历每条最短路径，按拓扑序定向（确保无环）
            for sink, path in shortest_paths_undir.items():
                print(f"  源 {source} 到汇 {sink} 的无向最短路径：{' -> '.join(path)}")
                # 路径中相邻节点按拓扑序定向（低层级→高层级）
                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i + 1]
                    edge_key = (u, v) if u < v else (v, u)
                    if edge_key not in processed_edges:
                        # 严格按层级定向：低层级→高层级
                        if node_level[u] < node_level[v]:
                            directed_graph[u].append(v)
                            print(f"    边 {u} -> {v} (最短路径+拓扑序)")
                        else:
                            directed_graph[v].append(u)
                            print(f"    边 {v} -> {u} (最短路径+拓扑序，路径方向调整)")
                        processed_edges.add(edge_key)

        # 步骤5：处理剩余未定向边（必须保留，按拓扑序定向）
        print("\n--- 步骤3：处理剩余边（保留所有边+拓扑序定向） ---")
        for u, v in edges:
            edge_key = (u, v) if u < v else (v, u)
            if edge_key not in processed_edges:
                # 严格按层级定向：低层级→高层级（根本无环）
                if node_level[u] < node_level[v]:
                    directed_graph[u].append(v)
                    print(f"    边 {u} -> {v} (拓扑序定向)")
                elif node_level[v] < node_level[u]:
                    directed_graph[v].append(u)
                    print(f"    边 {v} -> {u} (拓扑序定向)")
                else:
                    # 同层级节点：按节点名排序定向（避免环，同层级无父子关系）
                    from_node, to_node = (u, v) if u < v else (v, u)
                    directed_graph[from_node].append(to_node)
                    print(f"    边 {from_node} -> {to_node} (同层级按名称排序定向)")
                processed_edges.add(edge_key)

        # 验证所有边都已处理（确保无遗漏）
        assert len(processed_edges) == len(edges), f"错误：原始边 {len(edges)} 条，处理边 {len(processed_edges)} 条，存在遗漏"
        print(f"\n  所有 {len(edges)} 条边均已保留并定向 ✔️")

        # 步骤6：修正死胡同节点（保留边+无环）
        print("\n--- 步骤4：修正死胡同节点（保证源→汇可达） ---")
        graphOper.fix_dead_end_2(directed_graph, undirected_graph, node_level,  inflowNodes)

        # 步骤7：最终无环验证（兜底）
        print("\n--- 步骤5：最终无环验证 ---")
        graphOper.verify_acyclic(directed_graph)

        # 步骤8：基于有向图验证源→汇可达性（核心修正）
        print("\n--- 步骤6：基于有向图验证源→汇最短路径可达性 ---")
        unreachable_sources = []
        for source in outflowNodes:
            if source not in directed_graph:
                unreachable_sources.append(source)
                continue
            # 基于有向图计算最短路径（关键：用有向图判断）
            shortest_paths_dir = graphOper.bfs_shortest_path_directed(directed_graph, source, inflowNodes)
            if shortest_paths_dir:
                print(f"  源节点 {source} (有向图) 可到达汇节点：{list(shortest_paths_dir.keys())}")
                # 打印有向图下的最短路径
                for sink, path in shortest_paths_dir.items():
                    print(f"    最短路径：{' -> '.join(path)}")
            else:
                print(f"  警告：源节点 {source} (有向图) 无法到达任何汇节点")
                unreachable_sources.append(source)

        # 步骤9：自动调整不可达的源节点（调整中间路径边方向）
        if unreachable_sources:
            print(f"\n--- 步骤7：自动调整不可达源节点 ({unreachable_sources}) ---")
            for source in unreachable_sources:
                if source not in undirected_graph:
                    continue
                # 尝试调整中间路径边方向恢复可达性
                success = graphOper.adjust_unreachable_source(
                    directed_graph, undirected_graph, node_level, source, inflowNodes
                )
                if not success:
                    print(f"  错误：源节点 {source} 调整后仍无法到达汇节点（无向图本身不连通/拓扑序限制）")

        # 步骤10：最终有向图可达性二次验证
        print("\n--- 步骤8：最终有向图可达性二次验证 ---")
        for source in outflowNodes:
            if source not in directed_graph:
                continue
            shortest_paths_final = graphOper.bfs_shortest_path_directed(directed_graph, source, inflowNodes)
            if shortest_paths_final:
                print(f"  源节点 {source} 最终可达汇节点：{list(shortest_paths_final.keys())}")
                # 打印最终最短路径
                for sink, path in shortest_paths_final.items():
                    print(f"    最终最短路径：{' -> '.join(path)}")
            else:
                print(f"  警告：源节点 {source} 最终仍无法到达任何汇节点（无向图不连通/拓扑序限制）")

        # 更新边的流向信息
        for startNodeName, endNodesNameLst in directed_graph.items():
            for endNodeName in endNodesNameLst:
                for edge in self.networkConnDict.values():
                    if startNodeName in edge.nodes and endNodeName in edge.nodes:
                        edge.flowDirection = (startNodeName, endNodeName)

        print("\n--- 流向初始化完成（保留所有边+无环路+有向图可达） ---")
        return directed_graph

    def updateNeighborNodesEdges(self):
        """
        初始化节点的邻居信息，以字典形式存储
        :return: {节点实例对象:{'inflowNode','outflowNode','inflowConn','outflowConn'}}
        """
        # 构建工作图中节点索引到name的映射（方便后续查询）
        idx_to_name = {v.index: v["name"] for v in self.networkGraph.vs}

        result= {}

        for nodeIndex in idx_to_name:
            self.networkGraph.successors(nodeIndex)
            #查找当前节点的流入节点
            lstIndex=self.networkGraph.predecessors(nodeIndex)
            lstInflowNodeName=[idx_to_name[index] for index in lstIndex]

            lstInflowNodes=[]
            # 遍历键列表
            for nodeName in lstInflowNodeName:
                # 检查键是否在字典中
                if nodeName in self.networkNodesDict:
                    # 如果在，就把对应的值添加到结果列表中
                    lstInflowNodes.append(self.networkNodesDict[nodeName])

            #查找当前节点的流出节点
            lstIndex =self.networkGraph.successors(nodeIndex)
            lstOutflowNodeName=[idx_to_name[index] for index in lstIndex]

            lstOutflowNodes=[]
            # 遍历键列表
            for nodeName in lstOutflowNodeName:
                # 检查键是否在字典中
                if nodeName in self.networkNodesDict:
                    # 如果在，就把对应的值添加到结果列表中
                    lstOutflowNodes.append(self.networkNodesDict[nodeName])

            #查找当前节点的流入边
            inflowConnName=self.networkGraph.es.select(_target= nodeIndex)["name"]
            inflowConns=[]
            for connName in inflowConnName:
                if connName in self.networkConnDict:
                    inflowConns.append(self.networkConnDict[connName])

            #查找当前节点的流出边
            outflowConnName=self.networkGraph.es.select(_source= nodeIndex)["name"]
            outflowConns=[]
            for connName in outflowConnName:
                if connName in self.networkConnDict:
                    outflowConns.append(self.networkConnDict[connName])
            neighborParam= {'inflowNode': lstInflowNodes,
                            'outflowNode': lstOutflowNodes,
                            'inflowConn': inflowConns,
                            'outflowConn': outflowConns}
            result[self.networkNodesDict[idx_to_name[nodeIndex]]]=neighborParam
        return result

    def getNodesSortByDirection(self):
        """
        对输入有向图进行节点排序，，满足所有 u→v 中 u 在 v 之前，返回节点name属性
        """
        self.networkGraph = self.updateGraph()
        # 执行拓扑排序（返回节点ID列表，满足所有 u→v 中 u 在 v 之前）
        topo_order = self.networkGraph.topological_sorting()
        # print("拓扑排序结果（节点ID）:", topo_order)

        named_order = [self.networkGraph.vs[vid]["name"] for vid in topo_order]
        # print("拓扑排序结果（业务名称）:", named_order)

        nodesOrder=[]
        for nodeName in named_order:
            if nodeName in self.networkNodesDict:
                nodesOrder.append(self.networkNodesDict[nodeName])

        return nodesOrder


    def setConvergingNode(self):
        '''
        设置节点是否为汇合节点，连接边>2
        :return:
        '''
        convergingNodes=self.networkGraph["convergingNode"]
        for nodeName,node in self.networkNodesDict.items():
            if nodeName in convergingNodes:
                node.isConvergingNode=True

    def updateFlowDirectionByPressure(self,nodePressure:dict[str,float]):
        """
        根据节点压力更新节点压力和管道流体流向
        """
        for connName,conn in self.networkConnDict.items():
            startNodeName,endNodeName=conn.getNodes()
            self.networkNodesDict[startNodeName].pressure=nodePressure[startNodeName]
            self.networkNodesDict[endNodeName].pressure=nodePressure[endNodeName]
            if nodePressure[startNodeName] >= nodePressure[endNodeName]:
                conn.setFlowDirection((startNodeName,endNodeName))
            else:
                conn.setFlowDirection((endNodeName,startNodeName))
        #流体方向更新后依赖流体方向的有向图，邻居节点、边参数，求解顺序均需要更新
        self.networkGraph = self.updateGraph()
        self.neighborNodesEdges = self.updateNeighborNodesEdges()
        self.nodesCalOrder = self.getNodesSortByDirection()

    def updateFluidParam(self,nodePressure:dict[str,float],connFlowRate:dict[str,float])->bool:
        """
        更新水力热力计算参数，仅二级子网使用
        ---nodePressure:各节点的压力
        ---connFlowRate：各边的流量
        返回值：更新成功True,更新失败False
        """
        # if self.enNetworkLevel !=NetworkLevel.LEVEL_2:
        #
        #     raise TypeError(f"只有二级子网需要调用该接口更新水力热力参数，当前子网类型{self.enNetworkLevel}")

        #先更新流体流向及相关参数
        self.updateFlowDirectionByPressure(nodePressure)

        #根据节点计算顺序，计算各节点流体和管道的参数
        for node in self.nodesCalOrder:
            inflowNode=self.neighborNodesEdges[node]["inflowNode"]
            outflowNode=self.neighborNodesEdges[node]["outflowNode"]
            inflowConn=self.neighborNodesEdges[node]["inflowConn"]
            outflowConn=self.neighborNodesEdges[node]["outflowConn"]

            if node.type==NodeType.SOURCE:
                if len(inflowConn)!=0 or len(outflowConn)==0:
                    print(f'Cannot compute pipeline pressure drop,A source node "{node.name}"  must have outgoing edges, but must not have incoming edges.')
                    return False
            elif node.type==NodeType.SINK:
                if len(inflowConn)==0 or len(outflowConn)!=0:
                    print(f'Cannot compute pipeline pressure drop,A sink node "{node.name}"  must have incoming edges, but must not have outgoing edges.')
                    return False
            else:
                if len(inflowConn)==0 or len(outflowConn)== 0:
                    print(f'Cannot compute pipeline pressure drop,internal node "{node.name}"  must have both incoming edges and outgoing edges')
                    return False

            lstFluids=[]
            sumMassFlowRate=0

            for idx, neighborConn in enumerate(inflowConn):
                if isinstance(neighborConn,FlowLine):
                    fluid = neighborConn.flowlineSim.fluid
                    flowRate = neighborConn.flowlineSim.flowRate
                    #温度使用管道末端的温度，需要考虑管道沿程的温降特性
                    temperature = neighborConn.flowlineSim.T_end
                    param= neighborConn.flowlineSim.getFlowlineTwoEndParam(node.name)
                    sumMassFlowRate+=flowRate
                    lstFluids.append({'fluid':fluid,'massFlowRate':flowRate,'temperature':param['temperature']})
            if len(lstFluids) == 1:
                node.fluid = lstFluids[0]['fluid']
                node.temperature = lstFluids[0]['temperature']
                node.flowRate = lstFluids[0]['massFlowRate']
            elif len(lstFluids) > 1:
                newFluid, newTemp = fluidMix(lstFluids, fluidType='black_oil')
                node.fluid = newFluid
                node.temperature = newTemp
                node.flowRate = sumMassFlowRate

            #根据流出边及分流量计算各边流体的参数
            if node.type == NodeType.JUNCTION or node.type == NodeType.SOURCE:
                for idx, neighborConn in enumerate(outflowConn):
                    massFlowRate = 0.0
                    if node.type == NodeType.SOURCE:
                        massFlowRate = node.getMassFlowRateByPressure(nodePressure[node.name])
                    else:
                        if len(outflowConn) == 1:
                            massFlowRate = sumMassFlowRate
                        else:
                            massFlowRate = connFlowRate[neighborConn.name]

                    if isinstance(neighborConn, FlowLine):
                        neighborConn.flowlineSim.calculateProfile(node.fluid, massFlowRate,
                                                                  node.pressure, node.temperature,
                                                                  neighborConn.flowDirection)
            elif node.type == NodeType.THREE_PHASE_SEP:
                #设备节点需要先计算设备对流体的作用，再计算各管道流体参数
                pass
            else:
                pass

        return True


    def getNetworkGraph(self):
        return self.networkGraph

    def getFlowInNode(self)->list[str]:
        """
        获取仅流体流入的节点列表
        """
        lstFlowInNodesName=[]
        for nodeName,node in self.networkNodesDict.items():
            if node.type == NodeType.SINK:
                lstFlowInNodesName.append(nodeName)
        return lstFlowInNodesName

    def getNodeByName(self,name):
        """
        接受单个节点名称输入或者节点名称list列表输入获取节点对象
        :param name:
        :return:
        """
        if self.networkNodesDict is None:
            return None

        if isinstance(name,str):
            return self.networkNodesDict[name]
        elif isinstance(name,list):
            return [self.networkNodesDict.get(key) for key in name]
        else:
            return None

    def getConnByName(self,name):
        """
        接受单个边名称输入或者边名称list列表输入获取边对象
        :param name:
        :return:
        """
        # 先校验网络连接字典是否存在
        if self.networkConnDict is None:
            return None

        # 情况1：name是字符串（单个键），返回对应的值
        if isinstance(name, str):
            return self.networkConnDict[name]  # 严格模式，不存在抛KeyError

        # 情况2：name是列表（多个键），返回对应的值列表
        elif isinstance(name, list):
            return [self.networkConnDict.get(key) for key in name]

        else:
            # 可选：抛类型错误，明确告知输入不合法
            # raise TypeError("name must be str or list")
            return None

    def meshConnGrid(self,gridLength:float):

        for conn in self.networkConnDict.values():
            conn.meshGrid(gridLength)