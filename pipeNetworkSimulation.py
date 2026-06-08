from PipeNetworkModel.PipeNetwork.networkModel import *
from PipeNetworkModel.Components.networkNode import *
from PipeNetworkModel.Components.networkConnections import *
from PipeNetworkModel.PipeNetwork.graphPartitionComplete import GraphOperation
from PipeNetworkModel.PipeNetwork.pipeNetworkSolve import  PipeNetworkSolve
from Fluid.blackOil.blackoil import pvt_params
import  os
import time
import igraph as ig
import json
import numpy as np
import pandas as pd


class PipeNetworkSimulation:
    def __init__(self):
        self.originalNetworkModel:NetworkModel
        '''原始管网对象'''


    def _normalize_sink_pressures(self, sink_p):
        sink_names = [
            name for name, node in self.originalNetworkModel.networkNodesDict.items()
            if node.type == NodeType.SINK
        ]
        if not sink_names:
            return {}

        if sink_p is None:
            return {
                name: float(node.pressure)
                for name, node in self.originalNetworkModel.networkNodesDict.items()
                if node.type == NodeType.SINK and float(node.pressure or 0.0) > 0.0
            }

        if isinstance(sink_p, dict):
            sink_pressures = {
                str(name): float(pressure)
                for name, pressure in sink_p.items()
                if pressure is not None
            }
            unknown_sinks = sorted(set(sink_pressures) - set(sink_names))
            if unknown_sinks:
                raise ValueError(f"sink_p contains unknown sink node(s): {unknown_sinks}; available sinks: {sink_names}")
            return sink_pressures

        pressure = float(sink_p)
        return {name: pressure for name in sink_names}

    def _save_convergence_result(self, pipeNetworkSolve, elapsed_seconds, result_dir="./result"):
        os.makedirs(result_dir, exist_ok=True)
        solver = getattr(pipeNetworkSolve, "original_solver", None)
        history = getattr(solver, "residual_history", []) if solver is not None else []
        if not history:
            print(f"Solver timing: elapsed={elapsed_seconds:.3f}s; no convergence history recorded.")
            return

        df_history = pd.DataFrame(history)
        csv_path = os.path.join(result_dir, "convergence_history.csv")
        df_history.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df_plot = df_history.copy()
        for col in ("residual_norm", "node_norm", "edge_norm"):
            df_plot[col] = df_plot[col].astype(float).clip(lower=1e-12)

        png_path = os.path.join(result_dir, "convergence_history.png")
        try:
            import matplotlib
            matplotlib.use("Agg")
            matplotlib.rcParams["axes.unicode_minus"] = False
            import logging
            logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter, LogLocator

            fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
            ax.semilogy(df_plot["iteration"], df_plot["residual_norm"], marker="o", label="total residual")
            ax.semilogy(df_plot["iteration"], df_plot["node_norm"], marker="s", label="node residual")
            ax.semilogy(df_plot["iteration"], df_plot["edge_norm"], marker="^", label="edge residual")
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"1e{int(np.log10(value))}" if value > 0 else "")
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Residual norm")
            ax.set_title("Original Network Newton Convergence")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            ax.legend()
            fig.tight_layout()
            fig.savefig(png_path)
            plt.close(fig)
            plot_path = png_path
        except Exception:
            svg_path = os.path.join(result_dir, "convergence_history.svg")
            self._save_convergence_svg(df_plot, svg_path)
            plot_path = svg_path

        final_residual = float(df_history.iloc[-1]["residual_norm"])
        final_method = df_history.iloc[-1]["method"]
        converged = getattr(solver, "last_converged", False)
        print(
            f"Solver timing: elapsed={elapsed_seconds:.3f}s, "
            f"iterations={len(df_history)}, final_residual={final_residual:.6e}, "
            f"final_method={final_method}, converged={converged}, "
            f"plot={plot_path}, csv={csv_path}"
        )

    def _save_convergence_svg(self, df_history, svg_path):
        width, height = 900, 560
        left, right, top, bottom = 80, 30, 40, 70
        plot_w = width - left - right
        plot_h = height - top - bottom
        x_values = df_history["iteration"].astype(float).tolist()
        series = [
            ("total residual", df_history["residual_norm"].astype(float).tolist(), "#1f77b4"),
            ("node residual", df_history["node_norm"].astype(float).tolist(), "#2ca02c"),
            ("edge residual", df_history["edge_norm"].astype(float).tolist(), "#d62728"),
        ]
        positive_values = [max(v, 1e-30) for _, values, _ in series for v in values]
        log_min = np.floor(np.log10(min(positive_values)))
        log_max = np.ceil(np.log10(max(positive_values)))
        if log_min == log_max:
            log_min -= 1
            log_max += 1
        x_min, x_max = min(x_values), max(x_values)
        if x_min == x_max:
            x_min -= 1
            x_max += 1

        def sx(x):
            return left + (x - x_min) / (x_max - x_min) * plot_w

        def sy(y):
            y = max(y, 1e-30)
            return top + (log_max - np.log10(y)) / (log_max - log_min) * plot_h

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="18">Original Network Newton Convergence</text>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
            f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="14">Iteration</text>',
            f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="14">Residual norm (log)</text>',
        ]
        for exp in range(int(log_min), int(log_max) + 1):
            y = sy(10 ** exp)
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-dasharray="4 4"/>')
            lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">1e{exp}</text>')
        for label, values, color in series:
            points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_values, values))
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
            for x, y in zip(x_values, values):
                lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>')
        legend_x = left + plot_w - 180
        legend_y = top + 20
        for idx, (label, _, color) in enumerate(series):
            y = legend_y + idx * 22
            lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            lines.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-family="Arial" font-size="12">{label}</text>')
        lines.append("</svg>")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def readNetworkFromFile(self,excelPath:str)->NetworkModel:
        '''
        从excel文件中读取管网数据并创建管网模型
        :param excelPath-excel文件名及路径
        :return 管网模型类对象
        '''
        if os.path.exists(excelPath) is not True:
            print(excelPath + " is not exist")
            return False

        # 读取管网模型文件
        dfNetworkData = pd.read_excel(excelPath, sheet_name=None)


        dfProject = dfNetworkData["project"]
        dfSource = dfNetworkData["source"]
        dfSink = dfNetworkData["sink"]
        dfJunction = dfNetworkData["junction"]
        dfPipeline = dfNetworkData["pipeline"]
        dfPQCurve = dfNetworkData["pqCurve"]
        dfPipelineDetailed = dfNetworkData["Pipeline_detailed"]
        dfFluid = dfNetworkData["fluid"]
        dfBlackOil = dfNetworkData["black_oil"]
        dfBlackOilModelSet = dfNetworkData["black_oil_model_set"]

        nodes=[]
        for index, sourceData in dfSource.iterrows():
            sourceNode =Source(sourceData["source_name"])
            dfSourcePQ=dfPQCurve[dfPQCurve["source_id"]==sourceData["source_id"]]
            fluid_id=sourceData["fluid_id"]
            fluidType=dfFluid[dfFluid["fluid_id"]==fluid_id]["fluid_model"].item()
            if fluidType=='black_oil':
                Oil_API = 31.2
                Water_Cut = 0.72
                Water_Specific_Gravity = 1.01
                GOR = 60.23
                Gas_Specific_Gravity = 0.82
                Oil_C0 = 1884.06
                Gas_C0 = 2302.74
                Water_C0 = 4186.8
                fluid = pvt_params(Oil_API, Water_Cut,
                                   GOR, Gas_Specific_Gravity, Water_Specific_Gravity,
                                   Oil_C0, Gas_C0, Water_C0)
            else:
                Oil_API = 31.2
                Water_Cut = 0.72
                Water_Specific_Gravity = 1.01
                GOR = 60.23
                Gas_Specific_Gravity = 0.82
                Oil_C0 = 1884.06
                Gas_C0 = 2302.74
                Water_C0 = 4186.8
                fluid = pvt_params(Oil_API, Water_Cut,
                                   GOR, Gas_Specific_Gravity, Water_Specific_Gravity,
                                   Oil_C0, Gas_C0, Water_C0)

            sourceNode.setParam(sourceData["active"],
                                fluid,
                                sourceData["is_pq_curve"],
                                sourceData["pressure"],
                                sourceData["temperature"],
                                sourceData["boundary_type"],
                                sourceData["gas_flowrate"],
                                sourceData["liquid_flowrate"],
                                sourceData["Mass_flowrate"],
                                dfSourcePQ)
            nodes.append(sourceNode)

        for index, sinkData in dfSink.iterrows():
            sinkNode =Sink(sinkData["sink_name"])
            sinkNode.setParam(sinkData["active"],
                              sinkData["pressure"],
                              sinkData["boundary_type"],
                              sinkData["liquid_flowrate"],
                              sinkData["gas_flowrate"],
                              sinkData["mass_flowrate"],)
            nodes.append(sinkNode)

        for index, junctionData in dfJunction.iterrows():
            junctionNode =Junction(junctionData["junction_name"])
            nodes.append(junctionNode)

        connections=[]
        for index, pipelineData in dfPipeline.iterrows():

            dfCurrentDetailed=dfPipelineDetailed[dfPipelineDetailed['pipeline_id']==pipelineData["pipeline_id"]]
            startName = dfProject[dfProject['project_id'] == pipelineData["pipeline_start"]]['component_name'].item()
            endName = dfProject[dfProject['project_id'] == pipelineData["pipeline_end"]]['component_name'].item()
            node=(startName,endName)
            flowlineConn = FlowLine(pipelineData["pipeline_Name"],node)
            flowlineConn.setParam(pipelineData["Flow_type"],
                                  pipelineData["pipe_inside_diameter"],
                                  pipelineData["pipe_outside_diameter"],
                                  pipelineData["pipe_wall_thickness"],
                                  pipelineData["inner_pipe_outside_diameter"],
                                  pipelineData["pipe_roughness"],
                                  pipelineData["pipeline_Mode"],
                                  pipelineData["is_GIS_map"],
                                  pipelineData["Environment_type"],
                                  pipelineData["ambient_temperature"],
                                  pipelineData["U_value_type"],
                                  pipelineData["pipe_heat_transfer_coefficient"],
                                  dfCurrentDetailed.iloc[-1]['pipeline_measured_distance'],
                                  dfCurrentDetailed.iloc[-1]['pipeline_horizontal_distance'],
                                  dfCurrentDetailed)
            connections.append(flowlineConn)

        networkModel = NetworkModel(name="original",networkNodes= nodes,networkConnections= connections)
        return  networkModel

    def dictDataToExcel(self,dictData,excelPath:str):

        #确保输出目录存在
        output_dir = os.path.dirname(excelPath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)


        # 覆盖式写入Excel
        try:
            # engine="openpyxl" 是覆盖模式的核心引擎，不支持追加
            with pd.ExcelWriter(excelPath, engine="openpyxl") as writer:
                for sheet_name, data in dictData.items():
                    df=pd.DataFrame(data)
                    df.to_excel(writer, sheet_name=sheet_name)
            print(f"✅ 成功覆盖保存管网仿真数据到Excel：{excelPath}")
        except Exception as e:
            raise RuntimeError(f"❌ 保存Excel失败：{str(e)}")


    def createNetworkModel(self):

        # 创建管网节点信息
        # nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
        nodes = ["SWD", "well_A1", "well_A2", "well_A3", "well_A4", "well_A5", "well_B1",
                 "well_B2", "well_B3", "well_B4", "well_C1", "well_C2", "well_C3", "well_C4",
                 "well_C5", "well_C6", "Source_D", "Cond_Tank", "Gas_Sales",
                 "A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "C5", "C6", "D1", "G_S", "Manifold", "W_S",
                 "choke", "ck", "compressor", "hx", "pump", "separator"]
        networkNodes = []
        for node in nodes:
            source = Source(node)
            source.flow = 200
            networkNodes.append(source)

        # 创建管网连接信息
        # edges = [
        #     ('B','A'), ('A','E'), ('A','D'), ('A','C'),
        #     ('B','M'), ('M','G'), ('G','F'), ('H','F'),
        #     ('F','I'), ('L','K'), ('K','J'), ('J','B')
        # ]
        edges = [
            ('choke', 'A1'), ('well_A2', 'A1'), ('well_A3', 'A1'), ('well_A4', 'A2'),
            ('well_A5', 'A2'), ('A1', 'A2'), ('A2', 'D1'), ('well_B1', 'B1'),
            ('well_B2', 'B1'), ('well_B3', 'B1'), ('well_B4', 'B2'), ('B1', 'B2'), ('B2', 'Manifold'),
            ('well_C1', 'C1'), ('well_C2', 'C1'), ('well_C3', 'C1'),
            ('ck', 'C3'), ('well_C5', 'C4'), ('well_C6', 'C4'), ('C1', 'C2'), ('C2', 'C6'), ('C6', 'C5'), ('C3', 'C4'),
            ('C4', 'C5'), ('C5', 'Manifold'), ('G_S', 'compressor'),
            ('separator', 'Cond_Tank'), ('Source_D', 'D1'), ('D1', 'Manifold'), ('hx', 'Gas_Sales'), ('pump', 'SWD'),
            ('C4', 'D1'), ('W_S', 'pump'), ('Manifold', 'separator'),
            ('well_A1', 'choke'), ('well_C4', 'ck'), ('separator', 'G_S'), ('separator', 'W_S'), ('compressor', 'hx')
        ]
        connections = []
        for index, conn in enumerate(edges):
            inputNode, outputNode = conn
            fl = FlowLine(inputNode + "-" + outputNode + "-" + str(index), conn)
            fl.length = 600
            connections.append(fl)

        # 创建管网模型
        netWorkModel = NetworkModel("originalNetwork",networkNodes, connections)

        # 1. 处理 nodes 数据（单列）
        nodes_df = pd.DataFrame({"nodes": nodes})

        # 2. 处理 edges 数据（拆分为 start_node 和 end_node 两列）
        # 将元组列表转换为 DataFrame，自动分配列名，再重命名
        edges_df = pd.DataFrame(edges, columns=["start_node", "end_node"])

        # 3. 写入 Excel 文件（两个工作表）
        with pd.ExcelWriter("../../network_data.xlsx", engine="openpyxl") as writer:
            nodes_df.to_excel(writer, sheet_name="Nodes", index=False)  # index=False 不写入行索引
            edges_df.to_excel(writer, sheet_name="Edges", index=False)

        print("Excel 文件已生成：network_data.xlsx")

        return netWorkModel


    def drawToplogicalGraph(self,savePath = "./segGraphResult" ):
        """可视化原始管网并保存到指定文件夹目录"""
        graphOper = GraphOperation()
        # 检查文件夹是否存在
        if not os.path.exists(savePath):
            # 不存在则创建（parents=True允许创建多级目录，exist_ok=True避免重复创建报错）
            os.makedirs(savePath, exist_ok=True)
            print(f"文件夹不存在，已创建：{savePath}")
        else:
            print(f"文件夹已存在，直接使用：{savePath}")

        directedGraph =self.originalNetworkModel.getNetworkGraph()

        # 绘制完整原图
        graphOper.plot_original_directed_graph(
            original_graph=directedGraph,
            save_path=savePath + "/original_directed_graph.png",
            degree_type="total"
        )

    def jsonToNetworkModel(self,jsonNetworkModel)->NetworkModel:
        """
        解析JSON字典
        :param jsonNetworkModel: 待解析的JSON字典
        :return: 管网模型
        """
        # try:
        # dictNetworkData = jsonNetworkModel['jsonNetworkModel']
        dictNetworkData =jsonNetworkModel

        lstEquipments = dictNetworkData["equipments"]
        lstSource = dictNetworkData["source"]
        lstSink = dictNetworkData["sink"]
        lstWell = dictNetworkData['well']
        lstJunction = dictNetworkData["junction"]
        lstPipeline = dictNetworkData["pipeLine"]
        lstChoke = dictNetworkData["choke"]
        lstTwoPhase = dictNetworkData["twoPhase"]
        lstHeat = dictNetworkData["heat"]
        lstMultiPhase = dictNetworkData["multiphase"]
        lstCheckValve = dictNetworkData["check"]
        lstGenericPump = dictNetworkData["genericPump"]
        lstThreePhase = dictNetworkData["threePhase"]
        lstCompressor = dictNetworkData["compressor"]
        lstBoundaryConditon = dictNetworkData["boundaryCondition"]
        mapBoundaryConditon = {d["name"]: d for d in lstBoundaryConditon}
        lstRateConstraint = dictNetworkData.get("constraint",[])
        mapRateConstraint = {d["name"]: d for d in lstRateConstraint}
        # dfPQCurve = dfNetworkData["pqCurve"]
        # dfPipelineDetailed = dfNetworkData["Pipeline_detailed"]
        # dfFluid = dfNetworkData["fluid"]
        # dfBlackOil = dfNetworkData["black_oil"]
        # dfBlackOilModelSet = dfNetworkData["black_oil_model_set"]

        nodes = []
        for sourceData in lstSource:
            sourceNode = Source(sourceData["featureName"])
            lstPQ=sourceData['pqTable']
            dfPQTable=pd.DataFrame(lstPQ)
            # if not dfPQTable.empty:
                # dfPQTable=dfPQTable.rename(columns={'temperature':'temperature','pressure':'pressure','gasFlowrate':'gas_flowrate',
                #                                 'liquidFlowrate':'liquid_flowrate','massFlowrate':'Mass_flowrate'})
            # fluid_id = sourceData["fluid_id"]
            fluidModel = 'black_oil'
            if fluidModel == 'black_oil':
                dictBlackOil = sourceData['black_oil'][0]
                dictBlackOilSet = sourceData['black_oil_set'][0]

                Oil_API = dictBlackOil['oilApi']
                Water_Cut = dictBlackOil['waterRatio']
                Water_Specific_Gravity = dictBlackOil['waterSpecificGravity']
                GOR = dictBlackOil['gasOilRadio']
                Gas_Specific_Gravity = dictBlackOil['gasSpecificGravity']
                Oil_C0 = dictBlackOil['oilSpecificHeatCapacity']
                Gas_C0 = dictBlackOil['gasSpecificHeatCapacity']
                Water_C0 = dictBlackOil['waterSpecificHeatCapacity']
                fluid = pvt_params(Oil_API, Water_Cut,
                                   GOR, Gas_Specific_Gravity, Water_Specific_Gravity,
                                   Oil_C0, Gas_C0, Water_C0)
            else:
                Oil_API = 31.2
                Water_Cut = 0.72
                Water_Specific_Gravity = 1.01
                GOR = 60.23
                Gas_Specific_Gravity = 0.82
                Oil_C0 = 1884.06
                Gas_C0 = 2302.74
                Water_C0 = 4186.8
                fluid = pvt_params(Oil_API, Water_Cut,
                                   GOR, Gas_Specific_Gravity, Water_Specific_Gravity,
                                   Oil_C0, Gas_C0, Water_C0)

            sourceNode.setParam(sourceData.get("isActive"),
                                fluid,
                                sourceData.get("usePqCurve"),
                                sourceData.get("pressure"),
                                sourceData.get("temperature"),
                                sourceData.get("flowSelect"),
                                sourceData.get("gasFlowRate"),
                                sourceData.get("liquidFlowRate"),
                                sourceData.get("massFlowRate"),
                                dfPQTable)

            sourceNode.setBoundary(mapBoundaryConditon.get(sourceNode.name))
            sourceNode.volumeToMassFlow()
            nodes.append(sourceNode)

        for sinkData in lstSink:
            sinkNode = Sink(sinkData.get("featureName"))
            sinkNode.setParam(sinkData.get("isActive"),
                              sinkData.get("pressure"),
                              sinkData.get("flowSelect"),
                              sinkData.get("liquidFlowRate"),
                              sinkData.get("gasFlowRate"),
                              sinkData.get("massFlowRate"), )
            sinkNode.setBoundary(mapBoundaryConditon.get(sinkNode.name))

            nodes.append(sinkNode)

        for junctionData in lstJunction:
            junctionNode = Junction(junctionData.get("featureName"))
            nodes.append(junctionNode)

        connections = []
        for pipelineData in lstPipeline:
            dfCurrentDetailed = pd.DataFrame(pipelineData.get('profileTable'))
            if not dfCurrentDetailed.empty and pipelineData.get("flowMode").lower()!='simple':
                dfCurrentDetailed = dfCurrentDetailed.rename(columns={'horDistance': 'pipeline_horizontal_distance','measuredDistance': 'pipeline_measured_distance',
                                                                  'elevation': 'pipeline_elevation','depth': 'depth',
                                                                  'latitude': 'Latitude','longitude': 'Longitude'})
                dfCurrentDetailed = dfCurrentDetailed.sort_values(by='pipeline_measured_distance', ascending=True).reset_index(drop=True)

            startName=''
            startPointId=pipelineData['startPointId']
            nodeType = next((item['pointId'] for item in lstEquipments if item['featureId'] == startPointId), None)
            if nodeType == 'source':
                startName = next((item['featureName'] for item in lstSource if item['featureId'] == startPointId), None)
            elif nodeType == 'sink':
                startName = next((item['featureName'] for item in lstSink if item['featureId'] == startPointId), None)
            elif nodeType == 'junction':
                startName = next((item['featureName'] for item in lstJunction if item['featureId'] == startPointId), None)
            else:
                print(f'无法查找{pipelineData["featureName"]} 连接的端点设备！')
                return None
            if startName is None:
                print(f'未查找到{pipelineData["featureName"]} 连接的端点设备！')
                return None

            endName = ''
            endPointId = pipelineData['endPointId']
            nodeType = next((item['pointId'] for item in lstEquipments if item['featureId'] == endPointId), None)
            if nodeType == 'source':
                endName = next((item['featureName'] for item in lstSource if item['featureId'] == endPointId), None)
            elif nodeType == 'sink':
                endName = next((item['featureName'] for item in lstSink if item['featureId'] == endPointId), None)
            elif nodeType == 'junction':
                endName = next((item['featureName'] for item in lstJunction if item['featureId'] == endPointId), None)
            else:
                print(f'无法查找{pipelineData["featureName"]} 连接的端点设备！')
                return None

            if endName is None:
                print(f'未查找到{pipelineData["featureName"]} 连接的端点设备！')
                return None

            node = (startName, endName)
            flowlineConn = FlowLine(pipelineData["featureName"], node)
            flowlineConn.setParam(
                pipelineData.get("flowType"),
                pipelineData.get("insideDiameter"),
                pipelineData.get("outsideDiameter"),
                pipelineData.get("wallThickness"),
                pipelineData.get("innerOutsideDia"),
                pipelineData.get("roughness"),
                pipelineData.get("flowMode"),
                pipelineData.get("mapPopulated"),
                pipelineData.get("environment"),
                pipelineData.get("ambTemperature"),
                pipelineData.get("uValueType"),
                pipelineData.get("heatCoefficient"),
                pipelineData.get("measuredDifference"),
                pipelineData.get("horizontalDifference"),
                dfCurrentDetailed
            )
            connections.append(flowlineConn)

        networkModel = NetworkModel(name="original", networkNodes=nodes, networkConnections=connections)
        networkModel.setBoundaryConstraint(lstBoundaryConditon,lstRateConstraint)
        return networkModel
        # except json.JSONDecodeError as e:
        #     # 捕获JSON格式错误（最常见异常）
        #     print(f"JSON格式错误：{e}")
        #     return None
        # except Exception as e:
        #     # 捕获其他异常（如输入非字符串）
        #     print(f"解析失败：{e}")
        #     return None

    def dataframToJson(self,dictAllResult)->str:
        """管网仿真结果转换为json格式"""
        json_dict = {}
        for key, data in dictAllResult.items():
            if key == 'nodesResult':
                # DataFrame转成字典列表，支持JSON序列化
                json_dict[key] = data.to_dict(orient="records")
            elif key == 'branchResult':
                # DataFrame转成字典列表，支持JSON序列化
                json_dict[key] = data.to_dict(orient="records")
            elif key == 'profileResult':
                dictData = {}
                for connName, df in data.items():
                    dictData[connName] = df.to_dict(orient="records")
                json_dict[key] = dictData
            elif key == 'equipmentParam':
                json_dict[key] = data
        return json.dumps(json_dict, default=str, ensure_ascii=False, indent=2)

    def getNetworkSimResult(self)->(pd.DataFrame,pd.DataFrame,dict[str,pd.DataFrame],dict[str,dict[str,float]]):
        """
        返回节点信息，支线信息，剖面参数
        """

        lstNodesResult = []
        for nodeName,node in self.originalNetworkModel.networkNodesDict.items():
            type='Sink'
            if node.type == NodeType.SINK:
                type='Sink'
            elif node.type == NodeType.SOURCE:
                type='Source'
            elif node.type == NodeType.CHOKE:
                type='Choke'
            elif node.type == NodeType.JUNCTION:
                type='Junction'

            node_attrs = {
                "name": node.name,
                'type': type,
                "massFlowrate": node.flowRate,
                "temperatureOut": node.temperature,
                "pressureOut": node.pressure,
                # 如果需要pvt_params的子属性，可以在这里展开
                # "pvt_param_xxx": node.fluid.pvt_params.get("xxx") if node.fluid and node.fluid.pvt_params else None
            }
            lstNodesResult.append(node_attrs)


        lstProfileResult =[]
        lstProfileBranch=[]
        for connName, conn in self.originalNetworkModel.networkConnDict.items():
            pipelineProfileData=conn.flowlineSim.getProfileResult()
            pipelineProfileData.rename(columns={'measure':"totalDistance",'horizontalPosition': 'horizontalDistance',
                      'verticalPosition': 'elevation', 'pressure': 'pressure', 'temperature': 'temperature'},inplace=True)
            pipelineProfileData['branch'] = connName
            lstProfileResult.extend(pipelineProfileData.to_dict(orient="records"))
            lstProfileBranch.append({"branch":connName})
        return lstNodesResult,dict(),lstProfileBranch,lstProfileResult

    def readSimulatedResultFromJson(self):

        with open("./DB/模拟结果参数字段v2.json", 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        # return json.dumps(json_data, ensure_ascii=False, indent=2)
        return json_data

    def start(self, fileOrJson,jsonNetworkModel={}, excelPath = "./DB/NetworkProject.xlsx",isSavePicture=False,
              isSaveResult=False, savePath = "./segGraphResult",gridLength:float=10.0,
              solverMode: str = "direct", sink_p=None):
        """
        串联管网仿真过程
        :param fileOrJson: 从json字符串获取模型信息或者excel文件中获取，file/json
        :param jsonNetworkModel:存放管网模型参数的json字典
        :param excelPath:存放管网模型的excel文件
        :param isSavePicture:是否保存管网拓扑图分割结果
        :param savePath:管网拓扑图分割保存路径
        :param gridLength:管道网格划分长度
        """
        if fileOrJson=='file':
            self.originalNetworkModel = self.readNetworkFromFile(excelPath)
        else:
            self.originalNetworkModel = self.jsonToNetworkModel(jsonNetworkModel)


        if  isSavePicture:
            self.drawToplogicalGraph(savePath)

        self.originalNetworkModel.meshConnGrid(gridLength)

        # *************************牛顿求解算法**********************************************************
        pipeNetworkSolve =PipeNetworkSolve(self.originalNetworkModel)
        sink_pressures = self._normalize_sink_pressures(sink_p)
        solve_start_time = time.perf_counter()
        pipeNetworkSolve.solve(mode=solverMode, sink_pressures=sink_pressures)
        solve_elapsed = time.perf_counter() - solve_start_time
        # *************************牛顿求解算法**********************************************************

        #*************************二分求解，仅三节点管网可求，不需要删除，注释即可**********************************************************

        ##最小二乘优化求解方法
        # calculator = Calculator(self.originalNetworkModel,self.subNetworkModelLevel1,self.subNetworkModelLevel2)
        # calculator.calculate()

        #二分求解+顺序求解
        # computor = Computor(self.originalNetworkModel,self.subNetworkModelLevel1,self.subNetworkModelLevel2)
        # computor.compute()

        #*************************二分求解，仅三节点管网可求，不需要删除，注释即可**********************************************************

        lstNodesResult,lstBranchResult,lstProfileBranch,lstProfileResult= self.getNetworkSimResult()


        jsonResult={}
        jsonResult['nodesResult']=lstNodesResult
        jsonResult['branchResult'] = lstBranchResult
        jsonResult['profileBranch'] = lstProfileBranch
        jsonResult['profileResult'] = lstProfileResult

        if isSaveResult:
            self.dictDataToExcel(dictData= jsonResult,excelPath=savePath+'/networkSimResult.xlsx')


        # jsonResult = self.readSimulatedResultFromJson()
        self._save_convergence_result(pipeNetworkSolve, solve_elapsed, result_dir=savePath)

        return jsonResult



if __name__ == "__main__":

    # #*************************从excel文件读取管网模型数据进行计算，不需要删除，注释即可**********************************************************
    #
    # #存放管网结构的文件路径
    # excelPath = "./DB/NetworkProject-threeNodes.xlsx"
    # # 图分割目标文件夹路径
    # savePath = "./segGraphResult"
    #
    # gridLength = 100.0 #管网网格划分长度,m
    #
    # networkSim = PipeNetworkSimulation()
    # networkSim.start(fileOrJson='file',excelPath=excelPath,isSavePicture=True,isSaveResult=True, savePath=savePath,gridLength=gridLength)
    #
    # # *************************从excel文件读取管网模型数据进行计算，不需要删除，注释即可**********************************************************






    # *************************从json文件读取管网模型数据进行计算，不需要删除，注释即可**********************************************************

    #存放管网结构的文件路径
    filePath = "./DB/json数据模型数据.json"
    # 图分割目标文件夹路径
    savePath = "./segGraphResult"

    gridLength = 100.0 #管网网格划分长度,m
    sink_p = None

    # 打开JSON文件，直接用json.load解析
    with open(filePath, "r", encoding='utf-8') as f:
        json_data = json.load(f)
    print(f"成功读取JSON文件：{filePath}")

    jsonNetworkModel = json_data["jsonNetworkModel"]

    networkSim = PipeNetworkSimulation()
    networkSim.start(fileOrJson='json',jsonNetworkModel=jsonNetworkModel,isSavePicture=True,isSaveResult=True,
                     savePath=savePath,gridLength=gridLength, sink_p=sink_p)

    # *************************从json文件读取管网模型数据进行计算，不需要删除，注释即可**********************************************************
