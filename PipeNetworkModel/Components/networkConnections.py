import math
from enum import Enum
import pandas as pd
import numpy as np

from Fluid.blackOil.basciClassDef_blackOil import pvt_params
from Multiphaseflow.steady import *


class ConnectionType(Enum):
    CONNECTOR = 1
    FLOWLINE = 2
    RISER = 3


class Connection:
    """
    管网连接父类
    """

    def __init__(self, name, type, nodes: tuple[str, str]):
        self.name: str = name
        """管网连接名称"""
        self.type: ConnectionType = type
        """管网连接的类型"""
        self.nodes: tuple[str, str] = nodes
        """管网连接的两端节点，也表示管道距离标定的起始位置"""
        self.flowDirection: tuple[str, str] = nodes
        """管网中流体流向"""

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    def getNodes(self):
        return self.nodes

    def meshGrid(self, gridLength: float):
        pass

    def setFlowDirection(self, flowDirection: tuple[str, str]):
        pass


class FlowLine(Connection):
    """
    管网管线
    """

    def __init__(self, name, nodes: tuple[str, str]):
        super().__init__(name, ConnectionType.FLOWLINE, nodes)
        self.flowType = ''
        """管道类型，管道/环空"""
        self.pipe_inside_diameter = 0.0
        """管道内径，m"""
        self.pipe_outside_diameter = 0.0
        """管道外径，m"""
        self.pipe_wall_thickness: float = 0.0
        """管道壁厚，m"""
        self.inner_pipe_outside_diameter: float = 0.0
        """内管外径，m"""
        self.pipe_roughness: float = 0.0
        """管壁粗糙度,m"""
        self.pipeline_Mode: str = ''
        '''管道模型'''
        self.is_GIS_map: bool = False
        '''是否使用GIS数据'''
        self.Environment_type: str = ''
        '''管道所处环境类型'''
        self.ambient_temperature: float = 298.15
        '''管道环境温度,K '''
        self.U_value_type: str = ''
        '''U值类型 '''
        self.pipe_heat_transfer_coefficient: float = 0
        '''管道传热系数, J/(s.degC.m2)'''
        self.measuredDistance: float = 0.0
        '''简单模式下管道策略距离，m'''
        self.horizontalDistance: float = 0.0
        '''简单模式下管道水平距离,m'''
        self.pipeDetailedDataFrame = pd.DataFrame({"pipeline_horizontal_distance": pd.Series(dtype=float),
                                                   "pipeline_measured_distance": pd.Series(dtype=float),
                                                   "pipeline_elevation": pd.Series(dtype=float),
                                                   "Latitude": pd.Series(dtype=float),
                                                   "Longitude": pd.Series(dtype=float)})
        self.flowlineSim: FlowLineSim = FlowLineSim(self.nodes)
        """管道得计算实例类"""

    def setParam(self, flowType: str,
                 pipe_inside_diameter: float,
                 pipe_outside_diameter: float,
                 pipe_wall_thickness: float,
                 inner_pipe_outside_diameter: float,
                 pipe_roughness: float,
                 pipeline_Mode: str,
                 is_GIS_map: bool,
                 Environment_type: str,
                 ambient_temperature: float,
                 U_value_type: str,
                 pipe_heat_transfer_coefficient: float,
                 measuredDis,
                 horizontalDis,
                 pipeDetailedDataFrame: pd.DataFrame):
        """
       批量设置管网管线的核心参数（结构尺寸、传热特性、环境配置）
       功能说明：
           一次性配置管道的内径/外径/壁厚等结构参数、管壁粗糙度等流动阻力参数、
           环境温度/传热系数等热力参数，为管道流动阻力计算和热量损失计算提供基础数据。
       参数说明：
            :flowType:str - 管道类型 pipe/annulus
           :param pipe_inside_diameter: float - 管道内径，单位：米（m），需大于 0，
           :param pipe_outside_diameter: float - 管道外径，单位：米（m），需大于内径，
           :param pipe_wall_thickness: float - 管道壁厚，单位：米（m），
                                       理论上应满足：壁厚 = (外径 - 内径)/2，需与内外径匹配
           :param inner_pipe_outside_diameter: float - 内管外径，单位：米（m），
           :param pipe_roughness: float - 管壁粗糙度，单位：米（m），推荐取值范围：
           :param pipeline_Mode: str - 管道模型类型，需与实际管道结构匹配，可选值：
           :param is_GIS_map: bool - 是否启用 GIS 数据关联（True/False）：
           :param Environment_type: str - 管道所处环境类型，影响散热计算，可选值：
           :param ambient_temperature: float - 环境温度，单位：开尔文（K），
                                     转换公式：K = ℃ + 273.15，需与流体温度单位保持一致，
           :param U_value_type: str - 传热系数（U值）计算类型，可选值：
           :param pipe_heat_transfer_coefficient: float - 管道总传热系数，单位：焦耳/(秒·摄氏度·平方米)（J/(s·℃·m²)），
       返回值：
           :return: None - 无返回值，直接修改管线实例的属性

       注意事项：
           1. 尺寸一致性：管道外径必须大于内径，壁厚需与内外径满足几何关系（壁厚 = (外径 - 内径)/2），
              否则可能导致强度校核或流动计算异常；
           2. 单位统一性：所有长度单位为米（m）、温度为开尔文（K），需与管网整体仿真单位体系一致；
       使用示例：
           # 实例化地面单管管线（连接 well_A1 和 A1 节点）
           flow_line = FlowLine(name="Pipe_wellA1_A1", nodes=("well_A1", "A1"))
           # 配置管线参数
           flow_line.setParam(
               pipe_inside_diameter=0.1,          # 内径 100mm
               pipe_outside_diameter=0.114,       # 外径 114mm（对应壁厚 7mm）
               pipe_wall_thickness=0.007,         # 壁厚 7mm（(0.114-0.1)/2=0.007）
               inner_pipe_outside_diameter=0.0,   # 单管模型，内管外径设为 0
               pipe_roughness=2e-5,               # 普通碳钢管粗糙度
               pipeline_Mode="SinglePipe",        # 单管模型
               is_GIS_map=False,                  # 不启用 GIS 数据
               Environment_type="AboveGround",    # 地面管道
               ambient_temperature=293.15,        # 环境温度 20℃（20+273.15=293.15K）
               U_value_type="Custom",             # 自定义传热系数
               measuredDis,                       #测量距离
               horizontalDis,                     #水平距离
               pipe_heat_transfer_coefficient=30.0  # 传热系数 30 W/(℃·m²)
           )
        """
        # 结构尺寸参数赋值
        self.flowType = flowType
        self.pipe_inside_diameter = pipe_inside_diameter
        self.pipe_outside_diameter = pipe_outside_diameter
        self.pipe_wall_thickness = pipe_wall_thickness
        self.inner_pipe_outside_diameter = inner_pipe_outside_diameter

        # 流动阻力相关参数赋值
        self.pipe_roughness = pipe_roughness

        # 管道模型与GIS配置赋值
        self.pipeline_Mode = pipeline_Mode
        self.is_GIS_map = is_GIS_map

        # 环境与传热参数赋值
        self.Environment_type = Environment_type
        self.ambient_temperature = ambient_temperature
        self.U_value_type = U_value_type
        self.pipe_heat_transfer_coefficient = pipe_heat_transfer_coefficient

        self.measuredDistance=measuredDis
        self.horizontalDistance = horizontalDis
        self.pipeDetailedDataFrame = pipeDetailedDataFrame

    def meshGrid(self, gridLength: float):
        """
               按固定段长划分网格（垂直于中轴线，沿轴线等距划分）
               :param grid_length: 每个网格的固定长度（m）
               :return: 划分后的网格列表（self.grids）
       """
        pipe_total_length = 0

        if self.pipeDetailedDataFrame.empty and self.pipeline_Mode.strip().lower()=='detailed':
            print(f"管道的位置信息不能为空")
            return None
        elif self.pipeline_Mode.strip().lower()=='simple':
            pipe_total_length = self.measuredDistance
        else:
            pipe_total_length = self.pipeDetailedDataFrame['pipeline_measured_distance'].max()



        if gridLength <= 0:
            raise ValueError("网格长度必须大于0")
        # if gridLength >= pipe_total_length:
        #     raise ValueError("网格长度不能超过管道总长度")


        grids: list[GridCell] = []

        if self.pipeline_Mode.strip().lower() == 'simple':

            HorizontalDis = self.horizontalDistance
            measuredDis = self.measuredDistance


            ang =math.fabs(math.acos(HorizontalDis/measuredDis))

            # 计算网格数量（最后一段可能不足固定长度，自动调整为剩余长度）
            num_grids = int(np.ceil(pipe_total_length / gridLength))
            # 逐段生成网格
            for i in range(num_grids):
                start_pos = i * gridLength  # 第i个网格的起点（沿轴线距离）
                # 终点：最后一段取管道总长，其余取固定段长终点
                end_pos = min(start_pos + gridLength, pipe_total_length)
                startHorizontalDis = start_pos*math.cos(ang)
                endHorizontalDis = end_pos*math.cos(ang)
                startElevation = start_pos*math.sin(ang)
                endElevation = end_pos*math.sin(ang)
                # 创建网格单元并添加到列表
                grid = GridCell(
                    grid_id=i,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    start_horizontal_pos=startHorizontalDis,
                    end_horizontal_pos=endHorizontalDis,
                    start_vertical_pos=startElevation,
                    end_vertical_pos=endElevation,
                    ang=ang,
                    pipe_inside_diameter=self.pipe_inside_diameter,
                    pipe_outside_diameter=self.pipe_outside_diameter,
                    pipe_wall_thickness=self.pipe_wall_thickness,
                    inner_pipe_outside_diameter=self.inner_pipe_outside_diameter,
                    pipe_roughness=self.pipe_roughness,
                    pipeline_Mode=self.pipeline_Mode,
                    ambient_temperature=self.ambient_temperature,
                    U_value_type=self.U_value_type,
                    pipe_heat_transfer_coefficient=self.pipe_heat_transfer_coefficient,
                )
                grids.append(grid)
        else:
            count = 0
            measuredDis = 0
            index = 0

            if self.pipeDetailedDataFrame.empty :
                print(f"管道的位置信息不能为空")
                return None
            pipe_total_length = self.pipeDetailedDataFrame['pipeline_measured_distance'].max()

            while measuredDis < pipe_total_length:

                if index < len(self.pipeDetailedDataFrame) - 1:
                    firstMeasuredDis = self.pipeDetailedDataFrame.loc[index, 'pipeline_measured_distance']
                    firstHorizontalDis = self.pipeDetailedDataFrame.loc[index, 'pipeline_horizontal_distance']
                    firstElevation = self.pipeDetailedDataFrame.loc[index, 'pipeline_elevation']

                    secondMeasuredDis = self.pipeDetailedDataFrame.loc[index + 1, 'pipeline_measured_distance']
                    secondHorizontalDis = self.pipeDetailedDataFrame.loc[index + 1, 'pipeline_horizontal_distance']
                    secondElevation = self.pipeDetailedDataFrame.loc[index + 1, 'pipeline_elevation']

                else:
                    firstMeasuredDis = self.pipeDetailedDataFrame.iloc[-2][ 'pipeline_measured_distance']
                    firstHorizontalDis = self.pipeDetailedDataFrame.iloc[-2][ 'pipeline_horizontal_distance']
                    firstElevation = self.pipeDetailedDataFrame.iloc[-2]['pipeline_elevation']

                    secondMeasuredDis = self.pipeDetailedDataFrame.iloc[-1][ 'pipeline_measured_distance']
                    secondHorizontalDis = self.pipeDetailedDataFrame.iloc[-1][ 'pipeline_horizontal_distance']
                    secondElevation = self.pipeDetailedDataFrame.iloc[-1]['pipeline_elevation']

                if firstMeasuredDis <= measuredDis < secondMeasuredDis:
                    count += 1
                    if measuredDis + gridLength <= secondMeasuredDis:
                        ang = math.fabs(
                            math.asin((secondElevation - firstElevation) / (secondMeasuredDis - firstMeasuredDis)))
                        start_horizontal_pos = 0
                        end_horizontal_pos = 0
                        if secondHorizontalDis >= firstHorizontalDis:
                            start_horizontal_pos = firstHorizontalDis + (measuredDis - firstMeasuredDis) * math.cos(
                                ang)
                            end_horizontal_pos = firstHorizontalDis + (
                                    measuredDis + gridLength - firstMeasuredDis) * math.cos(ang)
                        else:
                            start_horizontal_pos = firstHorizontalDis - (measuredDis - firstMeasuredDis) * math.cos(
                                ang)
                            end_horizontal_pos = firstHorizontalDis - (
                                    measuredDis + gridLength - firstMeasuredDis) * math.cos(ang)

                        start_vertical_pos = 0
                        end_vertical_pos = 0
                        if secondElevation >= firstElevation:
                            start_vertical_pos = firstElevation + (measuredDis - firstMeasuredDis) * math.sin(ang)
                            end_vertical_pos = firstElevation + (
                                        measuredDis + gridLength - firstMeasuredDis) * math.sin(ang)
                        else:
                            start_vertical_pos = firstElevation - (measuredDis - firstMeasuredDis) * math.sin(ang)
                            end_vertical_pos = firstElevation - (
                                        measuredDis + gridLength - firstMeasuredDis) * math.sin(ang)

                        grid = GridCell(
                            grid_id=count,
                            start_pos=measuredDis,
                            end_pos=measuredDis + gridLength,
                            start_horizontal_pos=start_horizontal_pos,
                            end_horizontal_pos=end_horizontal_pos,
                            start_vertical_pos=start_vertical_pos,
                            end_vertical_pos=end_vertical_pos,
                            ang=ang,
                        )
                        grids.append(grid)
                        measuredDis += gridLength

                    else:
                        ang = math.fabs(
                            math.asin((secondElevation - firstElevation) / (secondMeasuredDis - firstMeasuredDis)))
                        start_horizontal_pos = 0
                        end_horizontal_pos = 0
                        if secondHorizontalDis >= firstHorizontalDis:
                            start_horizontal_pos = firstHorizontalDis + (measuredDis - firstMeasuredDis) * math.cos(
                                ang)
                            end_horizontal_pos = secondHorizontalDis
                        else:
                            start_horizontal_pos = firstHorizontalDis - (measuredDis - firstMeasuredDis) * math.cos(
                                ang)
                            end_horizontal_pos = secondHorizontalDis

                        start_vertical_pos = 0
                        end_vertical_pos = 0
                        if secondElevation >= firstElevation:
                            start_vertical_pos = firstElevation + (measuredDis - firstMeasuredDis) * math.sin(ang)
                            end_vertical_pos = secondElevation
                        else:
                            start_vertical_pos = firstElevation - (measuredDis - firstMeasuredDis) * math.sin(ang)
                            end_vertical_pos = secondElevation

                        grid = GridCell(
                            grid_id=count,
                            start_pos=measuredDis,
                            end_pos=measuredDis + gridLength,
                            start_horizontal_pos=start_horizontal_pos,
                            end_horizontal_pos=end_horizontal_pos,
                            start_vertical_pos=start_vertical_pos,
                            end_vertical_pos=end_vertical_pos,
                            ang=ang,
                        )
                        grids.append(grid)

                        measuredDis = secondMeasuredDis
                        index += 1
                else:
                    index += 1

        print(f"{self.name} 管道网格划分完成：共{len(grids)}个网格，最后一段长度{grids[-1].length:.3f}m")
        self.flowlineSim.setGrid(grids)
        return grids

    def setFlowDirection(self, flowDirection: tuple[str, str]):
        """更新流体流向"""
        self.flowDirection = flowDirection
        self.flowlineSim.flowDirection = flowDirection


class Riser(Connection):
    """
    管网立管
    """

    def __init__(self, name, nodes: tuple[str, str]):
        super().__init__(name, ConnectionType.RISER, nodes)


class GridCell:
    """网格单元类：封装单个网格的位置、几何、流动参数"""

    def __init__(self, grid_id: int, start_pos: float, end_pos: float, start_horizontal_pos: float,
                 end_horizontal_pos: float,
                 start_vertical_pos: float, end_vertical_pos: float, ang: float = 0.0,
                 pipe_inside_diameter: float = 0.0,
                 pipe_outside_diameter: float = 0.0, pipe_wall_thickness: float = 0.0,
                 inner_pipe_outside_diameter: float = 0.0,
                 pipe_roughness: float = 0.0, pipeline_Mode: str = '', ambient_temperature: float = 0.0,
                 U_value_type: str = '',
                 pipe_heat_transfer_coefficient: float = 0.0):
        """

        """
        self.id = grid_id
        """网格编号"""
        self.start_pos = start_pos
        "# 起点测量距离（m）"
        self.end_pos = end_pos
        "# 终点测量距离（m）"
        self.length = end_pos - start_pos
        " # 网格长度"
        self.start_horizontal_pos = start_horizontal_pos
        "起点水平位置，m"
        self.end_horizontal_pos = end_horizontal_pos
        "终点水平位置，m"
        self.start_vertical_pos = start_vertical_pos
        "起点垂直位置，m"
        self.end_vertical_pos = end_vertical_pos
        "终点垂直位置，m"
        self.ang = ang
        """管道倾角，弧度制"""
        self.pipe_inside_diameter = pipe_inside_diameter
        """管道内径，m"""
        self.pipe_outside_diameter = pipe_outside_diameter
        """管道外径，m"""
        self.pipe_wall_thickness: float = pipe_wall_thickness
        """管道壁厚，m"""
        self.inner_pipe_outside_diameter: float = inner_pipe_outside_diameter
        """内管外径，m"""
        self.pipe_roughness: float = pipe_roughness
        """管壁粗糙度,m"""
        self.pipeline_Mode: str = pipeline_Mode
        '''管道模型'''
        self.ambient_temperature: float = ambient_temperature
        '''管道环境温度,K '''
        self.U_value_type: str = U_value_type
        '''U值类型 '''
        self.pipe_heat_transfer_coefficient: float = pipe_heat_transfer_coefficient
        '''管道传热系数, J/(s.degC.m2)'''


    def update_params(self, **kwargs):
        """动态更新流动参数（支持任意参数，如velocity、pressure等）"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"网格单元无属性：{key}")

    def __repr__(self):
        """打印网格信息时更直观"""
        return (f"GridCell(id={self.id}, start={self.start_pos:.3f}m, end={self.end_pos:.3f}m, "
                f"len={self.length:.3f}m, velocity={self.velocity:.3f}m/s), pressure={self.pressure:.3f}Pa, temperature={self.temperature:.1f}K")

    def calParams(self, fluid, flowRate, pressureStart, temperatureStart, flowIndex):

        flowRate=flowRate/100
        pressureEnd, temperatureEnd, Vs_gas, Vs_liquid, Qm, Hp_Slip_liquid_k1 = calc_PT_numerical(flowRate,
                                                                                                  pressureStart,
                                                                                                  temperatureStart,
                                                                                                  fluid,
                                                                                                  self.ang, self.length,
                                                                                                  self.pipe_roughness,
                                                                                                  self.pipe_outside_diameter,
                                                                                                  self.pipe_inside_diameter,
                                                                                                  self.ambient_temperature,
                                                                                                  self.pipe_heat_transfer_coefficient,
                                                                                                  flowIndex)
        return pressureEnd, temperatureEnd, Vs_gas, Vs_liquid, Qm, Hp_Slip_liquid_k1


class FlowLineSim():
    def __init__(self,nodes: tuple[str, str]):
        self.nodes: tuple[str, str] = nodes
        """管网连接的两端节点，也表示管道距离标定的起始位置"""
        self.flowDirection: tuple[str, str] = nodes
        """管网中流体流向"""
        self.grids: list[GridCell] = []
        self.fluid = None
        self.flowRate = 0.0
        self.P_start = 0.0
        self.P_end = 0.0
        self.T_start = 0.0
        self.T_end = 0.0
        self.verbose = False

        self.ProfileResult:pd.DataFrame =pd.DataFrame({"id":pd.Series(dtype=int),
                                                        "measure":pd.Series(dtype=float),
                                                        "horizontalPosition":pd.Series(dtype=float),
                                                        "verticalPosition":pd.Series(dtype=float),
                                                        "pressure":pd.Series(dtype=float),
                                                        "temperature":pd.Series(dtype=float)})
        '''存放管道剖面参数'''

    def setGrid(self, grids: list[GridCell]):
        self.grids = grids

    def set_initP(self, P):
        self.P_start = P

    def return_Pend(self):
        return self.P_end

    def calculateProfile(self, fluid, flowRate, P_start, T_start, flowDirection: tuple[str, str]):
        """
        根据当前管道参数计算管道沿程的温度、压力等参数
        @param: fluid
        """
        self.p_start = P_start
        self.T_start = T_start
        self.P_end = P_start
        self.T_end = T_start
        self.fluid = fluid
        self.flowRate = flowRate
        self.flowDirection = flowDirection
        # Oil_API = 32
        # Water_Cut = 0.2
        # Water_Specific_Gravity = 1
        # GOR = 160
        # Gas_Specific_Gravity = 0.75
        # Oil_C0 = 1884.06
        # Gas_C0 = 1884.06
        # Water_C0 = 4186.8
        # fluid = pvt_params(Oil_API, Water_Cut, GOR, Gas_Specific_Gravity, Water_Specific_Gravity,
        #                    Oil_C0, Gas_C0, Water_C0)
        # 定义表格标题和数据

        rowValue=[]

        P = P_start
        T = T_start
        for idx, grid in enumerate(self.grids):
            if self.verbose:
                print(f'start to calculate profile for grid {idx}')
            if idx == 0:
                newRow={'id':grid.id,'measure':grid.start_pos,'horizontalPosition':grid.start_horizontal_pos,
                        'verticalPosition':grid.start_vertical_pos,'pressure':P,'temperature':T}
                rowValue.append(newRow)
            flowIndex = 1
            if grid.end_vertical_pos - grid.start_vertical_pos >= 0:
                flowIndex = -1
            else:
                flowIndex = 1
            if self.flowDirection != self.nodes:
                flowIndex = -flowIndex

            P, T, Vs_gas, Vs_liquid, Qm, Hp_Slip_liquid_k1 = grid.calParams(fluid, flowRate, P, T, flowIndex)
            self.P_end = P
            self.T_end = T
            newRow = {'id': grid.id, 'measure': grid.end_pos, 'horizontalPosition': grid.end_horizontal_pos,
                      'verticalPosition': grid.end_vertical_pos, 'pressure': P, 'temperature': T}
            rowValue.append(newRow)

        self.ProfileResult = pd.DataFrame(rowValue)


    def getProfileResult(self):
        return self.ProfileResult

    def getFlowlineTwoEndParam(self,nodeName:str):
        if nodeName == self.nodes[0]:
            param=self.ProfileResult.iloc[0]
        else:
            param=self.ProfileResult.iloc[-1]
        return  param

    def getFlowlinePressureDrop(self):
        '''
        获取改管道的管道压降，管道开始节点端压力减去管道终止节点端压力
        '''
        pressureStart=self.ProfileResult.iloc[0]['pressure']

        pressureEnd=self.ProfileResult.iloc[-1]['pressure']

        return pressureStart-pressureEnd
