from enum import Enum
import pandas as pd
from Multiphaseflow.steady import *


def binarySearch(arr: np.ndarray, value: float) -> int:
    """
    手动实现二分查找，与bisect.bisect_left行为完全一致
    找到value应该插入的位置，使得左边所有元素<value，右边所有元素>=value

    Args:
        arr: 严格递增的有序数组
        value: 要查找的值

    Returns:
        插入位置索引
    """
    low = 0
    high = len(arr)

    while low < high:
        mid = (low + high) // 2
        if arr[mid] < value:
            low = mid + 1
        else:
            high = mid

    return low


class NodeType(Enum):
    '''节点类型'''
    SOURCE=1
    SINK=2
    WELL=3
    JUNCTION=4
    CHOKE=5
    CHECK_VALVE=6
    TWO_PHASE_SEP=7
    THREE_PHASE_SEP=8
    GENERIC_PUMP=9
    MULTIPHASE_BOOSTER=10
    HEAT_EXCHANGER=11
    COMPRESSOR=12
    EXPANDER=13
    GENERIC_EQUIPMENT=14
    MULTIPLIER_ADDER=15
    INJECTION_POINT=16
    SPOT_REPORT=17
    ENGINE_KEYWORDS=18
    USER_EQUIPMENGT=19

class BoundaryCondition:
    def __init__(self,  dicBoundaryCondition):
        self.name=dicBoundaryCondition['name']
        self.type=dicBoundaryCondition['type']
        self.completion=dicBoundaryCondition['completion']
        self.active=dicBoundaryCondition['active']
        self.pressure_abs=dicBoundaryCondition['pressureAbs']
        self.flowrate_type= dicBoundaryCondition['flowrateType']
        self.flowrate_value=dicBoundaryCondition['flowrateValue']
        self.temperature=dicBoundaryCondition['temperature']
        self.zone=dicBoundaryCondition['zone']
        self.is_pq_table=dicBoundaryCondition['isPqTable']
        self.fluid_type=dicBoundaryCondition['fluidType']
        self.gas_ratio_type=dicBoundaryCondition['gasRatioType']
        self.gas_ratio_value=dicBoundaryCondition['gasRatioValue']
        self.water_ratio_type=dicBoundaryCondition['waterRatioType']
        self.water_ratio_value=dicBoundaryCondition['waterRatioValue']

class  RateConstraint:
    def __init__(self, dicRateConstraint):
        self.name=dicRateConstraint['name']
        self.type=dicRateConstraint['type']
        self.max_liquid_stand=dicRateConstraint['maxLiquidStand']
        self.max_oil_stand=dicRateConstraint['maxOilStand']
        self.max_water_stand=dicRateConstraint['maxWaterStand']
        self.max_gas_stand=dicRateConstraint['maxGasStand']
        self.max_mass=dicRateConstraint['maxMassStand']

class NetworkNode:
    """管网节点父类"""
    def __init__(self,name:str,type:NodeType,isBoundaryNode:bool):
        self.name = name
        '''节点名称'''
        self.type:NodeType = type
        '''节点类型'''
        self.isBoundaryNode = isBoundaryNode
        '''是否是边界节点'''
        self.isConvergingNode = None
        '''是否是汇合节点，连接边大于2'''

        self.fluid:pvt_params= None
        """流体模型"""
        self.flowRate=10.0
        """节点质量流量"""
        self.temperature=0.0
        """节点温度，k"""
        self.pressure=0.0
        """节点压力，Pa"""
    def getName(self):
        return self.name
    def getType(self):
        return self.type
    def isBoundaryNode(self):
        return self.isBoundaryNode

    def updateNodeParam(self,fluid,flowRate,temperature,pressure):
        """更新求解过程变化参数"""
        self.fluid = fluid
        self.flowRate = flowRate
        self.temperature = temperature
        self.pressure = pressure


class Source(NetworkNode):
    """
    管网源节点
    """

    def __init__(self,name):
        super().__init__(name,NodeType.SOURCE,True)
        self.isActive =True
        '''是否是激活状态'''
        self.fluid=""
        '''流体模型'''
        self.isPQCurve = False
        '''是否使用pq曲线'''
        self.pressure = 0
        """压力,pa a"""
        self.temperature = 0
        '''温度，k'''
        self.boundaryType = ''
        '''边界流量类型，Liquid flowrate/Gas flowrate/Mass FLowrate'''
        self.gasFlow = 100.0
        """气相体积流量 ,sm³/s"""
        self.liquidFlow = 0
        '''液相体积流量，sm³/s'''
        self.massFlow = 0
        '''质量流量，kg/s'''
        self.pqCurveDataFrame=pd.DataFrame({"temperature":pd.Series(dtype=float),
                                            "pressure":pd.Series(dtype=float),
                                            "pqType":pd.Series(dtype=str),
                                            "gasFlowrate":pd.Series(dtype=float),
                                            "liquidFlowrate":pd.Series(dtype=float),
                                            "MassFlowrate":pd.Series(dtype=float)})
        '''存放不同温度、压力下三种流量数据，单位同上'''

    def setParam(self,isActive,fluid,isPQCurve,pressure,temperature,boundaryType,gasFlow,liquidFlow,massFlow,pqCurveDataFrame):
        """
        批量设置管网源节点（Source）的核心供给参数

        功能说明：
            配置源节点的激活状态、流体模型、压力温度基准值、流量边界类型，
            支持固定流量设置或 PQ 曲线（变工况流量）设置，定义管网的初始供给条件，
            直接影响管网整体的流量分配和压力分布计算。

        参数说明：
            :param isActive: bool - 节点是否激活（True=启用该源节点，参与管网供给；False=禁用，停止流体输出）
            :param fluid: str - 流体模型名称（需与系统预设的流体模型库一致，如 "NaturalGas"、"CrudeOil"、"Water"），
                               用于匹配流体的物理性质（密度、粘度、压缩因子等）
            :param isPQCurve: bool - 是否启用 PQ 曲线模式（True=按变工况曲线提供流量；False=使用固定流量参数）
            :param pressure: float - 源节点出口绝对压力，单位：帕斯卡（Pa a），
                                   PQ 曲线模式下为基准压力，固定流量模式下为恒定出口压力
            :param temperature: float - 源节点出口流体温度，单位：开尔文（K），
                                      需输入热力学温度（摄氏度转换公式：K = ℃ + 273.15），
                                      用于计算流体的实际体积流量和物理性质
            :param boundaryType: str - 流量边界约束类型（需与流量参数或 PQ 曲线数据匹配），可选值：
                                      - "Liquid flowrate"：液相体积流量约束
                                      - "Gas flowrate"：气相体积流量约束
                                      - "Mass FLowrate"：总质量流量约束
            :param gasFlow: float - 固定气相体积流量（标准状态），单位：标准立方米/秒（sm³/s），
                                   仅当 isPQCurve=False 且 boundaryType="Gas flowrate" 时生效，默认值 0.0
            :param liquidFlow: float - 固定液相体积流量（标准状态），单位：标准立方米/秒（sm³/s），
                                     仅当 isPQCurve=False 且 boundaryType="Liquid flowrate" 时生效，默认值 0.0
            :param massFlow: float - 固定总质量流量，单位：千克/秒（kg/s），
                                   仅当 isPQCurve=False 且 boundaryType="Mass FLowrate" 时生效，默认值 0.0
            :param pqCurveDataFrame: pd.DataFrame - PQ 曲线数据（变工况流量数据），默认值为空 DataFrame，
                                      仅当 isPQCurve=True 时生效，需包含以下列（列名严格匹配）：
                                      - "temperature"：工况温度（K）
                                      - "pressure"：工况压力（Pa a）
                                      - "gas_flowrate"：对应工况下的气相流量（sm³/s）
                                      - "liquid_flowrate"：对应工况下的液相流量（sm³/s）
                                      - "Mass_flowrate"：对应工况下的总质量流量（kg/s）

        返回值：
            :return: None - 无返回值，直接修改节点实例的属性

        注意事项：
            1. 模式互斥规则：isPQCurve=True 时，固定流量参数（gasFlow/liquidFlow/massFlow）失效，
               仅读取 pqCurveDataFrame 中的变工况数据；反之则仅使用固定流量参数。
            2. 单位一致性：压力为绝对压力（非表压），温度为开尔文（非摄氏度），流量为标准状态体积流量，
               需与管网整体仿真的单位体系保持一致（标准状态通常为 101325 Pa、20℃）。
            3. 流体模型匹配：fluid 参数需与系统支持的模型名称完全一致，否则会导致流体性质计算异常。
            4. PQ 曲线数据要求：DataFrame 需包含所有必填列，且数据类型为数值型（无空值、字符串），
               否则变工况计算时会抛出数据错误。
            5. 边界类型匹配：固定流量模式下，boundaryType 需与配置的流量参数严格对应（如选气相约束则仅 gasFlow 生效）。

        使用示例：
            # 示例1：固定气相流量模式（如天然气井）
            source_node = Source(name="well_A1")
            source_node.setParam(
                isActive=True,
                fluid="NaturalGas",  # 流体模型为天然气
                isPQCurve=False,     # 禁用 PQ 曲线，使用固定流量
                pressure=10000000.0, # 出口绝对压力 10MPa（10×10^6 Pa）
                temperature=313.15,  # 温度 40℃（40+273.15=313.15K）
                boundaryType="Gas flowrate",
                gasFlow=80.0,        # 固定气相流量 80 sm³/s
                liquidFlow=0.0,
                massFlow=0.0
            )

            # 示例2：PQ 曲线模式（变工况供给）
            import pandas as pd
            # 构建 PQ 曲线数据
            pq_data = pd.DataFrame({
                "temperature": [313.15, 318.15, 323.15],  # 40℃、45℃、50℃
                "pressure": [10000000.0, 9500000.0, 9000000.0],  # 10MPa、9.5MPa、9MPa
                "gas_flowrate": [80.0, 78.0, 75.0],
                "liquid_flow": [0.5, 0.6, 0.7],
                "mass_flow": [25.3, 24.8, 24.2]
            })
            source_node.setParam(
                isActive=True,
                fluid="NaturalGas",
                isPQCurve=True,      # 启用 PQ 曲线
                pressure=10000000.0, # 基准压力（变工况中可覆盖）
                temperature=313.15,  # 基准温度（变工况中可覆盖）
                boundaryType="Gas flowrate",
                pqCurveDataFrame=pq_data  # 传入变工况数据
            )
        """
        self.isActive = isActive
        self.fluid = fluid
        self.isPQCurve = isPQCurve
        self.pressure = pressure
        self.temperature = temperature
        self.boundaryType = boundaryType
        self.gasFlow = gasFlow
        self.liquidFlow = liquidFlow
        self.massFlow = massFlow
        self.pqCurveDataFrame = pqCurveDataFrame

    def setBoundary(self, dicBoundaryCondition):

        self.isActive=dicBoundaryCondition['active']
        self.pressure=dicBoundaryCondition['pressureAbs']
        self.temperature=dicBoundaryCondition['temperature']

        self.isPQCurve=dicBoundaryCondition['isPqTable']
        if not self.isPQCurve:
            self.boundaryType= dicBoundaryCondition['flowrateType']
            if 'gas' in self.boundaryType.lower().replace(" ",""):
                self.gasFlow = dicBoundaryCondition['flowrateValue']
            elif 'liquid' in self.boundaryType.lower().replace(" ",""):
                self.liquidFlow = dicBoundaryCondition['flowrateValue']
            else:
                self.massFlow = dicBoundaryCondition['flowrateValue']



        # self.gas_ratio_type=dicBoundaryCondition['gas_ratio_type']
        # self.gas_ratio_value=dicBoundaryCondition['gas_ratio_value']
        # self.water_ratio_type=dicBoundaryCondition['water_ratio_type']
        # self.water_ratio_value=dicBoundaryCondition['water_ratio_value']


    def CalculateHead(self):
        pass

    def volumeToMassFlow(self):
        if self.pqCurveDataFrame.empty:
            return

        # pqType=self.pqCurveDataFrame.iloc[0]['pqType']
        vec_func = np.vectorize(volume_to_mass_flow)
        if 'liquid' in self.boundaryType.lower().replace(" ",""):
            if self.isPQCurve:
                _, _, _, massFlowrate = vec_func(self.pqCurveDataFrame['liquidFlowrate'], self.boundaryType,
                                              self.pqCurveDataFrame['pressure'], self.temperature, self.fluid)
                self.pqCurveDataFrame['massFlowrate'] = massFlowrate
        elif 'gas' in self.boundaryType.lower().replace(" ",""):
            if self.isPQCurve:
                _, _, _,  massFlowrate = vec_func(self.pqCurveDataFrame['gasFlowrate'], self.boundaryType,
                                              self.pqCurveDataFrame['pressure'], self.temperature, self.fluid)
                self.pqCurveDataFrame['massFlowrate'] = massFlowrate
        else:
            pass

    def getPQMinMax(self):
        """
        获取当前源节点压力流量的最大最小值，根据pq曲线计算得到
        """

        if self.isPQCurve:
            pMin = self.pqCurveDataFrame['pressure'].min()
            pMax = self.pqCurveDataFrame['pressure'].max()
            qMin = self.pqCurveDataFrame['massFlowrate'].min()
            qMax = self.pqCurveDataFrame['massFlowrate'].max()
        else:
            raise RuntimeError(f"当节点未启用PQ曲线时，无法获取压力和流量的极值！")

        return pMin,pMax,qMin,qMax


    def getMassFlowRateByPressure(self,pressure):
        """
        根据pq曲线获取产量压力关系
        """
        if self.isPQCurve:
            # massFlowRate=5.92e-7*pressure+0.14

            cleaned_df = self.pqCurveDataFrame.sort_values('pressure', ignore_index=True)


            pressures = cleaned_df['pressure'].values.astype(np.float64)
            flows = cleaned_df['MassFlowrate'].values.astype(np.float64)

            # 手动二分查找定位区间
            idx = binarySearch(pressures, pressure)

            # 线性插值计算
            p0, p1 = pressures[idx - 1], pressures[idx]
            q0, q1 = flows[idx - 1], flows[idx]
            massFlowRate = q0 + (q1 - q0) * (pressure - p0) / (p1 - p0)
            return massFlowRate
        else:
            raise RuntimeError(f'未启用PQ曲线时，无法根据节点压力获取节点流量！')


    def getPressureByMassFlowRate(self,massFlowRate):
        """
        根据pq曲线获取产量压力关系
        """
        if self.isPQCurve:

            cleaned_df = self.pqCurveDataFrame.sort_values('massFlowrate', ignore_index=True)


            pressures = cleaned_df['pressure'].values.astype(np.float64)
            flows = cleaned_df['massFlowrate'].values.astype(np.float64)

            # 手动二分查找定位区间
            idx = binarySearch(flows, massFlowRate)

            # 线性插值计算
            q0, q1 = flows[idx - 1], flows[idx]
            p0, p1 = pressures[idx - 1], pressures[idx]

            pressure = p0 + (p1 - p0) * (massFlowRate - q0) / (q1 - q0)
            return pressure
        else:
            raise RuntimeError(f'未启用PQ曲线时，无法根据节点流量获取节点压力！')


class Sink(NetworkNode):
    """
    管网末节点
    """
    def __init__(self,name):
        super().__init__(name,NodeType.SINK,True)
        self.isActive =True
        '''是否是激活状态'''
        self.pressure = 0
        """压力,pa a"""
        self.boundaryType = ''
        '''边界流量类型，Liquid flowrate/Gas flowrate/Mass FLowrate'''
        self.gasFlow = 100.0
        """气相体积流量 ,sm³/s"""
        self.liquidFlow = 0
        '''液相体积流量，sm³/s'''
        self.massFlow = 0
        '''质量流量，kg/s'''

    def setParam(self,isActive,pressure,boundaryType,gasFlow,liquidFlow,massFlow):
        """
        批量设置管网末节点（Sink）的核心运行参数

        功能说明：
            配置 Sink 节点的激活状态、出口压力边界、流量约束类型及对应流量值，
            定义管网终端的流体输出条件，直接影响整个管网的流动平衡计算。

        参数说明：
            :param isActive: bool - 节点是否激活（True=启用该末节点，参与管网计算；False=禁用，不参与流动分配）
            :param pressure: float - 节点出口绝对压力，单位：帕斯卡（Pa a），需输入绝对压力值（非表压）
            :param boundaryType: str - 边界流量约束类型（必须与实际配置的流量参数匹配），可选值：
                                      - "Liquid flowrate"：液相体积流量约束（需配置 liquidFlow 参数）
                                      - "Gas flowrate"：气相体积流量约束（需配置 gasFlow 参数）
                                      - "Mass FLowrate"：质量流量约束（需配置 massFlow 参数）
            :param gasFlow: float - 气相体积流量（标准状态），单位：标准立方米/秒（sm³/s），
                                   仅当 boundaryType 为 "Gas flowrate" 时生效，默认值 0.0
            :param liquidFlow: float - 液相体积流量（标准状态），单位：标准立方米/秒（sm³/s），
                                     仅当 boundaryType 为 "Liquid flowrate" 时生效，默认值 0.0
            :param massFlow: float - 总质量流量，单位：千克/秒（kg/s），
                                   仅当 boundaryType 为 "Mass FLowrate" 时生效，默认值 0.0

        返回值：
            :return: None - 无返回值，直接修改节点实例的属性

        注意事项：
            1. 压力单位为绝对压力（Pa a），若实际输入为表压（Pa g），需转换为绝对压力（绝对压力 = 表压 + 当地大气压，标准大气压取 101325 Pa）；
            2. boundaryType 必须与配置的流量参数严格匹配：
               - 选择 "Gas flowrate" 时，仅 gasFlow 生效，liquidFlow 和 massFlow 会被忽略（建议设为 0.0）；
               - 选择 "Liquid flowrate" 时，仅 liquidFlow 生效，其他流量参数建议设为 0.0；
               - 选择 "Mass FLowrate" 时，仅 massFlow 生效，需确保该值为气液两相总质量流量；
            3. 流量单位为标准状态下的体积流量（sm³/s），标准状态通常指 101325 Pa、20℃（需与管网整体仿真标准一致）；
            4. 若节点未激活（isActive=False），流量参数将失效，该节点不参与管网流动计算，需谨慎设置。

        使用示例：
            # 实例化 Sink 节点
            sink_node = Sink(name="Gas_Sales")
            # 配置：激活节点，出口绝对压力 5MPa，气相流量约束 150 sm³/s
            sink_node.setParam(
                isActive=True,
                pressure=5000000.0,  # 5MPa = 5×10^6 Pa（绝对压力）
                boundaryType="Gas flowrate",
                gasFlow=150.0,
                liquidFlow=0.0,
                massFlow=0.0
            )
        """
        self.isActive = isActive
        self.pressure = pressure
        self.boundaryType = boundaryType
        self.gasFlow = gasFlow
        self.liquidFlow = liquidFlow
        self.massFlow = massFlow

    def setBoundary(self, dicBoundaryCondition):

        self.isActive=dicBoundaryCondition['active']
        self.pressure=dicBoundaryCondition['pressureAbs']
        self.boundaryType= dicBoundaryCondition['flowrateType']
        if 'gas' in self.boundaryType.lower().replace(" ",""):
            self.gasFlow = dicBoundaryCondition['flowrateValue']
        elif 'liquid' in self.boundaryType.lower().replace(" ",""):
            self.liquidFlow = dicBoundaryCondition['flowrateValue']
        else:
            self.massFlow = dicBoundaryCondition['flowrateValue']


    def CalculateHead(self):
        pass

class Junction(NetworkNode):
    """
    连接节点
    """
    def __init__(self,name):
        super().__init__(name,NodeType.JUNCTION,False)
        self.isTreatAsSource =False
        '''是否视为源节点'''
        self.fluid=""
        '''流体模型'''
        self.pressure = 0
        """压力,pa a"""
        self.temperature = 0
        '''温度，k'''
        self.boundaryType = ''
        '''边界流量类型，Liquid flowrate/Gas flowrate/Mass FLowrate'''
        self.gasFlow = 100.0
        """气相体积流量 ,sm³/s"""
        self.liquidFlow = 0
        '''液相体积流量，sm³/s'''
        self.massFlow = 0
        '''质量流量，kg/s'''

class Choke(NetworkNode):
    """
    节流阀
    """
    def __init__(self,name):
        super().__init__(name,NodeType.CHOKE,False)
        self.isActive =True
        '''是否是激活状态'''
        self.subCriticalCorrelation = ''
        """亚临界相关性"""
        self.criticalCorrelation = ''
        '''临界相关性'''
        self.beanSize = 0
        """节流孔内径，m"""
        self.criticalPressureRatioType = 0
        '''临界压力比类型'''
        self.criticalPressureRatio = 0
        '''临界压力比'''
        self.tolerance = 0
        '''容错,%'''
        self.upstreamPipeID = 0
        '''上游管道内径,m'''
        self.gasPhaseFlowCoefficient = 0
        '''气相流量系数 '''
        self.liquidPhaseFlowCoefficient = 0
        '''液相流量系数'''
        self.dischargeCoefficient = 0
        '''排放系数'''
        self.fluidHeatCapacityRatio = 0
        '''流体比热容'''
        self.YCriticalPoint = 0
        '''临界点Y值'''
        self.isFlowrateIdentify = False
        '''临界和超临界流量识别'''
        self.isPressureRatioIdentify = False
        '''临界和超临界压力比识别'''
        self.isSonicUpstreamVelocityIdentify = False
        '''临界和超临界上游声速识别'''
        self.isSonicDownstreamVelocityIdentify = False
        '''临界和超临界下游声速识别'''
        self.isAdjustSubCriticalCorrelation = False
        ''' 是否调整亚临界相关性'''
        self.isPrintDetailedCalculations = False
        """是否打印详细计算结果"""

    def setParam(self,isActive,subCriticalCorrelation,criticalCorrelation,beanSize,
                 criticalPressureRatioType,criticalPressureRatio,tolerance,upstreamPipeID,
                 gasPhaseFlowCoefficient,liquidPhaseFlowCoefficient,dischargeCoefficient,
                 fluidHeatCapacityRatio,YCriticalPoint,isFlowrateIdentify,isPressureRatioIdentify,
                 isSonicUpstreamVelocityIdentify,isSonicDownstreamVelocityIdentify,
                 isAdjustSubCriticalCorrelation,isPrintDetailedCalculations):
        """
          批量设置节流阀（CHOKE）节点的核心参数

          功能说明：
              一次性配置节流阀的激活状态、流动相关性模型、结构尺寸、流量系数、临界判断条件等所有关键参数，
              替代逐个属性赋值，适用于初始化配置或动态调整节流阀参数场景。

          参数说明：
              :param isActive: bool - 节流阀是否激活（True=启用，False=禁用）
              :param subCriticalCorrelation: str - 亚临界流动相关性模型（如选用的经验公式/计算模型名称）
              :param criticalCorrelation: str - 临界流动相关性模型（如选用的经验公式/计算模型名称）
              :param beanSize: float - 节流孔内径，单位：米（m）
              :param criticalPressureRatioType: int - 临界压力比类型（0=默认类型，1/2=自定义类型，需结合业务场景定义）
              :param criticalPressureRatio: float - 临界压力比（节流阀临界状态下的上下游压力比值）
              :param tolerance: float - 计算容错率，单位：百分比（%），用于控制数值迭代计算的收敛精度
              :param upstreamPipeID: float - 上游管道内径，单位：米（m），用于匹配管道与节流阀的流道尺寸
              :param gasPhaseFlowCoefficient: float - 气相流量系数（修正气相通过节流阀的流量损失）
              :param liquidPhaseFlowCoefficient: float - 液相流量系数（修正液相通过节流阀的流量损失）
              :param dischargeCoefficient: float - 排放系数（综合修正节流阀的流量能力，与流道形状、流体性质相关）
              :param fluidHeatCapacityRatio: float - 流体比热容比（定压比热容与定容比热容的比值，用于临界流动判断）
              :param YCriticalPoint: float - 临界点Y值（临界流动状态下的流体膨胀因子，用于流量计算）
              :param isFlowrateIdentify: bool - 是否启用临界/超临界流量识别（True=自动判断流量状态，False=禁用自动判断）
              :param isPressureRatioIdentify: bool - 是否启用临界/超临界压力比识别（True=自动判断压力比状态，False=禁用）
              :param isSonicUpstreamVelocityIdentify: bool - 是否启用上游声速识别（True=检测上游流体是否达到声速，False=禁用）
              :param isSonicDownstreamVelocityIdentify: bool - 是否启用下游声速识别（True=检测下游流体是否达到声速，False=禁用）
              :param isAdjustSubCriticalCorrelation: bool - 是否调整亚临界相关性模型（True=动态修正模型参数，False=使用默认模型）
              :param isPrintDetailedCalculations: bool - 是否打印详细计算结果（True=输出迭代过程、中间参数等，False=仅输出最终结果）

          返回值：
              :return: None - 无返回值，直接修改实例属性

          注意事项：
              1. 数值型参数（如 beanSize、criticalPressureRatio）需确保单位一致性（均为国际单位制）；
              2. 相关性模型参数（subCriticalCorrelation/criticalCorrelation）需与系统支持的模型名称一致，否则可能导致计算异常；
              3. 布尔型参数（如 isActive、isFlowrateIdentify）直接控制功能开关，需根据仿真需求合理设置；
              4. 容错率（tolerance）建议设置为 0.01~1.0（对应 0.01%~1%），过小可能导致计算不收敛，过大可能影响结果精度。
          """
        self.isActive = isActive
        self.subCriticalCorrelation = subCriticalCorrelation
        self.criticalCorrelation = criticalCorrelation
        self.beanSize = beanSize
        self.criticalPressureRatioType = criticalPressureRatioType
        self.criticalPressureRatio = criticalPressureRatio
        self.tolerance = tolerance
        self.upstreamPipeID = upstreamPipeID
        self.gasPhaseFlowCoefficient = gasPhaseFlowCoefficient
        self.liquidPhaseFlowCoefficient = liquidPhaseFlowCoefficient
        self.dischargeCoefficient = dischargeCoefficient
        self.fluidHeatCapacityRatio = fluidHeatCapacityRatio
        self.YCriticalPoint = YCriticalPoint
        self.isFlowrateIdentify = isFlowrateIdentify
        self.isPressureRatioIdentify = isPressureRatioIdentify
        self.isSonicUpstreamVelocityIdentify = isSonicUpstreamVelocityIdentify
        self.isSonicDownstreamVelocityIdentify = isSonicDownstreamVelocityIdentify
        self.isAdjustSubCriticalCorrelation = isAdjustSubCriticalCorrelation
        self.isPrintDetailedCalculations = isPrintDetailedCalculations



