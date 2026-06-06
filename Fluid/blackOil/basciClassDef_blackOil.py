import math

def calc_Z(T, P, Gas_Specific_Gravity):
    '''Standing经验公式计算Z因子'''
    # 计算临界压力
    critical_Pressure = 709.604 - 58.718 * Gas_Specific_Gravity
    # 计算临界温度
    critical_Temperature = 170.491 + 307.344 * Gas_Specific_Gravity
    # 计算对比温度
    T_pr = T / critical_Temperature
    # 计算对比压力
    P_pr = P / critical_Pressure
    # 计算 A 参数
    A = 1.39 * (T_pr - 0.92) ** 0.5 - 0.36 * T_pr - 0.1
    # 计算 F 参数
    F = 0.3106 - 0.49 * T_pr + 0.1824 * T_pr ** 2
    # 计算 E 参数
    E = 9 * (T_pr - 1)
    # 计算 B 参数
    B = (0.62 - 0.23 * T_pr) * P_pr + (0.066 / (T_pr - 0.86) - 0.037) * P_pr ** 2 + (0.32 * P_pr ** 6 / (10 ** E))
    # 计算 C 参数
    C = 0.132 - 0.32 * math.log10(T_pr)
    # 计算 D 参数
    D = 10 ** F
    # 计算 Z 因子
    # 根据前面计算得到的 A、B、C、D 和 P_pr 参数，使用公式计算 Z 因子
    Z_Factor = A + (1 - A) / math.exp(B) + C * (P_pr ** D)
    # 返回计算得到的 Z 因子
    return Z_Factor

def calc_Rs(T, P, Gas_Specific_Gravity, Oil_API):
    '''
    天然气溶解气油比 Rs
    :param T: 温度
    :param P: 压力
    :param Gas_Specific_Gravity: 天然气相对密度
    :param Oil_API: 原油 API 度
    :return: 天然气溶解气油比 Rs
    '''
    # 将温度从华氏度转换为兰金度
    t = T - 460
    # 根据 Standing 公式计算天然气溶解气油比 Rs
    Rs = Gas_Specific_Gravity * ((P / 18) * (10 ** (0.0125 * Oil_API)) / (10 ** (0.00091 * t))) ** 1.2048
    # 返回计算得到的天然气溶解气油比 Rs
    return Rs

def calc_Bg(Z_Factor, T, P):
    '''
    气体体积因子（单位：RB/SCF），表示地下气体体积与标准条件的体积比
    :param Z_Factor: Z 因子
    :param T: 温度
    :param P: 压力
    :return: 气体体积因子 Bg
    '''
    # 根据公式计算气体体积因子 Bg
    Bg = 0.00504 * Z_Factor * T / P
    # 返回计算得到的气体体积因子 Bg
    return Bg

def calc_Bo(T, Rs, Gas_Specific_Gravity, Oil_Specific_Gravity):
    """
    计算原油体积系数 Bo
    :param T: 温度，单位为华氏度
    :param Rs: 天然气溶解气油比
    :param Gas_Specific_Gravity: 天然气相对密度
    :param Oil_Specific_Gravity: 原油相对密度
    :return: 原油体积系数 Bo
    """
    # 将温度从华氏度转换为兰金度
    t = T - 460
    # 使用经验公式计算原油体积系数 Bo
    Bo = 0.9759 + 0.00012 * (Rs * math.sqrt(Gas_Specific_Gravity / Oil_Specific_Gravity) + 1.25 * t) ** 1.2
    # 返回计算得到的原油体积系数 Bo
    return Bo


def calc_Bw(T, P):
    """
    计算水体积系数 Bw
    :param T: 华氏温度
    :param P: 压力
    :return: 水体积系数 Bw
    """
    # 将华氏温度转换为兰金温度
    t = T - 460
    # 计算 VwP，与温度相关的参数
    VwP = -1.0001 * 0.01 + 1.33391 * (10 ** (-4)) * t + 5.50654 * (10 ** (-7)) * (t ** 2)
    # 计算 VwT，与温度和压力相关的参数
    VwT = -1.95301 * (10 ** (-9)) * T * P - 1.72834 * (10 ** (-13)) * (
            P ** 2) * t - 3.58922 * (10 ** (-7)) * P - 2.25341 * (
                  10 ** (-10)) * (P ** 2)
    # 计算水体积系数 Bw，考虑温度和压力对水体积的影响
    Bw = (1 + VwT) * (1 + VwP)
    # 返回计算得到的水体积系数 Bw
    return Bw


def calc_Pb(Rs, Gas_Specific_Gravity, Oil_Specific_Gravity, T):
    """
    计算泡点压力 Pb
    :param Rs: 天然气溶解气油比
    :param Gas_Specific_Gravity: 天然气相对密度
    :param Oil_Specific_Gravity: 原油相对密度
    :param T: 温度
    :return: 泡点压力 Pb
    """
    # 根据经验公式计算泡点压力
    Pb = (5.38088 * (10 ** (-3))) * (Rs ** 0.715082) * (Gas_Specific_Gravity ** (-1.877840)) * (
            Oil_Specific_Gravity ** 3.1437) * (T ** 1.32657)
    return Pb

def calc_den_gas(T, P, Gas_Specific_Gravity, Z_Factor):
    """
    计算气体密度
    :param T: 温度
    :param P: 压力
    :param Gas_Specific_Gravity: 天然气相对密度
    :param Z_Factor: Z 因子
    :return: 气体密度
    """
    # 根据公式计算气体密度，公式基于理想气体状态方程变形而来
    Gas_Density = 28.967 * Gas_Specific_Gravity * P / (10.732 * T * Z_Factor)
    return Gas_Density

def calc_oil_viscosity(T, P, Oil_API, Rs, Pb):
    """
    计算油的黏度
    :param T: 温度
    :param P: 压力
    :param Oil_API: 原油 API 度
    :param Rs: 天然气溶解气油比
    :param Pb: 泡点压力
    :return: 油的黏度
    """
    # 计算 A 参数，用于后续脱气油黏度的计算
    A = math.pow(10, 0.43 + 8.33 / Oil_API)
    # 计算脱气油黏度，即不含溶解气时油的黏度
    DeadOil_Viscosity = (0.32 + (1.8 * math.pow(10, 7)) / (math.pow(Oil_API, 4.53))) * math.pow(
        360 / (T - 260), A)
    # 计算中间参数 a，与溶解气油比相关
    a = Rs * (2.2 * math.pow(10, -7) * Rs - 7.4 * math.pow(10, -4))
    # 计算中间参数 c，与溶解气油比相关
    c = 8.62 * math.pow(10, -5) * Rs
    # 计算中间参数 d，与溶解气油比相关
    d = 1.1 * math.pow(10, -3) * Rs
    # 计算中间参数 e，与溶解气油比相关
    e = 3.74 * math.pow(10, -3) * Rs
    # 计算中间参数 b，基于前面计算的 c、d、e 参数
    b = (0.68 / math.pow(10, c)) + (0.25 / math.pow(10, d)) + (0.062 / math.pow(10, e))
    # 计算泡点黏度，即压力等于泡点压力时油的黏度
    BubblePoint_Viscosity = math.pow(10, a) * math.pow(DeadOil_Viscosity, b)
    # 计算油黏度，考虑压力与泡点压力的差值对黏度的影响
    Oil_Viscosity = BubblePoint_Viscosity + 0.001 * (P - Pb) * (
            0.024 * math.pow(BubblePoint_Viscosity, 1.6) + 0.38 * math.pow(BubblePoint_Viscosity, 0.56))
    # 返回计算得到的油黏度
    return Oil_Viscosity


def calc_water_viscosity(T):
    """
    计算水的黏度
    :param T: 温度，单位为华氏度
    :return: 水的黏度
    """
    # 华氏温度转换为兰金温度，用于后续计算
    rankine_temp = T - 460
    # 使用经验公式计算水的黏度
    Water_Viscosity = math.exp(1.003 - 1.479 * 0.01 * rankine_temp + 1.982 * math.pow(10, -5) * math.pow(
        rankine_temp, 2))
    return Water_Viscosity

def calc_gas_viscosity(T, Gas_molecular_weight, Gas_Density):
    """
    计算气体的黏度
    :param T: 温度
    :param Gas_molecular_weight: 气体的分子量
    :param Gas_Density: 气体的密度
    :return: 气体的黏度
    """
    # 计算 K 参数，用于后续气体黏度的计算
    K = (9.4 + 0.02 * Gas_molecular_weight) * (T ** 1.5) / (
            209 + 19 * Gas_molecular_weight + T)
    # 计算 X 参数，用于后续气体黏度的计算
    X = 3.5 + (986 / T) + 0.01 * Gas_molecular_weight
    # 计算 Y 参数，用于后续气体黏度的计算
    Y = 2.4 - 0.2 * X
    # 根据前面计算的 K、X、Y 参数以及气体密度计算气体黏度
    Gas_Viscosity = (math.pow(10, -4)) * K * math.exp(X * math.pow((Gas_Density / 62.4), Y))
    # 返回计算得到的气体黏度
    return Gas_Viscosity


def calc_den_liquid(Oil_Specific_Gravity, Rs, Gas_Specific_Gravity, Bo, Water_Cut, Water_Specific_Gravity, Bw):
    """
    计算液体密度
    :param Oil_Specific_Gravity: 原油相对密度
    :param Rs: 天然气溶解气油比
    :param Gas_Specific_Gravity: 天然气相对密度
    :param Bo: 原油体积系数
    :param Water_Cut: 含水率
    :param Water_Specific_Gravity: 水的相对密度
    :param Bw: 水的体积系数
    :return: 液体密度
    """
    # 计算液体密度，公式考虑了原油和水的贡献
    Liquid_Density = (((62.4 * Oil_Specific_Gravity + (Rs * 0.0764 * Gas_Specific_Gravity / 5.615)) / Bo) * (
                1 - Water_Cut) + (
                              62.4 * Water_Specific_Gravity / Bw) * Water_Cut)
    return Liquid_Density

def calc_oil_gas_surface_tension(T, P, API):
    """
    计算油 - 气表面张力
    :param T: 温度，单位为华氏度(°F)
    :param P: 压力，单位为磅力每平方英寸(psia)
    :return: 水 - 气表面张力，单位为达因/厘米(dynes/cm)
    """
    part1 = 37.7 - 0.05 * (T - 100) - 0.26 * API
    part2 = 1 - 7.1 * 10 ** (-4) * P + 2.1 * 10 ** (-7) * P ** 2 + 2.37 * 10 ** (-11) * P ** 3
    sigma_og = part1 * part2
    return sigma_og


def calc_water_gas_surface_tension(T, P):
    """
    计算水 - 气表面张力
    :param T: 温度，单位为华氏度(°F)
    :param P: 压力，单位为磅力每平方英寸(psia)
    :return: 水 - 气表面张力，单位为达因/厘米(dynes/cm)
    """
    sigma_wg = 70 - 0.1 * (T - 74) - 0.002 * P
    return sigma_wg
def calc_liquid_gas_surface_tension(sigma_og, sigma_wg, water_cut):
    """
    计算液 - 气表面张力
    :param sigma_o: 油 - 气表面张力，单位为达因/厘米(dynes/cm)
    :param sigma_w: 水 - 气表面张力，单位为达因/厘米(dynes/cm)
    :param water_cut: 含水率(%), 即water cut
    :return: 液 - 气表面张力，单位为达因/厘米(dynes/cm)
    """
    if water_cut < 0.6:
        sigma_lg = sigma_og
    else:
        sigma_lg = sigma_og * (1 - water_cut ) + sigma_wg * water_cut
    return sigma_lg



class pvt_params:
    """
    该类用于存储和计算与 PVT（压力 - 体积 - 温度）相关的参数。
    这些参数包括油、水和气的属性，以及根据这些属性计算的各种物理量。
    """
    def __init__(self, oil_api, water_cut, GOR, gas_specific_gravity,Water_Specific_Gravity=1):
        """
        初始化 pvt_params 类的实例。

        :param oil_api: 原油 API 度，用于衡量油的相对密度。
        :param water_cut: 体积含水率，范围在 0 到 1 之间。
        :param GOR: 生产气油比，即总气（溶解气 + 游离气）产量除以油产量。
        :param gas_specific_gravity: 天然气相对密度。
        :param Water_Specific_Gravity: 水的相对密度，默认为 1。
        """
        # 油的属性
        self.Oil_API = oil_api
        '''油的 API 度，用于衡量油的相对密度'''
        self.Oil_Specific_Gravity = 141.5 / (oil_api + 131.5)
        '''API 度计算油的比重'''
        # 水的属性
        self.Water_Cut = water_cut
        '''体积含水率，范围在 0 到 1 之间'''
        self.Water_Specific_Gravity = Water_Specific_Gravity
        '''水比重'''
        # 气的属性
        self.GOR = GOR
        '''生产气油比，即总气（溶解气 + 游离气）产量除以油产量'''
        # 根据气油比和油产量计算气的产量，单位为标准立方英尺/天（scf/D）
        self.Gas_Specific_Gravity = gas_specific_gravity
        # 根据气的比重计算气的分子量，单位为磅/磅摩尔（Ibm/Ibmole）
        self.Gas_molecular_weight = 28.97 * gas_specific_gravity



    def calc(self,P: float,T: float):
        # 单位转换
        T = T * 1.8 + 32+460
        '''温度摄氏度转华氏度'''
        P = P*0.000145038
        '''Pa转psi'''
        Zg=calc_Z(T, P, self.Gas_Specific_Gravity)
        Rs=calc_Rs(T, P, self.Gas_Specific_Gravity, self.Oil_API)
        Bg=calc_Bg(Zg,T, P)
        Bo=calc_Bo(T, Rs, self.Gas_Specific_Gravity, self.Oil_Specific_Gravity)
        Bw=calc_Bw(T, P)
        Pb=calc_Pb(Rs, self.Gas_Specific_Gravity, self.Oil_Specific_Gravity, T)
        oil_viscosity=calc_oil_viscosity(T, P, self.Oil_API, Rs, Pb)
        water_viscosity=calc_water_viscosity(T)
        den_gas = calc_den_gas(T, P, self.Gas_Specific_Gravity, Zg)
        gas_viscosity=calc_gas_viscosity(T, self.Gas_molecular_weight, den_gas)
        den_liquid=calc_den_liquid(self.Oil_Specific_Gravity, Rs, self.Gas_Specific_Gravity, Bo, self.Water_Cut, self.Water_Specific_Gravity, Bw)
        # 单位转换
        den_gas = den_gas*16.02
        '''lbm/ft^3换算为kg/m^3'''
        den_liquid = den_liquid*16.02
        '''lbm/ft^3换算为kg/m^3'''
        # 粘度单位为cp
        T = T-460
        sigma_og = calc_oil_gas_surface_tension(T, P, self.Oil_API)
        sigma_wg = calc_water_gas_surface_tension(T, P)
        sigma_lg = calc_liquid_gas_surface_tension(sigma_og, sigma_wg, self.Water_Cut)
        # 修改代码，保存6位小数
        return (round(den_gas, 6), round(den_liquid, 6), round(oil_viscosity, 6),
                round(water_viscosity, 6), round(gas_viscosity, 6), round(Bo, 6),
                round(Bg, 6), round(Bw, 6), round(Zg, 6), round(sigma_lg, 6),
                round(sigma_og, 6), round(sigma_wg, 6), round(Pb, 6))


if __name__ == '__main__':
    Oil_API = 30
    Water_Cut = 0.5
    Water_Specific_Gravity = 1
    GOR = 345
    Gas_Specific_Gravity = 0.75
    fluid = pvt_params(Oil_API, Water_Cut, GOR, Gas_Specific_Gravity,Water_Specific_Gravity)
    P =6*1000000
    # Pa
    T = 25
    # ℃
    den_gas,den_liquid,oil_viscosity,water_viscosity,gas_viscosity,Bo,Bg,Bw,Zg,sigma_lg,sigma_og,sigma_wg,Pb = fluid.calc(P,T)
    # 定义表格标题和数据
    headers = ["参数", "值", "单位"]
    data = [
        ["气体密度", den_gas, "kg/m^3"],
        ["液体密度", den_liquid, "kg/m^3"],
        ["油黏度", oil_viscosity, "cp"],
        ["水黏度", water_viscosity, "cp"],
        ["气黏度", gas_viscosity, "cp"],
        ["油体积系数", Bo, "/"],
        ["水体积系数", Bw, "/"],
        ["气体积系数", Bg, "/"],
        ["气体压缩因子", Zg, "/"],
        ["油气表面张力", sigma_og, "dynes/cm"],
        ["水气表面张力", sigma_wg, "dynes/cm"],
        ["液气表面张力", sigma_lg, "dynes/cm"],
        ["饱和压力", round(Pb/0.000145038/1000000,4), "MPa"]
    ]

    # 把温度和压力信息添加到表头前
    header_prefix = f"温度: {T} ℃, 压力: {P} Pa"
    print(header_prefix)

    # 计算每列的最大宽度
    col_widths = [max(len(str(row[i])) for row in [headers] + data) for i in range(len(headers))]

    # 打印表头
    print("| " + " | ".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers)) + " |")
    print("|-" + "-|-".join("-" * width for width in col_widths) + "-|")

    # 打印数据行
    for row in data:
        print("| " + " | ".join(f"{str(value):<{col_widths[i]}}" for i, value in enumerate(row)) + " |")




