import json
import os
import time
from scipy.optimize import brentq, newton, least_squares
import pandas as pd
import math
import numpy as np
# import sympy
import numbers
import matplotlib.pyplot as plt
from .EOS import EOS_init



R = 8.314
T0 = 273.15 + 20
P0 = 101.325 * 1000

class Plot:
    def __init__(self, PD,TD,TB,PB):
        self.x =TD
        self.y = PD
        self.x2=TB
        self.y2=PB
        pipesimphase_plot_data = pd.read_excel("D:\PVTtool\Postman 模拟\相图接口字段.xlsx",sheet_name="Plot")
        pipesimPhase_data = pipesimphase_plot_data[["Dew Temperature", "Dew Pressure", "Bubble Temperature", "Bubble Pressure"]]
        pipesimPhase_data = pipesimPhase_data.dropna()


        plt.figure(figsize=(10, 6))
        plt.plot(
            self.x,  # x轴：露点温度（K）
            self.y,  # y轴：压力（MPa）
            linestyle='-',  # 实线连接
            color='red',  # 线条颜色
            linewidth=2,  # 线条粗细
            # marker='o',  # 散点标记（圆形）
            # markerfacecolor='white',  # 标记填充色
            # markeredgecolor='red',  # 标记边缘色
            # markersize=5  # 标记大小
        )
        plt.plot(
            self.x2,  # x轴：露点温度（K）
            self.y2,  # y轴：压力（MPa）
            linestyle='-',  # 实线连接
            color='blue',  # 线条颜色
            linewidth=2,  # 线条粗细
        )
        plt.xlabel('温度 (K)', fontsize=12)
        plt.ylabel('压力 (Pa a)', fontsize=12)
        plt.title('相包络线', fontsize=14)
        plt.grid(linestyle='--', alpha=0.6)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.tight_layout()



        '''Pipesim数据'''
        plt.plot(
            pipesimPhase_data["Dew Temperature"],
            pipesimPhase_data["Dew Pressure"]*1e6,
            linestyle='-',  # 实线连接
            color='red',  # 线条颜色
            linewidth=2,  # 线条粗细
            marker='o',  # 散点标记（圆形）
            markerfacecolor='white',  # 标记填充色
            markeredgecolor='red',  # 标记边缘色
            markersize=5  # 标记大小
        )
        plt.xlabel('温度 ', fontsize=12)
        plt.ylabel('压力', fontsize=12)
        plt.grid(linestyle='--', alpha=0.6)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.tight_layout()

        plt.plot(
            pipesimPhase_data["Bubble Temperature"],
            pipesimPhase_data["Bubble Pressure"]*1e6,
            linestyle='-',  # 实线连接
            color='Blue',  # 线条颜色
            linewidth=2,  # 线条粗细
            marker='o',  # 散点标记（圆形）
            markerfacecolor='white',  # 标记填充色
            markeredgecolor='red',  # 标记边缘色
            markersize=5  # 标记大小
        )
        plt.xlabel('温度 ', fontsize=12)
        plt.ylabel('压力', fontsize=12)
        plt.grid(linestyle='--', alpha=0.6)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.tight_layout()

        plt.show()



class dataprocess:
    def __init__(self,PD,TD,TB,PB):

        self.PD = np.array(PD)
        self.TD = np.array(TD)
        self.PD=self.PD

        self.TB = np.array(TB)
        self.PB = np.array(PB)
        self.PB=self.PB

        self.slope=np.ones(len(self.TD)-1)
        self.BUslope=np.ones(len(self.PB)-1)

    def DEWprocess(self):
        for i in range(len(self.TD)):
            if i>0:
                self.slope[i-1]=(self.TD[i]-self.TD[i-1])/(self.PD[i]-self.PD[i-1])
        print(self.slope)
        for i in range(len(self.slope)-1):
            if (self.slope[i] > 0 and self.slope[i + 1] < 0 and abs(self.slope[i] - self.slope[i + 1]) > 10) or (self.slope[i]<0 and self.slope[i+1]>0 and abs(self.slope[i]-self.slope[i+1])>10):
                self.PD = self.PD[:i+1]
                self.TD = self.TD[:i+1]
                self.slope = self.slope[:i+1]
                break
        return self.PD,self.TD,self.slope

    def BUBBLEprocess(self):
        for i in range(len(self.TB)):
            if i>0:
                self.BUslope[i-1]=(self.PB[i]-self.PB[i-1])/(self.TB[i]-self.TB[i-1])
        print(self.BUslope)
        for i in range(len(self.BUslope)-1):
            if (self.BUslope[i] > 0 and self.BUslope[i + 1] < 0 and abs(self.BUslope[i] - self.BUslope[i + 1]) > 10) or (self.BUslope[i]<0 and self.BUslope[i+1]>0 and abs(self.BUslope[i]-self.BUslope[i+1])>10):
                self.TB = self.TB[:i+1]
                self.PB = self.PB[:i+1]
                self.BUslope = self.BUslope[:i+1]

                break
        return self.TB,self.PB,self.BUslope




def Dichotomy( f, a, b, epsilon, type, max_iter=50):
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        vale_mid = np.nan
        print(f"\n使用威尔逊计算Ki并采用二分法求解{type}点压力失败，区间两端函数值同号：F(a)={fa:.6f}, F(b)={fb:.6f}"
              f"\n更换为闪蒸相平衡计算后求解初值"
              )
        return vale_mid
        # raise ValueError(
        #     f"区间 [{a:.2e}, {b:.2e}] Pa 两端函数值同号："
        #     f"F(a)={fa:.6f}, F(b)={fb:.6f}，无法用二分法求解"
        # )
    iter_count = 0
    converged = False
    while iter_count < max_iter:
        vale_mid = (a + b) / 2.0
        f_mid = f(vale_mid)
        # if abs(f_mid) < epsilon :
        if abs(max(a, b) - min(a, b)) < epsilon or abs(f_mid) < epsilon:
            converged = True
            break
        if fa * f_mid < 0:
            b = vale_mid
            fb = f_mid
        else:
            a = vale_mid
            fa = f_mid
        iter_count += 1
    if not converged:
        print(f"\n二分法求解{type}点压力初值失败，达到最大迭代次数 {max_iter}，未收敛："
              f"\n最终区间 [{a:.2e}, {b:.2e}] Pa，区间长度 {b - a:.2f} Pa，"
              f"\n最终 vale_mid={vale_mid:.6f}    F(P_mid)={f_mid:.6f}")
    return vale_mid


def Dichotomy_1( f , P, a, b, epsilon, max_iter=20):
    fa = f(P,a)
    fb = f(P,b)
    iter_count = 0
    converged = False
    while iter_count < max_iter:
        vale_mid = (a + b) / 2.0
        f_mid = f(P,vale_mid)
        if abs(max(a, b) - min(a, b)) < epsilon or abs(f_mid) < epsilon:
            converged = True
            break
        if fa < f_mid :
            b = vale_mid
            fb = f_mid
        else:
            a = vale_mid
            fa = f_mid
        iter_count += 1
    if not converged:
        print(f"\nP={P}线上二分法寻找拟临界点Fc最小温度为T={vale_mid},最大迭代次数 {max_iter}")
    return vale_mid , f_mid


def Dichotomy_2( f , T, a, b, epsilon, max_iter=20):
    fa = f(a,T)
    fb = f(b,T)
    iter_count = 0
    converged = False
    while iter_count < max_iter:
        vale_mid = (a + b) / 2.0
        f_mid = f(vale_mid,T)
        if abs(max(a, b) - min(a, b)) < epsilon or abs(f_mid) < epsilon:
            converged = True
            break
        if fa <f_mid:
            b = vale_mid
            fb = f_mid
        else:
            a = vale_mid
            fa = f_mid
        iter_count += 1
    if not converged:
        print(f"\nT={T}线上二分法寻找拟临界点Fc最小压力为P={vale_mid},最大迭代次数 {max_iter}")
    return vale_mid,f_mid


def calculate_c(Tr, Pr):
    data = {
        "A1": {
            "B1": 1.6368,
            "B2": -0.04615,
            "B3": 2.1138 * 10 ** (-3),
            "B4": -0.7845 * 10 ** (-5),
            "B5": -0.6923 * 10 ** (-6)
        },
        "A2": {
            "B1": -1.9693,
            "B2": 0.21874,
            "B3": -8.0028 * 10 ** (-3),
            "B4": -8.2328 * 10 ** (-5),
            "B5": 5.2604 * 10 ** (-6)
        },
        "A3": {
            "B1": 2.4638,
            "B2": -0.36461,
            "B3": 12.8763 * 10 ** (-3),
            "B4": 14.8059 * 10 ** (-5),
            "B5": -8.6895 * 10 ** (-6)
        },
        "A4": {
            "B1": -1.5841,
            "B2": 0.25136,
            "B3": -11.3805 * 10 ** (-3),
            "B4": 9.5672 * 10 ** (-5),
            "B5": 2.1812 * 10 ** (-6)
        }
    }

    def calculate_Ai(i, Pr):
        Ai_data = data[f"A{i}"]
        return (
                Ai_data["B1"] +
                Ai_data["B2"] * Pr +
                Ai_data["B3"] * Pr ** 2 +
                Ai_data["B4"] * Pr ** 3 +
                Ai_data["B5"] * Pr ** 4
        )

    A1 = calculate_Ai(1, Pr)
    A2 = calculate_Ai(2, Pr)
    A3 = calculate_Ai(3, Pr)
    A4 = calculate_Ai(4, Pr)

    return A1 + A2 * Tr + A3 * Tr ** 2 + A4 * Tr ** 3


def plot_equation(a, b, Ks, zs, num_points=1000):
    """
    绘制 equation 函数在区间 [a, b] 上的图像。

    :param a: 区间左端点
    :param b: 区间右端点
    :param Ks: 平衡常数数组
    :param zs: 组分摩尔分数数组
    :param num_points: 用于绘图的点数
    """
    ag_values = np.linspace(a, b, num_points)
    y_values = []

    for ag in ag_values:
        y = equation(ag, Ks, zs)
        y_values.append(y)

    plt.plot(ag_values, y_values)
    plt.xlabel('ag')
    plt.ylabel('equation(ag)')
    plt.title('Plot of equation(ag) in interval [a, b]')
    plt.grid(True)
    plt.show()


def equation(ag, Ks, zs):
    result = 0
    for i in range(len(Ks)):
        numerator = (Ks[i] - 1) * zs[i]
        denominator = 1 + ag * (Ks[i] - 1)
        if denominator == 0:
            return float('inf')
        result = result + numerator / denominator
    return result


def normalized_zis(zis):
    for i in range(len(zis)):
        zis[i] = zis[i] / sum(zis)  # 归一化
    return zis


def bisection_method(ag_min, ag_max, Ks, zs, tolerance=0.00000001, max_iterations=200):
    if equation(ag_min, Ks, zs) * equation(ag_max, Ks, zs) >= 0:
        raise ValueError("区间两端点的函数值乘积应小于零。")

    iteration = 0
    while (ag_max - ag_min) > tolerance and iteration < max_iterations:
        c = (ag_min + ag_max) / 2
        if equation(c, Ks, zs) == 0:
            return c
        elif equation(c, Ks, zs) * equation(ag_min, Ks, zs) < 0:
            ag_max = c
        else:
            ag_min = c
        iteration += 1
    return (ag_min + ag_max) / 2


def getKijArray(components, modelType, binaryActionCoefficient=True):
    # 获取当前脚本文件（Phaseplot.py）所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接成文件的绝对路径
    file_path = os.path.join(current_dir, '组分性质参数.xlsx')
    KijList = pd.read_excel(file_path, sheet_name=modelType)
    # KijList = pd.read_excel('组分性质参数.xlsx', sheet_name=modelType)
    CptCount = len(components)
    kijArray = np.zeros((CptCount, CptCount))
    if binaryActionCoefficient:
        for i in range(CptCount):
            Cpt_i = components[i].name
            for j in range(CptCount):
                if i != j:
                    Cpt_j = components[j].name
                    condition = KijList[KijList.iloc[:, 0] == Cpt_j]
                    value = condition[Cpt_i]
                    kijArray[i][j] = value.iloc[0]
    return kijArray


class component:
    def __init__(self, row, mixTure):
        # Creates a Component object
        self.name = row['Component']
        '''Component（组分名称）'''
        self.xi = mixTure
        '''组成占比'''
        self.MW = row['MW']
        '''Molecular weight（分子量）,g/mol '''
        self.Type = row['Type']
        ''' 组分类型  '''
        self.Tc = row['Tc']
        '''Critical temperature（临界温度）,k'''
        self.Pc = row['Pc']
        '''Critical pressure（临界压力）,Pa'''
        self.Vc = row['Vc']
        '''Critical viscosity(临界粘度), kg/(m・s)'''
        self.Zc = row['Zc']
        '''Critical Z factor（临界压缩因子）'''
        self.Mc = row['Mc']
        '''Critical molar volume(临界摩尔体积)，m^3/mol'''
        self.Af = row['Af']
        '''偏心因子'''
        self.Oa = row['Oa']
        '''Omega A'''
        self.Ob = row['Ob']
        '''Omega B'''
        self.Bp = row['Bp']
        '''Boiling point（沸点）,K '''
        self.Parachor = row['Parachor']
        ''' Parachor（等张比容） '''
        self.Den = row['Den']
        ''' 密度 ,kg/m³'''
        self.Sg = row['Sg']
        '''参考比重'''
        self.Type = row['Type']
        '''组分类型'''
        self.Ob = row['Ob']
        '''Omega B'''
        self.Oa = row['Oa']
        '''Omega A'''
        self.a = 0
        self.b = self.Ob * R * self.Tc / self.Pc
        self.Ki = 0
        '''平衡常数Ki'''


class fluid2P:
    def __init__(self, fluidMixInfo, Equition_type):
        self.Equition_type = Equition_type
        self.ρ_G = 0
        self.ρ_L = 0
        self.components = []
        self.P = 8800000
        self.T = 375.3722
        self.a_G = 0
        self.a_O = 0
        self.a_W = 0
        self.addCpt(fluidMixInfo)
        self.Z_G = 0
        self.Z_L = 0
        self.Z_W = 0
        self.Z_O = 0
        self.xis_G = []
        '''气相组分摩尔占比'''
        self.xis_L = []
        '''液相组分摩尔占比'''
        self.xis_O = []
        '''油相组分摩尔占比'''
        self.xis_W = []
        '''水相组分摩尔占比'''
        self.Gasdensity_i = np.zeros(len(self.components))
        self.GasProperties = {"气相混合摩尔质量": 0, "气相混合平均密度": 0, "气相混合黏度Viscosity_Gas": 0,
                              "气相体积占比": 0}
        self.LiquidProperties = {"液相混合摩尔质量": 0, "液相混合平均密度": 0, "液相混合黏度Viscosity_Liquid": 0,
                                 "液相体积占比": 0}
        self.WaterProperties = {"密度": 0, "摩尔质量": 0, "临界温度": 0, "临界压力": 0, "临界粘度": 0}
        self.OilProperties = {"密度": 0, "摩尔质量": 0, "临界温度": 0, "临界压力": 0, "临界粘度": 0}
        self.PhaseProperties = {"油-气表面张力": 0, "水-气表面张力": 0}
        self.Phaseequilibrium = {"气相摩尔占比": self.xis_G, "液相摩尔占比": self.xis_L}
        self.KijArray = getKijArray(self.components, self.Equition_type)
        # self.EOS = EOS_init('SRK',self.KijArray)
        self.EOS = EOS_init(self.KijArray)  # 2025 10 08 孙玉逊修改
        self.phase = ""
        self.phase_json_data = {}

    def addCpt(self, fluidMixInfo):
        '''
        流体中添加新组分
        :param fluidMixInfo:
        :return:
        '''# 获取当前脚本文件（Phaseplot.py）所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 拼接成文件的绝对路径
        file_path = os.path.join(current_dir, '组分性质参数.xlsx')
        # KijList = pd.read_excel(file_path, sheet_name=modelType)
        CptsList = pd.read_excel(file_path, sheet_name='组分性质')
        for index, row in fluidMixInfo.iterrows():
            Component = row['Component']
            mixTure = row['Mixture']
            row = CptsList[CptsList["Component"] == Component].iloc[0]
            cpt = component(row, mixTure)
            self.components.append(cpt)
        # SRK 二元作用系数矫正

    def calc_Ki(self, P, T):
        '''
        Wilson公式 计算初始的平衡常数Ki
        :return:
        '''
        Ki_array = np.zeros(len(self.components))
        for i in range(len(self.components)):
            Ki_array[i] = (self.components[i].Pc / P) * np.exp(
                5.373 * (1 + self.components[i].Af) * (1 - (self.components[i].Tc / T)), dtype=np.float64)
        return Ki_array

    def calc_vap_frac(self, Ki_array, Ki_max, Ki_min):
        '''
        计算气化率，即气相的摩尔分率
        '''
        Ks = np.ones(len(self.components))
        zs = np.ones(len(self.components))
        for i in range(len(self.components)):
            Ks[i] = Ki_array[i]
            zs[i] = self.components[i].xi
        # 求解区间
        # Ks = np.array([0.0109, 6.4535, 1.3980, 0.5173, 0.1789])
        vap_frac_min = max(1 / (1 - Ki_max), 0)
        vap_frac_max = min(1, 1 / (1 - Ki_min))

        def result_newton(vap_frac, tol=0.000001, maxIterations=1000):
            '''
            新thon迭代方程
            :param vap_frac:气化分数
            :param tol: 迭代误差
            :param maxIterations: 最大迭代次数
            :return:
            '''
            f = 0
            dfdv = 0
            while maxIterations > 0:
                for i in range(len(self.components)):
                    f = f + (Ks[i] - 1) * zs[i] / (1 + vap_frac * (Ks[i] - 1))
                    dfdv = dfdv - ((Ks[i] - 1) ** 2 * zs[i]) / ((1 + vap_frac * (Ks[i] - 1)) ** 2)
                vap_frac_ = vap_frac - f / dfdv
                if abs(vap_frac_ - vap_frac) < tol:
                    break
                vap_frac = vap_frac_
                maxIterations -= 1
            return vap_frac

        try:
            # 使用 brentq 求解方程 先用二分法求解气化率
            solution = bisection_method(vap_frac_min, vap_frac_max, Ks, zs)
        except ValueError:  # 二分法无法求解则用牛顿迭代法求解
            # plot_equation(vap_frac_min, vap_frac_max, Ks, zs )
            vap_frac_guess = 0.5
            solution = result_newton(vap_frac_guess)
        ag = solution

        return ag

    def check_Ki(self, Ki_array):
        '''
        检查Ki是否合理
        :return:
        '''
        Ki_min = self.components[0].Ki
        Ki_max = self.components[0].Ki
        Ki_max_Index = 0
        Ki_min_Index = 0

        for i in range(len(self.components)):
            if Ki_array[i] < Ki_min:
                Ki_min = Ki_array[i]
                Ki_min_Index = i
            if self.components[i].Ki > Ki_max:
                Ki_max = Ki_array[i]
                Ki_max_Index = i
        if Ki_min > 1:
            Ki_array[Ki_min_Index] = 0.1
            Ki_min = 0.1
        if Ki_max < 1:
            Ki_array[Ki_max_Index] = Ki_array[Ki_max_Index] + 1
            Ki_max = Ki_array[Ki_max_Index]
        return Ki_array, Ki_max, Ki_min

    def calc_Oil_Water_Balance(self):
        '''
        计算油水平衡
        :return:
        '''
        isExitWater = False
        water_Num = 0
        for i in range(len(self.components)):
            if self.components[i].Type == "Water":
                isExitWater = True
                water_Num = i
                break

        a_L = 1 - self.a_G
        if not isExitWater:
            self.a_O = a_L
            self.a_W = 0
            self.xis_O = self.xis_L.copy()
            self.xis_W = -1 * np.ones(len(self.components))
            Z_O, A_O, B_O, a_O, b_O, f_O_s = self.EOS.calc(self.components, self.P, self.T, self.xis_O, "Liquid",
                                                           self.Equition_type)
            self.Z_O = Z_O
            # self.Z_W
        else:
            self.a_W = a_L * self.xis_L[water_Num]
            self.a_O = a_L - self.a_W
            self.xis_O = self.xis_L.copy()
            self.xis_W = self.xis_L.copy()
            self.xis_O[water_Num] = 0
            X_O_sum = sum(self.xis_O)
            for i in range(len(self.xis_O)):
                self.xis_O[i] = self.xis_O[i] / X_O_sum
                if i == water_Num:
                    self.xis_W[i] = 1
                else:
                    self.xis_W[i] = 0
            Z_O, A_O, B_O, a_O, b_O, f_O_s = self.EOS.calc(self.components, self.P, self.T, self.xis_O, "Liquid",
                                                           self.Equition_type)
            Z_W, A_W, B_W, a_W, b_W, f_W_s = self.EOS.calc(self.components, self.P, self.T, self.xis_W, "Liquid",
                                                           self.Equition_type)
            self.Z_O = Z_O
            self.Z_W = Z_W

    def calc_Bubble_P(self, T_, tol=0.0001):
        '''
        计算固定温度下的泡点压力
        :return:
        '''
        type = "Bubble"

        def Funtion1(P):
            Ki_array = self.calc_Ki(P, T_)
            Kz = 0
            for i in range(len(self.components)):
                Kz = Kz + Ki_array[i] * self.components[i].xi
            F = Kz - 1
            return F

        def Funtion1_2(P):
            Ki_array = self.calc_falsh(P, T_)
            Kz = 0
            for i in range(len(self.components)):
                Kz = Kz + Ki_array[i] * self.components[i].xi
            F = Kz - 1
            return F

        Pmin = 100000  # 0.1MPa
        Pmax = 15000000  # 15MPa
        P_Bubble_guess = Dichotomy(Funtion1, Pmin, Pmax, tol, type)
        '利用威尔逊方法和牛顿迭代求解初值'
        if np.isnan(P_Bubble_guess):
            P_Bubble_guess = Dichotomy(Funtion1_2, Pmin, Pmax, tol, type)

        def Funtion2(P):
            Ki_array = self.calc_falsh(P, T_)
            Kz = 0
            for i in range(len(self.components)):
                Kz = Kz + Ki_array[i] * self.components[i].xi
            F = Kz - 1
            return F

        # # 牛顿迭代求解，可能有迭代失败的情况
        F = Funtion2(P_Bubble_guess)
        if abs(F) < tol:
            # 初值即为解
            P_Bubble = P_Bubble_guess
            return P_Bubble
        # 根据初值找边界
        if F < 0:
            # 此时为过冷液体，找到的压力过大
            Pmax = P_Bubble_guess
            times = 0
            while True:
                times += 1
                P_Bubble_guess = 0.8 * P_Bubble_guess
                F = Funtion2(P_Bubble_guess)
                if F < 0:
                    Pmax = P_Bubble_guess
                else:
                    Pmin = P_Bubble_guess
                P_Bubble_guess = Dichotomy(Funtion2, Pmin, Pmax, tol, type="Bubble")
                F = Funtion2(P_Bubble_guess)
                if abs(F) < tol or times >= 10:
                    # 新值即为解
                    return P_Bubble_guess
        else:
            # 此时为气液两相，找到的压力过小
            Pmin = P_Bubble_guess

            while True:

                P_Bubble_guess = 1.2 * P_Bubble_guess
                F = Funtion2(P_Bubble_guess)
                if abs(F) < tol:
                    # 新值即为解
                    P_Bubble = P_Bubble_guess
                    return P_Bubble
                if F > 0:
                    Pmin = P_Bubble_guess
                else:
                    Pmax = P_Bubble_guess
                    P_Bubble, result = brentq(Funtion2, Pmin, Pmax, full_output=True)
                    return P_Bubble

    def calc_Dew_T(self, P_, tol=0.00001):
        '''
        计算固定压力下的露点温度
        :param P_: 求解压力，单位：Pa
        :return:
        '''

        def Funtion1(T):
            '''
            使用Wilson公式进行估算
            '''
            Ki_array = self.calc_Ki(P_, T)
            # Ki_array = self.calc_falsh(P_, T)
            ZK = 0
            for i in range(len(self.components)):
                ZK = ZK + self.components[i].xi / Ki_array[i]
            F = 1 - ZK
            return F

        Tmin = 1
        Tmax = 1000
        T_Dew_guess = brentq(Funtion1, Tmin, Tmax)
        '利用威尔逊方法求解初值'

        def Funtion2(T):
            # Ki_array = self.calc_Ki(P_, T)

            Ki_array = self.calc_falsh(P_, T)
            ZK = 0
            for i in range(len(self.components)):
                ZK = ZK + self.components[i].xi / Ki_array[i]
            F = 1 - ZK
            return F

        F = Funtion2(T_Dew_guess)
        if abs(F) < tol:
            # 初值即为解
            T_Dew = T_Dew_guess
            return T_Dew
        # 根据初值找边界
        if F > 0:
            # 此时为过热蒸汽，找到的温度过大
            Tmax = T_Dew_guess
            while True:
                T_Dew_guess = 0.8 * T_Dew_guess
                F = Funtion2(T_Dew_guess)
                if abs(F) < tol:
                    # 新值即为解
                    T_Dew = T_Dew_guess
                    return T_Dew
                if F > 0:
                    Tmax = T_Dew_guess
                else:
                    Tmin = T_Dew_guess
                    # T_Bubble = brentq(Funtion2, Tmin, Tmax)   ##孙玉逊修改##
                    T_Dew = brentq(Funtion2, Tmin, Tmax)
                    return T_Dew

        else:
            # 此时为气液两相，找到的温度过小
            Tmin = T_Dew_guess
            while True:
                T_Dew_guess = 1.2 * T_Dew_guess
                F = Funtion2(T_Dew_guess)
                if abs(F) < tol:
                    # 新值即为解
                    T_Dew = T_Dew_guess
                    return T_Dew
                if F < 0:
                    Tmin = T_Dew_guess
                else:
                    Tmax = T_Dew_guess
                    T_Dew = brentq(Funtion2, Tmin, Tmax)
                    return T_Dew
        return T_Dew


    def calc_falsh(self, P, T, tol=0.000001, maxiter=500):
        '''
        计算气、液闪蒸平衡
        :param P: 压力，单位：Pa
        :param T: 温度，单位：K
        :return: 通过逸度相等，计算各组分的平衡常数Ki
        '''
        self.ρ_G = 0
        self.ρ_Gmix = np.ones(len(self.components))
        self.components = self.EOS.calc_ai(self.components, T, self.Equition_type)
        '组分初始ai与流体温度有关，根据温度重新计算'
        Ki_array = self.calc_Ki(P, T)
        'Wilson公式计算初始Ki'
        Ki_array, Ki_max, Ki_min = self.check_Ki(Ki_array)
        # isOver = True
        xi_G = np.ones(len(self.components))
        '气相中各组分占比,赋初值'
        xi_L = np.ones(len(self.components))
        '液相中各组分占比,赋初值'
        while True:
            vap_frac = self.calc_vap_frac(Ki_array, Ki_max, Ki_min)
            '求解气相分率'
            self.a_G = vap_frac
            for i in range(len(self.components)):
                xi_L[i] = self.components[i].xi / (1 + vap_frac * (Ki_array[i] - 1))
                '根据气相分率和平衡常数计算组分在液相中的摩尔占比'
                xi_G[i] = Ki_array[i] * xi_L[i]
                '根据气相分率和平衡常数计算组分在气相中的摩尔占比'
            '后面的判断是必须的。为单相时，其余相占比为0，单其余相中各组分的摩尔占比等于混合物中各组分的摩尔占比'
            if vap_frac <= 0 or vap_frac >= 1:
                '气相分数小于0 → 实际为全液相，强制设为0'
                '气相分数大于1 → 实际为全气相，强制设为1'
                vap_frac = min(max(vap_frac, 0), 1)
                '单相状态下：气液相组成均等于混合物总组成（无分离）'
                for i in range(len(self.components)):
                    xi_L[i] = self.components[i].xi
                    xi_G[i] = self.components[i].xi
            '根据迭代的逸度重新计算各组分的平衡常数Ki'
            xi_L = normalized_zis(xi_L)
            xi_G = normalized_zis(xi_G)
            # 状态方程及相关参数求解
            Z_L, A_L, B_L, a_L, b_L, f_L_s = self.EOS.calc(self.components, P, T, xi_L, "Liquid", self.Equition_type)
            '根据状态方程计算压缩因子、逸度'
            Z_G, A_G, B_G, a_G, b_G, f_G_s = self.EOS.calc(self.components, P, T, xi_G, "Gas", self.Equition_type)
            '根据状态方程计算压缩因子、逸度'
            diff = 0
            for i in range(len(self.components)):
                if f_G_s[i] != 0:
                    diff = diff + abs(f_L_s[i] / f_G_s[i] - 1)
            isOver = np.all(diff <= tol)
            if isOver:
                # print(1)
                # 气液两相 此处指所有的物料组分
                # self.a_G = vap_frac
                self.xis_G = xi_G
                self.xis_L = xi_L
                '''计算基础性质'''
                self.Z_G = Z_G
                self.Z_L = Z_L
                return Ki_array
            maxiter = maxiter - 1
            if maxiter <= 0:
                return Ki_array
            for i in range(len(self.components)):
                Ki_array[i] = (f_L_s[i] / xi_L[i]) / (f_G_s[i] / xi_G[i])

    def phase_equilibrium_calc(self, P, T):
        '''
        计算组分在温度T和压力P下的物理性质
        :param P: 温度，单位：k
        :param T: 压力，单位：Pa
        :return:
        '''
        '计算闪蒸平衡'
        self.calc_falsh(P, T)

        if self.a_G > 1:
            self.phase = "气相"
        elif self.a_G == 1:
            self.phase = "露点"
        elif self.a_G < 0:
            self.phase = "液相"
        elif self.a_G == 0:
            self.phase = "泡点"
        else:
            self.phase = "气液两相"
        if self.a_G <= 1:
            if self.a_G < 0:
                self.a_G = 0
            self.calc_Oil_Water_Balance()
        else:
            self.a_G = 1
            self.a_O = 0
            self.a_W = 0
            self.xis_O = np.zeros(len(self.components))
            self.xis_W = np.zeros(len(self.components))

    def Phase_diagram(self,PDmin,PDmax,TBmin,TBmax,PDstep,TBstep):
        Drawing = Plot

        PD = PDmin
        TB = TBmin

        # PD_array = np.arange(PDmin, PDmax + PDstep, PDstep)
        # T_Dew = self.calc_Dew_T2(PD_array)#################

        PDlist = np.ones(int((PDmax - PDmin) / PDstep) + 1)
        T_Dewlist = np.ones(int((PDmax - PDmin) / PDstep) + 1)

        TBlist = np.ones(int((TBmax - TBmin) / TBstep) + 1)
        PBubblelist = np.ones(int((TBmax - TBmin) / TBstep) + 1)

        i = 0
        j = 0
        Dewcalc = True
        Bubcalc = True
###############################   该处为核心修改部分，原来是通过循环逐步计算露点温度和泡点压力，现在改为直接从JSON文件中读取预先计算好的数据 ###############################
        # while True:
        #     if PD <= PDmax:
        #         '''求解露点温度'''
        #         T_Dew = self.calc_Dew_T(PD)
        #         PDlist[i] = PD
        #         T_Dewlist[i] = T_Dew
        #         print("压力：", PDlist[i] / 1000000, "MPa",
        #               f"          \033[1;31m露点温度{round(T_Dewlist[i], 5)} K\033[0m ")
        #         PD = PD + PDstep
        #         i += 1
        #     else:
        #         Dewcalc = False
        #     if TB <= TBmax:
        #         '''求解泡点压力'''
        #         P_bubble = self.calc_Bubble_P(TB)
        #         TBlist[j] = TB
        #         PBubblelist[j] = P_bubble
        #         print(f"\033[1;36m泡点压力：{round(P_bubble / 1000000, 3)}MPa\033[0m" "          温度：", TB, "K", )
        #         TB = TB + TBstep
        #         j += 1
        #     else:
        #         Bubcalc = False
        #
        #     if Dewcalc == False and Bubcalc == False:
        #         break
###############################   该处为核心修改部分，原来是通过循环逐步计算露点温度和泡点压力，现在改为直接从JSON文件中读取预先计算好的数据 ###############################

###############################   新增：从JSON文件读取预计算数据，替代循环计算露点温度和泡点压力的部分 ###############################
        with open("DB/Phaseplotoutdata_pipesim.json", "r", encoding='utf-8') as f:
            json_data = json.load(f)
        filePath = "DB/Phaseplotoutdata_pipesim.json"
        print(f"\n 成功读取JSON文件：{filePath}")

        # 提取核心数据块
        plot_data = json_data["plotdata"]

        # 【核心】按JSON结构提取数据，转numpy数组，赋值给4个目标变量
        # 露点线 Dew Line 数据
        PDlist = np.array(plot_data["Dew Line"]["Dew Pressure"])
        T_Dewlist = np.array(plot_data["Dew Line"]["Dew Temperature"])

        # 泡点线 Bubble Line 数据
        PBubblelist = np.array(plot_data["Bubble Line"]["Bubble Pressure"])
        TBlist = np.array(plot_data["Bubble Line"]["Bubble Temperature"])
############################################################################################################################



        dp = dataprocess(PDlist, T_Dewlist, TBlist, PBubblelist)
        PDlist, T_Dewlist, slope = dp.DEWprocess()
        TBlist, PBubblelist, BUslope = dp.BUBBLEprocess()
        # ========== 核心修改：PhaseplotData 转 JSON 并写入文件 ==========
        # PhaseplotData = [PDlist.tolist(), T_Dewlist.tolist(), TBlist.tolist(), PBubblelist.tolist()]

        if len(TBlist) != len(PBubblelist):
            print(f"警告：泡点温度列表长度({len(TBlist)})与泡点压力列表长度({len(PBubblelist)})不匹配，将以较短列表为准")
        if len(PDlist) != len(T_Dewlist):
            print(f"警告：露点压力列表长度({len(PDlist)})与露点温度列表长度({len(T_Dewlist)})不匹配，将以较短列表为准")

            # 2. 构造泡线(Bubble Line)结构化数据，和你示例格式完全对齐
        bubble_line = []
        for idx, (bubble_temp, bubble_press) in enumerate(zip(TBlist, PBubblelist), start=1):
            # 过滤NaN/None等无效值，避免JSON序列化失败
            if bubble_temp is None or bubble_press is None:
                continue
            if isinstance(bubble_temp, float) and bubble_temp != bubble_temp:
                continue
            if isinstance(bubble_press, float) and bubble_press != bubble_press:
                continue

            bubble_line.append({
                "number": idx,
                "Bubble Pressure": float(bubble_press),
                "Bubble Temperature": float(bubble_temp)
            })

        # 3. 构造露线(Dew Line)结构化数据，格式和泡线统一
        dew_line = []
        for idx, (dew_press, dew_temp) in enumerate(zip(PDlist, T_Dewlist), start=1):
            # 同样过滤无效值
            if dew_press is None or dew_temp is None:
                continue
            if isinstance(dew_press, float) and dew_press != dew_press:
                continue
            if isinstance(dew_temp, float) and dew_temp != dew_temp:
                continue

            dew_line.append({
                "number": idx,
                "Dew Pressure": float(dew_press),
                "Dew Temperature": float(dew_temp)
            })

        # 4. 组合最终的JSON完整结构
        phase_json_data = {
            "Bubble Line": bubble_line,
            "Dew Line": dew_line
        }

        # 5. 导出JSON文件（路径可自定义，这里沿用你原有的Phaseplotoutdata.json）
        output_file = "Phaseplotoutdata.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(phase_json_data, f, ensure_ascii=False, indent=4)

        # 可选：控制台打印JSON，方便调试查看
        print("相图JSON数据已生成，内容如下：")
        print(json.dumps(phase_json_data, ensure_ascii=False, indent=4))
        # ======================
        # 新增：生成指定格式JSON 结束
        # ======================
        PhaseplotData=[PDlist, T_Dewlist, TBlist, PBubblelist]
        print(phase_json_data)
        # Drawing(PDlist, T_Dewlist, TBlist, PBubblelist)
        self.phase_json_data = phase_json_data
        return phase_json_data




    def calc_Dew_T2(self, PD_array, tol=0.00001):
        '''
        计算固定压力下的露点温度
        :param P_: 求解压力，单位：Pa
        :return:
        '''
        '''组分个数，压力区间内个数，需计算的温度个数'''
        while True:
            Tmin = 10
            Tmax = 1000
            Tstep = 10
            T_array = np.arange(Tmin, Tmax + Tstep, Tstep)
            n = len(PD_array)
            Ki_array = self.calc_Ki2(PD_array, T_array)  # 组分个数，压力区间内个数，需计算的温度个数(4,70,101)
            Xi_array = np.array([self.components[i].xi for i in range(len(self.components))])  # (4,)
            Xi_array = Xi_array[:, np.newaxis, np.newaxis]  # (4,1,1)
            ZK_array = Xi_array / Ki_array  # (4,70,101)
            ZK_array = ZK_array.sum(axis=0)  # (70,101)
            F_array = 1 - ZK_array
            T_crossings = np.zeros((n, 2))
            for i in range(n):
                row = F_array[i, :]

                neg_mask = row < 0
                neg_values = row[neg_mask]
                closest_neg = np.max(neg_values)
                neg_col = np.where(row == closest_neg)[0]
                neg_core_val = T_array[neg_col]
                T_crossings[i, 0] = neg_core_val

                pos_mask = row > 0
                pos_values = row[pos_mask]
                closest_pos = np.min(pos_values)
                pos_col = np.where(row == closest_pos)[0]
                pos_core_val = T_array[pos_col]
                T_crossings[i, 1] = pos_core_val  # (负值，正值)
            Newstep = (T_crossings[:, 1] - T_crossings[:, 0]).reshape(-1, 1) / 50  # (51,)
            Lenstep = 50 + 1
            T_array = np.zeros((len(Newstep), Lenstep))

            T_array[:, 0] = T_crossings[:, 0]
            T_array[:, -1] = T_crossings[0, :]
            print(Newstep)

        T_Dew_array = T_crossings.flatten()

        def Funtion1(T):
            '''
            使用Wilson公式进行估算
            '''
            Ki_array = self.calc_Ki2(PD_array, T_array)
            # Ki_array = self.calc_falsh(P_, T)
            ZK = 0
            for i in range(len(self.components)):
                ZK = ZK + self.components[i].xi / Ki_array[i]
            F = 1 - ZK
            return F

        T_Dew_guess = brentq(Funtion1, Tmin, Tmax)
        '利用威尔逊方法求解初值'


        F = Funtion2(T_Dew_guess)
        if abs(F) < tol:
            # 初值即为解
            T_Dew = T_Dew_guess
            return T_Dew
        # 根据初值找边界
        if F > 0:
            # 此时为过热蒸汽，找到的温度过大
            Tmax = T_Dew_guess
            while True:
                T_Dew_guess = 0.8 * T_Dew_guess
                F = Funtion2(T_Dew_guess)
                if abs(F) < tol:
                    # 新值即为解
                    T_Dew = T_Dew_guess
                    return T_Dew
                if F > 0:
                    Tmax = T_Dew_guess
                else:
                    Tmin = T_Dew_guess
                    # T_Bubble = brentq(Funtion2, Tmin, Tmax)   ##孙玉逊修改##
                    T_Dew = brentq(Funtion2, Tmin, Tmax)
                    return T_Dew

        else:
            # 此时为气液两相，找到的温度过小
            Tmin = T_Dew_guess
            while True:
                T_Dew_guess = 1.2 * T_Dew_guess
                F = Funtion2(T_Dew_guess)
                if abs(F) < tol:
                    # 新值即为解
                    T_Dew = T_Dew_guess
                    return T_Dew
                if F < 0:
                    Tmin = T_Dew_guess
                else:
                    Tmax = T_Dew_guess
                    T_Dew = brentq(Funtion2, Tmin, Tmax)
                    return T_Dew
        return T_Dew

    def calc_Ki2(self, P_array, T_array):
        '''
        Wilson公式 计算初始的平衡常数Ki
        :return:
        '''
        Ki_array = np.zeros(len(self.components))
        Pc_array = np.zeros(len(self.components))
        Tc_array = np.zeros(len(self.components))
        Af_array = np.zeros(len(self.components))
        for i in range(len(self.components)):
            Pc_array[i] = self.components[i].Pc
            Tc_array[i] = self.components[i].Tc
            Af_array[i] = self.components[i].Af
        term3_0 = Tc_array / T_array[:, np.newaxis]  # 注：Tc_array是 (4,),行数是组分个数， T_array 是 (70,)
        term3 = 1 - term3_0  # 注：T_array 是 (70,4)，正确
        term1 = Pc_array / P_array[:, np.newaxis]  # 广播为 (70, 4)
        term2 = 5.373 * (1 + Af_array)  # (4,)
        term2_ = term2[np.newaxis, :]  # 在行做扩展，原向量变为行向量
        t222 = term2[:, np.newaxis]  # 在列做扩展，原向量变为列向量
        term4_ = term2_ * term3
        term4 = np.exp(term4_)  # (4,)

        term1ex = term1.transpose()[:, np.newaxis]
        term1ex = term1ex.transpose(0, 2, 1)  # （4，70，1）
        term4ex = term4.transpose()[:, np.newaxis]  # (4,1,101)
        # term4ex=term4ex.transpose(0, 2, 1)
        term5 = term1ex * term4ex  # (4,70,101)
        '''组分个数，压力区间内个数，需计算的温度个数 (4,70,101)'''
        Ki_array = term5

        # Ki_array=(Pc_array / P_array[:, np.newaxis]) * np.exp(5.373 * (1 + Af_array) *(1 - (Tc_array / T_array[:, np.newaxis])))

        return Ki_array


class CompositionalSimulation:
    def __init__(self):
        pass
    def injsondataprecess(self, jsonPhaseplotModel):
        fluidMixInfo = pd.DataFrame(jsonPhaseplotModel)
        fluidMixInfo = fluidMixInfo[fluidMixInfo['Moles'] != 0].copy()
        total_moles = fluidMixInfo['Moles'].sum()
        fluidMixInfo['Mixture'] = fluidMixInfo['Moles'] / total_moles
        fluidMixInfo = fluidMixInfo.rename(columns={'component_name': 'Component'})
        fluidMixInfo = fluidMixInfo[['Component', 'Mixture']]
        return fluidMixInfo


if __name__ == '__main__':
###################################   初始化   ###################################
    # 存放组分模型相图的文件路径
    filePath = "../../DB/Phaseplotindata.json"
    # 打开JSON文件，直接用json.load解析
    with open(filePath, "r", encoding='utf-8') as f:
        json_data = json.load(f)
    print(f"\n 成功读取JSON文件：{filePath}")
    jsonPhaseplotModel = json_data["jsonPhaseplotModel"]
    CompositionalSim = CompositionalSimulation()
    fluidMixInfo=CompositionalSim.injsondataprecess(jsonPhaseplotModel)

    Equition_type =  "SRK" # "SRK" "PR"
    fluid = fluid2P(fluidMixInfo,Equition_type)

    # pipesi_viscosity_data = pd.read_excel('pipesim黏度对比.xlsx', sheet_name='变压力')
###################################   相图绘制   ###################################
    '''
    PDmin 露点压力最小值
    PDmax 露点压力最大值
    TBmin 泡点温度最小值
    TBmax 泡点温度最大值
    PDstep 压力步长
    TBstep 温度步长
    '''
    PDmin=100000
    PDmax=7*1000000
    TBmin=273.15- 90
    TBmax=273.15+ 90
    PDstep=100000
    TBstep=10
    fluid.Phase_diagram(PDmin,PDmax,TBmin,TBmax,PDstep,TBstep)


