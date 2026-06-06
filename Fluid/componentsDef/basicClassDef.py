import time
from scipy.optimize import brentq, newton, least_squares
import pandas as pd
import math
import numpy as np
import sympy
import numbers
import matplotlib.pyplot as plt

from componentsDef.EOS import EOS_init
from scipy.optimize import minimize


R= 8.314
T0 = 273.15+20
P0 = 101.325*1000


def calculate_c(Tr, Pr):
    data = {
        "A1": {
            "B1": 1.6368,
            "B2": -0.04615,
            "B3": 2.1138 * 10**(-3),
            "B4": -0.7845 * 10**(-5),
            "B5": -0.6923 * 10**(-6)
        },
        "A2": {
            "B1": -1.9693,
            "B2": 0.21874,
            "B3": -8.0028 * 10**(-3),
            "B4": -8.2328 * 10**(-5),
            "B5": 5.2604 * 10**(-6)
        },
        "A3": {
            "B1": 2.4638,
            "B2": -0.36461,
            "B3": 12.8763 * 10**(-3),
            "B4": 14.8059 * 10**(-5),
            "B5": -8.6895 * 10**(-6)
        },
        "A4": {
            "B1": -1.5841,
            "B2": 0.25136,
            "B3": -11.3805 * 10**(-3),
            "B4": 9.5672 * 10**(-5),
            "B5": 2.1812 * 10**(-6)
        }
    }

    def calculate_Ai(i, Pr):
        Ai_data = data[f"A{i}"]
        return (
            Ai_data["B1"] +
            Ai_data["B2"] * Pr +
            Ai_data["B3"] * Pr**2 +
            Ai_data["B4"] * Pr**3 +
            Ai_data["B5"] * Pr**4
        )

    A1 = calculate_Ai(1, Pr)
    A2 = calculate_Ai(2, Pr)
    A3 = calculate_Ai(3, Pr)
    A4 = calculate_Ai(4, Pr)

    return A1 + A2 * Tr + A3 * Tr**2 + A4 * Tr**3


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


def getKijArray(components,modelType,binaryActionCoefficient=True):
    KijList = pd.read_excel('./componentsDef/组分性质参数.xlsx', sheet_name=modelType)
    CptCount = len(components)
    kijArray = np.zeros((CptCount,CptCount))
    if binaryActionCoefficient:
        for i in range(CptCount):
            Cpt_i = components[i].name
            for j in range(CptCount):
                if i!=j:
                    Cpt_j = components[j].name
                    condition = KijList[KijList.iloc[:, 0] == Cpt_j]
                    value = condition[Cpt_i]
                    kijArray[i][j] = value.iloc[0]
    return kijArray



class component:
    def __init__(self,row,mixTure):
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
    def __init__(self,fluidMixInfo):
        self.components=[]
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
        self.GasProperties = {"密度":0,"摩尔质量":0,"临界温度":0,"临界压力":0,"临界粘度":0}
        self.WaterProperties = {"密度":0,"摩尔质量":0,"临界温度":0,"临界压力":0,"临界粘度":0}
        self.OilProperties = {"密度":0,"摩尔质量":0,"临界温度":0,"临界压力":0,"临界粘度":0}
        self.KijArray = getKijArray(self.components, "SRK")
        self.EOS = EOS_init("SRK",self.KijArray)
        self.phase = ""




    def addCpt(self,fluidMixInfo):
        '''
        流体中添加新组分
        :param fluidMixInfo:
        :return:
        '''
        CptsList  =  pd.read_excel('./componentsDef/组分性质参数.xlsx', sheet_name='组分性质')
        for index, row in fluidMixInfo.iterrows():
            Component = row['Component']
            mixTure = row['Mixture']
            row = CptsList[CptsList["Component"] == Component].iloc[0]
            cpt = component(row,mixTure)
            self.components.append(cpt)
        #SRK 二元作用系数矫正






    def calc_Ki(self,P,T):
        '''
        Wilson公式 计算初始的平衡常数Ki
        :return:
        '''
        Ki_array = np.zeros(len(self.components))
        for i in range(len(self.components)):
            Ki_array[i] = (self.components[i].Pc / P) * np.exp(5.373 * (1 + self.components[i].Af) *(1 - (self.components[i].Tc / T)))
        return Ki_array

    def calc_vap_frac(self, Ki_array, Ki_max, Ki_min):
        '''
        计算气化率，即气相的摩尔分率
        '''
        Ks = np.ones(len(self.components))
        zs = np.ones(len(self.components))
        for i in range(len(self.components)):
            Ks[i] =  Ki_array[i]
            zs[i] = self.components[i].xi
        # 求解区间
        # Ks = np.array([0.0109, 6.4535, 1.3980, 0.5173, 0.1789])
        vap_frac_min = max(1 / (1 - Ki_max),0)
        vap_frac_max = min(1,1 / (1 - Ki_min))

        def result_newton(vap_frac,tol=0.000001,maxIterations=1000):
            '''
            新thon迭代方程
            :param vap_frac:气化分数
            :param tol: 迭代误差
            :param maxIterations: 最大迭代次数
            :return:
            '''
            f = 0
            dfdv = 0
            while maxIterations>0:
                for i in range(len(self.components)):
                    f = f + (Ks[i]-1)*zs[i] / (1 + vap_frac * (Ks[i] - 1))
                    dfdv = dfdv - ( (Ks[i] - 1) ** 2 * zs[i] ) / ( (1 + vap_frac * (Ks[i] - 1)) ** 2 )
                vap_frac_ = vap_frac - f/dfdv
                if abs(vap_frac_ - vap_frac) < tol:
                    break
                vap_frac = vap_frac_
                maxIterations -= 1
            return vap_frac

        try:
            # 使用 brentq 求解方程
            solution = bisection_method(vap_frac_min, vap_frac_max, Ks,zs)
        except ValueError:
            # plot_equation(vap_frac_min, vap_frac_max, Ks, zs )
            vap_frac_guess = 0.5
            solution = result_newton(vap_frac_guess)
        ag = solution

        return ag

    def check_Ki(self,Ki_array):
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
        if Ki_min > 1 :
            Ki_array[Ki_min_Index] = 0.1
            Ki_min = 0.1
        if Ki_max < 1 :
            Ki_array[Ki_max_Index] =   Ki_array[Ki_max_Index]+1
            Ki_max =  Ki_array[Ki_max_Index]
        return Ki_array,Ki_max, Ki_min

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
            self.xis_W = -1*np.ones(len(self.components))
            Z_O, A_O, B_O, a_O, b_O, f_O_s = self.EOS.calc(self.components, self.P, self.T, self.xis_O, "Liquid")
            self.Z_O = Z_O
            # self.Z_W
        else:
            self.a_W = a_L * self.xis_L[water_Num]
            self.a_O = a_L-self.a_W
            self.xis_O = self.xis_L.copy()
            self.xis_W = self.xis_L.copy()
            self.xis_O[water_Num] = 0
            X_O_sum = sum(self.xis_O)
            for i in range(len(self.xis_O)):
                self.xis_O[i] = self.xis_O[i]/X_O_sum
                if i ==  water_Num:
                    self.xis_W[i] = 1
                else:
                    self.xis_W[i] = 0
            Z_O, A_O, B_O, a_O, b_O, f_O_s = self.EOS.calc(self.components, self.P, self.T, self.xis_O, "Liquid")
            Z_W, A_W, B_W, a_W, b_W, f_W_s = self.EOS.calc(self.components, self.P, self.T, self.xis_W, "Liquid")
            self.Z_O = Z_O
            self.Z_W = Z_W

    def calc_ZK_KZ(self,P, T):
        '''
        计算ZK和KZ,用来判断混合液的相态
        :param P: 压力
        :param T: 温度
        :return: ZK, KZ
        '''
        Ki_array = self.calc_falsh(P, T)
        ZK = 0
        KZ = 0
        for i in range(len(self.components)):
            ZK = ZK + self.components[i].xi / Ki_array[i]
            KZ = KZ + Ki_array[i] * self.components[i].xi
        return ZK, KZ


    def calc_Bubble_P(self,T_,tol=1e-12,P0 = 0):
        '''
        计算固定温度下的泡点压力
        :return:
        '''
        def Funtion1(P):
            Ki_array = self.calc_Ki(P,T_)
            Kz = 0
            for i in range(len(self.components)):
                # if self.components[i].Type != "Water":
                Kz = Kz + Ki_array[i] * self.components[i].xi
            F = Kz-1
            return F
        if P0 != 0:
            P_Bubble_guess = P0
        else:
            Pmin = 100000
            Pmax = 15000000
            P_Bubble_guess = brentq(Funtion1, Pmin, Pmax)
        '利用威尔逊方法和牛顿迭代求解初值'

        def Funtion2(P):
            Ki_array = self.calc_falsh(P, T_)
            Kz = 0
            for i in range(len(self.components)):
                # if self.components[i].Type != "Water":
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
            Pmax =  P_Bubble_guess
            while True:
                P_Bubble_guess = 0.8 * P_Bubble_guess
                F = Funtion2(P_Bubble_guess)
                if abs(F) < tol:
                    # 新值即为解
                    P_Bubble = P_Bubble_guess
                    return P_Bubble
                if F < 0 :
                    Pmax = P_Bubble_guess
                else:
                    Pmin = P_Bubble_guess
                    P_Bubble = brentq(Funtion2, Pmin, Pmax)
                    return P_Bubble
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
                if F > 0 :
                    Pmin = P_Bubble_guess
                else:
                    Pmax = P_Bubble_guess
                    P_Bubble = brentq(Funtion2, Pmin, Pmax)
                    return P_Bubble


    def calc_Dew_T(self,P_,tol=1e-12,T0 = 0):
        '''
        计算固定压力下的露点温度
        :param P_: 求解压力，单位：Pa
        :param tol: 迭代精度
        :param T0: 迭代初值
        :return:
        '''
        def Funtion1(T):
            '''
            使用Wilson公式进行估算
            '''
            Ki_array = self.calc_Ki(P_,T)
            '''计算平衡常数'''
            ZK = 0
            for i in range(len(self.components)):
                ZK = ZK + self.components[i].xi / Ki_array[i]
            F = 1-ZK
            return F
        Tmin = 200
        Tmax = 500
        if T0 != 0:
            T_Dew_guess = T0
        else:
            T_Dew_guess = brentq(Funtion1, Tmin, Tmax)
        '利用威尔逊方法求解初值'

        def Funtion2(T):
            Ki_array = self.calc_falsh(P_,T)
            ZK = 0
            for i in range(len(self.components)):
                ZK = ZK + self.components[i].xi / Ki_array[i]
            F = 1-ZK
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
                    T_Bubble = brentq(Funtion2, Tmin, Tmax)
                    return T_Bubble
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
                    Tmax =T_Dew_guess
                    T_Dew = brentq(Funtion2, Tmin, Tmax)
                    return T_Dew


    def calc_critical_point(self,tol=0.00001):

        T_guess =  360.3083
        P_guess = 7308930#7.0793304239  * 1000000
        ZK,KZ = self.calc_ZK_KZ(P_guess,T_guess)
        print(1)

        def function(X):
            P = X[0]*100000
            T = X[1]
            Ki_array = self.calc_falsh(P,T)
            ZK,KZ = self.calc_ZK_KZ(P,T)

            F = sum(abs(np.log(Ki_array)))+abs(ZK-1)+abs(KZ-1)+T/400

            return F


        T_Dew_guess = 350
        P_guess = 60
        X0 = [P_guess,T_Dew_guess]
        Pmax = 100
        Tmax = 400
        Pmin = 100
        Tmin = 300
        bounds = [(Pmin, Pmax), (Tmin, Tmax)]

        result = minimize(function, x0=X0, bounds=bounds, method='Nelder-Mead', tol=1e-20)

    #     result = minimize(function, x0=X0, bounds=bounds, method='l-bfgs-b', tol=1e-12,options={
    #     'gtol': 1e-32,  # 降低梯度收敛阈值
    #     'maxiter': 1000,  # 增加最大迭代次数
    #     'eps': 1e-32  # 数值导数的步长，可能需要调整
    # })

        aa= 1


        # def Funtion(P):
        #     '''
        #     使用Wilson公式进行估算
        #     '''
        #     T_Dew = self.calc_Dew_T(P)
        #     T_guess = T_Dew
        #     Ki_array = self.calc_falsh(P, T_Dew)
        #     Kz = 0
        #     for i in range(len(self.components)):
        #         Kz = Kz + Ki_array[i] * self.components[i].xi
        #     F = Kz - 1
        #     print('压力：',P/1000000,'MPa','           温度：',round(T_Dew,5),'K','           F：',F)
        #     return F
        # P_guess = 5000000
        # P = newton(Funtion,x0=P_guess,tol=tol,maxiter=500)
        # T =  T_guess
        # print(1)

    def calc_phase_envelope(self):
        P0 = 1.013*100000
        P_step = 300000
        DewLine = []
        BubbleLine = []

        # 定义温度范围和点数
        start_temperature = 273.15  # 起始温度：150K
        end_temperature = 360.3083  # 结束温度：360.3083K
        num_points = 30  # 总点数
        temperatures = np.linspace(start_temperature,  end_temperature, num_points)
        # 计算每个温度对应的泡点压力
        pressures = [self.calc_Bubble_P(temp) for temp in temperatures]
        print(1)










        T_Dew = self.calc_Dew_T(P0)
        DewLine.append([P0, T_Dew])
        T_step = 2
        T0 = T_Dew
        P_Bubble = self.calc_Bubble_P(T0)
        BubbleLine.append([P_Bubble, T0])

        while True:
            try:
                if T0 >= T_Dew:
                    P0 = P0 + P_step
                    T_Dew = self.calc_Dew_T(P0,T0 = T_Dew)
                    DewLine.append([P0, T_Dew])
                else:
                    T0 = T0 + T_step
                    P_Bubble = self.calc_Bubble_P(T0)
                    BubbleLine.append([P_Bubble, T0])

                if P0 > P_Bubble:
                    print(1)
                    break
            except:
                print(1)
                break






















    def calc_falsh(self,P,T,tol=0.000001,maxiter=1000):
        '''
        计算气、液闪蒸平衡
        :param P: 压力，单位：Pa
        :param T: 温度，单位：K
        :return: 通过逸度相等，计算各组分的平衡常数Ki
        '''
        self.components = self.EOS.calc_ai(self.components, T)
        '组分初始ai与流体温度有关，根据温度重新计算'
        Ki_array = self.calc_Ki(P, T)
        'Wilson公式计算初始Ki'
        Ki_array,Ki_max, Ki_min = self.check_Ki(Ki_array)
        # isOver = True
        xi_G = np.ones(len(self.components))
        '气相中各组分占比,赋初值'
        xi_L = np.ones(len(self.components))
        '液相中各组分占比,赋初值'
        while True:
            vap_frac = self.calc_vap_frac(Ki_array,Ki_max, Ki_min)
            '求解气相分率'
            self.a_G = vap_frac
            for i in range(len(self.components)):
                xi_L[i]  = self.components[i].xi / (1 + vap_frac * (Ki_array[i] - 1))
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
            Z_L, A_L, B_L, a_L, b_L, f_L_s = self.EOS.calc(self.components, P, T, xi_L, "Liquid")
            '根据状态方程计算压缩因子、逸度'
            Z_G, A_G, B_G, a_G, b_G, f_G_s = self.EOS.calc(self.components, P, T, xi_G, "Gas")
            '根据状态方程计算压缩因子、逸度'
            diff = 0
            for i in range(len(self.components)):
                if f_G_s[i] != 0:
                    diff = diff + abs(f_L_s[i]/f_G_s[i]-1)
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

    def phase_equilibrium_calc(self,P,T):
        '''
        计算组分在温度T和压力P下的物理性质
        :param P: 温度，单位：k
        :param T: 压力，单位：Pa
        :return:
        '''
        '计算闪蒸平衡'
        self.calc_falsh(P, T)

        if self.a_G >1:
            self.phase = "气相"
        elif self.a_G == 1:
            self.phase = "露点"
        elif self.a_G < 0:
            self.phase = "液相"
        elif self.a_G ==0:
            self.phase = "泡点"
        else:
            self.phase = "气液两相"
        if  self.a_G <= 1:
            if self.a_G < 0:
                self.a_G = 0
            self.calc_Oil_Water_Balance()
        else:
            self.a_G = 1
            self.a_O = 0
            self.a_W = 0
            self.xis_O = np.zeros(len(self.components))
            self.xis_W = np.zeros(len(self.components))








        # self.components = self.EOS.calc_ai(self.components, self.T)
        # Bubble_P = self.calc_Bubble_P()
        # '''计算泡点压力'''
        # Dew_T = self.calc_Dew_T()
        # '''计算露点温度'''
        # if self.T >= Dew_T:
        #     Phase = "气相"
        #     zis = np.ones(len(self.components))  # 初始化 zis 为全 1 的数组
        #     for i in range(len(self.components)):
        #         zis[i] = self.components[i].xi
        #     Z_G, A_G, B_G, a_G, b_G, f_G_s = self.EOS.calc(self.components, self.P, self.T, zis, "Gas")
        #     self.xis_G = zis
        #     self.Z_G = Z_G
        #     self.a_G = 1
        #     self.a_O = 0
        #     self.a_W = 0
        #     self.xis_O = np.zeros(len(self.components))
        #     self.xis_W = np.zeros(len(self.components))
        #     self.phase = Phase
        # elif self.P >= Bubble_P:
        #     Phase = "液相"
        #     # 这里求解结束，求解完成，这里要对液相做单独处理
        #     zis = np.ones(len(self.components))
        #     for i in range(len(self.components)):
        #         zis[i] = self.components[i].xi
        #     Z_L, A_L, B_L, a_L, b_L, f_L_s = self.EOS.calc(self.components, self.P, self.T, zis, "Liquid")
        #     self.Z_L = Z_L
        #     self.a_G = 0
        #     self.phase = Phase
        #     self.xis_L = zis
        #     self.xis_G = np.zeros(len(self.components))
        #     self.calc_Oil_Water_Balance()
        # else:
        #     Phase = "气液两相"
        #     self.phase = Phase
        #     self.calc_falsh(self.P, self.T)













