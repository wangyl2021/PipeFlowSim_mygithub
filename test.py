import numpy as np
from scipy.optimize import root
import math





# 原始的calc_Z函数
def calc_Z(T, P, Gas_Specific_Gravity, model="Standing"):
    """
    使用Standing、McCabe-Thiele、Robinson方法计算气体压缩因子(Z-factor)

    参数:
        T (float): 温度(°R)
        P (float): 压力(psia)
        Gas_Specific_Gravity (float): 气体相对密度(空气=1)

    返回:
        float: 气体压缩因子Z

    注意事项:
        1. 方法适用范围: 对比温度Tr > 0.92
        2. 临界参数计算基于气体和凝析气体系经验公式
    """
    # 气体体系公式
    Tc_gas = 168 + 325 * Gas_Specific_Gravity - 12.5 * Gas_Specific_Gravity ** 2
    # 凝析气体系公式
    # Tc_gas_condensate = 187 + 330 * Gas_Specific_Gravity - 71.5 * Gas_Specific_Gravity ** 2
    # 计算临界压力(使用凝析气体系公式)
    pc = 706 - 51.7 * Gas_Specific_Gravity - 11.1 * Gas_Specific_Gravity ** 2
    Tr = T / Tc_gas  # 对比温度
    pr = P / pc  # 对比压力
    if model == "Standing":
        if Tr < 0.92:
            raise ValueError("方法适用范围为Tr >= 0.92，当前Tr值超出范围")

        # 计算四参数方程中的系数A(温度相关系数)
        A = 1.39 * (Tr - 0.92) ** 0.5 - 0.36 * Tr - 0.101

        # 计算系数B(压力相关系数，包含三项不同幂次的贡献)
        term1 = (0.62 - 0.23 * Tr) * pr  # 线性项
        term2 = ((0.666 / (Tr - 0.86)) - 0.037) * pr ** 2  # 二次项
        term3 = (0.32 * pr ** 6) / (10 ** (9 * (Tr - 1)))  # 高次项(对比温度校正)
        B = term1 + term2 + term3
        # 计算系数C(对数温度校正系数)
        C = 0.132 - 0.32 * math.log10(Tr)
        # 计算系数D(指数温度校正系数)
        D = 10 ** (0.3016 - 0.49 * Tr + 0.1824 * Tr ** 2)
        # 根据B值选择不同的计算公式(处理高对比压力情况)
        if B < 50:
            Z = A + C * pr ** D  # 简化公式
        else:
            Z = A + (1 - A) * math.exp(-B) + C * pr ** D  # 完整公式
    elif model == "McCabe-Thiele":
        # 计算对比参数(无因次化)
        if pr >= 0 and pr <= 1:
            raise ValueError("该方法不建议在对比压力pr范围[0, 1]内使用")
        t = 1 / Tr

        # 定义求解约化密度的方程
        def f(rhoR):
            A1 = 0.06125 * pr * t * np.exp(-1.2 * (1 - t) ** 2)
            A2 = 14.76 * t - 9.76 * t ** 2 + 4.85 * t ** 3
            A3 = 90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3
            A4 = 2.18 + 2.82 * t
            num = rhoR + rhoR ** 2 + rhoR ** 3 - rhoR ** 4
            den = (1 - rhoR) ** 3
            return -A1 * pr + num / den - A2 * rhoR ** 2 + A3 * rhoR ** A4

        # 初始猜测值
        rhoR_guess = 0.5
        # 求解约化密度
        sol = root(f, rhoR_guess)
        if not sol.success:
            raise Exception("求解约化密度失败")
        rhoR = sol.x[0]

        # 计算气体压缩因子Z
        A1 = 0.06125 * pr * t * np.exp(-1.2 * (1 - t) ** 2)
        Z = A1 / rhoR
    elif model == "Robinson":
        if Tr < 1.05 or Tr > 3.0 or pr < 0.2 or pr > 3.0:
            raise ValueError("该方法适用的对比温度范围为[1.05, 3.0]，对比压力范围为[0.2, 3.0]")
        A1 = 0.310506237
        A2 = -1.4067099
        A3 = -0.57832729
        A4 = 0.53530771
        A5 = -0.61232032
        A6 = -0.10488813
        A7 = 0.68157001
        A8 = 0.68446549
        D = 0.27 * pr / Tr
        t = 1 / Tr

        def f(rhoR):
            A1 = 0.06125 * pr * t * np.exp(-1.2 * (1 - t) ** 2)
            A2 = 14.76 * t - 9.76 * t ** 2 + 4.85 * t ** 3
            A3 = 90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3
            A4 = 2.18 + 2.82 * t
            num = rhoR + rhoR ** 2 + rhoR ** 3 - rhoR ** 4
            den = (1 - rhoR) ** 3
            return -A1 * pr + num / den - A2 * rhoR ** 2 + A3 * rhoR ** A4

        # 初始猜测值
        rhoR_guess = 0.5
        # 求解约化密度
        sol = root(f, rhoR_guess)
        if not sol.success:
            raise Exception("求解约化密度失败")
        rhoR = sol.x[0]
        term1 = (A1 + A2 / Tr + A3 / (Tr ** 3)) * rhoR
        term2 = (A4 + A5 / Tr) * rhoR ** 2
        term3 = (A5 * A6 / rhoR) * D ** 5
        term4 = (A7 / (Tr ** 3)) * rhoR ** 2 * (1 + A8 * rhoR ** 2) * math.exp(-A8 * rhoR ** 2)
        Z = 1 + term1 + term2 + term3 + term4

    return Z