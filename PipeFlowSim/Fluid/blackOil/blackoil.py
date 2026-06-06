import math
import numpy as np
from scipy.optimize import root
from Fluid.ConversionUnit import joule_kgk_to_btu_lbmdegf, \
    degf_psi_to_k_pa

den_standard_Air = 1.225
def dead_oil_viscosity(api, T, model="De Ghetto et al"):
    if model == "De Ghetto et al":
        if api < 10:
            y = 1.90296 - 0.012619 * api - 0.61748 * math.log10(T)
            x = 10 ** y
            return 10 ** x - 1
        elif 10 <= api < 22.3:
            y = 2.06492 - 0.0179 * api - 0.70226 * math.log10(T)
            x = 10 ** y
            return 10 ** x - 1
        elif 22.3 <= api < 31.1:
            c = 220.15 * (10 ** 9) * (T ** (-3.556))
            d = 12.5428 * math.log10(T) - 45.7874
            return c * (math.log10(api) ** d)
        else:
            y = 1.67083 - 0.017628 * api - 0.61304 * math.log10(T)
            x = 10 ** y
            U_od =  10 ** x - 1
    elif model == "Beggs and Robinson":
        z = 3.0324 - 0.02023 * api
        y = 10 ** z
        x = y * (T ** (-1.163))
        U_od = 10 ** x - 1
    return U_od
def live_oil_viscosity(api, T, Rs, model="De Ghetto et al"):
    U_od = dead_oil_viscosity(api, T)
    if model == "De Ghetto et al":
        if api < 10:
            B = 10 ** (-0.00081 * Rs)
            A = -0.0335 + 1.0875 * (10 ** (-0.000845 * Rs))
            F = A * (U_od ** (0.5798 + 0.3432 * B))
            U_ob = 2.3945 + 0.8927 * F + 0.01567 * (F ** 2)
        elif 10 <= api < 22.3:
            B = 10 ** (-0.00081 * Rs)
            A = 0.2478 + 0.6114 * (10 ** (-0.000845 * Rs))
            F = A * (U_od ** (0.4731 + 0.5664 * B))
            U_ob = -0.6311 + 1.078 * F - 0.003653 * (F ** 2)
        elif 22.3 <= api < 31.1:
            B = 10 ** (-0.00081 * Rs)
            A = 0.2038 + 0.8591 * (10 ** (-0.000845 * Rs))
            F = A * (U_od ** (0.3855 + 0.5664 * B))
            U_ob = 0.0132 + 0.9821 * F - 0.005215 * (F ** 2)
        else:
            A = 25.1921 * (Rs + 100) ** (-0.6487)
            B = 2.7516 * (Rs + 150) ** (-0.2135)
            U_ob = A * (U_od ** B)
    return U_ob
def undersaturated_oil_viscosity(api, p, pb, T, Rs, model="De Ghetto et al"):

    U_ob = live_oil_viscosity(api, T, Rs)
    if model == "De Ghetto et al":
        if api < 10:
            U_od = dead_oil_viscosity(api, T)  # 这里需先实现计算死油粘度函数来准确赋值，此处先占位
            A = 10 ** (-2.19) * (U_od ** 1.055) * (pb ** 0.3132)
            B = 10 ** (0.0099 * api)
            U_ou = U_ob - (1 - p / pb) * (A / B)
        elif 10 <= api < 22.3:
            A = -0.01153 * (U_ob ** 1.7933) + 0.03610 * (U_ob ** 1.5939)
            U_ou = 0.9886 * U_ob + 0.002763 * A * (p - pb)
        else:
            U_od = dead_oil_viscosity(api, T)  # 这里需先实现计算死油粘度函数来准确赋值，此处先占位
            A = 10 ** (-2.19) * (U_od ** 1.055) * (pb ** 0.3132)
            B = 10 ** (-0.00288 * api)
            U_ou = U_ob - (1 - p / pb) * (A / B)
    return U_ou
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
    if P < Pb:
        Oil_Viscosity = live_oil_viscosity(Oil_API, T, Rs)
    else:
        Oil_Viscosity = undersaturated_oil_viscosity(Oil_API, P, Pb, T, Rs)
    return Oil_Viscosity
def calc_Z(T, P, Gas_Specific_Gravity,model="Standing"):
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
    elif  model == "Robinson":
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
def calc_dZ_dT(T, P, Gas_Specific_Gravity, model="Standing"):
    """
    解析计算气体压缩因子Z对温度T的导数(dZ/dT)

    参数:
        T (float): 温度(°R)
        P (float): 压力(psia)
        Gas_Specific_Gravity (float): 气体相对密度(空气=1)
        model (str): 计算模型，可选"Standing"、"McCabe-Thiele"、"Robinson"

    返回:
        float: 气体压缩因子Z对温度T的导数(dZ/dT)
    """
    # 计算临界参数
    Tc_gas = 168 + 325 * Gas_Specific_Gravity - 12.5 * Gas_Specific_Gravity ** 2
    pc = 706 - 51.7 * Gas_Specific_Gravity - 11.1 * Gas_Specific_Gravity ** 2

    # 计算对比参数
    Tr = T / Tc_gas
    pr = P / pc

    # 对Tr求T的导数
    dTr_dT = 1 / Tc_gas

    if model == "Standing":
        if Tr < 0.92:
            raise ValueError("方法适用范围为Tr >= 0.92，当前Tr值超出范围")

        # 计算原始Z值所需的系数
        A = 1.39 * (Tr - 0.92) ** 0.5 - 0.36 * Tr - 0.101

        term1 = (0.62 - 0.23 * Tr) * pr
        term2 = ((0.666 / (Tr - 0.86)) - 0.037) * pr ** 2
        term3 = (0.32 * pr ** 6) / (10 ** (9 * (Tr - 1)))
        B = term1 + term2 + term3

        C = 0.132 - 0.32 * math.log10(Tr)
        D = 10 ** (0.3016 - 0.49 * Tr + 0.1824 * Tr ** 2)

        # 对各系数求Tr的导数
        dA_dTr = 1.39 * 0.5 * (Tr - 0.92) ** (-0.5) - 0.36

        dterm1_dTr = -0.23 * pr
        dterm2_dTr = (-0.666 / (Tr - 0.86) ** 2) * pr ** 2
        dterm3_dTr = (0.32 * pr ** 6) * (-9 * math.log(10) * 10 ** (-9 * (Tr - 1)))
        dB_dTr = dterm1_dTr + dterm2_dTr + dterm3_dTr

        dC_dTr = -0.32 / (Tr * math.log(10))
        dD_dTr = D * math.log(10) * (-0.49 + 0.3648 * Tr)

        # 计算dZ/dTr
        if B < 50:
            # Z = A + C * pr^D
            dZ_dTr = dA_dTr + dC_dTr * (pr ** D) + C * (pr ** D) * math.log(pr) * dD_dTr
        else:
            # Z = A + (1-A)*exp(-B) + C*pr^D
            dZ_dTr = dA_dTr + (-dA_dTr) * math.exp(-B) + (1 - A) * (-math.exp(-B)) * dB_dTr + \
                     dC_dTr * (pr ** D) + C * (pr ** D) * math.log(pr) * dD_dTr

        # 应用链式法则：dZ/dT = dZ/dTr * dTr/dT
        dZ_dT = dZ_dTr * dTr_dT

    elif model == "McCabe-Thiele":
        if pr >= 0 and pr <= 1:
            raise ValueError("该方法不建议在对比压力pr范围[0, 1]内使用")

        t = 1 / Tr
        dt_dTr = -1 / (Tr ** 2)

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

        # 对f(rhoR)关于Tr求导，用于隐函数求导
        def df_dTr(rhoR):
            # 计算A1及其导数
            A1 = 0.06125 * pr * t * np.exp(-1.2 * (1 - t) ** 2)
            dA1_dt = 0.06125 * pr * (np.exp(-1.2 * (1 - t) ** 2) + t * np.exp(-1.2 * (1 - t) ** 2) * 2.4 * (1 - t))
            dA1_dTr = dA1_dt * dt_dTr

            # 计算A2及其导数
            A2 = 14.76 * t - 9.76 * t ** 2 + 4.85 * t ** 3
            dA2_dt = 14.76 - 19.52 * t + 14.55 * t ** 2
            dA2_dTr = dA2_dt * dt_dTr

            # 计算A3及其导数
            A3 = 90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3
            dA3_dt = 90.7 - 484.4 * t + 127.2 * t ** 2
            dA3_dTr = dA3_dt * dt_dTr

            # 计算A4及其导数
            A4 = 2.18 + 2.82 * t
            dA4_dt = 2.82
            dA4_dTr = dA4_dt * dt_dTr

            # 计算f对Tr的导数
            return -dA1_dTr * pr - dA2_dTr * rhoR ** 2 + dA3_dTr * rhoR ** A4 + A3 * rhoR ** A4 * math.log(
                rhoR) * dA4_dTr

        # 对f(rhoR)关于rhoR求导
        def df_drhoR(rhoR):
            A1 = 0.06125 * pr * t * np.exp(-1.2 * (1 - t) ** 2)
            A2 = 14.76 * t - 9.76 * t ** 2 + 4.85 * t ** 3
            A3 = 90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3
            A4 = 2.18 + 2.82 * t

            # 计算f对rhoR的导数
            num = rhoR + rhoR ** 2 + rhoR ** 3 - rhoR ** 4
            den = (1 - rhoR) ** 3
            dnum_drhoR = 1 + 2 * rhoR + 3 * rhoR ** 2 - 4 * rhoR ** 3
            dden_drhoR = 3 * (1 - rhoR) ** 2
            d_frac_drhoR = (dnum_drhoR * den - num * dden_drhoR) / (den ** 2)

            return d_frac_drhoR - 2 * A2 * rhoR + A3 * A4 * rhoR ** (A4 - 1)

        # 使用隐函数求导法计算drhoR/dTr
        drhoR_dTr = -df_dTr(rhoR) / df_drhoR(rhoR)

        # 计算Z = A1/rhoR的导数
        A1 = 0.06125 * pr * t * np.exp(-1.2 * (1 - t) ** 2)
        dA1_dt = 0.06125 * pr * (np.exp(-1.2 * (1 - t) ** 2) + t * np.exp(-1.2 * (1 - t) ** 2) * 2.4 * (1 - t))
        dA1_dTr = dA1_dt * dt_dTr

        dZ_dTr = (dA1_dTr * rhoR - A1 * drhoR_dTr) / (rhoR ** 2)
        dZ_dT = dZ_dTr * dTr_dT

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
        dD_dTr = -0.27 * pr / (Tr ** 2)

        t = 1 / Tr
        dt_dTr = -1 / (Tr ** 2)

        # 定义求解约化密度的方程
        def f(rhoR):
            A1_term = 0.06125 * pr * t * np.exp(-1.2 * (1 - t) ** 2)
            A2_term = 14.76 * t - 9.76 * t ** 2 + 4.85 * t ** 3
            A3_term = 90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3
            A4_term = 2.18 + 2.82 * t
            num = rhoR + rhoR ** 2 + rhoR ** 3 - rhoR ** 4
            den = (1 - rhoR) ** 3
            return -A1_term * pr + num / den - A2_term * rhoR ** 2 + A3_term * rhoR ** A4_term

        # 初始猜测值
        rhoR_guess = 0.5
        # 求解约化密度
        sol = root(f, rhoR_guess)
        if not sol.success:
            raise Exception("求解约化密度失败")
        rhoR = sol.x[0]

        # 对f(rhoR)关于Tr求导，用于隐函数求导
        def df_dTr(rhoR):
            # 计算A1_term及其导数
            A1_term = 0.06125 * pr * t * np.exp(-1.2 * (1 - t) ** 2)
            dA1_dt = 0.06125 * pr * (np.exp(-1.2 * (1 - t) ** 2) + t * np.exp(-1.2 * (1 - t) ** 2) * 2.4 * (1 - t))
            dA1_dTr = dA1_dt * dt_dTr

            # 计算A2_term及其导数
            A2_term = 14.76 * t - 9.76 * t ** 2 + 4.85 * t ** 3
            dA2_dt = 14.76 - 19.52 * t + 14.55 * t ** 2
            dA2_dTr = dA2_dt * dt_dTr

            # 计算A3_term及其导数
            A3_term = 90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3
            dA3_dt = 90.7 - 484.4 * t + 127.2 * t ** 2
            dA3_dTr = dA3_dt * dt_dTr

            # 计算A4_term及其导数
            A4_term = 2.18 + 2.82 * t
            dA4_dt = 2.82
            dA4_dTr = dA4_dt * dt_dTr

            # 计算f对Tr的导数
            return -dA1_dTr * pr - dA2_dTr * rhoR ** 2 + dA3_dTr * rhoR ** A4_term + A3_term * rhoR ** A4_term * math.log(
                rhoR) * dA4_dTr

        # 对f(rhoR)关于rhoR求导
        def df_drhoR(rhoR):
            A1_term = 0.06125 * pr * t * np.exp(-1.2 * (1 - t) ** 2)
            A2_term = 14.76 * t - 9.76 * t ** 2 + 4.85 * t ** 3
            A3_term = 90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3
            A4_term = 2.18 + 2.82 * t

            # 计算f对rhoR的导数
            num = rhoR + rhoR ** 2 + rhoR ** 3 - rhoR ** 4
            den = (1 - rhoR) ** 3
            dnum_drhoR = 1 + 2 * rhoR + 3 * rhoR ** 2 - 4 * rhoR ** 3
            dden_drhoR = 3 * (1 - rhoR) ** 2
            d_frac_drhoR = (dnum_drhoR * den - num * dden_drhoR) / (den ** 2)

            return d_frac_drhoR - 2 * A2_term * rhoR + A3_term * A4_term * rhoR ** (A4_term - 1)

        # 使用隐函数求导法计算drhoR/dTr
        drhoR_dTr = -df_dTr(rhoR) / df_drhoR(rhoR)

        # 计算Z各部分的导数
        term1 = (A1 + A2 / Tr + A3 / (Tr ** 3)) * rhoR
        dterm1_dTr = (-A2 / (Tr ** 2) - 3 * A3 / (Tr ** 4)) * rhoR + (A1 + A2 / Tr + A3 / (Tr ** 3)) * drhoR_dTr

        term2 = (A4 + A5 / Tr) * rhoR ** 2
        dterm2_dTr = (-A5 / (Tr ** 2)) * rhoR ** 2 + (A4 + A5 / Tr) * 2 * rhoR * drhoR_dTr

        term3 = (A5 * A6 / rhoR) * D ** 5
        dterm3_dTr = (A5 * A6 / rhoR) * 5 * D ** 4 * dD_dTr + (A5 * A6 * (-1) / (rhoR ** 2)) * D ** 5 * drhoR_dTr

        term4 = (A7 / (Tr ** 3)) * rhoR ** 2 * (1 + A8 * rhoR ** 2) * math.exp(-A8 * rhoR ** 2)
        dterm4_dTr = (-3 * A7 / (Tr ** 4)) * rhoR ** 2 * (1 + A8 * rhoR ** 2) * math.exp(-A8 * rhoR ** 2) + \
                     (A7 / (Tr ** 3)) * (2 * rhoR * drhoR_dTr * (
                    1 + A8 * rhoR ** 2) + rhoR ** 2 * 2 * A8 * rhoR * drhoR_dTr) * math.exp(-A8 * rhoR ** 2) + \
                     (A7 / (Tr ** 3)) * rhoR ** 2 * (1 + A8 * rhoR ** 2) * math.exp(-A8 * rhoR ** 2) * (
                                 -2 * A8 * rhoR * drhoR_dTr)

        dZ_dTr = dterm1_dTr + dterm2_dTr + dterm3_dTr + dterm4_dTr
        dZ_dT = dZ_dTr * dTr_dT

    else:
        raise ValueError("不支持的模型类型，可选：'Standing', 'McCabe-Thiele', 'Robinson'")

    return dZ_dT
def calc_Rs(T, P, Gas_Specific_Gravity, API,Rs_max,C=1, P_sep=None, T_sep=None,model="De ghetto"):
    """
    根据不同油品API度计算Rs
    :param P: 压力，Psia
    :param T: 温度,华氏度°F
    :param API: 油品API度
    :param Gas_Specific_Gravity: 气体比重
    :param C: 校准常数
    :param P_sep: 分离器压力（可选，用于中质油计算）
    :param T_sep: 分离器温度（可选，用于中质油计算）
    :return: Rs的值,scf/STB
    """
    # T = T-460
    if model == "De ghetto":
        # 计算Rs
        if API <= 10:
            A = 10 ** (0.002 * T - 0.0142 * API)
            Rs = C * Gas_Specific_Gravity * (P / (10.7025 * A)) ** 1.1128
            Pb = 10.7025 * A * (Rs_max / (C * Gas_Specific_Gravity)) ** (1 / 1.1128)
        elif 10 < API <= 22.3:
            A = 10 ** (0.002 * T - 0.0142 * API)
            Rs = C * Gas_Specific_Gravity * (P / (15.7286 * A)) ** (1 / 0.7885)
            Pb = 15.7286 * A * (Rs_max / (C * Gas_Specific_Gravity)) ** 0.7885
        elif 22.3 < API <= 31.1:
            A = 10 ** (7.4576 * API / (T + 460))
            if P_sep and T_sep:
                g_corr = 0.1595 * API ** 0.4078 * T_sep ** (-0.2466) * math.log10(P_sep / 114.7)
            else:
                g_corr = 0
            C1 = 0.10084
            C2 = 0.2556
            C4 = 0.9868
            Rs = C * C1 * (Gas_Specific_Gravity * (1 + g_corr)) ** C2 * A * P ** C4
            Pb = (Rs_max / (C * C1 * (Gas_Specific_Gravity * (1 + g_corr)) ** C2 * A)) ** (1 / C4)
        elif API > 31.1:
            A = 10 ** (0.0009 * T - 0.0148 * API)
            Rs = C * Gas_Specific_Gravity * (P / (31.7648 * A)) ** (1 / 0.7885)
            Pb = 31.7648 * A * (Rs_max / (C * Gas_Specific_Gravity)) ** 0.7885
        else:
            raise ValueError("API值不在有效范围内")
    elif model == "Kartoatmodjo and Schmidt":
        # 计算Rs
        if API < 30:
            C1 = 0.05958
            C2 = 0.7972
            C3 = 13.1405
            C4 = 1.0014
        else:
            C1 = 0.0315
            C2 = 0.7587
            C3 = 11.2895
            C4 = 1.0937
        if P_sep is not None and T_sep is not None:
            g_corr = 0.1595 * (API ** 0.4078) * (T_sep ** -0.2466) * math.log10(P_sep / 114.7)
        else:
            g_corr = 0

        A_log = C3 * API / (T + 460)
        A = 10 ** A_log
        Rs = C * C1 * (Gas_Specific_Gravity * (1 + g_corr)) ** C2 * A * (P ** C4)
        Pb = ((Rs_max / (C * C1 * (Gas_Specific_Gravity * (1 + g_corr)) ** C2 * A)) ** (1 / C4))

    return Rs,Pb # SCF/STB，Psia
def calc_Bg(Z_Factor, T, P):
    '''
    气体体积因子（单位：RB/SCF），表示地下气体体积与标准条件的体积比
    :param Z_Factor: Z 因子
    :param T: 温度，兰金度
    :param P: 压力
    :return: 气体体积因子 Bg
    '''
    # 根据公式计算气体体积因子 Bg

    T_sc = 273.15
    P_sc = 14.69595
    T=T*5/9
    Bg = (Z_Factor * P_sc * T) / (P * T_sc)#0.00504 * Z_Factor * T / P
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
    # 将温度从兰金度转换为华氏度
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
    # print(T, Gas_molecular_weight, Gas_Density)
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
    :return: 油密度, 水密度, 混合流体密度
    """
    # 计算油的密度（考虑溶解气）
    den_oil = (62.4 * Oil_Specific_Gravity + (Rs * 0.0764 * Gas_Specific_Gravity / 5.615)) / Bo
    # 计算水的密度
    den_water = 62.4 * Water_Specific_Gravity / Bw
    # 计算混合流体密度
    den_liquid = den_oil * (1 - Water_Cut) + den_water * Water_Cut
    return den_oil, den_water, den_liquid
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
def calc_oil_H(cp_o,P,T):
    # Oil phase enthalpy
    cp_o = joule_kgk_to_btu_lbmdegf(cp_o)
    # T = kelvin_to_fahrenheit(T)# 开尔文（K）转华氏度（°F）
    # P = pa_to_psia(P) # Pa to Psi
    h_o = cp_o * T + 3.36449e-3 * P
    return h_o
def calc_gas_H(cp_g,P,T):
    # gas phase enthalpy
    cp_g = joule_kgk_to_btu_lbmdegf(cp_g)
    # T = kelvin_to_fahrenheit(T)  # 开尔文（K）转华氏度（°F）
    # P = pa_to_psia(P)  # Pa to Psi
    h_g = cp_g * T + P * ((1.619e-10 * P + 1.412e-6) * P - 0.02734)
    return h_g
def calc_water_H(cp_w,P,T,gamma_w):
    # Water phase enthalpy
    cp_w = joule_kgk_to_btu_lbmdegf(cp_w)
    # T = kelvin_to_fahrenheit(T)  # 开尔文（K）转华氏度（°F）
    # P = pa_to_psia(P)  # Pa to Psi
    h_w = cp_w * T + (2.9641e-3 / gamma_w) * P
    return h_w

def joule_thomson_gas(den_gas,T,c_pg,Z,dZ_dT):
    '''
    计算joule_thomson系数
    :param den_gas: 天然气密度
    :param c_pg: 天然气比热容
    :param P: 天然气相对密度
    :param T: 天然气温度
    :return: 焦耳-汤姆逊系数,K/Pa
    '''
    T = T - 459.67
    c_pg = joule_kgk_to_btu_lbmdegf(c_pg)
    joule_thomson_gas = (1 / (den_gas * c_pg)) * (T / Z) * dZ_dT / 5.40395
    # joule_thomson_gas = convert_fahrenheit_psia_to_kelvin_pa(joule_thomson_gas)
    return joule_thomson_gas

class pvt_params:
    """
    该类用于存储和计算与 PVT（压力 - 体积 - 温度）相关的参数。
    这些参数包括油、水和气的属性，以及根据这些属性计算的各种物理量。
    """
    def __init__(self, oil_api, water_cut, GOR,
                 gas_specific_gravity,
                 Water_Specific_Gravity,
                 Oil_C0, Gas_C0, Water_C0
                 ):
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
        self.Oil_C0 = Oil_C0
        '''油在常压下的比容'''
        # 水的属性
        self.Water_Cut = water_cut
        '''体积含水率，范围在 0 到 1 之间'''
        self.Water_Specific_Gravity = Water_Specific_Gravity
        self.Gas_C0 = Gas_C0
        '''气在常压下的比容'''
        self.GOR = GOR
        '''生产气油比，即总气（溶解气 + 游离气）产量除以油产量'''
        # 根据气油比和油产量计算气的产量，单位为标准立方英尺/天（scf/D）
        self.Gas_Specific_Gravity = gas_specific_gravity
        # 根据气的比重计算气的分子量，单位为磅/磅摩尔（Ibm/Ibmole）
        self.Gas_molecular_weight = 28.97 * gas_specific_gravity
        self.Water_C0 = Water_C0
        '''水在常压下的比容'''
        self.GLR = self.GOR / (1 / (1 - water_cut))
        self.OWR = (1-water_cut)/water_cut

        volume_OGW = self.GOR+1/(1-water_cut)
        self.G_volume_R = self.GOR/volume_OGW
        '''气体体积占比'''
        self.O_volume_R = 1/volume_OGW
        '''死油体积占比'''
        self.W_volume_R = (1/(1-water_cut)-1)/volume_OGW
        '''水体积占比'''


        weight_OGW = self.GOR*gas_specific_gravity*den_standard_Air+ (1-water_cut)*self.Oil_Specific_Gravity*1000 + water_cut*Water_Specific_Gravity*1000
        self.G_weight_R = self.GOR*gas_specific_gravity*den_standard_Air/weight_OGW
        '''气体质量占比'''
        self.O_weight_R = (1-water_cut)*self.Oil_Specific_Gravity*1000/weight_OGW
        '''死油质量占比'''
        self.W_weight_R = water_cut*Water_Specific_Gravity*1000/weight_OGW
        '''水质量占比'''






    def calc(self,P: float,T: float):
        # 单位转换
        T = T * 1.8
        '''开氏度转兰式度'''
        P = P * 0.000145038
        '''Pa转psi'''

        Zg=calc_Z(T, P, self.Gas_Specific_Gravity)
        Rs,Pb=calc_Rs(T-459.67, P, self.Gas_Specific_Gravity, self.Oil_API,self.GOR/0.1781076)
        # Pb = calc_Pb(Rs, self.Gas_Specific_Gravity, self.Oil_Specific_Gravity, T)
        Bg=calc_Bg(Zg,T,P)
        Bo=calc_Bo(T,Rs, self.Gas_Specific_Gravity, self.Oil_Specific_Gravity)
        Bw=calc_Bw(T, P)
        if P<Pb:
            dZg = calc_dZ_dT(T, P, self.Gas_Specific_Gravity)


            weight_OGW = self.GOR * self.Gas_Specific_Gravity * den_standard_Air + self.Oil_Specific_Gravity * 1000 + \
                         self.Water_Cut/(1-self.Water_Cut) * self.Water_Specific_Gravity * 1000
            #根据溶解气油比计算气体质量占比
            G_weight_R = (self.GOR-Rs*0.1781076) * self.Gas_Specific_Gravity * den_standard_Air / weight_OGW
            '''气体质量占比'''
            O_weight_R = (self.Oil_Specific_Gravity * 1000 + Rs*0.1781076* self.Gas_Specific_Gravity* den_standard_Air) / weight_OGW
            '''油质量占比'''
            W_weight_R = self.Water_Cut/(1-self.Water_Cut) * self.Water_Specific_Gravity * 1000 / weight_OGW
            '''水质量占比'''
            den_gas = calc_den_gas(T, P, self.Gas_Specific_Gravity, Zg)
            den_oil, den_water, den_liquid = calc_den_liquid(self.Oil_Specific_Gravity, Rs, self.Gas_Specific_Gravity,
                                                             Bo, self.Water_Cut, self.Water_Specific_Gravity, Bw)
            V_sum = G_weight_R/den_gas + O_weight_R/den_oil + W_weight_R/den_water
            O_volume_R = (O_weight_R/den_oil) / V_sum
            '''油体积占比'''
            W_volume_R = (W_weight_R/den_water) / V_sum
            G_volume_R = (G_weight_R/den_gas) / V_sum
            joule_thomson = joule_thomson_gas(den_gas, T,self.Gas_C0,  Zg, dZg)

            gas_viscosity = calc_gas_viscosity(T, self.Gas_molecular_weight, den_gas)
            gas_viscosity = gas_viscosity / 1000  # cp转Pa*s
            oil_viscosity = calc_oil_viscosity(T - 459.67, P, self.Oil_API, Rs, Pb)
            oil_viscosity = oil_viscosity / 1000  # cp转Pa*s
            water_viscosity = calc_water_viscosity(T)
            water_viscosity = water_viscosity / 1000  # cp转Pa*s
        else:
            # 大于饱和压力，此时没有气体析出
            weight_OGW = self.GOR * self.Gas_Specific_Gravity * den_standard_Air + self.Oil_Specific_Gravity * 1000 + \
                         self.Water_Cut / (1 - self.Water_Cut) * self.Water_Specific_Gravity * 1000
            G_weight_R = 0
            '''气体质量占比'''
            O_weight_R = (self.Oil_Specific_Gravity * 1000 + self.GOR * 0.1781076 * self.Gas_Specific_Gravity* den_standard_Air) / weight_OGW
            '''油质量占比'''
            W_weight_R = self.Water_Cut/(1-self.Water_Cut) * self.Water_Specific_Gravity * 1000 / weight_OGW
            '''水质量占比'''
            den_gas = calc_den_gas(T, P, self.Gas_Specific_Gravity, Zg)
            den_oil, den_water, den_liquid = calc_den_liquid(self.Oil_Specific_Gravity, Rs, self.Gas_Specific_Gravity,
                                                             Bo, self.Water_Cut, self.Water_Specific_Gravity, Bw)
            V_sum = G_weight_R / den_gas + O_weight_R / den_oil + W_weight_R / den_water
            O_volume_R = (O_weight_R / den_oil) / V_sum
            W_volume_R = (W_weight_R / den_water) / V_sum
            G_volume_R = (G_weight_R / den_gas) / V_sum
            gas_viscosity = 0 # cp转Pa*s
            oil_viscosity = calc_oil_viscosity(T - 459.67, P, self.Oil_API, Rs, Pb)
            oil_viscosity = oil_viscosity / 1000  # cp转Pa*s
            water_viscosity = calc_water_viscosity(T)
            water_viscosity = water_viscosity / 1000  # cp转Pa*s
            joule_thomson = 0.01


        #
        # if den_gas>den_liquid:
        #     print(1)
        # 粘度单位为cp
        # T = T-460
        sigma_og = calc_oil_gas_surface_tension(T-460, P, self.Oil_API)
        sigma_wg = calc_water_gas_surface_tension(T-460, P)
        sigma_lg = sigma_wg + O_volume_R/(O_volume_R+W_volume_R)*sigma_og
        sigma_wg = sigma_wg / 1000  # dynes/cm转换为N/m
        sigma_og = sigma_og / 1000  # dynes/cm转换为N/m
        sigma_lg = sigma_lg/1000 # dynes/cm转换为N/m
        # 修改代码，保存6位小数
        H_o = calc_oil_H(self.Oil_C0, P, T-460)
        H_g = calc_gas_H(self.Gas_C0, P, T-460)
        H_w = calc_water_H(self.Water_C0, P, T-460, self.Water_Specific_Gravity)
        # joule_thomson_g = joule_thomson_gas(den_gas,T-460, self.Gas_C0, Zg, dZ_dT)
        den_gas = den_gas * 16.02
        '''lbm/ft^3换算为kg/m^3'''
        den_oil = den_oil * 16.02
        '''lbm/ft^3换算为kg/m^3'''
        den_water = den_water * 16.02
        '''lbm/ft^3换算为kg/m^3'''
        den_liquid = den_liquid * 16.02
        '''lbm/ft^3换算为kg/m^3'''

        joule_thomson = degf_psi_to_k_pa(joule_thomson)
        result = {
            'den_gas': round(den_gas, 6),
            'den_oil': round(den_oil, 6),
            'den_water': round(den_water, 6),
            'den_liquid': round(den_liquid, 6),
            'oil_viscosity': round(oil_viscosity, 6),
            'water_viscosity': round(water_viscosity, 6),
            'gas_viscosity': round(gas_viscosity, 6),
            'Bo': round(Bo, 6),
            'Bg': round(Bg, 6),
            'Bw': round(Bw, 6),
            'Zg': round(Zg, 6),
            'sigma_lg': round(sigma_lg, 6),
            'sigma_og': round(sigma_og, 6),
            'sigma_wg': round(sigma_wg, 6),
            'Pb': round(Pb, 6),
            'H_o': round(H_o, 6),
            'H_g': round(H_g, 6),
            'H_w': round(H_w, 6),
            'G_weight_R': round(G_weight_R, 6),
            'O_weight_R': round(O_weight_R, 6),
            'W_weight_R': round(W_weight_R, 6),
            'G_volume_R': round(G_volume_R, 6),
            'O_volume_R': round(O_volume_R, 6),
            'W_volume_R': round(W_volume_R, 6),
            'joule_thomson' : round(joule_thomson, 10)
        }
        return result


if __name__ == '__main__':
    Oil_API = 31.2
    Water_Cut = 0.72
    Water_Specific_Gravity = 1.01
    GOR = 60.23
    Gas_Specific_Gravity = 0.82
    Oil_C0 = 1884.06
    Gas_C0 = 2302.74
    Water_C0 = 4186.8
    fluid = pvt_params(Oil_API, Water_Cut,
                       GOR, Gas_Specific_Gravity,Water_Specific_Gravity,
                       Oil_C0,Gas_C0,Water_C0)

    P = 5*1000000
    # Pa
    T = 100
    T = T+273.15
    # ℃
    fluid_properties = fluid.calc(P,T)
    # 定义表格标题和数据
    headers = ["参数", "值", "单位"]
    data = [
        ["气体质量占比", fluid_properties["G_weight_R"], "%"],
        ["油质量占比", fluid_properties["O_weight_R"], "%"],
        ["水质量占比", fluid_properties["W_weight_R"], "%"],
        ["气体体积占比", fluid_properties["G_volume_R"], "%"],
        ["油体积占比", fluid_properties["O_volume_R"], "%"],
        ["水体积占比", fluid_properties["W_volume_R"], "%"],
        ["气体密度", fluid_properties["den_gas"], "kg/m^3"],
        ["液体密度", fluid_properties["den_liquid"], "kg/m^3"],
        ["油黏度", fluid_properties["oil_viscosity"], "Pa*s"],
        ["水黏度", fluid_properties["water_viscosity"], "Pa*s"],
        ["气黏度", fluid_properties["gas_viscosity"], "Pa*s"],
        ["油体积系数", fluid_properties["Bo"], "/"],
        ["水体积系数", fluid_properties["Bw"], "/"],
        ["气体积系数", fluid_properties["Bg"], "/"],
        ["气体压缩因子", fluid_properties["Zg"], "/"],
        ["油气表面张力", fluid_properties["sigma_og"], "N/m"],
        ["水气表面张力", fluid_properties["sigma_wg"], "N/m"],
        ["液气表面张力", fluid_properties["sigma_lg"], "N/m"],
        ["饱和压力", round(fluid_properties["Pb"]/0.00145038/100000,4), "MPa"],
        ["油焓值", round(fluid_properties["H_o"] , 4), "BTU/lb"],
        ["气焓值", round(fluid_properties["H_g"], 4), "BTU/lb"],
        ["水焓值", round(fluid_properties["H_w"], 4), "BTU/lb"],
        ["气相Joule-Thomson系数", round(fluid_properties["joule_thomson"], 10), "Pa/K"]
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
