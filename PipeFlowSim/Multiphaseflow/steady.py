import  math
import numpy as np

from Fluid.blackOil.blackoil import pvt_params
from Multiphaseflow.holdupModle import drift_flux_Holdup


def calc_f(Absolute_Roughness, Pipe_Diameter, Reynolds_Number):
    """
    计算摩擦系数 Friction_Factor
    :param Absolute_Roughness: 绝对粗糙度
    :param Pipe_Diameter: 管道直径
    :param Reynolds_Number: 雷诺数
    :return: 摩擦系数 Friction_Factor
    """
    # 初始化摩擦系数的估计值
    FrictionFactor_estimate = 0.001
    # 初始化误差为无穷大
    E = float('inf')
    # 循环直到误差小于 0.0001
    while E > 0.0001:
        # 使用 Colebrook 方程计算摩擦系数
        Friction_Factor = (1 / (-2 * math.log10((Absolute_Roughness / (3.7 * Pipe_Diameter)) + (2.51 / (Reynolds_Number * math.sqrt(FrictionFactor_estimate)))))) ** 2
        # 计算当前估计值与新计算值之间的误差
        E = abs(Friction_Factor - FrictionFactor_estimate)
        # 更新估计值为新计算的摩擦系数
        FrictionFactor_estimate = Friction_Factor
    # 返回最终计算得到的摩擦系数
    return Friction_Factor


def temperature_drop(T_0, T_env, K, Do, pipeLen, G, c):
    """
    计算气液两相混输管路的温降
    :param T_0: 初始温度,K
    :param T_env: 环境温度,K
    :param K: 传热系数,W/(m²·K)
    :param Do: 管道直径,m
    :param pipeLen: 管路长度,m
    :param G: 质量流量,kg/s
    :param c: 比热容,J/(kg·K)
    :return: 计算后的温度 T1,K
    """
    T1 = T_0 + (T_env - T_0) * math.exp((-K * Do * pipeLen) / (G * c))
    return T1


def temperature_drop_f(T_env, T0, K, Do, pipeLen,
                       G, c_m, i_l, a_g, Joule_Thomson,
                       C0_g, Cl_g, P_1, P_2):
    """
    计算热油管道终点温度
    参数:
        T_env: 环境温度(K)
        T0: 管道起点温度(K)
        K: 总传热系数(W/(m²·K))
        D: 管道外径(m)
        x: 计算管段长度(m)
        G: 质量流量(kg/s)
        c: 混合流体比热容(J/(kg·K))
        i: 水力坡降(Pa/m)
        g: 重力加速度(m/s²)
        a_g: 质量含气率(%)
        D1: 焦耳逊系数
        cp: 油品定压比热容(J/(kg·K))
        PQ: 管道起点压力(MPa)
        P1: 管道终点压力(MPa)
    返回:
        T1: 管道终点温度(K)
    """
    g = 9.81
    # 计算传热系数与质量流量的比值
    a = (K * math.pi * Do) / (G * c_m)
    # 计算摩擦热修正系数
    b = (i_l * g) / (a * c_m)
    # 计算环境温度影响项
    term1 = T_env + b
    # 计算起点温度衰减项（指数衰减部分）
    term2 = (T0 - T_env - b) * math.exp(-a * pipeLen)
    # 计算焦耳-汤姆逊效应修正项
    pressure_diff = (P_1 - P_2)
    term3 = Joule_Thomson * (a_g * C0_g / Cl_g) * (pressure_diff / (a * pipeLen)) * (1 - math.exp(-a * pipeLen))
    # 综合三项计算终点温度
    T1 = term1 + term2 - term3
    return T1



def calc_dPf(Roughness,Di,
            den_fluid,Vm,Um,
            Hp_No_Slip_liquid,Hp_Slip_liquid):
    '''
    :param pipeLen: 管道长度，m
    :param Roughness: 管道粗糙度，m
    :param Di: 管道内直径，m
    :param den_fluid: 混合流体密度，kg/m3
    :param Vm: 混合流体速度，m/s
    :param Um: 混合流体粘度，Pa·s
    :param Hp_No_Slip_liquid: 无滑移持液率
    :param Hp_Slip_liquid: 滑移持液率
    '''
    NREN = den_fluid * Vm * Di / Um  # 雷诺数

    fn = calc_f(Roughness, Di, NREN)
    # fn = 0.0056 + 0.5 / math.pow(NREN, 0.32)
    # 计算摩阻系数矫正
    if (Hp_Slip_liquid == 0) or (Hp_Slip_liquid == 1):
        S = 0
    else:
        yy = Hp_No_Slip_liquid / np.square(Hp_Slip_liquid)
        if (yy > 1) and (yy < 1.2):
            S = math.log(2.2 * yy - 1.2)
        else:
            S = math.log(yy) / (
                    -0.0523 + 3.182 * math.log(yy) - 0.8725 * np.square(math.log(yy)) + 0.01853 * math.pow(
                math.log(yy), 4))
    Fm = math.exp(S) * fn
    # 计算压力损失
    dPf = - Fm * Vm * Vm * den_fluid / (2 * Di)
    return dPf



def calc_dPf_dPg(ang, Roughness, Di,
               Vs_gas, Vs_fulid,
               den_gas, den_liquid,
               sigma_lg,
               Um,
               Flow_Index):
    '''
    计算重力压降梯度和摩阻压降梯度
    :param ang: 管道倾角，弧度制
    :param pipeLen: 管道长度，m
    :param Roughness: 管道粗糙度，m
    :param Di: 管道直径，m
    :param Vs_gas: 气体表观速度，m/s
    :param Vs_fulid: 液体表观速度，m/s
    :param den_gas: 气体密度，kg/m3
    :param den_liquid: 液体密度，kg/m3
    :param sigma_lg: 气液界面张力，Pa
    :param Um: 混合液粘度，Pa·s
    :param Flow_Index: 1为下降流，-1 为上升流
    :return: 重力压降梯度 dPg 和 摩阻压降梯度 dPf，Pa/m
    '''
    Q_gas = Vs_gas * (math.pi * Di ** 2 / 4) # 气体体积流量
    Q_liquid = Vs_fulid * (math.pi * Di ** 2 / 4) # 液体体积流量
    Q_fluid = Q_gas + Q_liquid # 混合体积流量
    a_gas = Q_gas / Q_fluid # 气相体积比
    a_liquid = Q_liquid / Q_fluid # 液相体积比
    result = drift_flux_Holdup(ang, Di, Q_gas, Q_liquid, den_liquid, den_gas, sigma_lg, Flow_Index) # 计算持液率
    Hp_Slip_liquid = result["Hp_Slip_liquid"] # 滑移持液率
    Hp_No_Slip_liquid = 1-Q_gas / Q_fluid # 无滑移持液率
    Vm = Vs_gas + Vs_fulid # 混合速度
    den_fluid = den_gas * a_gas + den_liquid*a_liquid # 混合物密度
    # 计算摩阻压降
    dPf = calc_dPf(Roughness,Di,
            den_fluid,Vm,Um,
            Hp_No_Slip_liquid,Hp_Slip_liquid) # 计算摩阻压降
    den_fluid_real = den_liquid * Hp_Slip_liquid+den_gas*(1-Hp_Slip_liquid) # 实际密度
    dPg = den_fluid_real * 9.8 * math.sin(ang)*Flow_Index # 计算重力压降
    '''重力压降'''
    return dPf,dPg,Hp_Slip_liquid



def calc_flow_volume_PT(Gm,P,T,fluid):
    '''
    根据质量流量计算油气水体积流量和质量流量
    '''
    fluid_properties = fluid.calc(P, T)
    # 从计算得到的流体物性字典中提取气体、油和水的体积占比
    G_weight_R = fluid_properties["G_weight_R"]  # 温度压力P,T下：气体质量占比
    O_weight_R = fluid_properties["O_weight_R"]  # 温度压力P,T下：油的质量占比
    W_weight_R = fluid_properties["W_weight_R"]  # 温度压力P,T下：水的质量占比
    # 从计算得到的流体物性字典中提取气体、油和水的密度
    den_gas = fluid_properties["den_gas"]  # 初始温度压力下：气体密度
    den_oil = fluid_properties["den_oil"]  # 初始温度压力下：油的密度
    den_water = fluid_properties["den_water"]  # 初始温度压力下：水的密度

    G_gas = Gm*G_weight_R # 气体质量流量
    G_oil = Gm*O_weight_R # 油的质量流量
    G_water = Gm*W_weight_R # 水的质量流量

    Q_gas = G_gas/den_gas # 气体体积流量
    Q_oil = G_oil/den_oil # 油的体积流量
    Q_water = G_water/den_water # 水的体积流量
    return Q_gas,Q_oil,Q_water






def calc_PT_numerical(Q, P_start, T_start, fluid,
               ang, pipeLen, Roughness,Do,Di,
               T_env,K,
               Flow_Index,numerical_method='RK4'):
    '''
    数值差分计算压力和温度
    :param Q: 混合体积流量，m3/s
    :param P_start: 起始压力，Pa
    :param T_start: 起始温度，K
    :param fluid: 混合流体属性
    :param ang: 管道倾角，弧度制
    :param pipeLen: 管道长度，m
    :param Roughness: 管道粗糙度，m
    :param Do: 管道外径，m
    :param Di: 管道内径，m
    :param T_env: 环境温度，K
    :param K: 传热系数，W/(m²·K)
    :param Flow_Index: 1为下降流，-1 为上升流
    '''


    A = math.pi * (Di ** 2) / 4  # 管道截面积，m2
    fluid_properties = fluid.calc(P_start, T_start)
    '''流体物性计算体积占比'''
    # 从计算得到的流体物性字典中提取气体、油和水的体积占比
    G_volume_R0 = fluid_properties["G_volume_R"]  # 初始温度压力下：气体体积占比
    O_volume_R0 = fluid_properties["O_volume_R"]  # 初始温度压力下：油的体积占比
    W_volume_R0 = fluid_properties["W_volume_R"]  # 初始温度压力下：水的体积占比
    # 从计算得到的流体物性字典中提取气体、油和水的密度
    den_gas0 = fluid_properties["den_gas"]  # 初始温度压力下：气体密度
    den_oil0 = fluid_properties["den_oil"]  # 初始温度压力下：油的密度
    den_water0 = fluid_properties["den_water"]  # 初始温度压力下：水的密度
    den_liquid0 = fluid_properties["den_liquid"]  # 初始温度压力下：液体密度
    # 根据总体积流量 Q 和各相体积占比，计算气体、油和水的体积流量
    Q_gas_0 = Q * G_volume_R0  # 初始温度压力下：气体体积流量
    Q_oil_0 = Q * O_volume_R0  # 初始温度压力下：油的体积流量
    Q_water_0 = Q * W_volume_R0  # 初始温度压力下：水的体积流量
    # 根据各相体积流量和密度，计算气体、油和水的质量流量
    G_gas0 = Q_gas_0 * den_gas0  # 初始温度压力下：气体质量流量
    G_oil0 = Q_oil_0 * den_oil0  # 初始温度压力下：油的质量流量
    G_water0 = Q_water_0 * den_water0  # 初始温度压力下：水的质量流量
    Gm = G_gas0 + G_oil0 + G_water0  # 初始温度压力下：混合液的质量流量

    a_gas0 = Q_gas_0 / Q # 初始温度压力下： 气体质量比
    Vs_gas0 = Q_gas_0 / A # 初始温度压力下：气体表观速度
    Vs_liquid0 = (Q_oil_0 + Q_water_0) / A # 初始温度压力下： 液体表观速度
    # 计算气体、油和水的质量流量之和，得到总质量流量
    Gm = G_gas0 + G_oil0 + G_water0
    c0 = G_volume_R0*fluid.Gas_C0 + O_volume_R0*fluid.Oil_C0 + W_volume_R0*fluid.Water_C0 # 初始温度压力下：混合液的比热容
    cl0 = O_volume_R0/(O_volume_R0+W_volume_R0)*fluid.Oil_C0+W_volume_R0/(O_volume_R0+W_volume_R0)*fluid.Water_C0  # 初始温度压力下： 液体比热比
    Um0 = G_volume_R0*fluid_properties["gas_viscosity"]+O_volume_R0*fluid_properties["oil_viscosity"]+W_volume_R0*fluid_properties["water_viscosity"] # 初始温度压力下：混合液的粘度
    sigma_lg0 = fluid_properties["sigma_lg"] # 初始温度压力下：气液界面张力
    Joule_Thomson0 = fluid_properties["joule_thomson"]

    def calc_P(Pg, Pf, T1,
               den_gas, den_liquid, a_gas,
               Vs_gas, Vs_liquid):
        '''
        计算压力
        :param Pg: 重力压降
        :param Pf: 摩阻压降
        :param T1: 终点温度
        :param den_gas: 参考温度压力下：气体密度
        :param den_liquid:  参考温度压力下：液体密度
        :param a_gas:  参考温度压力下：气体体积比
        :param Vs_gas:  参考温度压力下：气体表观速度
        '''
        P1 = P_start + (Pg+Pf) # 猜测压力，不考虑流体的体积变化项
        P_guess = P1-10
        iteration = 0
        while abs(P1-P_guess)>1 or iteration<100:
            iteration += 1
            P_guess = P1
            Q_gas,Q_oil,Q_water = calc_flow_volume_PT(Gm, P1, T1, fluid)
            Vs_gas_P = Q_gas / A
            Vs_liquid_P = Q_oil / A+Q_water / A
            Vs_gas_delta = (Vs_gas_P-Vs_gas0)/pipeLen
            Vs_liquid_delta = (Vs_liquid_P-Vs_liquid0)/pipeLen
            Pv = den_gas*Vs_gas*a_gas*A*Vs_gas_delta+\
                 den_liquid*Vs_liquid*A*Vs_liquid_delta
            P1 = P_start + Pv + Pf + Pg
        return P1,Pv/pipeLen



    def calc_PT(P_,T_):
        '''
        计算终点压力和温度
         :param P_: 参考压力
         :param T_: 参考温度
         :return: 终点压力和温度
        '''
        fluid_properties = fluid.calc(P_, T_)
        '''流体物性计算体积占比'''
        # 从计算得到的流体物性字典中提取气体、油和水的体积占比
        G_weight_R_ = fluid_properties["G_weight_R"]  # 参考温度压力下：气体质量占比
        O_weight_R_ = fluid_properties["O_weight_R"]  # 参考温度压力下：油的质量占比
        W_weight_R_ = fluid_properties["W_weight_R"]  # 参考温度压力下：水的质量占比
        # 从计算得到的流体物性字典中提取气体、油和水的密度
        G_gas_ = Gm * G_weight_R_  # 参考温度压力下：气体质量流量
        G_oil_ = Gm * O_weight_R_  # 参考温度压力下：油的质量流量
        G_water_ = Gm * W_weight_R_  # 参考温度压力下：水的质量流量
        den_gas_ = fluid_properties["den_gas"]  # 参考温度压力下：气体密度
        den_oil_ = fluid_properties["den_oil"]  # 参考温度压力下：油的密度
        den_water_ = fluid_properties["den_water"]  # 参考温度压力下：水的密度
        Joule_Thomson = fluid_properties["joule_thomson"]  # 参考温度压力下：焦耳-汤姆逊效应

        Q_gas_ = G_gas_/den_gas_ # 参考温度压力下：气体体积流量
        Q_oil_ = G_oil_/den_oil_ # 参考温度压力下：油的体积流量
        Q_water_ = G_water_/den_water_ # 参考温度压力下：水的体积流量
        Qm_ = Q_gas_+Q_oil_+Q_water_ # 参考温度压力下：混合体积流量
        den_liquid_ = fluid_properties["den_liquid"] # 参考温度压力下：液体密度
        a_gas_ = Q_gas_/Qm_ # 参考温度压力下： 气体质量比
        G_volume_ = Q_gas_/Qm_ # 参考温度压力下：气体体积比
        O_volume_ = Q_oil_/Qm_ # 参考温度压力下：油的体积比
        W_volume_ = Q_water_/Qm_ # 参考温度压力下：水的体积比

        Vs_gas_ = Q_gas_/A # 参考温度压力下：气体表观速度
        Vs_liquid_ = (Q_oil_+Q_water_)/A # 参考温度压力下： 液体表观速度

        c0_ = G_volume_ * fluid.Gas_C0 + O_volume_ * fluid.Oil_C0 + W_volume_ * fluid.Water_C0  # 初始温度压力下：混合液的比热容
        c_l = O_volume_/(O_volume_+W_volume_)*fluid.Oil_C0 + W_volume_/(O_volume_+W_volume_)*fluid.Water_C0
        Um_ = G_volume_ * fluid_properties["gas_viscosity"] + O_volume_ * fluid_properties["oil_viscosity"] + W_volume_ * fluid_properties["water_viscosity"]  # 参考温度压力下：混合液的粘度
        sigma_lg_ = fluid_properties["sigma_lg"]  # 参考温度压力下：气液界面张力
        # 计算四阶龙格库塔K1 参考流速为初始值
        joule_thomson = fluid_properties["joule_thomson"]
        dPf_, dPg_,Hp_Slip_liquid_ = calc_dPf_dPg(ang, Roughness, Di,
                            Vs_gas_,Vs_liquid_,
                            den_gas_, den_liquid_,
                            sigma_lg_,
                            Um_,
                            Flow_Index)
        P_end_guess = P_start + (dPg_ + dPf_) * pipeLen
        i_l = (P_end_guess - P_start) / pipeLen / 9.81 / den_liquid_
        # T1 = temperature_drop(T_start, T_env, K, Do, pipeLen, Gm, c0_)
        T1 = temperature_drop_f(T_env, T_start, K, Do, pipeLen,
                       Gm, c0_, i_l,  G_weight_R_,joule_thomson,
                       fluid.Gas_C0, c_l, P_start, P_end_guess)
        Pg_ = dPg_*pipeLen
        Pf_ = dPf_*pipeLen

        P1,dPv_ = calc_P(Pg_, Pf_,T1,
                       den_gas_, den_liquid_, a_gas_,
                       Vs_gas_, Vs_liquid_)
        return P1,T1,dPf_,dPg_,dPv_,Vs_gas_,Vs_liquid_



    # 计算四阶龙格库塔K1 参考流速为初始值
    dPf_k1,dPg_k1,Hp_Slip_liquid_k1 =  calc_dPf_dPg(ang,Roughness, Di,
               Vs_gas0, Vs_liquid0,
               den_gas0, den_liquid0,
               sigma_lg0,
               Um0,
               Flow_Index)

    P_end_guess = P_start + (dPg_k1 + dPf_k1) * pipeLen
    # T1 = temperature_drop(T_start, T_env, K, Do, pipeLen, Gm, c0_)
    i_l = ( P_end_guess-P_start)/pipeLen/9.81/den_liquid0
    T1_k1 = temperature_drop_f(T_env, T_start, K, Do, pipeLen,
                            Gm, c0, i_l, G_volume_R0, Joule_Thomson0,
                            fluid.Gas_C0, cl0, P_start, P_end_guess)
    Pg_k1 = dPg_k1 * pipeLen
    Pf_k1 = dPf_k1 * pipeLen

    P1_k1,dPv_k1 = calc_P(Pg_k1,Pf_k1,T1_k1,
               den_gas0,den_liquid0,a_gas0,
               Vs_gas0,Vs_liquid0)

    Vs_gas_k1 = Vs_gas0
    Vs_liquid_k1 = Vs_liquid0

    # 计算四阶龙格库塔K2 参考流速为K1值
    P_ = (P_start + P1_k1)/2
    T_ = (T_start + T1_k1)/2
    P1_k2,T1_k2,dPf_k2,dPg_k2,dPv_k2,Vs_gas_k2,Vs_liquid_k2 = calc_PT(P_,T_)
    P_ = (P_start + P1_k2)/2
    T_ = (T_start + T1_k2)/2
    P1_k3,T1_k3,dPf_k3,dPg_k3,dPv_k3,Vs_gas_k3,Vs_liquid_k3  = calc_PT(P_,T_)
    P_ = P1_k3
    T_ = T1_k3
    P1_k4,T1_k4,dPf_k4,dPg_k4,dPv_k4,Vs_gas_k4,Vs_liquid_k4 = calc_PT(P_,T_)
    if numerical_method=='RK4':
        P1 = (P1_k1 + 2 * P1_k2 + 2 * P1_k3 + P1_k4) / 6
        T1 = (T1_k1 + 2 * T1_k2 + 2 * T1_k3 + T1_k4) / 6
    elif numerical_method=='Euler':
        P1 = P1_k1
        T1 = T1_k1
    elif numerical_method=='Implicit Euler':
        P1 = P1_k4
        T1 = T1_k4
    elif numerical_method=='Implicit Midpoint':
        P1 = P1_k3
        T1 = T1_k3


    Vs_gas = (Vs_gas_k1+ 2*Vs_gas_k2 + 2*Vs_gas_k3 + Vs_gas_k4)/6
    Vs_liquid = (Vs_liquid_k1 + 2*Vs_liquid_k2 + 2*Vs_liquid_k3 + Vs_liquid_k4)/6
    fluid_properties = fluid.calc(P1, T1)
    # 从计算得到的流体物性字典中提取气体、油和水的体积占比
    G_weight_R = fluid_properties["G_weight_R"]  # 参考温度压力下：气体质量占比
    O_weight_R = fluid_properties["O_weight_R"]  # 参考温度压力下：油的质量占比
    W_weight_R = fluid_properties["W_weight_R"]  # 参考温度压力下：水的质量占比
    G_gas = Gm * G_weight_R  # 参考温度压力下：气体质量流量
    G_oil = Gm * O_weight_R  # 参考温度压力下：油的质量流量
    G_water = Gm * W_weight_R  # 参考温度压力下：水的质量流量
    den_gas = fluid_properties["den_gas"]  # 参考温度压力下：气体密度
    den_oil = fluid_properties["den_oil"]  # 参考温度压力下：油的密度
    den_water = fluid_properties["den_water"]  # 参考温度压力下：水的密度
    Q_gas = G_gas / den_gas  # 参考温度压力下：气体体积流量
    Q_oil = G_oil / den_oil  # 参考温度压力下：油的体积流量
    Q_water = G_water / den_water  # 参考温度压力下：水的体积流量
    Qm = Q_gas + Q_oil + Q_water  # 参考温度压力下：混合体积流量
    return P1,T1,Vs_gas,Vs_liquid,Qm,Hp_Slip_liquid_k1




def print_results(P_end, T_end, NoSlip_Liquid_HoldUp, Slip_Liquid_HoldUp, Qo,Qg, Qw, Ql, Qtotal, GL, Gg, Gt, Flow_pattern_index):
    P_end = P_end/1000000
    Qo = Qo*86400
    Qg = Qg * 86400
    Qw = Qw*86400
    Ql = Ql*86400
    Qtotal = Qtotal*86400
    GL = GL*86400
    Gg = Gg*86400
    Gt = Gt*86400
    print("| 物理变量 | 数值 | 单位 |")
    print("| ---- | ---- | ---- |")
    print(f"| 结束点压力 | {P_end:.5f} | MPa |")
    print(f"| 结束点温度 | {T_end:.5f} | ℃ |")
    print(f"| 无滑脱持液率 | {NoSlip_Liquid_HoldUp * 100:.5f}% | 无 |")
    print(f"| 滑脱持液率 | {Slip_Liquid_HoldUp * 100:.5f}% | 无 |")
    print(f"| 油体积流量 | {Qo:.5f} | m^3/d |")
    print(f"| 气体积流量 | {Qg:.5f} | m^3/d |")
    print(f"| 水体积流量 | {Qw:.5f} | m^3/d |")
    print(f"| 液体总体积流量 | {Ql:.5f} | m^3/d |")
    print(f"| 总体积流量 | {Qtotal:.5f} | m^3/d |")
    print(f"| 液体质量流通量 | {GL:.5f} | kg/d |")
    print(f"| 气体质量流通量 | {Gg:.5f} | kg/d  |")
    print(f"| 总质量流通量 | {Gt:.5f} | kg/d  |")
    if Flow_pattern_index==1:
        flow_pattern="分层流"
    elif Flow_pattern_index==2:
        flow_pattern="过渡流"
    elif Flow_pattern_index==3:
        flow_pattern="间歇流"
    elif Flow_pattern_index==4:
        flow_pattern="分散流"
    print(f"| 流态指数 | {flow_pattern} | 无 |")


# ================= 新增体积流量转化质量流量函数 =================
def volume_to_mass_flow(Q_V ,  flowratetype  , P, T, fluid):
    """
    【独立接口】将体积流量转换为质量流量
    :param Q_V: 体积流量, m^3/s
    :param flowratetype: 气相或油相
    :param P: 压力, Pa
    :param T: 温度, K
    :param fluid: 流体对象 (pvt_params实例)
    :return: (G_gas, G_oil, G_water) 各相质量流量, kg/s
    """
    fluid_properties = fluid.calc(P, T)
    '''流体物性计算体积占比'''
    # 从计算得到的流体物性字典中提取气体、油和水的体积占比
    G_volume_R0 = fluid_properties["G_volume_R"]  # 初始温度压力下：气体体积占比
    O_volume_R0 = fluid_properties["O_volume_R"]  # 初始温度压力下：油的体积占比
    W_volume_R0 = fluid_properties["W_volume_R"]  # 初始温度压力下：水的体积占比
    # 从计算得到的流体物性字典中提取气体、油和水的密度
    den_gas0 = fluid_properties["den_gas"]  # 初始温度压力下：气体密度
    den_oil0 = fluid_properties["den_oil"]  # 初始温度压力下：油的密度
    den_water0 = fluid_properties["den_water"]  # 初始温度压力下：水的密度
    den_liquid0 = fluid_properties["den_liquid"]  # 初始温度压力下：液体密度
    "计算总的体积流量"
    if  flowratetype =="Gas flowrate" :
        Q = Q_V/ G_volume_R0
        Q_gas_0 = Q_V # 气体体积流量
        Q_oil_0 = Q * O_volume_R0  # 油的体积流量
        Q_water_0 = Q * W_volume_R0  # 水的体积流量
    elif flowratetype =="Liquid flowrate" :
        Q = Q_V/ (O_volume_R0+W_volume_R0)
        Q_gas_0 = Q * G_volume_R0  # 气体体积流量
        Q_oil_0 =  Q *  O_volume_R0 # 油的体积流量
        Q_water_0 = Q * W_volume_R0  # 水的体积流量
    # 根据各相体积流量和密度，计算气体、油和水的质量流量
    G_gas0 = Q_gas_0 * den_gas0  # 气体质量流量
    G_oil0 = Q_oil_0 * den_oil0  # 油的质量流量
    G_water0 = Q_water_0 * den_water0  # 水的质量流量
    Gm = G_gas0 + G_oil0 + G_water0  # 混合液的质量流量
    return G_gas0, G_oil0, G_water0,Gm


# ===================================================



if __name__ == '__main__':
    Oil_API = 32
    Water_Cut = 0.2
    Water_Specific_Gravity = 1
    GOR = 160
    Gas_Specific_Gravity = 0.75
    Oil_C0 = 1884.06
    Gas_C0 = 1884.06
    Water_C0 = 4186.8
    fluid = pvt_params(Oil_API, Water_Cut, GOR, Gas_Specific_Gravity, Water_Specific_Gravity,
                       Oil_C0, Gas_C0, Water_C0)
    # 定义表格标题和数据
    pipeLen = 20
    ang = math.pi / 2
    Ql_suf = 600 / 86400
    P_start = 20*1000000
    T_start = 60 + 273.15
    Roughness = 0.0001
    dT_env = 0.03
    T_env = T_start - pipeLen/2*dT_env
    Do = 0.101
    Di = 0.088
    Flow_Index = -1
    K = 101.25

    P2,T2,Vs_gas,Vs_liquid,Qm,Hp_Slip_liquid_k1 = calc_PT_numerical(Ql_suf, P_start, T_start, fluid,
               ang, pipeLen, Roughness,Do,Di,
               T_env,K,
               Flow_Index)
    """
    P1,            # 终点压力，Pa
    T1,            # 终点温度，K
    Vs_gas,        # 平均气相表观速度，m/s
    Vs_liquid,     # 平均液相表观速度，m/s
    Qm,            # 终点混合体积流量，m³/s
    Hp_Slip_liquid_k1  # 初始步长滑移持液率，无因次
    """

    print("开始压力：",P_start/1000000,"MPa  开始温度：",T_start-273.15,"℃")
    print("结束压力：",P2/1000000,"MPa  结束温度：",T2-273.15,"℃")

    # ================= 新增：测试 =================
    print("\n--- 测试体积转质量流量接口 ---")
    # 1. 准备测试数据 (假设在起始工况下)
    A = math.pi * (Di ** 2) / 4
    Q_gas_test = Vs_gas * A  # m^3/s
    Q_oil_test = (Vs_liquid * A) * (1 - Water_Cut)
    Q_water_test = (Vs_liquid * A) * Water_Cut



    ###  flowratetype  分为"Gas flowrate"和"Liquid flowrate"
    Q_V = Q_gas_test
    flowratetype="Gas flowrate" # 假设输入的是气相体积流量
    # 2. 调用独立接口
    G_gas, G_oil, G_water,Gm = volume_to_mass_flow( Q_V ,  flowratetype  , P_start, T_start, fluid)  #输入体积流量 是气相体积流量还是液相体积流量

    # 3. 打印结果
    print(f"\n===============输入体积流量形式：{flowratetype} ===============")
    print(f"测试工况: P={P_start / 1e6} MPa, T={T_start - 273.15} C")
    print(f"输入体积流量 (m^3/s): {Q_V:.6f}")
    print(f"输出质量流量 (kg/s): Gas={G_gas:.6f}, Oil={G_oil:.6f}, Water={G_water:.6f}")
    # =========================================================
    Q_V = Q_oil_test + Q_water_test
    flowratetype="Liquid flowrate" # 假设输入的是气相体积流量
    # 2. 调用独立接口
    G_gas, G_oil, G_water,Gm = volume_to_mass_flow( Q_V ,  flowratetype  , P_start, T_start, fluid)  #输入体积流量 是气相体积流量还是液相体积流量

    # 3. 打印结果
    print(f"\n===============输入体积流量形式：{flowratetype} ===============")
    print(f"测试工况: P={P_start / 1e6} MPa, T={T_start - 273.15} C")
    print(f"输入体积流量 (m^3/s): {Q_V:.6f}")
    print(f"输出质量流量 (kg/s): Gas={G_gas:.6f}, Oil={G_oil:.6f}, Water={G_water:.6f}")
    # =========================================================