import math
def Beggs_Brill_Holdup(ang, d, Qg, Ql, rho_l, rho_g, sigma_gl, Flow_Index):
    Ap = math.pi * (d ** 2) / 4.0
    Vsl = Ql / Ap
    ''' 液体表观速度'''
    Vsg = Qg / Ap
    '''气体表观速度'''
    Vm = Vsl + Vsg
    '''混合物表观速度'''
    NoSlip_Liquid_HoldUp = Vsl / (Vsg + Vsl)
    ''' 无滑脱携液率 '''
    Froude_number = Vm * Vm / (9.8 * d)
    '''弗雷德数'''
    NLv = Vsl * math.pow(rho_l / abs(sigma_gl) / 9.8, 0.25)
    '''液相折算速度准数'''
    L1 = 316 * (NoSlip_Liquid_HoldUp ** 0.302)
    L2 = 0.0009252 * (NoSlip_Liquid_HoldUp ** -2.4684)
    L3 = 0.1 * (NoSlip_Liquid_HoldUp ** -1.4516)
    L4 = 0.5 * (NoSlip_Liquid_HoldUp ** -6.738)
    A = (L3 - Froude_number) / (L3 - L2)

    # 确定流型和倾斜管道持液率
    Flow_pattern_index = None
    Liquid_HoldUp0 = None
    C = None
    '''
    # 流型索引:Flow_pattern_index
    # 1.    分层流（SEGREGATED）
    # 2.    过渡流（TRANSITION）
    # 3.    间歇流（INTERMITTENT）
    # 4.    分散流（DISTRIBUTED）
    '''
    if (NoSlip_Liquid_HoldUp < 0.01 and Froude_number < L1) or ( NoSlip_Liquid_HoldUp >= 0.01 and Froude_number < L2):
        "分层流（SEGREGATED）"
        Flow_pattern_index = 1
        Liquid_HoldUp0 = 0.98 * (NoSlip_Liquid_HoldUp ** 0.4846) / (Froude_number ** 0.0868)
        if Flow_Index == 1:
            C = (1 - NoSlip_Liquid_HoldUp) * math.log(
                0.011 * (NLv ** 3.539) * (NoSlip_Liquid_HoldUp ** -3.768) * (Froude_number ** -1.614))
        elif Flow_Index == 2:
            C = (1 - NoSlip_Liquid_HoldUp) * math.log(
                4.7 * (NLv ** 0.1244) * (NoSlip_Liquid_HoldUp ** -0.3692) * (Froude_number ** -0.5056))
    elif NoSlip_Liquid_HoldUp >= 0.01 and Froude_number >= L2 and Froude_number <= L3:
        "过渡流（TRANSITION）"
        Flow_pattern_index = 2
    elif (NoSlip_Liquid_HoldUp >= 0.01 and NoSlip_Liquid_HoldUp < 0.4 and Froude_number > L3 and Froude_number <= L1) or(NoSlip_Liquid_HoldUp >= 0.4 and Froude_number > L3 and Froude_number <= L4):
        "间歇流（INTERMITTENT）"
        Flow_pattern_index = 3
        Liquid_HoldUp0 = 0.845 * (NoSlip_Liquid_HoldUp ** 0.5351) / (Froude_number ** 0.0173)
        if Flow_Index == 1:
            C = (1 - NoSlip_Liquid_HoldUp) * math.log(
                2.96 * (NLv ** -0.4473) * (NoSlip_Liquid_HoldUp ** 0.305) * (Froude_number ** 0.0978))
        elif Flow_Index == 2:
            C = (1 - NoSlip_Liquid_HoldUp) * math.log(
                4.7 * (NLv ** 0.1244) * (NoSlip_Liquid_HoldUp ** -0.3692) * (Froude_number ** -0.5056))
    elif (NoSlip_Liquid_HoldUp < 0.4 and Froude_number >= L1) or (NoSlip_Liquid_HoldUp >= 0.4 and Froude_number > L4):
        Flow_pattern_index = 4
        Liquid_HoldUp0 = 1.065 * (NoSlip_Liquid_HoldUp ** 0.5824) / (Froude_number ** 0.0609)
        if Flow_Index == 1:
            C = 0
        elif Flow_Index == 2:
            C = (1 - NoSlip_Liquid_HoldUp) * math.log(
                4.7 * (NLv ** 0.1244) * (NoSlip_Liquid_HoldUp ** -0.3692) * (Froude_number ** -0.5056))
    Slip_Liquid_HoldUp = None
    if ang==90:
        inclination_correction_factor = 1+0.3*C
        Slip_Liquid_HoldUp = inclination_correction_factor * Liquid_HoldUp0
    else:
    # 计算倾斜管道持液率

        if Flow_pattern_index in [1, 3, 4]:
            inclination_correction_factor = 1 + C * (math.sin(1.8 * (ang * math.pi / 180)) - 0.333 * (
                    math.sin(1.8 * (ang * math.pi / 180)) ** 3))
            Slip_Liquid_HoldUp = inclination_correction_factor * Liquid_HoldUp0
        elif Flow_pattern_index == 2:
            if Flow_Index == 1:
                C_Segregated = (1 - NoSlip_Liquid_HoldUp) * math.log(
                    0.011 * (NLv ** 3.539) * (NoSlip_Liquid_HoldUp ** -3.768) * (Froude_number ** -1.614))
            elif Flow_Index == 2:
                C_Segregated = (1 - NoSlip_Liquid_HoldUp) * math.log(
                    4.7 * (NLv ** 0.1244) * (NoSlip_Liquid_HoldUp ** -0.3692) * (Froude_number ** -0.5056))
            inclination_correction_factor_Segregated = 1 + C_Segregated * (
                    math.sin(1.8 * (ang * math.pi / 180)) - 0.333 * (
                    math.sin(1.8 * (ang * math.pi / 180)) ** 3))
            Liquid_HoldUp0_Segregated = 0.98 * (NoSlip_Liquid_HoldUp ** 0.4846) / (Froude_number ** 0.0868)
            HoldUp_Segregated = inclination_correction_factor_Segregated * Liquid_HoldUp0_Segregated

            if Flow_Index == 1:
                C_Intermittent = (1 - NoSlip_Liquid_HoldUp) * math.log(
                    2.96 * (NLv ** -0.4473) * (NoSlip_Liquid_HoldUp ** 0.305) * (Froude_number ** 0.0978))
            elif Flow_Index == 2:
                C_Intermittent = (1 - NoSlip_Liquid_HoldUp) * math.log(
                    4.7 * (NLv ** 0.1244) * (NoSlip_Liquid_HoldUp ** -0.3692) * (Froude_number ** -0.5056))
            inclination_correction_factor_Intermittent = 1 + C_Intermittent * (
                    math.sin(1.8 * (ang * math.pi / 180)) - 0.333 * (
                    math.sin(1.8 * (ang * math.pi / 180)) ** 3))
            Liquid_HoldUp0_Intermittent = 0.845 * (NoSlip_Liquid_HoldUp ** 0.5351) / (Froude_number ** 0.0173)
            HoldUp_Intermittent = inclination_correction_factor_Intermittent * Liquid_HoldUp0_Intermittent
            Slip_Liquid_HoldUp = A * HoldUp_Segregated + (1 - A) * HoldUp_Intermittent
    # 限制持液率范围
    if Slip_Liquid_HoldUp >= 1:
        Slip_Liquid_HoldUp = 1
    elif Slip_Liquid_HoldUp <= 0:
        Slip_Liquid_HoldUp = 0
    return 1-Slip_Liquid_HoldUp,Slip_Liquid_HoldUp,1-NoSlip_Liquid_HoldUp,NoSlip_Liquid_HoldUp


def drift_flux_Holdup(ang, d, Qg, Ql, rho_l, rho_g, sigma_gl,Flow_Index):
    """
    使用计算气液两相流的持液率和持气率。
    参数:
        ang (float): 井斜角（弧度）
        d (float): 管道内径（米）
        q_g (float): 气体体积流量（立方米/秒）
        q_l (float): 液体体积流量（立方米/秒）
        rho_l (float): 液体密度（千克/立方米）
        rho_g (float): 气体密度（千克/立方米）
        sigma_gl (float): 气液界面张力（牛/米）
    返回:
        H_g (float): 持气率（-）
        H_l (float): 持液率（-）
        lambda_l (float): 无滑脱条件下的液相体积分数（-）
        lambda_g (float): 无滑脱条件下的气相体积分数（-）
    异常:
        ValueError: 如果无量纲管道直径N_d不在[2, 70]范围内
    """
    Qg = Qg * Flow_Index
    Ql = Ql * Flow_Index
    # 计算管道截面积
    ang = math.radians(ang)
    A = math.pi * (d ** 2) / 4.0

    # 计算表观速度
    v_sg = Qg / A  # 气体表观速度（米/秒）
    v_sl = Ql / A  # 液体表观速度（米/秒）
    v_ms = v_sg + v_sl  # 混合表观速度（米/秒）
    # 计算无量纲管道直径N_d
    g = 9.81  # 重力加速度（米/秒²）
    delta_rho = rho_l - rho_g
    try:
        N_d = d * math.sqrt(abs(g * delta_rho/sigma_gl))
    except :
        raise ValueError("sigma_gl不能为零")

    # 检查N_d范围
    if not (2 <= N_d <= 70):
        raise ValueError(f"无量纲管道直径N_d={N_d:.2f}不在有效范围[2, 70]内")
    else:
        K_u = (1.0152e-5 * N_d ** 3 - 2.3396e-3 * N_d ** 2 + 8.085e-1 * N_d - 1.5934) / (1.9551e-1 * N_d + 1)

    # 计算临界速度V_c和气体泛点速度V_sgf
    V_c = -((sigma_gl * g * (rho_l - rho_g) / (rho_l ** 2)) ** (1 / 4)) # 临界速度（米/秒）
    V_sgf = K_u * math.sqrt(rho_l / rho_g) * V_c  # 气体泛点速度（米/秒）
    # 计算无滑脱体积分数
    sum_q = Qg + Ql
    if sum_q == 0:
        raise ValueError("气体和液体流量之和不能为零")
    lambda_g = Qg / sum_q  # 无滑脱持气率
    lambda_l = 1.0 - lambda_g  # 无滑脱持液率
    # 根据管道直径选择模型参数
    if d < 0.1:
        # 小管径参数
        c_o_bub = 1.2
        # beta_av = 0.6
        a1 = 0.06
        a2 = 0.12
        m0 = 1.27
        n1 = 0.24
        n2 = 1.08
    else:
        # 大管径参数
        c_o_bub = 1.0
        # beta_av = 1.0
        a1 = 0.06
        a2 = 0.21
        m0 = 1.85
        n1 = 0.21
        n2 = 0.95
    # 计算倾斜修正系数m_alpha
    m_alpha = m0 * (math.cos(ang) ** n1) * ((1 + math.sin(ang)) ** n2)
    # 迭代参数初始化
    H_g_0 = lambda_g  # 初始猜测值
    f_0 = 0.5  # 阻尼因子
    tolerance = 1e-4
    H_g = H_g_0  # 初始化返回值
    beta_av = lambda_g
    # Picard迭代求解持气率
    for _ in range(1000):
        # 计算beta和gamma
        beta = max(H_g_0, H_g_0 * (v_ms / V_sgf))
        gamma = (beta - beta_av) / (1 - beta_av)
        gamma = max(0.0, min(gamma, 1.0))  # 限制在[0,1]之间
        # 计算分布系数C_0
        C_0 = c_o_bub / (1 + (c_o_bub - 1) * (gamma ** 2))#????
        # 确定K值（插值处理）
        if H_g_0 < a1:
            K = 1.53 / C_0
        elif H_g_0 > a2:
            K = K_u
        else:
            # 线性插值
            K_low = 1.53 / C_0
            K_high = K_u
            K = K_low + (H_g_0 - a1) * (K_high - K_low) / (a2 - a1)
        # 计算漂移速度v_d
        numerator_vd = m_alpha * (1 - H_g_0 * C_0) * C_0 * K * V_c
        denominator_vd = H_g_0 * C_0 * math.sqrt(rho_g / rho_l) + (1 - H_g_0 * C_0)
        v_d = numerator_vd / denominator_vd
        # 更新持气率估计值
        H_g_new = (1 - f_0) * H_g_0 + f_0 * (v_sg / (v_d + C_0 * v_ms))
        # 检查收敛性
        error = abs(H_g_new - H_g_0)
        if error <= tolerance:
            H_g = H_g_new
            break
        H_g_0 = H_g_new
    else:  # 循环正常结束（未触发break）
        print("警告：未在1000次迭代内收敛")

    # 计算持液率并验证结果
    H_l = 1.0 - H_g

    # 处理向下流动的特殊情况
    if Qg > 0 and H_l > lambda_l:
        # print("警告：向下流动且持液率超过无滑脱值，强制调整")
        H_l = lambda_l
        H_g = 1.0 - H_l

    result = {"Hp_Slip_gas":H_g,"Hp_Slip_liquid":H_l}
    return result

# 调用示例
if __name__ == "__main__":
    import math

    alpha = 90 # 60度转弧度
    d = 0.088  # 管道内径（米）
    q_g = -50 / 86400  # 气体体积流量（立方米/秒）
    q_l = -100 / 86400  # 液体体积流量（立方米/秒）
    rho_l = 762  # 液体密度（千克/立方米）
    rho_g = 69  # 气体密度（千克/立方米）
    sigma_gl = 0.06  # 气液界面张力（牛/米
    flowIndex = 1
    # 调用函数
    result  = drift_flux_Holdup(alpha, d, q_g, q_l, rho_l, rho_g, sigma_gl,flowIndex)

    # 打印结果
    print("漂移流模型计算结果：",result)
    # print(f"滑脱持气率 H_g = {H_g:.4f}",f"无滑脱气相分数 lambda_g = {lambda_g:.4f}")
    # print(f"滑脱持液率 H_l = {H_l:.4f}",f"无滑脱液相分数 lambda_l = {lambda_l:.4f}")
    # print("Beggs_Brill关联式计算结果：")
    # H_g, H_l, lambda_g, lambda_l = Beggs_Brill_Holdup(alpha, d, abs(q_g), abs(q_l), rho_l, rho_g, sigma_gl,flowIndex)
    # print(f"滑脱持气率 H_g = {H_g:.4f}", f"无滑脱气相分数 lambda_g = {lambda_g:.4f}")
    # print(f"滑脱持液率 H_l = {H_l:.4f}", f"无滑脱液相分数 lambda_l = {lambda_l:.4f}")



