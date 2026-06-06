
import pandas as pd
import componentsDef.basicClassDef



if __name__ == '__main__':
    fluidMixInfo= pd.read_excel('./流体组分.xlsx', sheet_name='1')
    fluid = componentsDef.basicClassDef.fluid2P(fluidMixInfo)
    P =  3000000
    T =  300  #354.5761135008132#273.15+60
    # point1 = fluid.calc_critical_point()
    # point = fluid.calc_phase_envelope()
    T_Dew = fluid.calc_Dew_T(P)
    print("压力：",P/1000000,"MPa","           露点温度：",round(T_Dew,5),"K")
    P_bubble = fluid.calc_Bubble_P(T)
    print("温度：",T,"K","           泡点压力：",round(P_bubble/1000000,3),"MPa")
    #


    fluid.phase_equilibrium_calc(P, T)
    # 假设 fluid 对象有相应属性存储相态、压缩因子和摩尔占比

    phase = fluid.phase  # 相态信息
    Z_G = fluid.Z_G  # 气相压缩因子
    Z_O = fluid.Z_O  # 油相压缩因子
    Z_W = fluid.Z_W  # 水相压缩因子

    zis_G = fluid.xis_G  # 气相各组分摩尔占比
    zis_L = fluid.xis_L  # 液相各组分摩尔占比
    zis_O = fluid.xis_O  # 油相各组分摩尔占比

    print("压力：",P,"Pa","           温度：",T,"K")

    print("                  气                   油                    水                ")
    print("摩尔占比        ",f"{fluid.a_G:.4f}               {fluid.a_O:.4f}               {fluid.a_W:.4f}")
    print("压缩因子        ",f"{Z_G.real:.4f}               {Z_O.real:.4f}               {Z_W.real:.4f}")
    print("相态:-------------------------", phase,"--------------------------")

    print(f"{'组分名称':<11}{'气相摩尔占比':<19}{'油相摩尔占比':<18}{'水相摩尔占比':<15}")
    for i in range(len(fluid.components)):
        name = fluid.components[i].name
        gas_ratio = zis_G[i]
        oil_ratio = zis_O[i]
        water_ratio = fluid.xis_W[i]
        print(f"{name:<15}{gas_ratio:.5f}{'':<15}{oil_ratio:.5f}{'':<15}{water_ratio:.5f}")
