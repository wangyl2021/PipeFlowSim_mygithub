from Fluid.blackOil.blackoil import pvt_params


def calc_mass_flow(fulid,Q,P,T):
    '''
    计算温度T和P下已知流体fulid下的质量流量
    '''
    fluid_properties = fulid.calc(P, T)
    den_gas = fluid_properties["den_gas"]
    den_liquid = fluid_properties["den_liquid"]
    den_oil = fluid_properties["den_oil"]
    GLR = fluid_properties["GLR_PT"]
    OWR = fluid_properties["OWR_PT"]
    Q_gas = Q * GLR/(GLR+1)
    Q_liquid = Q /(GLR+1)
    Q_oil = Q_liquid*OWR/(OWR+1)
    Q_water = Q_liquid/(OWR+1)
    Gg = Q_gas*den_gas
    Gl = Q_liquid*den_liquid
    Go = Q_oil*den_oil
    Gw = Q_water*den_oil
    Gt = Go+Gg+Gw
    return Gt

if __name__ == "__main__":
    Oil_API = 30
    Water_Cut = 0.5
    Water_Specific_Gravity = 1
    GOR = 345
    Gas_Specific_Gravity = 0.75
    Oil_C0 = 1884.06
    Gas_C0 = 1884.06
    Water_C0 = 4186.8
    fluid = pvt_params(Oil_API, Water_Cut,
                       GOR, Gas_Specific_Gravity, Water_Specific_Gravity,
                       Oil_C0, Gas_C0, Water_C0)

    P = 6 * 1000000
    # Pa
    T = 40
    # ℃
    fluid_properties = fluid.calc(P, T)
    Q = 200/86400
    # m3/s
    Gt = calc_mass_flow(fluid, Q, P, T)
    print(Gt)


