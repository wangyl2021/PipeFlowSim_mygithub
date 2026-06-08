#### 流体混合规则文件
from Fluid.blackOil import blackoil
import numpy as np

VERBOSE = False
import math



def fluidParamMix(Multiple_fluids,fluidType)-> blackoil.pvt_params:
    """
    多流体混合后，参数计算
    多流体混合基本物性参数
        :param:  GSG_mix: 混合气体比重
        :param:  WSG_mix: 混合水比重Gas_Specific_Gravity_mix
        :param:  Gas_Specific_Gravity: 混合气体比重
        :param:  Multiple_fluids[i].Q_Liquid: 各流体液相体积流量
        :param:  Q_vgmix: 混合气相流量
        :param:  Q_vwmix: 混合水相流量
        :param:  Q_vomix: 混合油相流量
        :param:  Q_momix: 混合油相质量流量
        :param:  Q_mgmix: 混合气相质量流量
        :param:  Q_mwmix: 混合水相质量流量
        :param:  储罐条件下的  温度519.75°R； 压力14.69psia
        :param: Multiple_fluids:list[{'fluid':pvt_params,'massFlowRate':float,'temperature':float}]
    """
    if fluidType == 'black_oil':

        def Den_cal(GSG, WSG, API, WCUT, GOR):
            '''计算流体各相密度 用于体积流量与质量流量之间的转化'''
            Zg_stock_tank = blackoil.calc_Z(519.75, 14.69, GSG)  # 储罐条件下的  温度519.75°R； 压力14.69psia
            den_gas = blackoil.calc_den_gas(519.75, 14.69, GSG, Zg_stock_tank)  # 储罐条件下各流体的气体密度
            Rs, Pb =  blackoil.calc_Rs(519.75 - 459.67, 14.69, GSG, API, GOR / 0.1781076)
            Oil_Specific_Gravity = 141.5 / (API + 131.5)
            Bo =  blackoil.calc_Bo(519.75, Rs, GSG, Oil_Specific_Gravity)
            Bw =  blackoil.calc_Bw(519.75, 14.69)
            den_oil, den_water, den_liquid =  blackoil.calc_den_liquid(Oil_Specific_Gravity, Rs, GSG, Bo, WCUT, WSG,
                                                             Bw)  # 计算各流体的液相密度
            return den_gas*16.2, den_oil*16.2, den_water*16.2, den_liquid*16.2  #密度单位由lbm/ft^3转换成Kg/m³

        VolumeFlowRate_list=np.zeros(len(Multiple_fluids))#每股流体的体积流量（油+水）
        den_gasi_list = np.zeros(len(Multiple_fluids))#每股流体的气相密度
        den_oili_list = np.zeros(len(Multiple_fluids))#每股流体的油相密度
        den_wateri_list = np.zeros(len(Multiple_fluids))#每股流体的水相密度
        for i in range(len(Multiple_fluids)):
            '''计算每股流体的体积流量（质量流量转换成体积流量）'''
            GSG_i = Multiple_fluids[i]['fluid'].Gas_Specific_Gravity
            WSG_i = Multiple_fluids[i]['fluid'].Water_Specific_Gravity
            API_i = Multiple_fluids[i]['fluid'].Oil_API
            WCUT_i = Multiple_fluids[i]['fluid'].Water_Cut
            GOR_i = Multiple_fluids[i]['fluid'].GOR
            den_gas, den_oil, den_water, den_liquid = Den_cal(GSG_i, WSG_i, API_i, WCUT_i, GOR_i) #密度单位Kg/m³
            Q_Liquid_Vi= Multiple_fluids[i]['massFlowRate'] / den_liquid #质量流量单位Kg/d
            VolumeFlowRate_list[i]=Q_Liquid_Vi  #储罐条件下的体积液量单位是  m³/d
            den_gasi_list[i]=den_gas
            den_oili_list[i]=den_oil
            den_wateri_list[i]=den_water

        # 混合计算
        GSG_mix = 0  # 混合气体比重Gas_Specific_Gravity_mix
        WSG_mix = 0  # 混合水比重Gas_Specific_Gravity_mix
        Q_vgmix = 0
        Q_vwmix = 0
        Q_vomix = 0
        API_mix = 0

        """GOR_mix WCUT_mix GSG_mix WSG_mix API_mix均采用体积流量计算"""
        for i in range(len(Multiple_fluids)):
            Q_vomix = Q_vomix + VolumeFlowRate_list[i] * (1 - Multiple_fluids[i]['fluid'].Water_Cut)
            Q_vgmix = Q_vgmix + VolumeFlowRate_list[i] * (1 - Multiple_fluids[i]['fluid'].Water_Cut) * Multiple_fluids[i]['fluid'].GOR
            Q_vwmix = Q_vwmix + VolumeFlowRate_list[i]* Multiple_fluids[i]['fluid'].Water_Cut
        GOR_mix = Q_vgmix / Q_vomix  # 混合气油比GOR_mix
        WCUT_mix = Q_vwmix / (Q_vwmix + Q_vomix)  # 混合含水率WCUT_mix
        for i in range(len(Multiple_fluids)):
            API_mix = API_mix + 141.5 / (Multiple_fluids[i]['fluid'].Oil_API + 131.5) * (
                        VolumeFlowRate_list[i] * (1 - Multiple_fluids[i]['fluid'].Water_Cut))
            GSG_mix = GSG_mix + Multiple_fluids[i]['fluid'].Gas_Specific_Gravity * (
                        VolumeFlowRate_list[i] * (1 - Multiple_fluids[i]['fluid'].Water_Cut) * Multiple_fluids[i]['fluid'].GOR)
            WSG_mix = WSG_mix + Multiple_fluids[i]['fluid'].Water_Specific_Gravity * (
                        VolumeFlowRate_list[i] * Multiple_fluids[i]['fluid'].Water_Cut)
        GSG_mix = GSG_mix / Q_vgmix
        WSG_mix = WSG_mix / Q_vwmix
        API_mix = (141.5 / (API_mix / Q_vomix)) - 131.5
        Oil_C0 = 0
        Water_C0 = 0
        Gas_C0 = 0
        Q_liquidm_mix = 0
        """热性质参数均采用质量流量计算"""
        # 使用上面的混合流体参数计算混合气液的密度参数
        for i in range(len(Multiple_fluids)):
            Q_liquidm_mix = Q_liquidm_mix + Multiple_fluids[i]['massFlowRate']  # 混合液相质量流量 Kg/d
            Oil_C0 = Oil_C0 + (Multiple_fluids[i]['fluid'].Oil_C0 * (
                        Multiple_fluids[i]['massFlowRate'] * (1 - Multiple_fluids[i]['fluid'].Water_Cut)))
            Water_C0 = Water_C0 + (Multiple_fluids[i]['fluid'].Water_C0 * (
                        Multiple_fluids[i]['massFlowRate'] * Multiple_fluids[i]['fluid'].Water_Cut))
            Gas_C0 = Gas_C0 + (Multiple_fluids[i]['fluid'].Gas_C0 * (
                        Multiple_fluids[i]['massFlowRate'] * (1 - Multiple_fluids[i]['fluid'].Water_Cut) *
                        Multiple_fluids[i]['fluid'].GOR))

        Q_momix = Q_liquidm_mix * (1 - WCUT_mix)  # 混合油质量流量 Kg/d
        Q_mgmix = Q_liquidm_mix * (1 - WCUT_mix) * GOR_mix  # 混合气相质量流量
        Q_mwmix = Q_liquidm_mix * WCUT_mix  # 混合水质量流量

        Oil_C0_mix = Oil_C0 / Q_momix
        Water_C0_mix = Water_C0 / Q_mwmix
        Gas_C0_mix = Gas_C0 / Q_mgmix

        newFluid = blackoil.pvt_params(API_mix, WCUT_mix, GOR_mix, GSG_mix, WSG_mix,
                                    Oil_C0_mix, Gas_C0_mix, Water_C0_mix)

        return newFluid
    else:
        pass


def fluidTemperatureMix(Multiple_fluids,fluidType,newFluid):
    '''
    多流体混合后，温度计算
    @param: Multiple_fluids:list[{'fluid':pvt_params,'massFlowRate':float,'temperature':float}]
    '''
    if fluidType == 'black_oil':

        """计算混合温度"""
        # 计算单股流体的各相密度，用于计算质量流量
        CO_i = np.zeros(len(Multiple_fluids))
        Q_m_i = np.zeros(len(Multiple_fluids))
        for i in range(len(Multiple_fluids)):
            '''计算每股流体的各相质量流量，从而计算每股流体的平均比热容 '''
            Q_moi = Multiple_fluids[i]['massFlowRate'] * (1 - Multiple_fluids[i]['fluid'].Water_Cut)
            Q_mwi = Multiple_fluids[i]['massFlowRate'] * Multiple_fluids[i]['fluid'].Water_Cut
            Q_mgi =  Multiple_fluids[i]['massFlowRate'] * (1 - Multiple_fluids[i]['fluid'].Water_Cut) * Multiple_fluids[i]['fluid'].GOR
            Q_mi = Q_moi + Q_mwi + Q_mgi #每股流体的总质量流量
            COi = (Q_moi * Multiple_fluids[i]['fluid'].Oil_C0 + Q_mwi * Multiple_fluids[i]['fluid'].Water_C0 + Q_mgi * Multiple_fluids[i]['fluid'].Gas_C0) / Q_mi#每股流体的平均比热容
            CO_i[i] = COi
            Q_m_i[i] = Q_mi
        numerator = 0
        denominator = 0
        for i in range(len(Multiple_fluids)):
            numerator = numerator + Q_m_i[i] * Multiple_fluids[i]['temperature'] * CO_i[i]
            denominator = denominator + Q_m_i[i] * CO_i[i]
        Temperature_mix = numerator / denominator

        Q_liquidm_mix =0
        for i in range(len(Multiple_fluids)):
            Q_liquidm_mix = Q_liquidm_mix + Multiple_fluids[i]['massFlowRate']  # 混合液相质量流量 Kg/d

        dicFluidParam = {'fluid': newFluid, 'massFlowRate': Q_liquidm_mix,
                         'temperature': Temperature_mix}  # 质量流量转换成Kg/d

        return Temperature_mix
    else:
        pass


def fluidMix(Multiple_fluids,fluidType):
    """
        :param: Multiple_fluids:list[{'fluid':pvt_params,'massFlowRate':float,'temperature':float}]
    """
    newFluid = fluidParamMix(Multiple_fluids,fluidType)
    mixTempe = fluidTemperatureMix(Multiple_fluids,fluidType,newFluid)
    if VERBOSE:
        print(f'multiple fluids mixed! temperature of new fluid is : {mixTempe}')
    return newFluid,mixTempe


if __name__ == '__main__':
    '''黑油模型多流体参数文档说明
            Oil_API： 原油 API 度
            Water_Cut： 体积含水率 %
            Water_Specific_Gravity： 水的比重
            GOR	Gas_Specific_Gravity： 气体比重
            Oil_C0： 油的比热容J/(kg.k)
            Gas_C0： 气的比热容J/(kg.k)
            Water_C0： 水的比热容J/(kg.k)
            Q_Liquid： 储罐条件下的质量液量单位是  Kg/s
            Temperature: 温度 ℃
            '''
    lstFluids= list()
    for i in range(3):
        if i==0:
            Oil_API = 31.2
            Water_Cut = 0.72
            Water_Specific_Gravity = 1.01
            GOR = 60.23
            Gas_Specific_Gravity = 0.82
            Oil_C0 = 1884.06
            Gas_C0 = 2302.74
            Water_C0 = 4186.8
            massFlowRate=0.9
            temperature=60+273.15
        if i==1:
            Oil_API = 36
            Water_Cut = 0.6
            Water_Specific_Gravity = 1
            GOR = 355
            Gas_Specific_Gravity = 0.8
            Oil_C0 = 1884.06
            Gas_C0 = 2302.74
            Water_C0 = 4186.8
            massFlowRate=1.01
            temperature=80+273.15
        if i==2:
            Oil_API = 46
            Water_Cut = 0.2
            Water_Specific_Gravity = 1
            GOR = 420
            Gas_Specific_Gravity = 0.66
            Oil_C0 = 1884.06
            Gas_C0 = 2302.74
            Water_C0 = 4186.8
            massFlowRate=0.8
            temperature=40+273.15

        fluid = blackoil.pvt_params(Oil_API, Water_Cut, GOR, Gas_Specific_Gravity, Water_Specific_Gravity,
                           Oil_C0, Gas_C0, Water_C0)

        # massFlowRate=i+100
        # temperature=i+200

        dicFluidParam={'fluid':fluid,'massFlowRate':massFlowRate*86400,'temperature':temperature} #质量流量转换成Kg/d

        lstFluids.append(dicFluidParam)

    fluidMix(lstFluids,fluidType='black_oil')
