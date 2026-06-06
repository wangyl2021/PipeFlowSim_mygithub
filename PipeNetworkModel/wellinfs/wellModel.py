import math

import numpy as np
import pandas as pd
from WellTrack.WellTrackInterpolation import rawWellTrackCal, newPoint_interpolation
from wellinfs.base_class import grid
import matplotlib.pyplot as plt
# 设置字体为支持中文的字体，例如微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


def read_data(Exlstr):
    '''
    :param Exlstr: 井名
    :return: productionPipe:生产管柱, LayerInfo：层段信息,ESPbaseINfo：泵基础信息,ESPFeature,WellTrack
    '''
    excel =r'E:\myCode\OilWellSimulation\WellDB\C02.xlsx'
    # 读取数据
    productionPipe = pd.read_excel(excel, sheet_name='采油管柱')
    LayerInfo = pd.read_excel(excel, sheet_name='地层信息')
    ESPbaseINfo = pd.read_excel(excel, sheet_name='电潜泵基础信息')
    ESPFeature = pd.read_excel(excel, sheet_name='电潜泵特性')
    WellTrack = pd.read_excel(excel, sheet_name='井眼轨迹')
    # 将井斜列转换为弧度制
    Casing = pd.read_excel(excel, sheet_name='套管信息')
    # realdata = pd.read_excel(excel, sheet_name='真实数据')

    return productionPipe,Casing,LayerInfo,ESPbaseINfo,ESPFeature,WellTrack
def wellTrackInterpolation(welltracks, depth):
    '''
    :param welltracks: 井眼轨迹
    :param depth: 目标深度
    :return: 插值后的目标深度、角度、方位角和垂深
    '''
    # 假设 welltracks 中有一个 '斜深' 列表示深度
    depths = welltracks['井深']
    n = None
    # 找到 depth 所在的区间
    for i in range(1, len(depths)):
        if depths[i - 1] <= depth <= depths[i]:
            n = i
            break

    # 检查是否找到合适的区间
    if n is None:
        if depth < depths[0]:
            # 如果 depth 小于最小深度，取前两个点
            n = 1
        elif depth > depths.iloc[-1]:
            # 如果 depth 大于最大深度，取最后两个点
            n = len(depths)
        else:
            raise ValueError("目标深度不在井眼轨迹数据范围内。")

    point1 = welltracks.loc[n - 1].values.tolist()
    point2 = welltracks.loc[n].values.tolist()

    # 处理 depth 与 point1 或 point2 井深相等的情况
    if depth == depths[n - 1]:
        target_depth = depth
        target_Ang = point1[1]  # 假设角度在第二列
        target_azimuth = point1[2]  # 假设方位角在第三列
        target_vDepth = point1[3]  # 假设垂深在第四列
    elif depth == depths[n]:
        target_depth = depth
        target_Ang = point2[1]  # 假设角度在第二列
        target_azimuth = point2[2]  # 假设方位角在第三列
        target_vDepth = point2[3]  # 假设垂深在第四列
    else:
        target_depth, target_Ang, target_azimuth, target_vDepth = newPoint_interpolation(depth, point1, point2)

    return target_depth, target_Ang, target_azimuth, target_vDepth


class well_info:
    def __init__(self,Exlstr):
        '''
        :param Exlstr: 井名
        :return: productionPipe:生产管柱, LayerInfo：层段信息,ESPbaseINfo：泵基础信息,ESPFeature,WellTrack
        '''
        ProductionPipe,Casing,LayerInfo, ESPbaseINfo, ESPFeature, WellTrack = read_data(Exlstr)
        self.ProductionPipe = ProductionPipe
        '''管柱数据'''
        self.LayerInfo = LayerInfo
        '''层段数据'''
        self.ESPbaseINfo = ESPbaseINfo
        '''泵基础数据'''
        self.ESPFeature = ESPFeature
        ''' 泵特性数据 '''
        welltrack = rawWellTrackCal(WellTrack)
        #补全垂深
        self.WellTrack = welltrack  # 井眼轨迹
        '''井眼轨迹'''
        self.CasingInfo = Casing
        self.grid_Len = 0





def calc_thermal_resistance(r_out, r_in, lambda_i, L):
    """
    该函数用于计算热阻，根据圆柱状物体的内外半径、热传导系数和长度进行计算。

    参数:
    r_out (float): 物体的外半径。
    r_in (float): 物体的内半径。
    lambda_i (float): 物体材料的热传导系数。
    L (float): 物体的长度。

    返回:
    float: 计算得到的热阻。
    """
    # 计算外半径与内半径比值的自然对数，这是热阻公式的分子部分
    outer_inner_radius_ratio_ln = math.log(r_out / r_in)
    # 计算热阻公式的分母部分
    thermal_resistance_denominator = 2 * math.pi * lambda_i * L
    # 计算最终的热阻
    thermal_resistance = outer_inner_radius_ratio_ln / thermal_resistance_denominator
    return thermal_resistance


class well:
    def __init__(self,wellname):
        self.grids = []
        self.wellinfo = well_info(wellname)
    def CalgridsThermalResistance(self,dth_aver,pipeNum,Len):
        '''
        计算管道网格的热阻
        :return:
        '''
        Rt = []
        r_out = self.wellinfo.ProductionPipe['外径'][pipeNum]
        r_in = self.wellinfo.ProductionPipe['内径'][pipeNum]
        lambda_i = self.wellinfo.ProductionPipe['热导率'][pipeNum]
        Rt.append(calc_thermal_resistance(r_out, r_in, lambda_i, Len))
        '''油管自身热阻'''

        CasingNum = len(self.wellinfo.CasingInfo) - 1
        while CasingNum > 0:
            if dth_aver >= self.wellinfo.CasingInfo['套管顶深'][CasingNum] and\
                    dth_aver <=self.wellinfo.CasingInfo['套管底深'][CasingNum]:
                r_out = self.wellinfo.CasingInfo['内径'][CasingNum]
                break
            CasingNum = CasingNum - 1
        # 找到微元段对应环空的内径
        r_in = self.wellinfo.ProductionPipe['外径'][pipeNum]
        lambda_i = self.wellinfo.ProductionPipe['环空介质热导率'][pipeNum]
        Rt.append(calc_thermal_resistance(r_out, r_in, lambda_i, Len))
        '''油管外部环空热阻'''
        CasingNum = len(self.wellinfo.CasingInfo)-1
        while CasingNum>=0:
            if dth_aver>=self.wellinfo.CasingInfo['套管顶深'][CasingNum] and dth_aver<=self.wellinfo.CasingInfo['套管底深'][CasingNum] and self.wellinfo.CasingInfo['套管类型'][CasingNum]!='裸眼':
                r_out = self.wellinfo.CasingInfo['外径'][CasingNum]
                r_in = self.wellinfo.ProductionPipe['内径'][CasingNum]
                lambda_i = self.wellinfo.ProductionPipe['热导率'][CasingNum]
                Rt.append(calc_thermal_resistance(r_out, r_in, lambda_i, Len))
                '''套管自身热阻'''
                r_out = self.wellinfo.CasingInfo['井眼直径'][CasingNum]
                r_in = self.wellinfo.ProductionPipe['外径'][CasingNum]
                lambda_i = self.wellinfo.ProductionPipe['环空介质热导率'][CasingNum]
                Rt.append(calc_thermal_resistance(r_out, r_in, lambda_i, Len))
                '''套管外部介质热阻'''
            CasingNum = CasingNum - 1
        Rt_sum = sum(Rt)
        # print(dth_aver,Rt)
        return Rt_sum


    def setgrids(self,gridLenth):
        # PipesCount = len(self.ProductioPpipe) - 1
        count = self.wellinfo.ProductionPipe.shape[0]
        # n = self.wellinfo.WellTrack.shape[0]-1
        # 井轨迹总点数
        for i in range(count):
            pipe_num = count - i-1
            dth_init = self.wellinfo.ProductionPipe["斜深"][pipe_num]
            ''' 起始井深 '''
            dth_end = self.wellinfo.ProductionPipe["斜深"][pipe_num] - self.wellinfo.ProductionPipe["长度"][pipe_num]
            ''' 终止井深 '''
            if self.wellinfo.ProductionPipe['管柱类型'][pipe_num] == 'ICV' or self.wellinfo.ProductionPipe['管柱类型'][pipe_num] == 'ESP':
                dth_aver = (dth_init + dth_end) / 2
                Len = abs(dth_init - dth_end)
                init_depth, init_Ang, init_azimuth,init_vDepth=wellTrackInterpolation(self.wellinfo.WellTrack,dth_init)
                end_depth, end_Ang, end_azimuth,end_vDepth=wellTrackInterpolation(self.wellinfo.WellTrack,dth_end)
                Ang =(init_Ang+end_Ang)/2
                Dia = float(self.wellinfo.ProductionPipe['内径'][pipe_num])
                Rt = self.CalgridsThermalResistance(dth_aver,pipe_num,Len)
                newgrid = grid(gridType=self.wellinfo.ProductionPipe['管柱类型'][pipe_num], initDepth=dth_init, endDepth=dth_end,
                               initVDepth=init_vDepth, endVDepth=end_vDepth, Len=Len, Ang=Ang, Dia=Dia,K=Rt)
                self.grids.append(newgrid)
                # print(Rt)
            else:
                n = math.ceil((dth_init - dth_end) / gridLenth)
                nLen = round((dth_init-dth_end)/n,2)
                self.grid_Len = nLen
                for j in range(n):
                    if j==n-1:
                        init_depth = dth_init - j * nLen
                        end_depth = dth_end
                        Len = nLen
                        init_depth, init_Ang, init_azimuth, init_vDepth = wellTrackInterpolation(
                            self.wellinfo.WellTrack, init_depth)
                        end_depth, end_Ang, end_azimuth, end_vDepth = wellTrackInterpolation(self.wellinfo.WellTrack,
                                                                                             end_depth)
                        dth_aver = (init_depth + end_depth) / 2
                        # Rt = self.CalgridsThermalResistance(dth_aver, pipe_num, Len)
                        Rt = 48
                        Tenv = 31.2 + 273.15 + (init_vDepth + end_vDepth) / 2 * 0.031
                        Ang = (init_Ang + end_Ang) / 2
                        Dia = float(self.wellinfo.ProductionPipe['内径'][pipe_num])
                        Wall_thickness = float(self.wellinfo.ProductionPipe['壁厚'][pipe_num])
                        newgrid = grid(gridType='pipe', initDepth=init_depth, endDepth=end_depth,
                                       initVDepth=init_vDepth, endVDepth=end_vDepth, Len=Len, Ang=Ang,Dia=Dia,Wallthickness=Wall_thickness,K=Rt,T_env=Tenv)
                        self.grids.append(newgrid)
                    else:
                        init_depth = dth_init-j*nLen
                        end_depth = dth_init-(j+1)*nLen
                        Len = nLen
                        init_depth, init_Ang, init_azimuth, init_vDepth = wellTrackInterpolation(
                            self.wellinfo.WellTrack, init_depth)
                        end_depth, end_Ang, end_azimuth, end_vDepth = wellTrackInterpolation(self.wellinfo.WellTrack,
                                                                     end_depth)
                        Ang = (init_Ang + end_Ang) / 2
                        Dia = float(self.wellinfo.ProductionPipe['内径'][pipe_num])
                        dth_aver = (init_depth + end_depth) / 2
                        Rt = 48
                        Tenv = 31.2 + 273.15 + (init_vDepth+end_vDepth)/2*0.031
                        Wall_thickness = float(self.wellinfo.ProductionPipe['壁厚'][pipe_num])
                        newgrid = grid(gridType='pipe', initDepth=init_depth, endDepth=end_depth,
                                       initVDepth=init_vDepth, endVDepth=end_vDepth, Len=Len, Ang=Ang,Dia=Dia,Wallthickness=Wall_thickness,K=Rt,T_env=Tenv)
                        self.grids.append(newgrid)
                    # print(Rt)
                # 对 self.grids 列表进行倒序操作
        self.grids.reverse()
        return self.grids

    @property
    def showGrids(self):
        # 定义不同 gridType 对应的颜色
        color_map = {
            'ICV': 'red',
            'ESP': 'blue',
            'pipe': 'green',
            # 可以根据需要添加更多的 gridType 和对应的颜色
        }

        # 创建一个宽高比为 1:3 的图形
        fig,ax = plt.subplots(figsize=(4, 12))

        # 定义 'pipe' 类型的线宽为其他类型的一半
        default_linewidth = 2  # 其他类型的线宽
        pipe_linewidth = default_linewidth / 2  # 'pipe' 类型的线宽

        # 按 gridType 分组绘制散点，设置较大的 s 参数来让线变长
        for grid_type, color in color_map.items():
            end_depths = []
            for grid_obj in self.grids:
                if grid_obj.gridType == grid_type:
                    end_depths.append(grid_obj.endDepth)
            if end_depths:
                # 根据 gridType 设置线宽
                if grid_type == 'pipe':
                    linewidth = pipe_linewidth
                else:
                    linewidth = default_linewidth
                # 增大 s 参数的值，例如设为 200，可根据需求调整
                ax.scatter([0] * len(end_depths), end_depths, c=color, marker='_', label=grid_type, s=200, linewidths=linewidth)

        ax.set_ylabel('End Depth')
        ax.set_title('网格划分示意')

        # 颠倒 y 轴刻度
        ax.invert_yaxis()

        # 隐藏 x 轴刻度标签
        ax.set_xticklabels([])

        # 显示图例
        if ax.legend_ is None:
            ax.legend()

        plt.show()



