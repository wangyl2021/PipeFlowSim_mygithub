import numpy as np
import pandas as pd
import math


def DetaAzimStandard(rad_angle):
    while abs(rad_angle) > math.pi:
        rad_angle = rad_angle - math.copysign(2 * math.pi, rad_angle)
    return rad_angle

def cta(x, y):
    '''
    两个实数X和Y的反正切值的函数
    :param x:
    :param y:
    :return:
    '''
    if x == 0:
        if y > 0:
            return math.pi / 2
        elif y == 0:
            return 0
        else:
            return 1.5 * math.pi
    elif x > 0:
        result = math.atan(y / x)
        if y < 0:
            result += 2 * math.pi
        return result
    else:
        return math.atan(y / x) + math.pi



# def cylindrical_thread_newPointInterpolation(point1, point2, target_depth):
#     """
#      圆柱螺线插值
#      :param point1: 插值参考点1，上部点
#      :param point2: 插值参考点2，下部点
#      :param target_depth:插值目标点，单位m
#      :return: target_Ang: 目标点井斜角，弧度；target_azimuth: 目标点方位角，弧度；target_vDepth: 目标点垂深，单位m
#      """
#     Len_delta = (point2[0] - point1[0])  ## 井深差
#     Ang_delta = (point2[1] - point1[1]) / Len_delta  ## 井斜变化率
#     target_Ang = point1[1] + Ang_delta * (target_depth - point1[0])  ## 目标点井斜
#     detaL = target_depth - point1[0]
#
#     if abs(Ang_delta) < 0.0001:  ##如果井斜变化太小，直接计算垂深
#         vDepth_delta = math.cos(point1[1]) * detaL
#         ds = math.sin(point1[1]) * detaL
#         DDS = math.sin(point1[1]) * detaL
#     else:
#         R = detaL / (target_Ang - point1[1])
#         DDS = (math.cos(point1[1]) - math.cos(point2[1])) * R
#         vDepth_delta = R * (math.sin(target_Ang) - math.sin(point1[1]))
#         ds = (math.cos(target_Ang) - math.cos(point2[1])) * R
#     target_vDepth = vDepth_delta + point1[3]
#
#     if abs(ds) < 0.0001:
#         target_azimuth = point1[2]
#     else:
#         Kh = DetaAzimStandard(point2[2] - point1[2]) / DDS
#         target_azimuth = DetaAzimStandard(point2[2] - Kh * ds)
#     # if target_azimuth<0:
#     #     target_azimuth=target_azimuth+2*math.pi
#     # print("目标井深",+str(target_depth))
#     # print("目标井斜角变化率", +str(target_depth))
#     return target_Ang, target_azimuth, target_vDepth





def space_Curve_newPointInterpolation(point1, point2, target_depth):
    '''

    :param point1:  插值参考点1，上部点
    :param point2:  插值参考点2，下部点
    :param target_depth: 插值目标点，单位m
    :return: 目标点井斜角，弧度；target_azimuth: 目标点方位角，弧度；target_vDepth: 目标点垂深，单位m
    '''
    Len_delta = (point2[0] - point1[0])  ## 井深差
    Azi_delta = DetaAzimStandard(point1[2] - point2[2])
    angY= math.cos(point2[1]) * math.cos(point1[1]) + math.sin(point2[1]) * math.sin(point1[1]) *  math.cos(Azi_delta)

    if abs(angY) <= 1:
        if abs(angY) < 0.0001:
            angY = math.pi / 2 if angY >= 0 else math.pi * 1.5
        else:
            angY = math.atan(math.sqrt(1 - angY * angY) / angY)
            if angY < 0:
                angY += math.pi
    else:
        print('error01,插值数据不合理！')

    if abs(math.sin(Azi_delta) * math.sin(point2[1] - angY)) < 0.0001:
        if abs(math.sin(Azi_delta)) < 0.0001:
            W_arc = 0 if point1[1] > point2[1] else math.pi
        else:
            fz = math.sin(Azi_delta) * math.cos(point2[1])
            fm = math.cos(Azi_delta)
            W_arc = 2 * cta(fm, fz)
    else:
        fz = math.pow(math.sin(angY), 2) - math.pow(math.sin(point2[1]) * math.sin(Azi_delta), 2)
        if fz < -0.000001:
            raise ValueError('error02,数据不合理！')
        else:
            fz = math.sqrt(abs(fz))
        if point1[1] <= point2[1]:
            fz = -fz
        fz = math.sin(angY) * math.cos(Azi_delta) - fz
        fm = math.sin(Azi_delta) * math.sin(point2[1] - angY)
        W_arc = 2 * cta(fm, fz)

    K_arc = angY / Len_delta
    if W_arc > 2 * math.pi:
        W_arc -= 2 * math.pi

    detaL = point2[0]-target_depth  ##ll不知道是啥

    if abs(K_arc) < 0.0001:
        target_Ang = point1[1]
    else:
        target_Ang = (math.cos(point2[1]) * math.cos(K_arc * detaL) - math.sin(point2[1]) * math.cos(W_arc) * math.sin(
            K_arc * detaL))
        if abs(target_Ang) < 0.0001:
            target_Ang = math.pi / 2
        else:
            target_Ang = math.atan(math.sqrt(1 - target_Ang * target_Ang) / target_Ang)
            if target_Ang < 0:
                target_Ang += math.pi

    # 求方位
    if abs(K_arc) < 0.0001:
        target_azimuth = point1[2]
    else:
        # 判断装置角是否接近高边(0)或低边(PI)
        wTmp = W_arc
        if wTmp > math.pi:
            wTmp = wTmp - math.pi  # (0-2PI)→(0,PI)
        if wTmp > math.pi / 2:
            wTmp = wTmp - math.pi  # (0-PI)→(-PI/2,PI/2)

        if abs(wTmp) < 0.0001:  # 表明装置角接近高边(0)或低边(PI)
            target_azimuth = point1[2]
        else:
            wTmp = math.tan(K_arc * detaL)  # wTmp被赋予新的含义
            # 定义如下变量可以加速运算，因为1次三角函数运算量相当于多次乘除运算
            sinInc = math.sin(point2[1])
            cosInc = math.cos(point2[1])
            sinAzm = math.sin(point2[2])
            cosAzm = math.cos(point2[2])
            sinW = math.sin(W_arc)
            cosW = math.cos(W_arc)
            cosIncCosW = cosInc * cosW

            fm = sinInc * cosAzm + (cosIncCosW * cosAzm - sinAzm * sinW) * wTmp
            fz = sinInc * sinAzm + (cosIncCosW * sinAzm + cosAzm * sinW) * wTmp
            if fm == 0:
                target_azimuth = math.pi / 2
            else:
                target_azimuth = math.atan(fz / fm)
                if target_azimuth < 0:
                    target_azimuth = target_azimuth + math.pi
                if fz < 0:
                    target_azimuth = target_azimuth + math.pi


    Ang_delta = (point2[1] - point1[1]) / Len_delta  ## 井斜变化率
    target_Ang = point1[1] + Ang_delta * (target_depth - point1[0])  ## 目标点井斜
    # detaL = target_depth - point2[0]
    target_vDepth= point2[3]-detaL*math.cos(target_Ang/2+point2[1]/2)

    return target_Ang,target_azimuth,target_vDepth


def curvature_radius_newPointInterpolation(point1, point2, target_depth):
    """
    曲率半径插值
    :param point1: 插值参考点1，上部点
    :param point2: 插值参考点2，下部点
    :param target_depth:插值目标点，单位m
    :return: target_Ang: 目标点井斜角，弧度；target_azimuth: 目标点方位角，弧度；target_vDepth: 目标点垂深，单位m
    """
    Len_delta = (point2[0]-point1[0])## 井深差
    Ang_delta  = (point2[1]-point1[1]) / Len_delta ## 井斜变化率
    target_Ang = point1[1] + Ang_delta * (target_depth - point1[0]) ## 目标点井斜
    detaL = target_depth - point1[0]

    if abs(Ang_delta) < 0.0001:##如果井斜变化太小，直接计算垂深
        vDepth_delta = math.cos(point1[1])*detaL
        ds=math.sin(point1[1])*detaL
        DDS = math.sin(point1[1])*detaL
    else:
        R = detaL/(target_Ang - point1[1])
        DDS = (math.cos(point1[1]) - math.cos(point2[1])) * R
        vDepth_delta = R*(math.sin(target_Ang)-math.sin(point1[1]))
        ds=(math.cos(target_Ang) - math.cos(point2[1]))*R
    target_vDepth = vDepth_delta + point1[3]

    if abs(ds) < 0.0001:
        target_azimuth=point1[2]
    else:
        Kh = DetaAzimStandard(point2[2] - point1[2]) /   DDS
        target_azimuth = DetaAzimStandard(point2[2] -Kh * ds)
    # if target_azimuth<0:
    #     target_azimuth=target_azimuth+2*math.pi
    # print("目标井深",+str(target_depth))
    # print("目标井斜角变化率", +str(target_depth))
    return target_Ang,target_azimuth,target_vDepth

def curvature_radius_vDepthCal(point1, point2):
    """
       曲率半径确定垂深
       :param point1: 插值参考点1，上部点
       :param point2: 插值参考点1，下部点
       :return: target_vDepth: 目标点垂深，单位m
       """
    Len_delta = (point2[0] - point1[0])
    Ang_delta = (point2[1] - point1[1]) / Len_delta ## 井斜变化率
    if abs(Ang_delta) < 0.0001:
        vDepth_delta = math.cos(point1[1]) * Len_delta
    else:
        R = Len_delta / (point2[1] - point1[1])
        vDepth_delta = R * (math.sin(point2[1]) - math.sin(point1[1]))

    target_vDepth = vDepth_delta + point1[3]
    return target_vDepth


def newPoint_interpolation(target_depth,point1, point2, MathModle=0 ):
    """
    井眼轨迹插值
    :param MathModle: 插值方法
    :param point1: 插值参考点1，上部点
    :param point2: 插值参考点1，下部点
    :param target_depth:插值目标点，单位m
    :return: target_Ang: 目标点井斜角，弧度；target_azimuth: 目标点方位角，弧度；target_vDepth: 目标点垂深，单位m
    """
    target_Ang, target_vDepth, target_azimuth=0,0,0
    if MathModle == 0:  ## 曲率半径法
        target_Ang, target_azimuth,target_vDepth = curvature_radius_newPointInterpolation(point1,point2, target_depth)
    elif MathModle == 1: ## 空间圆弧
        target_Ang, target_azimuth, target_vDepth = space_Curve_newPointInterpolation(point1, point2, target_depth)
    return target_depth, target_Ang, target_azimuth, target_vDepth


def Vdepth_cal(MathModle,point1, point2):
    """
    垂深计算
    :param MathModle: 插值方法
    :param point1: 插值参考点1，上部点
    :param point2: 插值参考点2，下部点
    :return: target_vDepth: 目标点垂深，单位m
    """
    target_vDepth=0
    if MathModle==0:##曲率半径法
        target_vDepth = curvature_radius_vDepthCal(point1, point2)
    return target_vDepth

def curvatureCal(point1, point2):
    '''
    曲率计算
    :param point1:上部点
    :param point2: 下部点
    :return: 曲率
    '''
    detaL= (point2[0] - point1[0])  ## 井深差
    Ang_delta = (point2[1] - point1[1]) / detaL  ## 井斜变化率
    Ang_avg = (point2[1] + point1[1])/2
    if abs(Ang_delta) < 0.0001:  ##如果井斜变化太小，直接计算垂深
        ds = math.sin(point2[1]) * detaL
    else:
        R = detaL / (point2[1] - point1[1])
        ds = (math.cos(point1[1]) - math.cos(point2[1])) * R
    if abs(ds) < 0.0001:
        Kh = 0
    else:
        Kh = DetaAzimStandard(point2[2] - point1[2]) / ds
    invPI = 180 / math.pi  ## 转弧度制
    kb= math.sqrt(math.pow(Ang_delta, 2)+math.pow(math.sin(Ang_avg)*math.sin(Ang_avg)*Kh,2))
    return kb


def rawWellTrackCal(Welltracks,MathModle=0):
    Welltracks = Welltracks.round(10)
    PointNum = Welltracks.shape[0]  ##获取井轨迹数据的点数
    invPI = 180 / math.pi  ## 转弧度制
    Welltracks['井斜'] = Welltracks['井斜'].apply(lambda x: x / invPI)
    Welltracks['方位'] = Welltracks['方位'].apply(lambda x: x / invPI)
    Welltracks['垂深'] = None
    Welltracks['狗腿度'] = None
    Welltracks.loc[0, '垂深'] = 0.0
    Welltracks['狗腿度'] = 0.0
    for i in range(PointNum - 1):
        point1 = Welltracks.loc[i].values.tolist()
        point2 = Welltracks.loc[i + 1].values.tolist()
        Welltracks.loc[i + 1, '垂深'] = Vdepth_cal(MathModle, point1, point2)
        Welltracks.loc[i + 1, '狗腿度'] = curvatureCal(point1, point2)
    Welltracks['方位'] = Welltracks['方位'].apply(lambda x: DetaAzimStandard(x))
    Welltracks = Welltracks.round(10)
    return Welltracks


def wellTrackInterpolation(MathModle,Welltracks,Step,EndDepth):
    """
    根据已有井眼轨迹，根据目标深度和计算步长，计算新的井眼轨迹表
    :param MathModle: 插值方法
    :param Welltracks: 已有井眼轨迹
    :param Step: 插值步长
    :return: EndDepth: 插值终止井深
    """
    PointNum = Welltracks.shape[0]
    StepCount = math.ceil(EndDepth/Step)+1##确定分割的行数
    Welltracks_new = pd.DataFrame({'井深': [0.0] * StepCount,'井斜': [0.0] * StepCount, '方位': [0.0] * StepCount,'垂深': [0.0] * StepCount,'狗腿度':[0.0] * StepCount,"方位角变化率":0.0,"井斜角变化率":0.0})
    WelltracksRowCount=0
    i=0
    while(i<StepCount):
        if i == 0:
            Welltracks_new['井深'][i] = 0.0
            Welltracks_new['井斜'][i] = Welltracks['井斜'][WelltracksRowCount]
            Welltracks_new['方位'][i] = Welltracks['方位'][WelltracksRowCount]
            Welltracks_new['垂深'][i] = 0.0
            Welltracks_new['狗腿度'][i] = 0.0
            WelltracksRowCount = WelltracksRowCount + 1
            Welltracks_new['方位角变化率'][i] = 0
            Welltracks_new['井斜角变化率'][i] = 0
        else:
            thisPointDepth = i*Step
            if i==StepCount-1:
                thisPointDepth=EndDepth
            if  WelltracksRowCount<PointNum:#判断内插还是外插 逻辑上 此处为内插
                if thisPointDepth < Welltracks['井深'][WelltracksRowCount]:
                    point1 = Welltracks.loc[WelltracksRowCount-1].values.tolist()
                    point2 = Welltracks.loc[WelltracksRowCount].values.tolist()
                    target_depth, target_Ang, target_azimuth, target_vDepth = newPoint_interpolation(MathModle,point1, point2, thisPointDepth)
                    Welltracks_new['井深'][i] = target_depth
                    Welltracks_new['井斜'][i] = target_Ang
                    Welltracks_new['方位'][i] = target_azimuth
                    Welltracks_new['垂深'][i] = target_vDepth
                    Welltracks_new['狗腿度'][i] = point2[4]
                    Welltracks_new['方位角变化率'][i] = (point2[2]-point1[2])/ (point2[0]-point1[0])
                    Welltracks_new['井斜角变化率'][i] = (point2[1] - point1[1])/ (point2[0]-point1[0])


                elif thisPointDepth == Welltracks['井深'][WelltracksRowCount]:
                    point1 = Welltracks.loc[WelltracksRowCount-1].values.tolist()
                    point2 = Welltracks.loc[WelltracksRowCount].values.tolist()
                    Welltracks_new['井深'][i]=thisPointDepth
                    Welltracks_new['井斜'][i] = point2[1]
                    Welltracks_new['方位'][i] = point2[2]
                    Welltracks_new['垂深'][i] = Vdepth_cal(MathModle,point1, point2)
                    Welltracks_new['狗腿度'][i] = point2[4]
                    Welltracks_new['方位角变化率'][i] = (point2[2] - point1[2]) / (point2[0] - point1[0])
                    Welltracks_new['井斜角变化率'][i] =  (point2[1] - point1[1])/ (point2[0]-point1[0])
                    WelltracksRowCount = WelltracksRowCount + 1
                elif thisPointDepth > Welltracks['井深'][WelltracksRowCount]:
                    WelltracksRowCount = WelltracksRowCount + 1
                    i-=1
            else:#判断内插还是外插 逻辑上 此处为外插
                point1 = Welltracks_new.loc[i-1].values.tolist()
                Welltracks_new['井深'][i] = thisPointDepth
                Welltracks_new['井斜'][i] = point1[1]
                Welltracks_new['方位'][i] = point1[2]
                Welltracks_new['垂深'][i] = abs(thisPointDepth-point1[0])*math.cos(point1[1])+point1[3]
                Welltracks_new['狗腿度'][i] = point1[4]
                Welltracks_new['方位角变化率'][i] = Welltracks_new['方位角变化率'][i-1]
                Welltracks_new['井斜角变化率'][i] = Welltracks_new['井斜角变化率'][i-1]
        i+=1
    # Welltracks_new['井斜'] = Welltracks_new['井斜'].apply(lambda x: x * invPI)
    # Welltracks_new['方位'] = Welltracks_new['方位'].apply(lambda x: x * invPI)
    Welltracks_new['方位角变化率'] = Welltracks_new['方位角变化率'].round(10)
    Welltracks_new['井斜角变化率'] = Welltracks_new['井斜角变化率'].round(10)
    # for i in range(len(Welltracks_new)):
    #     if Welltracks_new['方位'][i]<0:
    #         Welltracks_new['方位'][i]= Welltracks_new['方位'][i]+math.pi*2
    return Welltracks_new


