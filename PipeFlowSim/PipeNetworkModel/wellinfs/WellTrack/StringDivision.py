import numpy as np
import pandas as pd
import math
import WellTrack


def stringAndCasingProcess(pipstrings,casingpipes,BitDepth):
    """
   根据目标钻头井深处理管柱结构和套管程序
   :param pipstrings: 管柱
   :param casingpipes: 套管
   :param BitDepth: 钻头位置
   :return: pipstrings,casingpipes
   """
    ##判断钻头在套管的位置,移除下部多余套管
    i=casingpipes.shape[0]-1
    while(i>0):
        if BitDepth < casingpipes["套管顶深"][i]:
            casingpipes=casingpipes.drop(i)
            i-=1
        else:
            break
    casingpipes.loc[casingpipes.shape[0]-1, "套管底深"] = BitDepth
    ##判断管住长度位置,滑动补齐管柱长度
    i = pipstrings.shape[0] - 1
    stringLengthDelta=BitDepth-pipstrings['钻柱底深'][i]
    while(i>=0):
        pipstrings.loc[i,'钻柱顶深']=pipstrings['钻柱顶深'][i]+ stringLengthDelta
        pipstrings.loc[i,'钻柱底深'] = pipstrings['钻柱底深'][i] + stringLengthDelta
        if pipstrings['钻柱顶深'][i]<0:
            pipstrings.loc[i,'钻柱顶深'] = 0
            pipstrings.loc[i,'长度'] = pipstrings['钻柱底深'][i]
            break
        if i==0:
            pipstrings.loc[i, '钻柱顶深'] = 0
            pipstrings.loc[i, '长度'] = pipstrings['钻柱底深'][i]
            break
        i-=1
    for j in range(i):
        pipstrings=pipstrings.drop(0)
    return pipstrings,casingpipes


def string_division(welltracks,pipstrings,casingpipes):
    """
    根据已有井眼轨迹，套管程序、管柱，根据目标深度，划分微元段
    :param MathModle: 插值方法
    :param Welltracks: 已有井眼轨迹
    :param Step: 插值步长
    :return: StringSegments: 划分好的微元段
    """
    PointCount = welltracks.shape[0]
    StringSegments = pd.DataFrame({'管串编号':[0.0],'底深': [0.0] ,'顶深':  [0.0],
                                   '底井斜':  [0.0], '顶井斜': [0.0],
                                   '底方位': [0.0] ,'顶方位': [0.0],
                                   '底垂深': [0.0],  '顶垂深': [0.0],
                                   '狗腿度':[0.0], '管柱':[0.0],
                                   '套管':  [0.0],'方位角变化率':[0.0],'井斜角变化率':[0.0]})
    StringSegments = StringSegments.drop(0)
    i=1
    casingpipeNum=casingpipes.shape[0]-1
    stringNum=pipstrings.shape[0]-1
    splitpoint = 0
    splitpointType = 'Casing'
    StartDepth = welltracks['井深'][PointCount - i]
    while(i<PointCount):
        EndDepth=welltracks['井深'][PointCount - i-1]
        if casingpipes["套管顶深"][casingpipeNum]>pipstrings["钻柱顶深"][stringNum]:
            splitpoint=casingpipes["套管顶深"][casingpipeNum]
            splitpointType='Casing'
        elif casingpipes["套管顶深"][casingpipeNum]<pipstrings["钻柱顶深"][stringNum]:
            splitpoint=pipstrings["钻柱顶深"][stringNum]
            splitpointType = 'Pips'
        else:
            splitpoint = pipstrings["钻柱顶深"][stringNum]
            splitpointType = 'Pips&Casing'

        if EndDepth>splitpoint:#没有分割点
            new_row = {'管串编号': PointCount - 1 - i, '底深': StartDepth, '顶深': EndDepth,
                       '底井斜': welltracks['井斜'][PointCount - i], '顶井斜': welltracks['井斜'][PointCount - i - 1],
                       '底方位': welltracks['方位'][PointCount - i], '顶方位': welltracks['方位'][PointCount - i - 1],
                       '底垂深': welltracks['垂深'][PointCount - i], '顶垂深': welltracks['垂深'][PointCount - i - 1],
                       '狗腿度': welltracks['狗腿度'][PointCount - i],
                       '管柱': stringNum + 1, '套管': casingpipeNum + 1,"方位角变化率":welltracks['方位角变化率'][PointCount - i],"井斜角变化率":welltracks['井斜角变化率'][PointCount - i]}
            StartDepth =  welltracks['井深'][PointCount - i-1]
            StringSegments =StringSegments.append(new_row,ignore_index=True)
            i+=1
        elif EndDepth <splitpoint:#有分割点
            point1 = [welltracks['井深'][PointCount - i - 1],welltracks['井斜'][PointCount - i - 1],
                      welltracks['方位'][PointCount - i - 1],welltracks['垂深'][PointCount - i - 1]]
            '''取上部点'''
            point2 = [welltracks['井深'][PointCount - i], welltracks['井斜'][PointCount - i],
                      welltracks['方位'][PointCount - i], welltracks['垂深'][PointCount - i]]
            '''取下部点'''
            target_depth, target_Ang, target_azimuth, target_vDepth=WellTrack.WellTrackInterpolation.newPoint_interpolation(0,point1,point2, splitpoint)
            '''利用插值求取splitpoint处的井斜信息'''

            # new_row = {'管串编号': PointCount - 1 - i, '底深': StartDepth, '顶深': splitpoint,
            #            '底井斜': welltracks['井斜'][PointCount - i], '顶井斜': target_Ang,
            #            '底方位': welltracks['方位'][PointCount - i], '顶方位': target_azimuth,
            #            '底垂深': welltracks['垂深'][PointCount - i], '顶垂深': target_vDepth,
            #            '狗腿度': welltracks['狗腿度'][PointCount - i],
            #            '管柱': stringNum + 1, '套管': casingpipeNum + 1,"方位角变化率":welltracks['方位角变化率'][PointCount - i],"井斜角变化率":welltracks['井斜角变化率'][PointCount - i]}
            # StartDepth = splitpoint

            new_row = {'管串编号': PointCount - 1 - i, '底深': StartDepth, '顶深': EndDepth,
                       '底井斜': welltracks['井斜'][PointCount - i], '顶井斜': welltracks['井斜'][PointCount - i - 1],
                       '底方位': welltracks['方位'][PointCount - i], '顶方位': welltracks['方位'][PointCount - i - 1],
                       '底垂深': welltracks['垂深'][PointCount - i], '顶垂深': welltracks['垂深'][PointCount - i - 1],
                       '狗腿度': welltracks['狗腿度'][PointCount - i],
                       '管柱': stringNum + 1, '套管': casingpipeNum + 1, "方位角变化率": welltracks['方位角变化率'][PointCount - i],
                       "井斜角变化率": welltracks['井斜角变化率'][PointCount - i]}
            StartDepth = welltracks['井深'][PointCount - i - 1]

            if  splitpointType == 'Casing':
                casingpipeNum= casingpipeNum-1
            elif splitpointType == 'Pips':
                stringNum = stringNum - 1
            else:
                casingpipeNum = casingpipeNum - 1
                stringNum = stringNum - 1

            if (splitpoint - EndDepth)  > 0.5*(StartDepth - EndDepth) :
                new_row['管柱']=stringNum + 1
                new_row['套管']=casingpipeNum + 1

            StringSegments = StringSegments.append(new_row, ignore_index=True)
            i += 1

        else:#有分割点，而且分割点与微元段顶部位置相等
            new_row = {'管串编号': PointCount - 1 - i, '底深': StartDepth, '顶深': EndDepth,
                       '底井斜': welltracks['井斜'][PointCount - i], '顶井斜': welltracks['井斜'][PointCount - i - 1],
                       '底方位': welltracks['方位'][PointCount - i], '顶方位': welltracks['方位'][PointCount - i - 1],
                       '底垂深': welltracks['垂深'][PointCount - i], '顶垂深': welltracks['垂深'][PointCount - i - 1],
                       '狗腿度': welltracks['狗腿度'][PointCount - i],
                       '管柱': stringNum + 1, '套管': casingpipeNum + 1,"方位角变化率":welltracks['方位角变化率'][PointCount - i],"井斜角变化率":welltracks['井斜角变化率'][PointCount - i]}
            StartDepth = welltracks['井深'][PointCount - i - 1]
            StringSegments = StringSegments.append(new_row, ignore_index=True)
            if splitpointType == 'Casing':
                casingpipeNum = casingpipeNum - 1
            elif splitpointType == 'Pips':
                stringNum = stringNum - 1
            else:
                casingpipeNum = casingpipeNum - 1
                stringNum = stringNum - 1
            i += 1
    StringSegments["管串编号"]= StringSegments["管串编号"].astype(int)
    StringSegments["套管"] = StringSegments["套管"].astype(int)
    StringSegments["管柱"] = StringSegments["管柱"].astype(int)
    return StringSegments
