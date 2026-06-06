import math
import numbers

import numpy as np


R= 8.314
T0 = 273.15+20
P0 = 101.325*1000

def solve_equation(coefficients):
    roots = np.roots(coefficients)
    root_reals = []
    for root in roots:
        if root.dtype == np.complex128:
            if root.imag == 0:
                root_reals.append(root.real)
        else:
            root_reals.append(root)
    return np.array(root_reals)

class EOS():
    def __init__(self):
        self.a
        pass

class State_Equation(EOS):   # 2025 10 08 孙玉逊修改类名SRK--》State_Equation
    def __init__(self,KijArray):
        self.KijArray = KijArray
        pass

    def calc_Z(self,A, B, phase,Equition_type):
        '''
        计算压缩因子
        :param A:
        :param B:
        :param phase:
        :return:
        '''
        if Equition_type=="SRK":
            a = 1
            b = -1
            c = A - B - B * B
            d = -A * B
        elif Equition_type=="PR":
            a = 1
            b = B-1
            c = A - 3 * B - 2 * B * B
            d = -A * B+ B * B + B * B * B
        coefficients = [a, b, c, d]
        # roots = np.roots(coefficients)
        root_reals = solve_equation(coefficients)
        if len(root_reals) == 3:
            if phase == "Gas":
                Z = max(root_reals)
            else:
                Z = min(root_reals)
        elif len(root_reals) == 1 :
            Z = root_reals[0]
            # if phase == "Gas":
            #     if Z <  (1/3):
            #         a = 3
            #         b = -2
            #         c = A - B - B * B
            #         coefficients = [a, b, c]
            #         root_reals = solve_equation(coefficients)
            #         Z = max(root_reals)
            # if phase == "Liquid":
            #     if Z > (1/3):
            #         a = 3
            #         b = -2
            #         c = A - B - B * B
            #         coefficients = [a, b, c]
            #         root_reals = solve_equation(coefficients)
            #         Z = min(root_reals)
        return Z

    def calc_ai(self,components,T,Equition_type):
        '''
        根据温度，计算初始的各组分ai
        :return:
        '''
        if Equition_type=="SRK":
            for i in range(len(components)):
                Tr = T / components[i].Tc
                if components[i].name == 'Hydrogen':
                    ah = math.pow(1.096 * math.exp(-1.15114 * Tr), 2)
                else:
                    ah = math.pow(1 + (1 - math.sqrt(Tr)) * (
                            0.48508 + 1.55171 * components[i].Af - 0.15613 * math.pow(components[i].Af, 2)),
                                  2)
                components[i].a = components[i].Oa * (R ** 2) * (components[i].Tc ** 2) / components[i].Pc * ah  # SRK公式
            return components
        elif Equition_type=="PR":
            for i in range(len(components)):
                Tr = T / components[i].Tc
                if components[i].name == 'Hydrogen':
                    ah = math.pow(1.096 * math.exp(-1.15114 * Tr), 2)
                else:
                    ah = math.pow(1 + (1 - math.sqrt(Tr)) * (
                                0.37464 + 1.54226 * components[i].Af - 0.26992 * math.pow(components[i].Af, 2)),
                                  2)
                components[i].a = components[i].Oa * (R ** 2) * (components[i].Tc ** 2) / components[i].Pc * ah #SRK公式
            return components

    def calc_ab(self, components, xs, isConsiderKij=False):
        """
        计算混合物的参数 a 和 b。
        :param components: 组分列表，每个组分对象应包含属性 a 和 b
        :param xs: 各组分的摩尔分数列表
        :return: 混合物的参数 a 和 b
        """
        # 初始化混合物参数 a 和 b 为 0
        a = 0
        b = 0
        # 遍历所有组分，计算混合物的参数 b
        for i in range(len(components)):
            # 根据混合规则累加计算参数 b
            b = b + xs[i] * components[i].b
            # 双重循环遍历所有组分对，计算混合物的参数 a
            if isConsiderKij:
                for j in range(len(components)):
                    # 从 Kij 数组中获取交互系数 Kij
                    Kij = self.KijArray[i][j]
                    # 根据混合规则累加计算参数 a
                    a = a + xs[i] * xs[j] * math.sqrt(components[i].a * components[j].a) * (1 - Kij)
            else:
                a = a + xs[i] * math.sqrt(components[i].a)
        if isConsiderKij==False:
            a = a*a
        return a, b

    def calc_fi(self, P,T,zis,a,b,A,B,Z,components,mixingRule=1):
        if mixingRule == 1:
            '''matlab'''
            fis = np.ones(len(components))
            for j in range(len(components)):
                term = Z - B
                term1 = (components[j].b / b) * (Z - 1)
                term2 = 0
                for i in range(len(components)):
                    term2 = term2 + zis[i] * np.sqrt(components[i].a *components[j].a)
                term3 = (A / B) * ((2 /a*term2) - (components[j].b / b)) * np.log(1 + (B / Z))
                exponent = (term1 - term3) #/ R / T
                fis[j] =  np.exp(exponent)/term*zis[j]*P
        elif mixingRule == 2:
            """李长俊"""
            fis = np.ones(len(components))
            for i in range(len(components)):
                Term=(components[i].b / b) * (Z - 1)-np.log(Z-B)
                Term1=(A / B) *(2*np.sqrt(components[i].a /a)-(components[i].b / b))* np.log(1 + (B / Z))
                fis[i]=np.exp((Term-Term1))*zis[i]*P
        elif mixingRule == 3:
            """马高强"""
            fis = np.ones(len(components))
            for j in range(len(components)):
                term1 = (components[j].b / b) * (Z - 1)-np.log(Z-B)
                term2 = 0
                for i in range(len(components)):
                    term2 = term2 + zis[i] * np.sqrt(components[i].a * components[j].a)
                term3 = (A / B) * ((2 / a * term2) - (components[j].b / b)) * np.log((Z+B)/Z)
                exponent = (term1 - term3)  # / R / T
                fis[j] = np.exp(exponent) * zis[j] * P
        return fis

    def calc(self,components,P,T,zis,phase,Equition_type):
        a, b = self.calc_ab(components, zis)
        A = a * P / (math.pow(R * T, 2))
        B = b * P / (R * T)
        Z = self.calc_Z(A, B, phase,Equition_type)
        fis = self.calc_fi(P,T,zis,a,b,A,B,Z,components,mixingRule=1)
        return Z, A, B, a, b, fis




class SRK(EOS):
    def __init__(self,KijArray):
        self.KijArray = KijArray
        pass

    def calc_Z(self,A, B, phase):
        '''
        计算压缩因子
        :param A:
        :param B:
        :param phase:
        :return:
        '''
        a = 1
        b = -1
        c = A - B - B * B
        d = -A * B
        coefficients = [a, b, c, d]
        # roots = np.roots(coefficients)
        root_reals = solve_equation(coefficients)
        if len(root_reals) == 3:
            if phase == "Gas":
                Z = max(root_reals)
            else:
                Z = min(root_reals)
        elif len(root_reals) == 1 :
            Z = root_reals[0]
            # if phase == "Gas":
            #     if Z <  (1/3):
            #         a = 3
            #         b = -2
            #         c = A - B - B * B
            #         coefficients = [a, b, c]
            #         root_reals = solve_equation(coefficients)
            #         Z = max(root_reals)
            # if phase == "Liquid":
            #     if Z > (1/3):
            #         a = 3
            #         b = -2
            #         c = A - B - B * B
            #         coefficients = [a, b, c]
            #         root_reals = solve_equation(coefficients)
            #         Z = min(root_reals)
        return Z

    def calc_ai(self,components,T):
        '''
        根据温度，计算初始的各组分ai
        :return:
        '''
        for i in range(len(components)):
            Tr = T / components[i].Tc
            if components[i].name == 'Hydrogen':
                ah = math.pow(1.096 * math.exp(-1.15114 * Tr), 2)
            else:
                ah = math.pow(1 + (1 - math.sqrt(Tr)) * (
                            0.48508 + 1.55171 * components[i].Af - 0.15613 * math.pow(components[i].Af, 2)),
                              2)
            components[i].a = components[i].Oa * (R ** 2) * (components[i].Tc ** 2) / components[i].Pc * ah #SRK公式
        return components

    def calc_ab(self, components, xs, isConsiderKij=False):
        """
        计算混合物的参数 a 和 b。
        :param components: 组分列表，每个组分对象应包含属性 a 和 b
        :param xs: 各组分的摩尔分数列表
        :return: 混合物的参数 a 和 b
        """
        # 初始化混合物参数 a 和 b 为 0
        a = 0
        b = 0
        # 遍历所有组分，计算混合物的参数 b
        for i in range(len(components)):
            # 根据混合规则累加计算参数 b
            b = b + xs[i] * components[i].b
            # 双重循环遍历所有组分对，计算混合物的参数 a
            if isConsiderKij:
                for j in range(len(components)):
                    # 从 Kij 数组中获取交互系数 Kij
                    Kij = self.KijArray[i][j]
                    # 根据混合规则累加计算参数 a
                    a = a + xs[i] * xs[j] * math.sqrt(components[i].a * components[j].a) * (1 - Kij)
            else:
                a = a + xs[i] * math.sqrt(components[i].a)
        if isConsiderKij==False:
            a = a*a
        return a, b

    def calc_fi(self, P,T,zis,a,b,A,B,Z,components,mixingRule=1):
        if mixingRule == 1:
            '''matlab'''
            fis = np.ones(len(components))
            for j in range(len(components)):
                term = Z - B
                term1 = (components[j].b / b) * (Z - 1)
                term2 = 0
                for i in range(len(components)):
                    term2 = term2 + zis[i] * np.sqrt(components[i].a *components[j].a)
                term3 = (A / B) * ((2 /a*term2) - (components[j].b / b)) * np.log(1 + (B / Z))
                exponent = (term1 - term3) #/ R / T
                fis[j] =  np.exp(exponent)/term*zis[j]*P
        elif mixingRule == 2:
            """李长俊"""
            fis = np.ones(len(components))
            for i in range(len(components)):
                Term=(components[i].b / b) * (Z - 1)-np.log(Z-B)
                Term1=(A / B) *(2*np.sqrt(components[i].a /a)-(components[i].b / b))* np.log(1 + (B / Z))
                fis[i]=np.exp((Term-Term1))*zis[i]*P
        elif mixingRule == 3:
            """马高强"""
            fis = np.ones(len(components))
            for j in range(len(components)):
                term1 = (components[j].b / b) * (Z - 1)-np.log(Z-B)
                term2 = 0
                for i in range(len(components)):
                    term2 = term2 + zis[i] * np.sqrt(components[i].a * components[j].a)
                term3 = (A / B) * ((2 / a * term2) - (components[j].b / b)) * np.log((Z+B)/Z)
                exponent = (term1 - term3)  # / R / T
                fis[j] = np.exp(exponent) * zis[j] * P
        return fis






    def calc(self,components,P,T,zis,phase):
        a, b = self.calc_ab(components, zis)
        A = a * P / (math.pow(R * T, 2))
        B = b * P / (R * T)
        Z = self.calc_Z(A, B, phase)
        fis = self.calc_fi(P,T,zis,a,b,A,B,Z,components,mixingRule=1)
        return Z, A, B, a, b, fis




class PR(EOS):
    def __init__(self):
        pass
    def calc_Z(self):
        print("PR")



def EOS_init(kijArray):
    # 2025 10 08 孙玉逊修改注释
    # if method=='SRK':
    #     EOS = State_Equation(kijArray)
    # elif method=="PR":
    #     EOS = PR(kijArray)
    return State_Equation(kijArray)




