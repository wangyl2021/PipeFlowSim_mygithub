class grid:
    def __init__(self,gridType,initDepth,endDepth,initVDepth,endVDepth,Len,Ang,Dia,T_env,Wallthickness=0,roughness=0,K=0):
        self.gridType = gridType
        '''网格类型'''
        self.initDepth = initDepth
        '''初始井深'''
        self.endDepth = endDepth
        '''结束井深'''
        self.initVDepth = initVDepth
        '''初始垂深'''
        self.endVDepth = endVDepth
        '''结束垂深'''
        self.Len = Len
        '''长度'''
        self.Ang = Ang
        '''管道倾斜角'''
        self.Dia = Dia
        '''管道内直径'''
        self.Wallthickness=Wallthickness
        '''管道壁厚'''
        self.roughness = roughness
        '''管道粗糙度'''
        self.K = K
        '''油管的热导率'''
        self.T_env= T_env