def joule_kgk_to_btu_lbmdegf(value):
    """
    将J/(kg*K)转换为Btu/(lbm*degF)

    参数:
    value (float): 以J/(kg*K)为单位的值

    返回:
    float: 转换后以Btu/(lbm*degF)为单位的值
    """
    conversion_factor = 0.000238846
    return value * conversion_factor


def degf_psi_to_k_pa(value):
    """
    将°F/psi转换为K/Pa
    参数:
        value (float): 以°F/psi为单位的温度梯度值

    返回:
        float: 转换后的K/Pa单位值
    """
    # 组合转换系数：(5/9) / 6894.76 ≈ 8.0641e-5
    conversion_factor = 8.0641e-5
    return value * conversion_factor

def kelvin_to_fahrenheit(kelvin: float) -> float:
    """
    将开尔文温度转换为华氏温度

    参数:
    kelvin (float): 开尔文温度值

    返回:
    float: 对应的华氏温度值
    """
    return (kelvin - 273.15) * 9 / 5 + 32

def pa_to_psia(pascal: float) -> float:
    """
    将帕斯卡压力转换为磅力每平方英寸(绝对压力)

    参数:
    pascal (float): 帕斯卡压力值

    返回:
    float: 对应的psia压力值
    """
    return pascal / 6894.75729


def convert_fahrenheit_psia_to_kelvin_pa(value):
    """
    将组合单位（∘F/psia）转换为（K/Pa）

    参数:
    value (float): 以（∘F/psia）为单位的数值

    返回:
    float: 转换为（K/Pa）单位的数值
    """
    # 温度转换：∘F 到 K
    # T(K) = (T(°F) + 459.67) × 5/9
    temperature_conversion = 5 / 9

    # 压力转换：psia 到 Pa
    # 1 psi = 6894.75729 Pa
    pressure_conversion = 6894.75729

    # 组合单位转换系数
    conversion_factor = temperature_conversion / pressure_conversion

    # 应用转换系数
    return value * conversion_factor


def jkgk_to_btulbmdegf(value):
    """
    将 J/(kg·K) 单位的值转换为 Btu/(lbm·°F) 单位的值

    参数:
        value (float): 以 J/(kg·K) 为单位的数值

    返回:
        float: 转换后以 Btu/(lbm·°F) 为单位的数值
    """
    # 换算系数，1 J/(kg·K) ≈ 0.000238846 Btu/(lbm·°F)
    conversion_factor = 0.000238846
    return value * conversion_factor