"""nm↔DBU 的十进制精确换算（全部落格点参数的唯一换算入口）。"""

from decimal import Decimal  # 十进制除法无二进制浮点误差


def exact_dbu(value_nm: Decimal, dbu_nm: Decimal, name: str) -> int:
    """把必须落在版图格点上的 nm 参数精确转换为整数 DBU。"""
    quotient = value_nm / dbu_nm  # 十进制除法，无二进制浮点误差
    if quotient != quotient.to_integral_value():  # 非整数倍即无法精确落格点
        # 报错必须写明参数名、nm 值与当前 dbu_nm
        raise ValueError(
            f"{name}={value_nm} nm 无法精确换算为 {dbu_nm} nm/DBU 的整数倍")
    return int(quotient)  # 精确整数 DBU
