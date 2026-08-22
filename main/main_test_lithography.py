"""GDS → 光刻结果留档入口：任意版图输入，逐 tile nominal/binary PNG + manifest。"""

import argparse
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]  # main/ 的上一级即仓库根
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/lithography 可导入

from common.units import exact_dbu
from layout import LayerNotFoundError, LayerSpec, LayoutDB
from lithography import ICCAD13Lithography
from main._macro_pipeline import save_lithography_pngs
from opc.input import MaskPolarity


def _decimal_argument(text: str) -> Decimal:
    """把命令行文本解析为 Decimal（十进制语义，不经二进制浮点）。"""
    try:
        return Decimal(text)
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"非法数值：{text}") from None


def _layer_argument(text: str) -> LayerSpec:
    """解析 "layer/datatype"（如 11/0）为层规格。"""
    parts = text.split("/")
    if len(parts) != 2 or not all(part.strip().lstrip("-").isdigit() for part in parts):
        raise argparse.ArgumentTypeError(f"层格式应为 N/D（如 11/0）：{text}")
    return LayerSpec(int(parts[0]), int(parts[1]))


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """解析命令行；用法错误由 argparse 以退出码 2 终止并打印提示。"""
    parser = argparse.ArgumentParser(
        description="GDS → 光刻结果留档：逐 tile nominal/binary PNG + manifest（与迭代管线 final_lithography 产物同款）"
    )
    parser.add_argument("gds", type=Path, help="输入 GDS 文件路径")
    parser.add_argument("--top", default=None, metavar="CELL", help="顶层 Cell 名（缺省=版图唯一顶层；多顶层必须指名）")
    parser.add_argument(
        "--layer", type=_layer_argument, default="11/0", help="目标层 layer/datatype（默认 11/0，TestReticle 惯例）"
    )
    parser.add_argument("--polarity", choices=("clear", "opaque"), default="clear", help="掩模极性（默认 clear）")
    parser.add_argument(
        "--core-nm", type=_decimal_argument, default=Decimal(1024), help="tile 核心边长 nm（默认 1024）"
    )
    parser.add_argument(
        "--context-nm", type=_decimal_argument, default=Decimal(512), help="tile 上下文边长 nm（默认 512）"
    )
    parser.add_argument("--pixel-nm", type=_decimal_argument, default=Decimal(8), help="光刻像素 nm（默认 8）")
    parser.add_argument("--batch", type=int, default=4, help="光刻前向批大小（默认 4）")
    parser.add_argument("--device", default="auto", help="计算设备 auto/cpu/cuda（默认 auto）")
    parser.add_argument(
        "--out", type=Path, default=None, help="留档目录（默认 output/lithography/<GDS 主干名>/，仓库根锚定）"
    )
    args = parser.parse_args(argv)
    if args.batch < 1:  # range 步长 0 是脏崩溃，边界值在解析层拦截
        parser.error("--batch 必须 ≥ 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """打开版图、解析网格参数，用与迭代管线同一内核留档光刻结果。"""
    args = _parse_arguments(argv)
    # 留档目录：显式 --out 优先；缺省锚定仓库根（output/ 已 gitignore）
    output_dir = args.out if args.out is not None else _REPO_ROOT / "output" / "lithography" / args.gds.stem
    with LayoutDB.open(args.gds, args.top) as database:  # 文件/顶层歧义在此报错
        dbu_nm = Decimal(str(database.dbu_um)) * 1000  # DBU 的 nm 值（十进制精确）
        try:
            bounds = database.layer_bbox(args.layer)  # 层在子树内无图形时 None
        except LayerNotFoundError:  # 层号在版图中根本不存在——同一口径报错
            bounds = None
        if bounds is None:  # 空层无法出图（与迭代管线 prepare 阶段同口径）
            raise ValueError(f"目标层 {args.layer.layer}/{args.layer.datatype} 不含任何图形")
    # nm→DBU 三个网格参数精确换算（非整数倍报错自带 flag 名）；
    # core+2*context 与像素的画布契约由留档内核内 plan_macros 校验
    core_dbu = exact_dbu(args.core_nm, dbu_nm, "--core-nm")
    context_dbu = exact_dbu(args.context_nm, dbu_nm, "--context-nm")
    pixel_dbu = exact_dbu(args.pixel_nm, dbu_nm, "--pixel-nm")
    # 最贵的一步放最后：全部输入合法才加载模型资产
    model = ICCAD13Lithography(device=args.device)
    print(f"device={model.device}")
    manifest = save_lithography_pngs(
        args.gds,
        args.layer,
        MaskPolarity(args.polarity),
        core_dbu,
        context_dbu,
        pixel_dbu,
        model.config.canvas,
        model,
        args.batch,
        output_dir,
    )
    print(f"tile 数：{manifest['tile_count']}")
    print(f"manifest：{output_dir / 'manifest.json'}")
    print(f"已保存 {output_dir}")
    return 0


if __name__ == "__main__":  # 直接运行入口
    raise SystemExit(main())
