"""ICCAD13 光刻模型独立验证入口：真实 raster 画布 → 三工艺角 → batch → backward。"""

import sys  # 仓库根路径引导（免安装直接运行）
import time  # 前向计时（唯一保留的统计）
from pathlib import Path  # 路径工具

import matplotlib.pyplot as plt  # 阶段 6 可视化面板（Agg 后端下 show 无操作）
import numpy as np  # raster 画布的 numpy 类型
import torch  # 张量、no_grad 与原生 autograd

_REPO_ROOT = Path(__file__).resolve().parents[1]  # main/ 的上一级即仓库根
if str(_REPO_ROOT) not in sys.path:  # 避免重复插入
    sys.path.insert(0, str(_REPO_ROOT))  # 使 layout/opc/lithography 可导入

import klayout.db as kdb  # 原生 Region/Box 构造演示几何

from layout import DbuBox  # DBU 整数坐标框
from lithography import ICCAD13Lithography  # 固定 ICCAD13 光刻模型
from opc.input.raster import rasterize_mask_canvas  # 居中透光率画布


def _build_demo_canvas() -> np.ndarray:
    """用当前 raster 公共接口生成非对称矩形加孔洞的 256×256 透光率画布。"""
    region = (kdb.Region(kdb.Box(200, 200, 1400, 1300)) -  # 非对称实心矩形
              kdb.Region(kdb.Box(500, 500, 900, 700)))  # 减去中心孔洞（带孔图形）
    # context 1824 DBU / pixel 8 DBU = 228×228 局部窗口，居中后四边各留
    # 14 像素零 padding 恰满 256 画布——与 macro-core 管线同一画布契约。
    canvas = rasterize_mask_canvas(  # 栅格化为透光率
        region, DbuBox(0, 0, 1824, 1824), 8, 256, polarity="clear")  # clear=源图形透光
    return canvas  # float32[256,256]，1=透光，行 0=最低 Y


def run_demo(device: str = "auto") -> None:
    """依次演示模型加载、三条件前向、batch、二值阈值和真实 backward。"""
    print("=" * 72)  # 演示总分隔线
    # 阶段 1：生成真实模型输入（几何 → raster → 透光率画布）。
    canvas = _build_demo_canvas()  # 真实几何的 256 画布
    print("阶段 1 · raster 画布")  # 阶段标题
    print(f"  shape={canvas.shape} dtype={canvas.dtype} "  # 形状与精度
          f"min={canvas.min():.1f} max={canvas.max():.1f} sum={canvas.sum():.1f}")  # 数值摘要
    print("  行 0 = 最低 Y（左下原点），全程未做图片翻转")  # 方向声明
    mask = torch.from_numpy(canvas)  # 转 torch 张量（CPU 侧，模型内部再搬运）
    # 阶段 2：加载模型与四个资产 buffer（只打印元数据，不打印张量内容）。
    print("阶段 2 · 模型加载")  # 阶段标题
    model = ICCAD13Lithography(device=device)  # auto=有 CUDA 用 CUDA 否则 CPU
    config = model.config  # 冻结数值配置
    print(f"  device={model.device}")  # 实际设备
    print(f"  kernel_count={config.kernel_count} canvas={config.canvas} "  # 网格契约
          f"resolution={config.resolution} print_threshold={config.print_threshold}")  # 二值阈值
    for name, buffer in model.named_buffers():  # 恰四个资产 buffer
        print(f"  buffer {name}: {tuple(buffer.shape)} {buffer.dtype}")  # 只看元数据
    # 阶段 3：三工艺条件一次前向（no_grad 纯推理）。
    print("阶段 3 · 三工艺条件推理（torch.no_grad）")  # 阶段标题
    conditions = [model.condition(name) for name in  # 三个默认条件
                  ("nominal", "dose_max", "defocus_min")]  # 标称/大剂量/离焦
    on_cuda = model.device.type == "cuda"  # GPU 才有显存峰值可报
    if on_cuda:  # GPU 计时前准备
        torch.cuda.synchronize()  # 排空此前异步操作
        torch.cuda.reset_peak_memory_stats()  # 重置峰值计量起点
    started = time.perf_counter()  # 前向计时起点
    with torch.no_grad():  # 推理不建 autograd 图
        images = model.forward_many(mask, conditions)  # 一次 mask FFT 三条件共享
    if on_cuda:  # 等 GPU 真正完成再停表
        torch.cuda.synchronize()  # 显式同步
    elapsed = time.perf_counter() - started  # 前向耗时
    threshold = config.print_threshold  # 二值化阈值（仅统计，不回写模型）
    for condition in conditions:  # 逐条件打印摘要
        image = images[condition.name]  # 该条件输出
        exposed = int((image >= threshold).sum().item())  # 二值曝光像素数
        print(f"  {condition.name:12s} kernel={condition.kernel:7s} "  # 条件与 bank
              f"dose={condition.dose:.2f} shape={tuple(image.shape)} "  # 剂量与形状
              f"range=[{image.min():.4f}, {image.max():.4f}] "  # 连续范围
              f"sum={image.sum():.1f} 曝光像素={exposed}")  # 总强度与二值统计
    print(f"  前向耗时 {elapsed * 1000:.1f} ms")  # 计时结果
    if on_cuda:  # 报告 GPU 显存峰值
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2  # 换算 MiB
        print(f"  CUDA peak allocated {peak:.1f} MiB")  # 峰值显存
    # 阶段 4：batch 前向——模型接受 [B,H,W] 但不决定 B。
    print("阶段 4 · batch")  # 阶段标题
    shifted = torch.roll(mask, shifts=(20, -30), dims=(0, 1))  # 平移变体
    variant = 0.6 * shifted + 0.4 * (1.0 - shifted)  # 压成 0.4~0.6 连续值
    batch = torch.stack((mask, variant))  # 组成 [2,256,256]
    with torch.no_grad():  # 推理
        batched = model(batch, model.condition("nominal"))  # 整批一次前向
    print(f"  输入 [2,256,256] → 输出 {tuple(batched.shape)}")  # 形状直通
    print("  模型不在内部拆分 batch；B 由调用方按显存决定")  # 边界声明
    # 阶段 5：真实 backward（原生 autograd，非均匀权重防对称掩盖）。
    print("阶段 5 · 真实 backward（原生 autograd）")  # 阶段标题
    leaf = mask.clone().requires_grad_(True)  # 复制为可求导叶子
    weights = torch.linspace(  # 非均匀上游权重
        -0.7, 1.3, mask.numel(), dtype=torch.float32).reshape_as(mask)  # 与画布同形
    weights = weights.to(model.device)  # 权重搬到输出所在设备
    loss = torch.sum(model(leaf, model.condition("nominal")) * weights)  # 标量损失
    loss.backward()  # 反向传播到 mask
    gradient = leaf.grad  # 输入梯度
    finite = bool(torch.all(torch.isfinite(gradient)).item())  # 有限性
    nonzero = int(torch.count_nonzero(gradient).item())  # 非零元素数
    norm = float(gradient.norm().item())  # L2 范数
    print(f"  梯度 finite={finite} 非零元素={nonzero}/{gradient.numel()} "  # 梯度摘要
          f"L2范数={norm:.4f}")  # 摘要续
    print("  仅演示梯度传播；不更新 mask，不引入优化器")  # 边界声明
    # 阶段 6：把光刻计算结果可视化（2×2 灰度面板：输入与三工艺角胶图）。
    print("阶段 6 · 可视化（matplotlib）")  # 阶段标题
    panels = (  # 四联面板数据：输入透光率 + 三个条件的连续 printed image
        ("输入 mask（1=透光）", canvas),
        ("nominal（focus × 1.00²）", images["nominal"].cpu().numpy()),
        ("dose_max（focus × 1.02²）", images["dose_max"].cpu().numpy()),
        ("defocus_min（defocus × 0.98²）", images["defocus_min"].cpu().numpy()))
    figure, axes = plt.subplots(  # 建 2×2 面板
        2, 2, figsize=(10, 9), layout="constrained")  # 紧凑排版不重叠
    for ax, (title, image) in zip(axes.flat, panels):  # 逐面板绘制
        drawn = ax.imshow(  # origin=lower 让行 0（最低 Y）显示在底部
            image, origin="lower", extent=(0, 1824, 0, 1824),  # 轴=context DBU 坐标
            cmap="gray", vmin=0.0, vmax=1.0)  # 固定灰阶便于跨面板对比
        ax.set_title(title)  # 面板标题
        ax.set_xlabel("context X（DBU）")  # 横轴说明
        ax.set_ylabel("context Y（DBU）")  # 纵轴说明
        figure.colorbar(drawn, ax=ax, shrink=0.85)  # 灰阶色条
    output_dir = _REPO_ROOT / "output" / "lithography"  # 留档目录（gitignored）
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    figure_path = output_dir / "main_test_lithography.png"  # PNG 产物路径
    figure.savefig(str(figure_path), dpi=150)  # 写盘留档（锚定仓库根，不依赖 cwd）
    print(f"  已保存 {figure_path}")  # 打印留档路径
    plt.show()  # 弹窗交互查看；测试用 Agg 无头后端时此调用不阻塞
    print("=" * 72)  # 演示结束


def main() -> int:
    """在自动选择的设备上运行完整正常流程，成功返回 0。"""
    run_demo(device="auto")  # 自动设备选择
    return 0  # 成功退出码


if __name__ == "__main__":  # 直接运行入口
    raise SystemExit(main())  # 以 main 返回值退出
