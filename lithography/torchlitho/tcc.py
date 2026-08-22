"""TorchLitho Hopkins 传输交叉系数（TCC）构造与本征核分解（Option 2 忠实迁移）。

迁移自 TorchLitho-2.0 pylitho/sim/hopkins/tcc.py：randomized SVD 与 TCC 大矩阵
构造保持原数值链；genTCC 的 size>64 频域 resize 分支中 cv2.resize 用 torch
F.interpolate(bilinear) 同构替换（MyOPC 环境无 opencv，插值核差异在一致性
测试报告中量化）。
"""

from __future__ import annotations

import numpy as np
import torch

from .source import frequency_grid, pupil_function, source_points

# TCC 网格边长上限（原库 MAX_TCC_SIZE）：限制 [N²,N²] TCC 矩阵的内存规模。
MAX_TCC_SIZE = 64

# randomized SVD 保留分量数与随机参数（原库写死；seed 固定保证确定性）。
_SVD_COMPONENTS = 64
_SVD_SEED = 0


def randomized_svd(
    matrix: np.ndarray, n_components: int, n_iter: int = 4, n_oversamples: int = 16, seed: int = _SVD_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算实/复 numpy 矩阵的截断 SVD（随机化范围搜索算法，支持复数）。

    scikit-learn 的实现拒绝复数输入，而 Hopkins 的 TCC 矩阵是复数；本实现
    使用带共轭转置的标准随机范围搜索，与原库逐算法一致。
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    max_rank = min(matrix.shape)
    n_components = min(n_components, max_rank)
    sample_size = min(n_components + n_oversamples, max_rank)
    rng = np.random.default_rng(seed)
    sample = rng.standard_normal((matrix.shape[1], sample_size))
    if np.iscomplexobj(matrix):
        sample = sample + 1j * rng.standard_normal(sample.shape)
    sample = sample.astype(matrix.dtype, copy=False)
    basis, _ = np.linalg.qr(matrix @ sample, mode="reduced")
    for _ in range(n_iter):
        basis, _ = np.linalg.qr(matrix.conj().T @ basis, mode="reduced")
        basis, _ = np.linalg.qr(matrix @ basis, mode="reduced")
    projected = basis.conj().T @ matrix
    projected_u, singular_values, vh = np.linalg.svd(projected, full_matrices=False)
    u = basis @ projected_u
    return u[:, :n_components], singular_values[:n_components], vh[:n_components, :]


def compute_tcc(src: np.ndarray, pupil: np.ndarray, thresh: float = 1.0e-6) -> tuple[list[np.ndarray], list[float]]:
    """构造 [N²,N²] TCC 矩阵并做本征分解，返回超过阈值的核与权重。

    块构造公式与原库一致：w[i,j] = flip(roll(J,(i,j))) · h[i,j] · h* / N⁴，
    其中 J = fftshift(fft2(src/Σsrc))、h = fftshift(fft2(pupil))；核 = 左奇异
    向量按 size² 重排。数值精度链对齐原库：瞳/源按 float64/complex128 计算，
    TCC 大矩阵按 complex64 存储（原库 w 的 dtype）。双循环保持原结构
    （N=64 时约 0.1 秒，仅构造期执行）。
    """
    src = np.asarray(src, dtype=np.float64)
    pupil = np.asarray(pupil, dtype=np.complex128)
    size = pupil.shape[0]
    pupil_fft = np.fft.fftshift(np.fft.fft2(pupil))
    pupil_star = pupil_fft.conj()
    src_fft = np.fft.fftshift(np.fft.fft2(src / np.sum(src)))
    # 大矩阵峰值内存：N=64 时 [4096,4096] complex64 ≈ 128 MiB，函数返回后释放。
    blocks = np.zeros((size, size, size, size), dtype=np.complex64)
    scale = float(np.prod(pupil.shape) * np.prod(src.shape))
    for idx in range(size):
        for jdx in range(size):
            src_shifted = np.flip(np.roll(src_fft, shift=(idx, jdx), axis=(0, 1)), axis=(0, 1))
            blocks[idx, jdx] = src_shifted * pupil_fft[idx, jdx] * pupil_star / scale
    matrix = blocks.reshape(size * size, size * size)
    mat_u, mat_s, _ = randomized_svd(matrix, n_components=_SVD_COMPONENTS)
    phis: list[np.ndarray] = []
    weights: list[float] = []
    for idx, weight in enumerate(mat_s):
        if weight >= thresh:
            phis.append(mat_u[:, idx].reshape(size, size) * (size * size))
            weights.append(float(weight))
    return phis, weights


def build_tcc_kernels(
    shape: str,
    sigma: float,
    pole_center: float,
    na: float,
    wavelength_nm: float,
    refractive_index: float,
    size: int,
    pixel_nm: float,
    defocus_nm: float,
    thresh: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成一个 defocus 值的 Hopkins 本征核，返回 phis [K,size,size] complex64 与 weights [K]。

    网格缩放规则与原库 genTCC 一致：仿真网格超过 64 时优先把 TCC 像素翻倍
    （padding 因子），视场超过 2048nm 才把 TCC 画布减半（resize 因子）；TCC
    恒在 ≤64 的网格上构造，随后把低分辨核的频谱零嵌入（必要时双线性插值）
    还原到仿真网格，末尾乘 padding²·resize² 补偿。
    """
    canvas_nm = size * pixel_nm
    tcc_canvas_nm, tcc_pixel_nm = float(canvas_nm), float(pixel_nm)
    resize, padding = 1, 1
    while tcc_canvas_nm / tcc_pixel_nm > MAX_TCC_SIZE:
        if tcc_canvas_nm > 2048.0:
            tcc_canvas_nm /= 2.0
            resize *= 2
        else:
            tcc_pixel_nm *= 2.0
            padding *= 2
    tcc_size = round(tcc_canvas_nm / tcc_pixel_nm)
    if tcc_size * padding != size and resize == 1:
        # padding 路径要求 n·padding 恰等于 size（像素翻倍不改变视场）；不成立
        # 说明 size 不是 2 的整数倍缩放链可达，属于未定义的网格组合。
        raise ValueError(f"TCC 网格缩放链无法对齐仿真网格：size={size}, n={tcc_size}, padding={padding}")
    # TCC 网格频率坐标（格距 1/tcc_canvas_nm，float64 对齐原库精度）；瞳与源
    # 掩膜与仿真网格共用同一套物理判定函数，保证两方法描述同一光源与光瞳。
    cpu = torch.device("cpu")
    fx, fy, freq = frequency_grid(tcc_size, tcc_pixel_nm, cpu, dtype=torch.float64)
    pupil = pupil_function(freq, na, wavelength_nm, defocus_nm, refractive_index).numpy()
    _, src_mask = source_points(shape, fx, fy, sigma, pole_center, na, wavelength_nm)
    phis_low, weights = compute_tcc(src_mask.to(torch.float64).numpy(), pupil, thresh)
    # 频域还原：fftshift(fft2(fftshift(phi))) → 居中零嵌入 [n·padding]² →（必要时
    # 双线性插值到 size）→ fftshift(ifft2(fftshift(·)))；同尺寸时跳过插值（恒等）。
    padded_size = tcc_size * padding
    phis = torch.empty((len(phis_low), size, size), dtype=torch.complex64)
    for idx, phi_low in enumerate(phis_low):
        low = torch.from_numpy(np.asarray(phi_low, dtype=np.complex64))
        spectrum = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(low)))
        padded = torch.zeros((padded_size, padded_size), dtype=torch.complex64)
        begin = (padded_size - tcc_size) // 2
        padded[begin : begin + tcc_size, begin : begin + tcc_size] = spectrum
        if padded_size != size:
            real = torch.nn.functional.interpolate(padded.real[None, None], size=(size, size), mode="bilinear")[0, 0]
            imag = torch.nn.functional.interpolate(padded.imag[None, None], size=(size, size), mode="bilinear")[0, 0]
            padded = torch.complex(real, imag)
        restored = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(padded)))
        phis[idx] = (restored * float(padding * padding * resize * resize)).to(torch.complex64)
    return phis, torch.tensor(weights, dtype=torch.float32)
