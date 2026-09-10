from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .common import synchronize_device, write_csv


def square_wave_np(t: np.ndarray, f0: float = 1.0) -> np.ndarray:
    # 理想方波只根据 sine 的正负取 +1/-1，这里作为 Fourier 重构图的 ground truth。
    return np.sign(np.sin(2.0 * np.pi * f0 * t))


def square_wave_fourier_np(t: np.ndarray, f0: float = 1.0, n_harmonics: int = 50) -> np.ndarray:
    # 用 Fourier series 近似方波：叠加越多奇次谐波，边缘越陡，但跳变处会有 Gibbs ringing。
    result = np.zeros_like(t, dtype=np.float64)
    for k in range(n_harmonics):
        # 方波的 Fourier series 只包含奇次谐波：1, 3, 5, ...
        n = 2 * k + 1
        result += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result

#NumPy 普通 DFT，直接实现公式，外层循环计算全部 N 个频率，内层循环读取全部 N 个采样点
def naive_dft_np(x: np.ndarray) -> np.ndarray:
    n_samples = len(x)
    result = np.zeros(n_samples, dtype=np.complex128)
    # 这里故意不用 np.fft.fft，因为任务要求展示按公式直接实现 DFT 的代价。
    # 外层 k 是频率 bin，内层 n 是时域采样点，每一个频率都要扫完整个信号。
    for k in range(n_samples):
        for n in range(n_samples):
            # 直接按 DFT 定义双重循环计算，复杂度是 O(N^2)。
            angle = -2j * np.pi * k * n / n_samples
            result[k] += x[n] * np.exp(angle)
    return result


def square_wave_torch(t, f0: float = 1.0):
    import torch

    # PyTorch 版本和 NumPy 版本等价，方便后面在 CPU/GPU tensor 上生成信号。
    return torch.sign(torch.sin(2.0 * math.pi * f0 * t))


def square_wave_fourier_torch(t, f0: float = 1.0, n_harmonics: int = 50):
    import torch

    result = torch.zeros_like(t)
    # 这个函数保留为 tensor 版本，说明同样的数学表达式可以迁移到 PyTorch。
    for k in range(n_harmonics):
        n = 2 * k + 1
        result = result + torch.sin(2 * math.pi * n * f0 * t) / n
    return (4 / math.pi) * result

#PyTorch DFT 核心，使用 PyTorch tensor 和矩阵乘法实现 DFT，没有调用 torch.fft.fft
def torch_matrix_dft(x, device=None):
    """DFT implemented by explicit tensor matrix multiplication, not torch.fft."""
    import torch

    if device is None:
        device = x.device
    # 这里使用 float64/complex128，避免大矩阵累积误差导致和 NumPy FFT 对不上。
    x = x.to(device=device, dtype=torch.float64)
    n_samples = x.shape[0]
    n = torch.arange(n_samples, device=device, dtype=torch.float64)
    k = n[:, None]
    # DFT 矩阵的第 (k,n) 项是 exp(-j*2*pi*k*n/N)，矩阵乘信号后得到所有频率 bin。
    # 这种实现仍是 O(N^2)，但把循环交给底层矩阵运算，能体现 tensor library 的加速能力。
    angle = -2.0 * math.pi * k * n[None, :] / n_samples
    matrix = torch.cos(angle).to(torch.complex128) + 1j * torch.sin(angle).to(torch.complex128)
    return matrix @ x.to(torch.complex128)

#测试运行时间
def _time_call(fn, device=None) -> tuple[float, object]:
    # GPU 操作默认是异步的，所以计时前后都 synchronize，避免只测到提交 kernel 的时间。
    synchronize_device(device)
    start = time.perf_counter()
    result = fn()
    synchronize_device(device)
    return time.perf_counter() - start, result

#比较三种方法
def benchmark_dft(
    sample_sizes: list[int],
    n_harmonics: int = 50,
    max_naive_n: int = 2048,
    include_gpu: bool = True,
) -> list[dict[str, str | int | float | bool]]:
    import torch

    rows: list[dict[str, str | int | float | bool]] = []
    for n_samples in sample_sizes:
        # endpoint=False 避免 0 和 1 两个周期端点重复采样，频谱会更干净。
        t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        signal = square_wave_fourier_np(t, f0=1.0, n_harmonics=n_harmonics)

        # NumPy FFT 是库函数优化实现，理论复杂度约 O(N log N)，作为正确性基准。
        fft_time, fft_result = _time_call(lambda: np.fft.fft(signal))
        rows.append(
            {
                "sample_count": n_samples,
                "method": "numpy_fft",
                "seconds": fft_time,
                "close_to_numpy_fft": True,
                "device": "cpu",
            }
        )

        if n_samples <= max_naive_n:
            # naive DFT 很慢，所以只在样本数不太大时运行。
            naive_time, naive_result = _time_call(lambda: naive_dft_np(signal))
            rows.append(
                {
                    "sample_count": n_samples,
                    "method": "numpy_naive_dft",
                    "seconds": naive_time,
                    "close_to_numpy_fft": bool(np.allclose(naive_result, fft_result, atol=1e-7)),
                    "device": "cpu",
                }
            )

        x_cpu = torch.from_numpy(signal.astype(np.float32))
        # PyTorch 版本用显式 DFT 矩阵乘法，展示 tensor/matrix 运算和 Python 循环的区别。
        torch_cpu_time, torch_cpu_result = _time_call(lambda: torch_matrix_dft(x_cpu, torch.device("cpu")))
        rows.append(
            {
                "sample_count": n_samples,
                "method": "torch_matrix_dft",
                "seconds": torch_cpu_time,
                "close_to_numpy_fft": bool(
                    #np.allclose() 比较其他方法与 NumPy FFT
                    np.allclose(torch_cpu_result.detach().cpu().numpy(), fft_result, atol=1e-3)
                ),
                "device": "cpu",
            }
        )

        if include_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            x_gpu = x_cpu.to(device)
            # 如果在 Rangpur/A100 上运行，这一行会产生 GPU 计时；本地 Mac 通常只会显示 CPU。
            gpu_time, gpu_result = _time_call(lambda: torch_matrix_dft(x_gpu, device), device=device)
            rows.append(
                {
                    "sample_count": n_samples,
                    "method": "torch_matrix_dft",
                    "seconds": gpu_time,
                    "close_to_numpy_fft": bool(
                        np.allclose(gpu_result.detach().cpu().numpy(), fft_result, atol=1e-3)
                    ),
                    "device": torch.cuda.get_device_name(device),
                }
            )
    return rows


def save_dft_plots(output: str | Path, sample_count: int = 2048, n_harmonics: int = 50) -> None:
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    square = square_wave_np(t)

    # 第一张图展示谐波数量增加时方波重构如何逐渐变好，适合 Part 1 讲 Fourier series。
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(t, square, "k", label="square wave")
    ax.set_title("Original square wave")
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True)
    ax.legend()
    for i, harmonics in enumerate([1, 3, 5, 20, 50], start=2):
        ax = fig.add_subplot(2, 3, i)
        y = square_wave_fourier_np(t, n_harmonics=harmonics)
        ax.plot(t, y, label=f"{harmonics} harmonics")
        ax.plot(t, square, "k--", alpha=0.5, label="square wave")
        ax.set_title(f"Fourier approximation: {harmonics}")
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "part1_square_wave_harmonics.png", dpi=160)
    plt.close(fig)

    # 第二张图展示 DFT magnitude spectrum：方波能量主要集中在奇数频率位置。
    signal = square_wave_fourier_np(t, n_harmonics=n_harmonics)
    spectrum = np.fft.fft(signal)
    xf = np.fft.fftfreq(sample_count, d=1.0 / sample_count)[: sample_count // 2]
    magnitude = 2.0 / sample_count * np.abs(spectrum[: sample_count // 2])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(t, signal, color="c")
    ax1.set_title(f"Square wave reconstructed with {n_harmonics} harmonics")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    ax2.stem(xf, magnitude, basefmt=" ")
    ax2.set_title("Discrete Fourier transform magnitude spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlim(0, min(60, sample_count // 2))
    ax2.grid(True)
    for harmonic in range(1, min(40, sample_count // 2), 2):
        ax2.axvline(harmonic, color="r", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "part1_dft_spectrum.png", dpi=160)
    plt.close(fig)


def write_dft_timings(path: str | Path, rows: list[dict[str, str | int | float | bool]]) -> Path:
    return write_csv(path, rows)
