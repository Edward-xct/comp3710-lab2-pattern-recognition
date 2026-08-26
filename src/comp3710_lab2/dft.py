from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .common import synchronize_device, write_csv


def square_wave_np(t: np.ndarray, f0: float = 1.0) -> np.ndarray:
    return np.sign(np.sin(2.0 * np.pi * f0 * t))


def square_wave_fourier_np(t: np.ndarray, f0: float = 1.0, n_harmonics: int = 50) -> np.ndarray:
    result = np.zeros_like(t, dtype=np.float64)
    for k in range(n_harmonics):
        n = 2 * k + 1
        result += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result


def naive_dft_np(x: np.ndarray) -> np.ndarray:
    n_samples = len(x)
    result = np.zeros(n_samples, dtype=np.complex128)
    for k in range(n_samples):
        for n in range(n_samples):
            angle = -2j * np.pi * k * n / n_samples
            result[k] += x[n] * np.exp(angle)
    return result


def square_wave_torch(t, f0: float = 1.0):
    import torch

    return torch.sign(torch.sin(2.0 * math.pi * f0 * t))


def square_wave_fourier_torch(t, f0: float = 1.0, n_harmonics: int = 50):
    import torch

    result = torch.zeros_like(t)
    for k in range(n_harmonics):
        n = 2 * k + 1
        result = result + torch.sin(2 * math.pi * n * f0 * t) / n
    return (4 / math.pi) * result


def torch_matrix_dft(x, device=None):
    """DFT implemented by explicit tensor matrix multiplication, not torch.fft."""
    import torch

    if device is None:
        device = x.device
    x = x.to(device=device, dtype=torch.float32)
    n_samples = x.shape[0]
    n = torch.arange(n_samples, device=device, dtype=torch.float32)
    k = n[:, None]
    angle = -2.0 * math.pi * k * n[None, :] / n_samples
    matrix = torch.cos(angle).to(torch.complex64) + 1j * torch.sin(angle).to(torch.complex64)
    return matrix @ x.to(torch.complex64)


def _time_call(fn, device=None) -> tuple[float, object]:
    synchronize_device(device)
    start = time.perf_counter()
    result = fn()
    synchronize_device(device)
    return time.perf_counter() - start, result


def benchmark_dft(
    sample_sizes: list[int],
    n_harmonics: int = 50,
    max_naive_n: int = 2048,
    include_gpu: bool = True,
) -> list[dict[str, str | int | float | bool]]:
    import torch

    rows: list[dict[str, str | int | float | bool]] = []
    for n_samples in sample_sizes:
        t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        signal = square_wave_fourier_np(t, f0=1.0, n_harmonics=n_harmonics)

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
        torch_cpu_time, torch_cpu_result = _time_call(lambda: torch_matrix_dft(x_cpu, torch.device("cpu")))
        rows.append(
            {
                "sample_count": n_samples,
                "method": "torch_matrix_dft",
                "seconds": torch_cpu_time,
                "close_to_numpy_fft": bool(
                    np.allclose(torch_cpu_result.detach().cpu().numpy(), fft_result, atol=1e-3)
                ),
                "device": "cpu",
            }
        )

        if include_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            x_gpu = x_cpu.to(device)
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
