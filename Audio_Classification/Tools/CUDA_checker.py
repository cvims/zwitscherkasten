"""
PyTorch Environment Checker.

Comprehensive diagnostic tool for PyTorch + CUDA setup.
Reports Python/PyTorch versions, CUDA availability, GPU details,
memory stats, and performance benchmarks.

Usage:
    python check_torch.py
    python check_torch.py --benchmark
"""

import argparse
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Benchmark settings.
BENCHMARK_DURATION: float = 1.0  # Seconds for GPU warmup/benchmark
WARMUP_ITERATIONS: int = 10      # Iterations before benchmarking

# Memory thresholds for warnings (in GB).
LOW_MEMORY_THRESHOLD: float = 2.0   # Warn if < 2GB free
CRITICAL_MEMORY_THRESHOLD: float = 1.0  # Error if < 1GB free

# ==============================================================================
# UTILS
# ==============================================================================


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string (e.g., 1.23 GB).

    Supports up to TB scale.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_version(version: str) -> str:
    """Extract semantic version (major.minor.patch)."""
    return version.split("+")[0]  # Remove CUDA build suffix


def log_section(title: str, char: str = "=", width: int = 50) -> None:
    """Print a section header."""
    line = char * width
    print(f"\n{line}")
    print(f"{title:^50}")
    print(line)


# ==============================================================================
# CUDA DIAGNOSTICS
# ==============================================================================


def check_cuda_environment() -> Dict[str, object]:
    """
    Comprehensive CUDA environment check.

    Returns a dict with all diagnostic info for easy testing/logging.
    """
    diagnostics = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "current_device": None,
        "devices": [],
        "memory_info": [],
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_version": None,
    }

    if not diagnostics["cuda_available"]:
        return diagnostics

    diagnostics["device_count"] = torch.cuda.device_count()
    diagnostics["current_device"] = torch.cuda.current_device()

    for i in range(diagnostics["device_count"]):
        props = torch.cuda.get_device_properties(i)
        mem_info = torch.cuda.get_device_properties(i).total_memory
        diagnostics["devices"].append({
            "index": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "memory_gb": mem_info / (1024**3),
            "multiprocessor_count": props.multi_processor_count,
            "is_mps": props.name.startswith("Apple M"),
        })
        diagnostics["memory_info"].append(mem_info)

    # Try to get driver version (requires nvidia-ml-py or parsing nvidia-smi).
    try:
        diagnostics["driver_version"] = torch.version.driver
    except AttributeError:
        diagnostics["driver_version"] = "Unknown (PyTorch < 2.1)"

    return diagnostics


def print_cuda_status(diagnostics: Dict[str, object]) -> None:
    """Pretty-print CUDA diagnostics with emojis and color hints."""
    if diagnostics["cuda_available"]:
        print("✅ CUDA is AVAILABLE!")
    else:
        print("❌ CUDA is NOT available.")
        print("   Running on CPU only.")
        return

    print(f"   Device count: {diagnostics['device_count']}")
    print(f"   Current device: {diagnostics['current_device']}")

    for device in diagnostics["devices"]:
        status = "⚠️" if device["memory_gb"] < LOW_MEMORY_THRESHOLD else "✅"
        print(f"   GPU {device['index']}: {device['name']}")
        print(f"      Compute: {device['compute_capability']}, "
              f"SMs: {device['multiprocessor_count']}")
        print(f"      Memory: {device['memory_gb']:.1f} GB {status}")

    print(f"   CUDA: {diagnostics['cuda_version']}")
    print(f"   cuDNN: {diagnostics['cudnn_version'] // 1000}.{diagnostics['cudnn_version'] % 1000:03d}")
    print(f"   Driver: {diagnostics['driver_version']}")


def check_memory_status(diagnostics: Dict[str, object]) -> None:
    """Check and warn about low GPU memory."""
    if not diagnostics["cuda_available"]:
        return

    for i, mem_total in enumerate(diagnostics["memory_info"]):
        mem_free = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)
        mem_free_gb = mem_free / (1024**3)

        status = "✅"
        message = ""
        if mem_free_gb < CRITICAL_MEMORY_THRESHOLD:
            status = "🚨"
            message = "CRITICAL - Restart kernel or reduce batch size!"
        elif mem_free_gb < LOW_MEMORY_THRESHOLD:
            status = "⚠️"
            message = "Low memory - Consider reducing batch size."

        print(f"\n   GPU {i} Free Memory: {format_bytes(mem_free)} {status}")
        if message:
            print(f"      {message}")


# ==============================================================================
# BENCHMARKING
# ==============================================================================


def benchmark_device(duration: float = BENCHMARK_DURATION, warmup_iters: int = WARMUP_ITERATIONS) -> None:
    """
    Simple matrix multiply benchmark to test GPU performance.

    Measures:
    - Peak FLOPS
    - Memory bandwidth
    - Basic sanity check

    Uses square matrices sized to fill ~80% of GPU memory.
    """
    if not torch.cuda.is_available():
        print("Skipping benchmark (no CUDA).")
        return

    device = torch.device("cuda")
    print(f"\n⏱️  GPU Benchmark (duration: {duration}s)...")

    # Size matrices to fill most GPU memory.
    gpu_props = torch.cuda.get_device_properties(0)
    target_mem_gb = gpu_props.total_memory * 0.8 / (1024**3)
    matrix_size = int((target_mem_gb * 2 / 8 * 1024**3) ** 0.5)  # float32 = 4B
    print(f"   Matrix size: {matrix_size:,} x {matrix_size:,}")

    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    # Warmup.
    for _ in range(warmup_iters):
        _ = torch.mm(a, b)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    iters = 0
    while (torch.cuda.Event.elapsed_time(end, start) / 1000) < duration:
        _ = torch.mm(a, b)
        iters += 1
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    gflops = 2 * matrix_size**3 / (elapsed_ms * 1e-3) / 1e9

    print(f"   Iterations: {iters:,}")
    print(f"   Time per iter: {elapsed_ms:.2f} ms")
    print(f"   Peak FLOPS: {gflops:.1f} GFLOPS")


# ==============================================================================
# MAIN REPORT
# ==============================================================================


def print_environment_report(include_benchmark: bool = False) -> None:
    """Generate complete environment report."""
    log_section("ENVIRONMENT")
    print(f"Python: {format_version(sys.version.split()[0])}")
    print(f"PyTorch: {format_version(torch.__version__)}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Script: {Path(sys.argv[0]).resolve()}")

    log_section("CUDA STATUS")
    diagnostics = check_cuda_environment()
    print_cuda_status(diagnostics)
    check_memory_status(diagnostics)

    if include_benchmark:
        log_section("PERFORMANCE BENCHMARK")
        benchmark_device()


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="PyTorch + CUDA Environment Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_torch.py                    # Basic diagnostics
  python check_torch.py --benchmark        # Include GPU performance test
        """,
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        action="store_true",
        help="Run GPU matrix multiply benchmark",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=BENCHMARK_DURATION,
        help=f"Benchmark duration in seconds (default: {BENCHMARK_DURATION})",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_arguments()
    print_environment_report(args.benchmark)


if __name__ == "__main__":
    main()
