"""
Mel Spectrogram Plotter for .npy files.

Converts uint8 [0, 255] mel spectrograms to dB scale [-80, 0] and displays
up to 4 files in a single figure with shared x-axis.

Usage:
    python plot_mels.py file1.npy
    python plot_mels.py file1.npy file2.npy
    python plot_mels.py file1.npy file2.npy file3.npy
"""

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib backend for interactive plotting.
matplotlib.use("TkAgg")  # Use "Qt5Agg" if PyQt5 is available.

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Expected data range for uint8 mel spectrograms.
UINT8_MIN: int = 0
UINT8_MAX: int = 255

# dB scale mapping: uint8 [0, 255] -> dB [-80, 0]
DB_MIN: float = -80.0
DB_MAX: float = 0.0

# Plotting limits.
MAX_FILES: int = 4
FIG_WIDTH: float = 10.0
FIG_HEIGHT_PER_PLOT: float = 3.0

# Colormap for mel spectrograms (magma is good for audio).
CMAP: str = "magma"

# ==============================================================================
# UTILS
# ==============================================================================


def validate_and_load_mel(path: Path) -> np.ndarray:
    """
    Load and validate a .npy mel spectrogram file.

    Validation checks:
    - File exists and is readable.
    - Shape has at least 2 dimensions (time, freq).
    - dtype is uint8 in expected range [0, 255].

    Parameters
    ----------
    path : pathlib.Path
        Path to the .npy file.

    Returns
    -------
    numpy.ndarray
        Loaded spectrogram as float32.

    Raises
    ------
    ValueError
        If validation fails.
    """
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    try:
        mel = np.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to load {path}: {exc}") from exc

    if mel.ndim < 2:
        raise ValueError(f"Invalid shape {mel.shape} in {path} (expected 2D or more)")

    if mel.dtype != np.uint8:
        raise ValueError(f"Unexpected dtype {mel.dtype} in {path} (expected uint8)")

    if (mel < UINT8_MIN).any() or (mel > UINT8_MAX).any():
        raise ValueError(f"Values outside [0, 255] in {path}")

    # Squeeze extra dimensions and cast to float32 for processing.
    mel_squeezed = np.squeeze(mel).astype(np.float32)
    return mel_squeezed


def uint8_to_db(mel: np.ndarray, db_min: float = DB_MIN, db_max: float = DB_MAX) -> np.ndarray:
    """
    Convert uint8 mel spectrogram [0, 255] to dB scale [db_min, db_max].

    Linear scaling formula:
        db = (uint8 / 255) * (db_max - db_min) + db_min

    Parameters
    ----------
    mel : numpy.ndarray
        uint8 spectrogram.
    db_min : float
        Minimum dB value (default: -80.0).
    db_max : float
        Maximum dB value (default: 0.0).

    Returns
    -------
    numpy.ndarray
        dB-scaled spectrogram.
    """
    normalized = mel / UINT8_MAX
    return normalized * (db_max - db_min) + db_min


# ==============================================================================
# PLOTTING
# ==============================================================================


def plot_mel_spectrograms(
    files: List[Path],
    figsize: tuple[float, float],
    cmap: str = CMAP,
) -> None:
    """
    Create a single figure with subplots for multiple mel spectrograms.

    Features:
    - Shared x-axis for easy comparison.
    - Individual colorbars per subplot.
    - Proper labels and layout.
    - Tight layout and automatic sizing.

    Parameters
    ----------
    files : list of pathlib.Path
        List of .npy files to plot (1-4).
    figsize : tuple
        Figure size (width, height).
    cmap : str
        Matplotlib colormap name.
    """
    n_files = len(files)
    if n_files == 0:
        print("No files to plot.")
        return

    # Create subplots: n_files rows, 1 column.
    fig, axes = plt.subplots(n_files, 1, figsize=figsize, sharex=True)
    if n_files == 1:
        axes = [axes]  # Ensure axes is always iterable.

    for ax, path in zip(axes, files):
        # Load and convert to dB.
        mel = validate_and_load_mel(path)
        mel_db = uint8_to_db(mel)

        # Plot as imshow with audio-friendly settings.
        im = ax.imshow(
            mel_db,
            origin="lower",  # Time increases left-to-right, freq low-to-high.
            aspect="auto",   # Stretch to fill subplot height.
            cmap=cmap,
            interpolation="nearest",  # Crisp pixels for spectrograms.
        )

        # Labels and title.
        ax.set_title(str(path), fontsize=10, pad=10)
        fig.colorbar(im, ax=ax, format="%+2.1f dB", shrink=0.8)

    # Global labels.
    axes[-1].set_xlabel("Time Frames")
    axes[0].set_ylabel("Mel Frequency Bins")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# CLI INTERFACE
# ==============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Supports:
    - Positional file arguments (1-4).
    - --help for usage info.
    - File validation (must be .npy).
    """
    parser = argparse.ArgumentParser(
        description="Plot mel spectrograms from uint8 .npy files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_mels.py audio1.npy
  python plot_mels.py audio1.npy audio2.npy
  python plot_mels.py audio1.npy audio2.npy audio3.npy
        """,
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="file.npy",
        type=Path,
        help="Path to .npy mel spectrogram file (1-4 files)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=MAX_FILES,
        help=f"Maximum number of files to plot (default: {MAX_FILES})",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    # Validate number of files.
    if len(args.files) > args.max_files:
        print(
            f"Error: Too many files. Maximum is {args.max_files}, got {len(args.files)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate all files are .npy.
    for path in args.files:
        if path.suffix != ".npy":
            print(f"Error: '{path}' is not a .npy file.", file=sys.stderr)
            sys.exit(1)

    # Calculate figure size dynamically.
    figsize = (FIG_WIDTH, FIG_HEIGHT_PER_PLOT * len(args.files))

    try:
        plot_mel_spectrograms(args.files, figsize)
    except Exception as exc:  # noqa: BLE001
        print(f"Error while plotting: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
