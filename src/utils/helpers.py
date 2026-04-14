"""
utils/helpers.py

Shared utilities: logging setup, pretty printing, result export,
and convergence analysis tools.
"""

from __future__ import annotations
import logging
import csv
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("quantum_drug_discovery")


def setup_logging(level: str = "INFO"):
    """Configure the root logger with a clean, readable format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("quantum_drug_discovery")
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)
    root_logger.propagate = False


def print_banner():
    """Print a startup banner."""
    banner = """
╔══════════════════════════════════════════════════════╗
║        Quantum Drug Discovery — VQE Pipeline         ║
║        Built with Qiskit + PySCF                     ║
╚══════════════════════════════════════════════════════╝
    """
    print(banner)
    logger.info(f"Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save a list of result dicts to a CSV file."""
    if not results:
        logger.warning("No results to save.")
        return

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved to {output_path} ({len(results)} rows)")


def hartree_to_kcal(hartree: float) -> float:
    """Convert energy from Hartree to kcal/mol."""
    return hartree * 627.5094740631


def hartree_to_ev(hartree: float) -> float:
    """Convert energy from Hartree to electron-Volts."""
    return hartree * 27.211386245988


def kcal_to_kj(kcal: float) -> float:
    """Convert kcal/mol to kJ/mol."""
    return kcal * 4.184


def check_dependencies():
    """
    Check whether all required packages are installed.
    Prints a clear report of what's missing.
    """
    required = {
        "qiskit": "pip install qiskit",
        "qiskit_aer": "pip install qiskit-aer",
        "qiskit_nature": "pip install qiskit-nature",
        "qiskit_algorithms": "pip install qiskit-algorithms",
        "pyscf": "pip install pyscf",
        "numpy": "pip install numpy",
        "scipy": "pip install scipy",
        "matplotlib": "pip install matplotlib",
    }
    optional = {
        "rdkit": "pip install rdkit  (for SMILES input)",
        "qiskit_ibm_runtime": "pip install qiskit-ibm-runtime  (for real hardware)",
        "tqdm": "pip install tqdm  (for progress bars)",
        "pubchempy": "pip install pubchempy  (for PubChem queries)",
    }

    missing_required = []
    missing_optional = []

    for pkg, install_cmd in required.items():
        try:
            __import__(pkg)
        except ImportError:
            missing_required.append((pkg, install_cmd))

    for pkg, install_cmd in optional.items():
        try:
            __import__(pkg)
        except ImportError:
            missing_optional.append((pkg, install_cmd))

    if missing_required:
        logger.error("Missing required packages:")
        for pkg, cmd in missing_required:
            logger.error(f"  {pkg:25s}  →  {cmd}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    else:
        logger.info(f"All required packages installed.")

    if missing_optional:
        logger.info("Optional packages not installed:")
        for pkg, cmd in missing_optional:
            logger.info(f"  {pkg:25s}  →  {cmd}")

    return True


def analyze_convergence(energy_history: list) -> dict:
    """
    Analyze the convergence behavior of a VQE run.

    Returns a dict with:
        - converged        : bool
        - convergence_iter : int (iteration at which convergence was detected)
        - final_energy     : float
        - energy_range     : float (max - min over last 20 iters)
        - is_noisy         : bool (high variance in last 20 iters)
    """
    import numpy as np

    if not energy_history:
        return {"converged": False, "error": "empty history"}

    energies = np.array(energy_history)
    final = energies[-1]
    tail = energies[-20:] if len(energies) >= 20 else energies

    tail_range = float(tail.max() - tail.min())
    is_noisy = tail_range > 1e-3
    converged = tail_range < 1e-6

    # Find convergence iteration
    convergence_iter = len(energies)
    for i in range(10, len(energies)):
        if abs(energies[i] - energies[i - 1]) < 1e-6:
            convergence_iter = i
            break

    return {
        "converged": converged,
        "convergence_iter": convergence_iter,
        "final_energy": float(final),
        "energy_range_last20": tail_range,
        "is_noisy": is_noisy,
        "total_iterations": len(energies),
        "best_energy": float(energies.min()),
    }


def plot_energy_landscape(
    energies_by_restart: List[List[float]],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot all VQE restarts on one chart to visualize energy landscape trapping.
    Different restarts are plotted in different colors.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not installed.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.tab10.colors

    for i, history in enumerate(energies_by_restart):
        color = colors[i % len(colors)]
        ax.plot(
            history,
            color=color,
            linewidth=1.2,
            alpha=0.75,
            label=f"Restart {i+1} (final: {history[-1]:.4f} Ha)",
        )

    best = min(h[-1] for h in energies_by_restart)
    ax.axhline(best, color="black", linestyle="--", linewidth=1, label=f"Best: {best:.6f} Ha")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Energy (Hartree)", fontsize=12)
    ax.set_title("VQE energy landscape — all restarts", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Energy landscape plot saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
