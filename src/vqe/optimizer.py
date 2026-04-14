"""
vqe/optimizer.py

Classical optimizer configuration for VQE.

Optimizer guide:
    cobyla      Gradient-free. Robust. Great default for noise-free simulators.
                Works well with 10–500 parameters.

    spsa        Simultaneous Perturbation Stochastic Approximation.
                Gradient-free. Noise-robust. The best choice for real hardware.
                Can handle thousands of parameters.

    lbfgsb      Gradient-based (uses parameter-shift rule for quantum gradients).
                Fastest on noise-free simulators. Fails on noisy hardware.

    neldermead  Simplex gradient-free method. Slow but robust for small problems.

    adam        Adaptive gradient-based. Good for large parameter counts.
"""

from __future__ import annotations
import logging
from typing import Optional, Callable, List

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """
    Creates and configures classical optimizers for VQE.

    Examples
    --------
    >>> factory = OptimizerFactory()
    >>> optimizer = factory.create("cobyla", max_iterations=500)
    >>> optimizer = factory.create("spsa", max_iterations=300, learning_rate=0.01)
    """

    @staticmethod
    def create(
        name: str,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
        **kwargs,
    ):
        """
        Create a Qiskit-compatible optimizer.

        Parameters
        ----------
        name : str
            Optimizer name: "cobyla" | "spsa" | "lbfgsb" | "neldermead" | "adam"
        max_iterations : int
            Maximum number of function evaluations / iterations
        convergence_threshold : float
            Stop when |ΔE| < threshold between iterations
        **kwargs
            Extra optimizer-specific settings (see individual methods below)
        """
        name = name.lower()
        creators = {
            "cobyla": OptimizerFactory._cobyla,
            "spsa": OptimizerFactory._spsa,
            "lbfgsb": OptimizerFactory._lbfgsb,
            "neldermead": OptimizerFactory._neldermead,
            "adam": OptimizerFactory._adam,
            "slsqp": OptimizerFactory._slsqp,
        }

        if name not in creators:
            raise ValueError(
                f"Unknown optimizer '{name}'. "
                f"Choose from: {list(creators.keys())}"
            )

        optimizer = creators[name](
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            **kwargs,
        )

        logger.info(f"Optimizer: {name} | max_iter={max_iterations} | tol={convergence_threshold:.2e}")
        return optimizer

    @staticmethod
    def _cobyla(max_iterations, convergence_threshold, **kwargs):
        try:
            from qiskit_algorithms.optimizers import COBYLA
        except ImportError:
            raise ImportError("qiskit-algorithms is required. Run: pip install qiskit-algorithms")

        return COBYLA(
            maxiter=max_iterations,
            tol=convergence_threshold,
            rhobeg=kwargs.get("rhobeg", 1.0),
        )

    @staticmethod
    def _spsa(max_iterations, convergence_threshold, **kwargs):
        """
        SPSA with optional learning rate and perturbation schedules.
        For hardware runs, start with max_iterations=300.
        """
        try:
            from qiskit_algorithms.optimizers import SPSA
        except ImportError:
            raise ImportError("qiskit-algorithms is required. Run: pip install qiskit-algorithms")

        return SPSA(
            maxiter=max_iterations,
            # a, c: learning rate and perturbation magnitude (auto-calibrated if None)
            learning_rate=kwargs.get("learning_rate", None),
            perturbation=kwargs.get("perturbation", None),
            termination_checker=_make_termination_checker(convergence_threshold),
        )

    @staticmethod
    def _lbfgsb(max_iterations, convergence_threshold, **kwargs):
        try:
            from qiskit_algorithms.optimizers import L_BFGS_B
        except ImportError:
            raise ImportError("qiskit-algorithms is required. Run: pip install qiskit-algorithms")

        return L_BFGS_B(
            maxiter=max_iterations,
            ftol=convergence_threshold,
            gtol=convergence_threshold * 1e-3,
            iprint=kwargs.get("iprint", -1),
        )

    @staticmethod
    def _neldermead(max_iterations, convergence_threshold, **kwargs):
        try:
            from qiskit_algorithms.optimizers import NELDER_MEAD
        except ImportError:
            raise ImportError("qiskit-algorithms is required. Run: pip install qiskit-algorithms")

        return NELDER_MEAD(
            maxiter=max_iterations,
            tol=convergence_threshold,
            adaptive=kwargs.get("adaptive", True),
        )

    @staticmethod
    def _adam(max_iterations, convergence_threshold, **kwargs):
        try:
            from qiskit_algorithms.optimizers import ADAM
        except ImportError:
            raise ImportError("qiskit-algorithms is required. Run: pip install qiskit-algorithms")

        return ADAM(
            maxiter=max_iterations,
            tol=convergence_threshold,
            lr=kwargs.get("lr", 0.01),
            beta_1=kwargs.get("beta_1", 0.9),
            beta_2=kwargs.get("beta_2", 0.99),
            noise_factor=kwargs.get("noise_factor", 1e-8),
        )

    @staticmethod
    def _slsqp(max_iterations, convergence_threshold, **kwargs):
        try:
            from qiskit_algorithms.optimizers import SLSQP
        except ImportError:
            raise ImportError("qiskit-algorithms is required. Run: pip install qiskit-algorithms")

        return SLSQP(
            maxiter=max_iterations,
            tol=convergence_threshold,
            ftol=convergence_threshold,
        )

    @staticmethod
    def recommended_for_backend(backend: str) -> str:
        """Return the recommended optimizer for a given backend."""
        if "ibm_real" in backend:
            return "spsa"      # noise-robust
        elif "noise" in backend:
            return "spsa"
        else:
            return "cobyla"    # fast, gradient-free, good default


def _make_termination_checker(threshold: float) -> Callable:
    """
    Creates a termination function for SPSA that stops when
    the energy change is smaller than the threshold.
    """
    previous_value = [None]

    def checker(nfev, x, fx, dx, metadata) -> bool:
        if previous_value[0] is not None:
            delta = abs(fx - previous_value[0])
            if delta < threshold:
                logger.debug(f"SPSA converged: |ΔE|={delta:.2e} < {threshold:.2e}")
                return True
        previous_value[0] = fx
        return False

    return checker
