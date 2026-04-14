"""
vqe/runner.py

Main VQE execution engine. Orchestrates:
    1. Backend selection (simulator vs IBM hardware)
    2. Hamiltonian encoding
    3. Ansatz construction
    4. Optimizer configuration
    5. VQE loop with multiple restarts
    6. Optional Zero Noise Extrapolation post-processing
    7. Result analysis and convergence plotting

Robustness features:
    - Automatic retry on transient hardware/runtime errors
    - Adaptive active space selection when qubit count is too large
    - Graceful fallback from noisy → statevector backend
    - Convergence detection using moving-average window
"""

from __future__ import annotations
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

HARTREE_TO_KCAL = 627.5094740631

# Qiskit 1.x changed the StatevectorEstimator to use (circuit, observable, params) pubs
_QISKIT_V1 = False
try:
    import qiskit
    _QISKIT_V1 = int(qiskit.__version__.split(".")[0]) >= 1
except Exception:
    pass


@dataclass
class VQEResult:
    """Complete result from a VQE run."""
    ground_energy: float                   # in Hartree
    ground_energy_kcal: float             # in kcal/mol
    optimal_params: np.ndarray
    num_iterations: int
    num_function_evaluations: int
    converged: bool
    num_qubits: int
    num_parameters: int
    ansatz_type: str
    optimizer_name: str
    backend: str
    runtime_seconds: float
    energy_history: List[float] = field(default_factory=list)
    restart_energies: List[float] = field(default_factory=list)
    zne_energy: Optional[float] = None     # ZNE-corrected energy (if enabled)
    num_retries: int = 0

    def summary(self) -> str:
        lines = [
            f"{'─'*55}",
            f"  VQE Result Summary",
            f"{'─'*55}",
            f"  Ground energy    : {self.ground_energy:.6f} Ha",
            f"                   : {self.ground_energy_kcal:.2f} kcal/mol",
        ]
        if self.zne_energy is not None:
            lines.append(f"  ZNE energy       : {self.zne_energy:.6f} Ha")
        lines += [
            f"  Converged        : {'Yes' if self.converged else 'No'}",
            f"  Iterations       : {self.num_iterations}",
            f"  Fn evaluations   : {self.num_function_evaluations}",
            f"  Qubits           : {self.num_qubits}",
            f"  Parameters       : {self.num_parameters}",
            f"  Ansatz           : {self.ansatz_type}",
            f"  Optimizer        : {self.optimizer_name}",
            f"  Backend          : {self.backend}",
            f"  Runtime          : {self.runtime_seconds:.1f}s",
        ]
        if self.num_retries:
            lines.append(f"  Retries          : {self.num_retries}")
        lines.append(f"{'─'*55}")
        return "\n".join(lines)


class VQERunner:
    """
    Runs VQE to find the ground-state energy of a molecular Hamiltonian.

    Examples
    --------
    >>> runner = VQERunner(backend="aer_statevector", ansatz_type="efficient_su2")
    >>> result = runner.run(molecule_problem)
    >>> print(result.summary())
    >>> runner.plot_convergence()
    """

    def __init__(
        self,
        backend: str = "aer_statevector",
        ansatz_type: str = "efficient_su2",
        ansatz_reps: int = 2,
        optimizer_name: str = "cobyla",
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
        num_restarts: int = 3,
        initial_params_strategy: str = "random",
        qubit_mapping: str = "jordan_wigner",
        seed: Optional[int] = 42,
        ibm_token: Optional[str] = None,
        ibm_instance: str = "ibm-q/open/main",
        enable_zne: bool = False,
        zne_noise_factors: Optional[List[int]] = None,
        max_retries: int = 2,
        auto_active_space_max_qubits: int = 12,
    ):
        self.backend_name = backend
        self.ansatz_type = ansatz_type
        self.ansatz_reps = ansatz_reps
        self.optimizer_name = optimizer_name
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.num_restarts = num_restarts
        self.initial_params_strategy = initial_params_strategy
        self.qubit_mapping = qubit_mapping
        self.seed = seed
        self.ibm_token = ibm_token
        self.ibm_instance = ibm_instance
        self.enable_zne = enable_zne
        self.zne_noise_factors = zne_noise_factors or [1, 2, 3]
        self.max_retries = max_retries
        self.auto_active_space_max_qubits = auto_active_space_max_qubits

        self._last_result: Optional[VQEResult] = None

    def run(self, problem) -> VQEResult:
        """
        Run VQE on the given molecular problem.

        Parameters
        ----------
        problem : MoleculeProblem or ElectronicStructureProblem

        Returns
        -------
        VQEResult with ground energy, convergence info, and energy history.
        """
        from src.molecule.encoder import MoleculeEncoder
        from src.vqe.ansatz import AnsatzBuilder
        from src.vqe.optimizer import OptimizerFactory

        start_time = time.time()
        num_retries = 0

        # Step 1: Encode molecule as qubit Hamiltonian
        encoder = MoleculeEncoder(mapping=self.qubit_mapping)
        encoded = encoder.encode(problem)
        qubit_op = encoded.qubit_op

        logger.info(f"Hamiltonian: {encoded.num_qubits} qubits, {len(qubit_op)} Pauli terms")

        # Auto active space: if too many qubits, apply active space truncation
        if encoded.num_qubits > self.auto_active_space_max_qubits:
            logger.warning(
                f"Qubit count {encoded.num_qubits} exceeds limit "
                f"{self.auto_active_space_max_qubits}. "
                f"Applying automatic active space truncation."
            )
            problem, encoded, qubit_op = self._apply_auto_active_space(
                problem, encoded
            )

        # Step 2: Build ansatz circuit
        builder = AnsatzBuilder()
        if self.ansatz_type == "uccsd":
            ansatz = builder.build_uccsd(
                num_qubits=encoded.num_qubits,
                num_particles=encoded.num_particles,
                num_spatial_orbitals=encoded.num_spatial_orbitals,
            )
        else:
            ansatz = builder.build(
                ansatz_type=self.ansatz_type,
                num_qubits=encoded.num_qubits,
                reps=self.ansatz_reps,
            )

        # Step 3: Set up backend / estimator (with retry on failure)
        estimator = self._build_estimator_with_retry()

        # Step 4: Set up optimizer
        optimizer = OptimizerFactory.create(
            name=self.optimizer_name,
            max_iterations=self.max_iterations,
            convergence_threshold=self.convergence_threshold,
        )

        # Step 5: Run VQE with multiple restarts (avoids local minima)
        best_energy = float("inf")
        best_params = None
        best_energy_history = []
        restart_energies = []
        total_iters = 0
        total_evals = 0

        rng = np.random.default_rng(self.seed)

        for restart in range(self.num_restarts):
            logger.info(f"\n[Restart {restart + 1}/{self.num_restarts}]")

            seed_i = int(rng.integers(0, 100_000))
            initial_params = builder.get_initial_params(
                ansatz,
                strategy=self.initial_params_strategy,
                seed=seed_i,
            )

            # Retry loop for transient hardware/runtime errors
            for attempt in range(self.max_retries + 1):
                try:
                    energy, params, iters, evals, history = self._run_single(
                        estimator=estimator,
                        ansatz=ansatz,
                        qubit_op=qubit_op,
                        optimizer=optimizer,
                        initial_params=initial_params,
                        restart_idx=restart,
                    )
                    break
                except Exception as exc:
                    if attempt < self.max_retries:
                        logger.warning(
                            f"  VQE attempt {attempt + 1} failed: {exc}. "
                            f"Retrying ({attempt + 2}/{self.max_retries + 1})..."
                        )
                        num_retries += 1
                        time.sleep(2 ** attempt)  # exponential back-off
                        # Rebuild estimator in case of connection issue
                        estimator = self._build_estimator_with_retry()
                    else:
                        logger.error(f"  All {self.max_retries + 1} attempts failed.")
                        raise

            restart_energies.append(energy)
            total_iters += iters
            total_evals += evals

            if energy < best_energy:
                best_energy = energy
                best_params = params
                best_energy_history = history
                logger.info(f"  New best energy: {best_energy:.6f} Ha")

        # Step 6: Convergence detection using moving-average window
        converged = self._check_convergence(best_energy_history)

        # Step 7: Optional Zero Noise Extrapolation
        zne_energy = None
        if self.enable_zne and best_params is not None:
            zne_energy = self._run_zne(ansatz, qubit_op, best_params, estimator)

        runtime = time.time() - start_time

        result = VQEResult(
            ground_energy=best_energy,
            ground_energy_kcal=best_energy * HARTREE_TO_KCAL,
            optimal_params=best_params,
            num_iterations=total_iters,
            num_function_evaluations=total_evals,
            converged=converged,
            num_qubits=encoded.num_qubits,
            num_parameters=ansatz.num_parameters,
            ansatz_type=self.ansatz_type,
            optimizer_name=self.optimizer_name,
            backend=self.backend_name,
            runtime_seconds=runtime,
            energy_history=best_energy_history,
            restart_energies=restart_energies,
            zne_energy=zne_energy,
            num_retries=num_retries,
        )

        self._last_result = result
        logger.info(f"\n{result.summary()}")
        return result

    def plot_convergence(self, save_path: Optional[str] = None, show: bool = False):
        """Plot the VQE energy convergence curve."""
        if self._last_result is None:
            logger.warning("No VQE result to plot. Run run() first.")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed. Run: pip install matplotlib")
            return

        result = self._last_result
        history = result.energy_history

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(range(len(history)), history, color="#1D9E75", linewidth=1.5, alpha=0.9)
        ax.axhline(
            result.ground_energy,
            color="#D85A30",
            linestyle="--",
            linewidth=1,
            label=f"Final: {result.ground_energy:.6f} Ha",
        )
        if result.zne_energy is not None:
            ax.axhline(
                result.zne_energy,
                color="#5B4FCF",
                linestyle=":",
                linewidth=1.5,
                label=f"ZNE: {result.zne_energy:.6f} Ha",
            )
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Energy (Hartree)", fontsize=12)
        ax.set_title(
            f"VQE Convergence — {result.ansatz_type} / {result.optimizer_name}",
            fontsize=13,
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#FAFAFA")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Convergence plot saved: {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    # ─── Internal ────────────────────────────────────────────────────────────

    def _check_convergence(self, energy_history: List[float], window: int = 10) -> bool:
        """
        Robust convergence check using a moving-average window.
        Converged when the std-dev of the last `window` energies < threshold.
        """
        if len(energy_history) < window:
            return False
        tail = np.array(energy_history[-window:])
        return float(tail.std()) < self.convergence_threshold

    def _run_single(
        self,
        estimator,
        ansatz,
        qubit_op,
        optimizer,
        initial_params,
        restart_idx: int,
    ):
        """Run one VQE optimization from given initial parameters."""
        energy_history = []
        iteration_counter = [0]

        def callback(nfev, x, fx, metadata, stddev=None):
            iteration_counter[0] += 1
            energy_history.append(float(fx))
            if iteration_counter[0] % 50 == 0 or iteration_counter[0] <= 5:
                delta = (
                    abs(energy_history[-1] - energy_history[-2])
                    if len(energy_history) > 1
                    else float("nan")
                )
                logger.info(
                    f"  [iter {iteration_counter[0]:>4}] "
                    f"E = {fx:.8f} Ha  |ΔE| = {delta:.2e}"
                )

        try:
            from qiskit_algorithms import VQE as QiskitVQE
        except ImportError:
            raise ImportError(
                "qiskit-algorithms is required. Run: pip install qiskit-algorithms"
            )

        vqe = QiskitVQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            callback=callback,
            initial_point=initial_params,
        )

        vqe_result = vqe.compute_minimum_eigenvalue(qubit_op)

        energy = float(vqe_result.eigenvalue.real)
        params = vqe_result.optimal_point
        evals = vqe_result.cost_function_evals or len(energy_history)

        return energy, params, iteration_counter[0], evals, energy_history

    def _build_estimator_with_retry(self):
        """Build estimator, falling back gracefully on import/connection errors."""
        try:
            return self._build_estimator()
        except Exception as exc:
            if self.backend_name != "aer_statevector":
                logger.warning(
                    f"Failed to build '{self.backend_name}' estimator: {exc}. "
                    f"Falling back to statevector."
                )
                return self._statevector_estimator()
            raise

    def _build_estimator(self):
        """Build the appropriate Qiskit Estimator based on the backend setting."""
        if self.backend_name == "aer_statevector":
            return self._statevector_estimator()
        elif self.backend_name == "aer_qasm":
            return self._qasm_estimator()
        elif self.backend_name == "aer_noise":
            return self._noise_estimator()
        elif self.backend_name == "ibm_real":
            return self._ibm_estimator()
        else:
            logger.warning(f"Unknown backend '{self.backend_name}', defaulting to statevector.")
            return self._statevector_estimator()

    def _statevector_estimator(self):
        """Exact statevector simulation — no shot noise."""
        # Qiskit 1.x: StatevectorEstimator is the preferred primitive
        try:
            from qiskit.primitives import StatevectorEstimator
            return StatevectorEstimator()
        except ImportError:
            pass
        # Fallback for older Qiskit
        try:
            from qiskit.primitives import Estimator
            return Estimator()
        except ImportError:
            raise ImportError("qiskit is required. Run: pip install qiskit")

    def _qasm_estimator(self, shots: int = 8192):
        """Shot-based simulation — mimics real hardware measurement noise."""
        try:
            from qiskit_aer.primitives import Estimator as AerEstimator
            estimator = AerEstimator()
            estimator.set_options(shots=shots)
            return estimator
        except ImportError:
            logger.warning("qiskit-aer not installed, falling back to statevector.")
            return self._statevector_estimator()

    def _noise_estimator(self, shots: int = 8192):
        """
        Shot-based simulation WITH a fake noise model based on a real IBM device.
        Good for testing noise mitigation before real hardware runs.
        """
        try:
            from qiskit_aer.primitives import Estimator as AerEstimator
            from qiskit_aer.noise import NoiseModel

            # Try several fake backends in order of availability
            noise_model = None
            for fake_cls_path in [
                ("qiskit_ibm_runtime.fake_provider", "FakeManilaV2"),
                ("qiskit_ibm_runtime.fake_provider", "FakeNairobiV2"),
                ("qiskit_aer.noise", "NoiseModel"),
            ]:
                try:
                    mod = __import__(fake_cls_path[0], fromlist=[fake_cls_path[1]])
                    fake_backend = getattr(mod, fake_cls_path[1])()
                    noise_model = NoiseModel.from_backend(fake_backend)
                    logger.info(f"Using fake backend: {fake_cls_path[1]}")
                    break
                except Exception:
                    continue

            estimator = AerEstimator()
            if noise_model is not None:
                estimator.set_options(
                    shots=shots,
                    noise_model=noise_model,
                    optimization_level=0,
                )
            else:
                estimator.set_options(shots=shots)
            return estimator
        except ImportError:
            logger.warning("Noise model unavailable, falling back to qasm estimator.")
            return self._qasm_estimator(shots)

    def _ibm_estimator(self):
        """
        Real IBM Quantum hardware estimator using Qiskit Runtime.
        Requires IBM_QUANTUM_TOKEN to be set.
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, Session
            from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
            from qiskit_ibm_runtime.options import EstimatorOptions
        except ImportError:
            raise ImportError(
                "qiskit-ibm-runtime is required. Run: pip install qiskit-ibm-runtime"
            )

        token = self.ibm_token
        if not token:
            # Try environment variable as last resort
            import os
            token = os.environ.get("IBM_QUANTUM_TOKEN", "")

        if not token:
            raise ValueError(
                "IBM Quantum token is required for ibm_real backend. "
                "Set IBM_QUANTUM_TOKEN environment variable or pass ibm_token= to VQERunner."
            )

        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=token,
            instance=self.ibm_instance,
        )

        backend = service.least_busy(operational=True, simulator=False)
        logger.info(f"Using IBM Quantum backend: {backend.name}")

        options = EstimatorOptions()
        options.resilience_level = 2        # Zero Noise Extrapolation
        options.optimization_level = 3      # Transpile + layout optimization

        session = Session(backend=backend)
        estimator = RuntimeEstimator(mode=session, options=options)
        return estimator

    def _apply_auto_active_space(self, problem, encoded):
        """
        Automatically apply active space truncation to keep qubit count manageable.
        Selects the HOMO/LUMO window around the Fermi level.
        """
        from src.molecule.loader import MoleculeLoader
        from src.molecule.encoder import MoleculeEncoder

        target_qubits = self.auto_active_space_max_qubits
        # Each spatial orbital = 2 qubits (spin up + spin down)
        n_orb = target_qubits // 2

        # Use half the target orbitals as active electrons (conservative)
        n_elec = min(encoded.num_particles[0] + encoded.num_particles[1], n_orb)
        # Ensure even number of electrons for closed-shell
        if n_elec % 2 != 0:
            n_elec = max(2, n_elec - 1)

        logger.info(
            f"Auto active space: ({n_elec} electrons, {n_orb} orbitals) "
            f"→ {n_orb * 2} qubits"
        )

        try:
            from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

            qiskit_problem = (
                problem.qiskit_problem if hasattr(problem, "qiskit_problem") else problem
            )
            transformer = ActiveSpaceTransformer(
                num_electrons=n_elec,
                num_spatial_orbitals=n_orb,
            )
            reduced_problem = transformer.transform(qiskit_problem)

            # Re-encode with reduced problem
            encoder = MoleculeEncoder(mapping=self.qubit_mapping)

            # Wrap in a simple object that encoder.encode() can handle
            class _WrappedProblem:
                def __init__(self, p):
                    self.qiskit_problem = p

            wrapped = _WrappedProblem(reduced_problem)
            new_encoded = encoder.encode(wrapped)
            return wrapped, new_encoded, new_encoded.qubit_op

        except Exception as exc:
            logger.warning(f"Auto active space failed: {exc}. Using full space.")
            return problem, encoded, encoded.qubit_op

    def _run_zne(self, ansatz, qubit_op, optimal_params, estimator) -> Optional[float]:
        """Run Zero Noise Extrapolation on the converged VQE solution."""
        try:
            from src.noise.mitigation import ZeroNoiseExtrapolator
            zne = ZeroNoiseExtrapolator(
                noise_factors=self.zne_noise_factors,
                extrapolation="linear",
            )
            zne_energy = zne.extrapolate(ansatz, qubit_op, optimal_params, estimator)
            logger.info(f"ZNE-corrected energy: {zne_energy:.6f} Ha")
            return zne_energy
        except Exception as exc:
            logger.warning(f"ZNE failed: {exc}. Skipping.")
            return None
