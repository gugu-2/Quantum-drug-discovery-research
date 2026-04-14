"""
noise/mitigation.py

Noise mitigation utilities for running VQE on real quantum hardware.

Techniques implemented:
    1. Zero Noise Extrapolation (ZNE)
       Run the circuit at multiple noise levels by gate folding,
       then extrapolate the energy back to zero noise.

    2. Readout Error Mitigation
       Calibrate measurement errors using a full bit-flip calibration
       matrix, then invert it to correct measurement outcomes.

    3. Probabilistic Error Cancellation (PEC) — via IBM Runtime resilience=3
       Most accurate but exponentially expensive in shots. Use sparingly.
"""

from __future__ import annotations
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

# Gate types that can be safely folded (unitary, invertible)
_FOLDABLE_GATE_NAMES = {
    "cx", "cy", "cz", "ch", "swap", "iswap",
    "rx", "ry", "rz", "r", "u", "u1", "u2", "u3",
    "x", "y", "z", "h", "s", "sdg", "t", "tdg",
    "ccx", "cswap", "rzz", "rxx", "ryy",
    "p", "cp", "crx", "cry", "crz",
}


class ZeroNoiseExtrapolator:
    """
    Manual Zero Noise Extrapolation (ZNE).

    Runs a parameterized circuit at 3+ noise scale factors (obtained by
    gate folding), then extrapolates to the zero-noise limit using
    polynomial regression.

    How gate folding works:
        A gate G is "folded" by replacing it with G·G†·G.
        This multiplies the gate's noise by the fold factor without
        changing the circuit's ideal unitary.

    Examples
    --------
    >>> zne = ZeroNoiseExtrapolator(noise_factors=[1, 2, 3])
    >>> energy_zero_noise = zne.extrapolate(ansatz, qubit_op, params, estimator)
    """

    def __init__(
        self,
        noise_factors: List[int] = None,
        extrapolation: str = "linear",
    ):
        """
        Parameters
        ----------
        noise_factors : list of int
            Odd integers: [1, 2, 3] or [1, 3, 5]. Must start with 1 (unmitigated).
        extrapolation : str
            "linear"      : fit E = a + b·λ, extrapolate to λ=0
            "quadratic"   : fit E = a + b·λ + c·λ², better for large noise
            "exponential" : fit E = a·exp(b·λ), best for depolarizing noise
        """
        self.noise_factors = noise_factors or [1, 2, 3]
        self.extrapolation = extrapolation

    def extrapolate(
        self,
        ansatz,
        qubit_op,
        optimal_params: np.ndarray,
        estimator,
    ) -> float:
        """
        Compute the ZNE-mitigated ground energy.

        Parameters
        ----------
        ansatz : QuantumCircuit
            The ansatz circuit with parameters bound to optimal_params.
        qubit_op : SparsePauliOp
            The Hamiltonian.
        optimal_params : np.ndarray
            Optimal VQE parameters from the unmitigated run.
        estimator : BaseEstimator
            An Aer or IBM Runtime estimator.

        Returns
        -------
        float : Zero-noise extrapolated energy estimate.
        """
        energies_at_factors = []
        bound_circuit = ansatz.assign_parameters(optimal_params)

        for factor in self.noise_factors:
            if factor == 1:
                noisy_circuit = bound_circuit
            else:
                noisy_circuit = self._fold_gates(bound_circuit, scale_factor=factor)

            try:
                # Qiskit 1.x StatevectorEstimator uses pub format: (circuit, observable)
                job = estimator.run([(noisy_circuit, qubit_op)])
                result = job.result()
                # Handle both old and new result formats
                try:
                    energy = result[0].data.evs
                except AttributeError:
                    energy = result.values[0]
            except Exception as exc:
                logger.warning(f"ZNE factor={factor} failed: {exc}. Skipping this factor.")
                continue

            energies_at_factors.append(float(energy))
            logger.debug(f"  ZNE factor={factor} | E={energy:.6f} Ha")

        if len(energies_at_factors) < 2:
            logger.warning("Not enough ZNE data points for extrapolation. Returning raw energy.")
            return energies_at_factors[0] if energies_at_factors else float("nan")

        # Use only the factors for which we have data
        valid_factors = self.noise_factors[: len(energies_at_factors)]
        extrapolated = self._extrapolate_to_zero(valid_factors, energies_at_factors)
        logger.info(
            f"ZNE energies: {[f'{e:.6f}' for e in energies_at_factors]} "
            f"→ extrapolated: {extrapolated:.6f} Ha"
        )
        return extrapolated

    def _fold_gates(self, circuit, scale_factor: int):
        """
        Gate folding: replace each foldable gate G with G·(G†·G)^n
        where scale_factor = 2n + 1.

        Non-foldable instructions (barriers, measurements, resets) are
        passed through unchanged.
        """
        from qiskit import QuantumCircuit

        n_folds = (scale_factor - 1) // 2
        if n_folds == 0:
            return circuit

        folded = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)

        for instruction in circuit.data:
            gate = instruction.operation
            qubits = instruction.qubits
            clbits = instruction.clbits

            # Always append the original instruction
            folded.append(gate, qubits, clbits)

            # Only fold unitary gates — skip barriers, measurements, resets
            gate_name = gate.name.lower()
            if gate_name in _FOLDABLE_GATE_NAMES:
                try:
                    inv_gate = gate.inverse()
                    for _ in range(n_folds):
                        folded.append(inv_gate, qubits, clbits)
                        folded.append(gate, qubits, clbits)
                except Exception:
                    # Gate doesn't support inverse — skip folding for this gate
                    pass

        return folded

    def _extrapolate_to_zero(
        self,
        noise_factors: List[float],
        energies: List[float],
    ) -> float:
        """Fit the noise-energy relationship and extrapolate to λ=0."""
        x = np.array(noise_factors, dtype=float)
        y = np.array(energies, dtype=float)

        if self.extrapolation == "linear":
            coeffs = np.polyfit(x, y, deg=1)
            return float(np.polyval(coeffs, 0.0))

        elif self.extrapolation == "quadratic":
            if len(x) < 3:
                logger.warning("Need ≥3 points for quadratic fit. Falling back to linear.")
                coeffs = np.polyfit(x, y, deg=1)
            else:
                coeffs = np.polyfit(x, y, deg=2)
            return float(np.polyval(coeffs, 0.0))

        elif self.extrapolation == "exponential":
            # Fit log(E - E_min) = log(a) + b·λ using linear regression
            # This assumes E > 0; for negative energies, shift and unshift.
            shift = min(y) - 1.0
            log_y = np.log(np.clip(y - shift, 1e-12, None))
            coeffs = np.polyfit(x, log_y, deg=1)
            return float(np.exp(coeffs[1]) + shift)

        else:
            raise ValueError(
                f"Unknown extrapolation '{self.extrapolation}'. "
                f"Use 'linear', 'quadratic', or 'exponential'."
            )


class ReadoutMitigator:
    """
    Readout (measurement) error mitigation.

    Calibrates a confusion matrix M where M[i][j] = P(measure i | prepared j).
    Then inverts M to correct the raw measurement distribution.

    This is particularly important for near-term hardware where readout
    errors can be 1-5% per qubit.

    Examples
    --------
    >>> mitigator = ReadoutMitigator(num_qubits=4)
    >>> mitigator.calibrate(sampler, shots=4096)
    >>> corrected_counts = mitigator.apply(raw_counts)
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.calibration_matrix: Optional[np.ndarray] = None
        self.inverse_matrix: Optional[np.ndarray] = None

    def calibrate(self, sampler, shots: int = 4096):
        """
        Run calibration circuits (prepare each computational basis state,
        measure, record the results) to build the confusion matrix.
        """
        from qiskit import QuantumCircuit

        num_states = 2 ** self.num_qubits
        cal_matrix = np.zeros((num_states, num_states))

        logger.info(f"Calibrating readout errors for {self.num_qubits} qubits ({num_states} states)...")

        for state_idx in range(num_states):
            qc = QuantumCircuit(self.num_qubits, self.num_qubits)
            bits = format(state_idx, f"0{self.num_qubits}b")
            for qubit_idx, bit in enumerate(reversed(bits)):
                if bit == "1":
                    qc.x(qubit_idx)
            qc.measure_all()

            try:
                job = sampler.run([qc], shots=shots)
                result = job.result()
                # Handle both old and new result formats
                try:
                    counts = result[0].data.meas.get_counts()
                except AttributeError:
                    counts = result.quasi_dists[0]

                for measured_state, count in counts.items():
                    if isinstance(measured_state, int):
                        meas_idx = measured_state
                    else:
                        meas_idx = int(measured_state.replace(" ", ""), 2)
                    if meas_idx < num_states:
                        cal_matrix[meas_idx][state_idx] = count / shots

            except Exception as exc:
                logger.warning(f"Calibration circuit {state_idx} failed: {exc}. Using identity.")
                cal_matrix[state_idx][state_idx] = 1.0

        self.calibration_matrix = cal_matrix
        # Pseudo-inverse is more stable than direct inverse for noisy matrices
        self.inverse_matrix = np.linalg.pinv(cal_matrix)
        logger.info("Readout calibration complete.")
        return cal_matrix

    def apply(self, raw_counts: dict) -> dict:
        """
        Apply the calibrated correction to a raw measurement count dictionary.

        Parameters
        ----------
        raw_counts : dict
            e.g. {"0000": 512, "0001": 120, ...}

        Returns
        -------
        dict : corrected counts
        """
        if self.inverse_matrix is None:
            raise RuntimeError("Calibrate first with mitigator.calibrate(sampler).")

        num_states = 2 ** self.num_qubits
        raw_vector = np.zeros(num_states)
        total_shots = sum(raw_counts.values())

        if total_shots == 0:
            return raw_counts

        for state, count in raw_counts.items():
            if isinstance(state, int):
                idx = state
            else:
                idx = int(state.replace(" ", ""), 2)
            if idx < num_states:
                raw_vector[idx] = count / total_shots

        corrected_vector = self.inverse_matrix @ raw_vector
        # Clip negatives (can appear from matrix inversion noise)
        corrected_vector = np.clip(corrected_vector, 0, None)
        total = corrected_vector.sum()
        if total > 0:
            corrected_vector /= total

        corrected_counts = {}
        for idx, prob in enumerate(corrected_vector):
            if prob > 1e-6:
                state = format(idx, f"0{self.num_qubits}b")
                corrected_counts[state] = int(prob * total_shots)

        return corrected_counts


def apply_ibm_runtime_mitigation(
    estimator,
    resilience_level: int = 2,
    zne_noise_factors: Optional[List[int]] = None,
):
    """
    Configure IBM Runtime Estimator with built-in noise mitigation.

    IBM Runtime handles ZNE and PEC natively — this is much easier than
    manual ZNE when you're already using IBM hardware.

    Parameters
    ----------
    estimator : IBM Runtime EstimatorV2
    resilience_level : int
        0 = no mitigation
        1 = readout error mitigation only
        2 = ZNE (recommended default for VQE)
        3 = PEC (probabilistic error cancellation — most accurate, most shots)
    zne_noise_factors : list or None
        Custom noise factors for ZNE (default: [1, 2, 3])
    """
    try:
        from qiskit_ibm_runtime.options import EstimatorOptions
    except ImportError:
        logger.warning("qiskit-ibm-runtime not installed — skipping runtime mitigation config.")
        return estimator

    try:
        options = EstimatorOptions()
        options.resilience_level = resilience_level
        options.optimization_level = 3

        if resilience_level >= 2 and zne_noise_factors:
            options.resilience.zne_noise_factors = zne_noise_factors

        logger.info(f"IBM Runtime noise mitigation: resilience_level={resilience_level}")
    except Exception as exc:
        logger.warning(f"Could not configure IBM Runtime mitigation options: {exc}")

    return estimator
