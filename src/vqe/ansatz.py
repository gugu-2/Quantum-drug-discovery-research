"""
vqe/ansatz.py

Builds parameterized quantum circuits (ansätze) for VQE.

Available ansätze:
    efficient_su2    Hardware-efficient. Shallow circuit. Best for NISQ hardware.
    uccsd            Unitary Coupled Cluster Singles & Doubles. Chemistry-inspired.
                     High accuracy but deep circuit. Best for simulators.
    real_amplitudes  Real-valued rotation + CNOT layers. Simple, fast.
    two_local        Flexible general-purpose hardware-efficient ansatz.
    custom           Build your own layer-by-layer.
"""

from __future__ import annotations
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class AnsatzBuilder:
    """
    Factory class that builds the right ansatz circuit for VQE.

    Examples
    --------
    >>> builder = AnsatzBuilder()
    >>> ansatz = builder.build("efficient_su2", num_qubits=4, reps=2)
    >>> print(ansatz.num_parameters)

    >>> # For UCCSD (chemistry-inspired)
    >>> ansatz = builder.build_uccsd(num_qubits=4, num_particles=(1,1), num_orbitals=2)
    """

    def build(
        self,
        ansatz_type: str,
        num_qubits: int,
        reps: int = 2,
        entanglement: str = "linear",
        insert_barriers: bool = True,
    ):
        """
        Build an ansatz circuit.

        Parameters
        ----------
        ansatz_type : str
            "efficient_su2" | "real_amplitudes" | "two_local" | "uccsd"
        num_qubits : int
            Number of qubits in the circuit
        reps : int
            Number of repetition layers (depth). Higher = more expressive.
        entanglement : str
            Entanglement pattern: "linear" | "full" | "circular" | "pairwise"
            - linear   : CNOT chain q0→q1→q2→...  (shallow, default)
            - full     : all-to-all CNOTs          (deep, expressive)
            - circular : linear + wrap-around CNOT (good for periodic systems)
        """
        builders = {
            "efficient_su2": self._efficient_su2,
            "real_amplitudes": self._real_amplitudes,
            "two_local": self._two_local,
        }

        if ansatz_type == "uccsd":
            raise ValueError(
                "For UCCSD, use build_uccsd() — it requires chemistry-specific parameters "
                "(num_particles, num_spatial_orbitals) from the molecular problem."
            )

        if ansatz_type not in builders:
            raise ValueError(
                f"Unknown ansatz '{ansatz_type}'. "
                f"Choose from: {list(builders.keys()) + ['uccsd']}"
            )

        ansatz = builders[ansatz_type](
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=insert_barriers,
        )

        logger.info(f"Ansatz: {ansatz_type} | qubits={num_qubits} | reps={reps} | params={ansatz.num_parameters}")
        return ansatz

    def build_uccsd(
        self,
        num_qubits: int,
        num_particles,
        num_spatial_orbitals: int,
        reps: int = 1,
    ):
        """
        Build a UCCSD (Unitary Coupled Cluster Singles & Doubles) ansatz.

        UCCSD is the chemistry-standard ansatz — it's derived from the
        structure of electron excitations, so it converges faster for
        molecules. The trade-off is a much deeper circuit.

        Parameters
        ----------
        num_particles : tuple
            (alpha_electrons, beta_electrons), e.g. (1, 1) for H2
        num_spatial_orbitals : int
            Number of spatial orbitals in the active space
        """
        try:
            from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
            from qiskit_nature.second_q.mappers import JordanWignerMapper
        except ImportError:
            raise ImportError("qiskit-nature is required. Run: pip install qiskit-nature")

        mapper = JordanWignerMapper()

        # Hartree-Fock initial state — electrons fill lowest orbitals
        # This is a much better starting point than |0...0⟩
        hf_state = HartreeFock(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
        )

        ansatz = UCCSD(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
            initial_state=hf_state,
            reps=reps,
        )

        logger.info(
            f"UCCSD ansatz | qubits={num_qubits} | params={ansatz.num_parameters} | "
            f"particles={num_particles} | orbitals={num_spatial_orbitals}"
        )
        return ansatz

    def get_initial_params(
        self,
        ansatz,
        strategy: str = "random",
        seed: Optional[int] = None,
    ):
        """
        Generate initial parameters for the ansatz.

        Parameters
        ----------
        strategy : str
            "random"  : random in [-π, π]  (default, avoids barren plateaus)
            "zero"    : all zeros (can cause symmetry trapping)
            "small"   : small random values near zero
        """
        import numpy as np
        num_params = ansatz.num_parameters
        rng = np.random.default_rng(seed)

        if strategy == "random":
            params = rng.uniform(-np.pi, np.pi, num_params)
        elif strategy == "zero":
            params = np.zeros(num_params)
        elif strategy == "small":
            params = rng.uniform(-0.1, 0.1, num_params)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'random', 'zero', or 'small'.")

        logger.debug(f"Initial params ({strategy}): shape={params.shape}, range=[{params.min():.3f}, {params.max():.3f}]")
        return params

    def draw(self, ansatz, output: str = "text", fold: int = 80):
        """Print or return a circuit diagram."""
        return ansatz.decompose().draw(output=output, fold=fold)

    # ─── Internal builders ────────────────────────────────────────────────────

    def _efficient_su2(self, num_qubits, reps, entanglement, insert_barriers):
        try:
            from qiskit.circuit.library import EfficientSU2
        except ImportError:
            raise ImportError("qiskit is required. Run: pip install qiskit")

        return EfficientSU2(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=insert_barriers,
        )

    def _real_amplitudes(self, num_qubits, reps, entanglement, insert_barriers):
        try:
            from qiskit.circuit.library import RealAmplitudes
        except ImportError:
            raise ImportError("qiskit is required. Run: pip install qiskit")

        return RealAmplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=insert_barriers,
        )

    def _two_local(self, num_qubits, reps, entanglement, insert_barriers):
        try:
            from qiskit.circuit.library import TwoLocal
        except ImportError:
            raise ImportError("qiskit is required. Run: pip install qiskit")

        return TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cx",
            reps=reps,
            entanglement=entanglement,
            insert_barriers=insert_barriers,
        )
