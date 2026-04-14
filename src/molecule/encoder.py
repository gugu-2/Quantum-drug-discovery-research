"""
molecule/encoder.py

Converts a MoleculeProblem into a qubit Hamiltonian (SparsePauliOp)
using various fermion-to-qubit mappings.

Supported mappings:
    - jordan_wigner  : simple, standard, more qubits
    - bravyi_kitaev  : fewer 2-qubit gates, better for hardware
    - parity         : compact, needs 2 fewer qubits
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EncodedHamiltonian:
    """A qubit Hamiltonian ready for VQE."""
    qubit_op: object              # SparsePauliOp
    num_qubits: int
    num_particles: tuple          # (alpha_electrons, beta_electrons)
    num_spatial_orbitals: int
    mapping: str
    nuclear_repulsion_energy: float
    hartree_fock_energy: float


class MoleculeEncoder:
    """
    Encodes a MoleculeProblem into a qubit Hamiltonian.

    Examples
    --------
    >>> encoder = MoleculeEncoder(mapping="jordan_wigner")
    >>> ham = encoder.encode(problem)
    >>> print(f"Qubits: {ham.num_qubits}")
    >>> print(f"Pauli terms: {len(ham.qubit_op)}")
    """

    def __init__(self, mapping: str = "jordan_wigner"):
        """
        Parameters
        ----------
        mapping : str
            Fermion-to-qubit mapping. One of:
            "jordan_wigner", "bravyi_kitaev", "parity"
        """
        self.mapping = mapping

    def encode(self, problem) -> EncodedHamiltonian:
        """
        Convert a MoleculeProblem or raw Qiskit-Nature problem into
        a qubit Hamiltonian (SparsePauliOp).

        Parameters
        ----------
        problem : MoleculeProblem or ElectronicStructureProblem
        """
        # Handle both MoleculeProblem wrapper and raw Qiskit-Nature problem
        if hasattr(problem, "qiskit_problem"):
            qiskit_problem = problem.qiskit_problem
        else:
            qiskit_problem = problem

        mapper = self._get_mapper()

        # Get second-quantized Hamiltonian
        second_q_op = qiskit_problem.hamiltonian.second_q_op()

        # Map to qubits
        qubit_op = mapper.map(second_q_op)

        # Reduce qubit count for parity mapping (2-qubit reduction possible)
        if self.mapping == "parity":
            qubit_op = self._apply_parity_reduction(qubit_op, qiskit_problem)

        num_qubits = qubit_op.num_qubits
        num_particles = qiskit_problem.num_particles
        num_spatial_orbitals = qiskit_problem.num_spatial_orbitals

        # Hartree-Fock reference energy (classical upper bound on ground energy)
        try:
            hf_energy = qiskit_problem.reference_energy
        except AttributeError:
            hf_energy = 0.0

        # Nuclear repulsion energy (constant offset, added back to VQE energy)
        try:
            nuclear_energy = qiskit_problem.nuclear_repulsion_energy
        except AttributeError:
            nuclear_energy = 0.0

        logger.info(f"Encoded Hamiltonian ({self.mapping}):")
        logger.info(f"  Qubits       : {num_qubits}")
        logger.info(f"  Pauli terms  : {len(qubit_op)}")
        logger.info(f"  Particles    : {num_particles}")
        logger.info(f"  Orbitals     : {num_spatial_orbitals}")
        logger.info(f"  HF energy    : {hf_energy:.6f} Ha (upper bound)")

        return EncodedHamiltonian(
            qubit_op=qubit_op,
            num_qubits=num_qubits,
            num_particles=num_particles,
            num_spatial_orbitals=num_spatial_orbitals,
            mapping=self.mapping,
            nuclear_repulsion_energy=nuclear_energy,
            hartree_fock_energy=hf_energy,
        )

    def get_hamiltonian_summary(self, encoded: EncodedHamiltonian) -> dict:
        """Return a summary dict useful for logging and reporting."""
        pauli_counts = {}
        for pauli, coeff in encoded.qubit_op.to_list():
            weight = pauli.count("I")  # I-weight
            k = encoded.num_qubits - weight
            pauli_counts[k] = pauli_counts.get(k, 0) + 1

        return {
            "num_qubits": encoded.num_qubits,
            "num_pauli_terms": len(encoded.qubit_op),
            "mapping": encoded.mapping,
            "hf_energy": encoded.hartree_fock_energy,
            "k_local_breakdown": pauli_counts,
        }

    def print_pauli_terms(self, encoded: EncodedHamiltonian, max_terms: int = 20):
        """Print the largest Pauli terms of the Hamiltonian."""
        terms = sorted(
            encoded.qubit_op.to_list(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        logger.info(f"\nTop {min(max_terms, len(terms))} Pauli terms by coefficient magnitude:")
        for pauli, coeff in terms[:max_terms]:
            logger.info(f"  {pauli:>{encoded.num_qubits}}  {coeff.real:+.6f}")

    # ─── Internal ────────────────────────────────────────────────────────────

    def _get_mapper(self):
        """Instantiate the Qiskit-Nature mapper for the chosen mapping."""
        try:
            from qiskit_nature.second_q.mappers import (
                JordanWignerMapper,
                BravyiKitaevMapper,
                ParityMapper,
            )
        except ImportError:
            raise ImportError("qiskit-nature is required. Run: pip install qiskit-nature")

        mappers = {
            "jordan_wigner": JordanWignerMapper,
            "bravyi_kitaev": BravyiKitaevMapper,
            "parity": ParityMapper,
        }

        if self.mapping not in mappers:
            raise ValueError(
                f"Unknown mapping '{self.mapping}'. "
                f"Choose from: {list(mappers.keys())}"
            )

        return mappers[self.mapping]()

    def _apply_parity_reduction(self, qubit_op, problem):
        """
        Apply the 2-qubit reduction for the parity mapping.
        This exploits particle number conservation to remove 2 qubits.
        """
        try:
            from qiskit_nature.second_q.mappers import ParityMapper
            mapper = ParityMapper(num_particles=problem.num_particles)
            second_q_op = problem.hamiltonian.second_q_op()
            return mapper.map(second_q_op)
        except Exception as e:
            logger.warning(f"2-qubit parity reduction failed: {e}. Using full parity mapping.")
            return qubit_op
