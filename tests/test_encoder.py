"""
tests/test_encoder.py

Unit tests for molecule loading and Hamiltonian encoding.
Run with: pytest tests/test_encoder.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def h2_problem():
    """Load the H2 molecule problem once for all tests in this module."""
    from src.molecule.loader import MoleculeLoader
    loader = MoleculeLoader()
    return loader.from_name("h2", basis="sto-3g")


@pytest.fixture(scope="module")
def h2_encoded(h2_problem):
    """Encode H2 with Jordan-Wigner mapping."""
    from src.molecule.encoder import MoleculeEncoder
    encoder = MoleculeEncoder(mapping="jordan_wigner")
    return encoder.encode(h2_problem)


# ─── Loader tests ──────────────────────────────────────────────────────────────

class TestMoleculeLoader:
    def test_load_h2_by_name(self):
        from src.molecule.loader import MoleculeLoader
        loader = MoleculeLoader()
        problem = loader.from_name("h2")
        assert problem is not None
        assert problem.name.upper() == "H2"
        assert problem.qiskit_problem is not None

    def test_load_lih_by_name(self):
        from src.molecule.loader import MoleculeLoader
        loader = MoleculeLoader()
        problem = loader.from_name("lih", basis="sto-3g")
        assert problem is not None

    def test_load_from_geometry(self):
        from src.molecule.loader import MoleculeLoader
        loader = MoleculeLoader()
        problem = loader.from_geometry(
            geometry="H 0.0 0.0 0.0; H 0.0 0.0 0.735",
            basis="sto-3g",
            name="H2_manual",
        )
        assert problem is not None
        assert problem.name == "H2_manual"

    def test_load_with_active_space(self):
        from src.molecule.loader import MoleculeLoader
        loader = MoleculeLoader()
        # LiH with active space — should use fewer qubits
        problem_full = loader.from_name("lih", basis="sto-3g", active_space=None)
        problem_as = loader.from_name("lih", basis="sto-3g", active_space=(2, 3))
        assert problem_full is not None
        assert problem_as is not None

    def test_invalid_molecule_name(self):
        from src.molecule.loader import MoleculeLoader
        loader = MoleculeLoader()
        with pytest.raises(ValueError, match="not in built-in library"):
            loader.from_name("unobtanium")

    def test_load_from_smiles_fallback(self):
        """Even without RDKit, from_smiles should return a problem (using fallback)."""
        from src.molecule.loader import MoleculeLoader
        loader = MoleculeLoader()
        # The fallback returns H2 geometry if RDKit is missing
        problem = loader.from_smiles("CCO", basis="sto-3g")
        assert problem is not None

    def test_molecule_library_completeness(self):
        from src.molecule.loader import MOLECULE_LIBRARY
        required_keys = ["h2", "lih", "water", "nh3"]
        for key in required_keys:
            assert key in MOLECULE_LIBRARY, f"'{key}' missing from MOLECULE_LIBRARY"


# ─── Encoder tests ─────────────────────────────────────────────────────────────

class TestMoleculeEncoder:
    def test_encode_h2_jordan_wigner(self, h2_problem):
        from src.molecule.encoder import MoleculeEncoder
        encoder = MoleculeEncoder(mapping="jordan_wigner")
        encoded = encoder.encode(h2_problem)
        assert encoded is not None
        assert encoded.num_qubits == 4      # H2 with STO-3G → 4 qubits
        assert encoded.mapping == "jordan_wigner"
        assert len(encoded.qubit_op) > 0    # Hamiltonian has at least 1 Pauli term

    def test_encode_h2_bravyi_kitaev(self, h2_problem):
        from src.molecule.encoder import MoleculeEncoder
        encoder = MoleculeEncoder(mapping="bravyi_kitaev")
        encoded = encoder.encode(h2_problem)
        assert encoded is not None
        assert encoded.num_qubits == 4
        # BK typically has fewer Pauli terms than JW
        assert len(encoded.qubit_op) > 0

    def test_qubit_operator_type(self, h2_encoded):
        """The qubit operator should be a SparsePauliOp."""
        try:
            from qiskit.quantum_info import SparsePauliOp
            assert isinstance(h2_encoded.qubit_op, SparsePauliOp)
        except ImportError:
            pytest.skip("qiskit not installed")

    def test_num_qubits_correct(self, h2_encoded):
        assert h2_encoded.num_qubits == h2_encoded.qubit_op.num_qubits

    def test_pauli_terms_nonzero(self, h2_encoded):
        assert len(h2_encoded.qubit_op) > 0

    def test_invalid_mapping(self, h2_problem):
        from src.molecule.encoder import MoleculeEncoder
        encoder = MoleculeEncoder(mapping="invalid_mapping")
        with pytest.raises(ValueError, match="Unknown mapping"):
            encoder.encode(h2_problem)

    def test_hamiltonian_summary(self, h2_encoded):
        from src.molecule.encoder import MoleculeEncoder
        encoder = MoleculeEncoder(mapping="jordan_wigner")
        summary = encoder.get_hamiltonian_summary(h2_encoded)
        assert "num_qubits" in summary
        assert "num_pauli_terms" in summary
        assert summary["num_qubits"] == 4
        assert summary["num_pauli_terms"] > 0

    def test_hermitian_hamiltonian(self, h2_encoded):
        """The Hamiltonian must be Hermitian (all coefficients must be real)."""
        for pauli, coeff in h2_encoded.qubit_op.to_list():
            assert abs(coeff.imag) < 1e-10, f"Non-real coefficient for {pauli}: {coeff}"
