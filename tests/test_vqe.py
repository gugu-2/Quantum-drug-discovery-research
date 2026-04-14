"""
tests/test_vqe.py

Tests for VQE components: ansatz, optimizer, and end-to-end runner.
Run with: pytest tests/test_vqe.py -v

The H2 benchmark is the canonical VQE validation:
    Exact ground energy: -1.137270 Ha
    VQE should get within 5 mHa (0.4%) of this value.
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

H2_EXACT_ENERGY = -1.137270      # Hartree, FCI/STO-3G reference
H2_TOLERANCE_HA = 0.005          # 5 mHartree tolerance


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def h2_problem():
    from src.molecule.loader import MoleculeLoader
    return MoleculeLoader().from_name("h2", basis="sto-3g")


@pytest.fixture(scope="module")
def h2_encoded(h2_problem):
    from src.molecule.encoder import MoleculeEncoder
    return MoleculeEncoder(mapping="jordan_wigner").encode(h2_problem)


# ─── Ansatz tests ─────────────────────────────────────────────────────────────

class TestAnsatzBuilder:
    def test_efficient_su2_4_qubits(self):
        from src.vqe.ansatz import AnsatzBuilder
        builder = AnsatzBuilder()
        ansatz = builder.build("efficient_su2", num_qubits=4, reps=2)
        assert ansatz.num_qubits == 4
        assert ansatz.num_parameters > 0

    def test_real_amplitudes(self):
        from src.vqe.ansatz import AnsatzBuilder
        builder = AnsatzBuilder()
        ansatz = builder.build("real_amplitudes", num_qubits=4, reps=1)
        assert ansatz.num_qubits == 4
        assert ansatz.num_parameters > 0

    def test_two_local(self):
        from src.vqe.ansatz import AnsatzBuilder
        builder = AnsatzBuilder()
        ansatz = builder.build("two_local", num_qubits=4, reps=2)
        assert ansatz.num_qubits == 4

    def test_uccsd_requires_special_builder(self):
        from src.vqe.ansatz import AnsatzBuilder
        builder = AnsatzBuilder()
        with pytest.raises(ValueError, match="use build_uccsd"):
            builder.build("uccsd", num_qubits=4, reps=1)

    def test_uccsd_chemistry_ansatz(self):
        from src.vqe.ansatz import AnsatzBuilder
        builder = AnsatzBuilder()
        ansatz = builder.build_uccsd(
            num_qubits=4,
            num_particles=(1, 1),
            num_spatial_orbitals=2,
        )
        assert ansatz.num_qubits == 4
        assert ansatz.num_parameters > 0

    def test_invalid_ansatz_type(self):
        from src.vqe.ansatz import AnsatzBuilder
        builder = AnsatzBuilder()
        with pytest.raises(ValueError, match="Unknown ansatz"):
            builder.build("magic_ansatz", num_qubits=4, reps=1)

    def test_initial_params_random(self):
        from src.vqe.ansatz import AnsatzBuilder
        builder = AnsatzBuilder()
        ansatz = builder.build("efficient_su2", num_qubits=4, reps=2)
        params = builder.get_initial_params(ansatz, strategy="random", seed=42)
        assert params.shape == (ansatz.num_parameters,)
        assert not np.allclose(params, 0)

    def test_initial_params_zero(self):
        from src.vqe.ansatz import AnsatzBuilder
        builder = AnsatzBuilder()
        ansatz = builder.build("efficient_su2", num_qubits=4, reps=2)
        params = builder.get_initial_params(ansatz, strategy="zero")
        assert np.allclose(params, 0)

    def test_depth_increases_with_reps(self):
        from src.vqe.ansatz import AnsatzBuilder
        builder = AnsatzBuilder()
        a1 = builder.build("efficient_su2", num_qubits=4, reps=1)
        a2 = builder.build("efficient_su2", num_qubits=4, reps=3)
        # More reps = more parameters
        assert a2.num_parameters > a1.num_parameters


# ─── Optimizer tests ──────────────────────────────────────────────────────────

class TestOptimizerFactory:
    def test_cobyla_creation(self):
        from src.vqe.optimizer import OptimizerFactory
        opt = OptimizerFactory.create("cobyla", max_iterations=100)
        assert opt is not None

    def test_spsa_creation(self):
        from src.vqe.optimizer import OptimizerFactory
        opt = OptimizerFactory.create("spsa", max_iterations=100)
        assert opt is not None

    def test_lbfgsb_creation(self):
        from src.vqe.optimizer import OptimizerFactory
        opt = OptimizerFactory.create("lbfgsb", max_iterations=100)
        assert opt is not None

    def test_invalid_optimizer(self):
        from src.vqe.optimizer import OptimizerFactory
        with pytest.raises(ValueError, match="Unknown optimizer"):
            OptimizerFactory.create("gradient_magic", max_iterations=100)

    def test_recommended_for_simulator(self):
        from src.vqe.optimizer import OptimizerFactory
        rec = OptimizerFactory.recommended_for_backend("aer_statevector")
        assert rec == "cobyla"

    def test_recommended_for_hardware(self):
        from src.vqe.optimizer import OptimizerFactory
        rec = OptimizerFactory.recommended_for_backend("ibm_real")
        assert rec == "spsa"


# ─── VQE runner tests ─────────────────────────────────────────────────────────

class TestVQERunner:
    def test_result_dataclass(self):
        from src.vqe.runner import VQEResult
        result = VQEResult(
            ground_energy=-1.1,
            ground_energy_kcal=-1.1 * 627.5,
            optimal_params=np.zeros(4),
            num_iterations=50,
            num_function_evaluations=100,
            converged=True,
            num_qubits=4,
            num_parameters=16,
            ansatz_type="efficient_su2",
            optimizer_name="cobyla",
            backend="aer_statevector",
            runtime_seconds=5.2,
            energy_history=[-0.9, -1.0, -1.1],
        )
        summary = result.summary()
        assert "-1.100000" in summary
        assert "converged" in summary.lower()

    def test_h2_vqe_energy_accuracy(self, h2_problem):
        """
        INTEGRATION TEST: VQE must converge on H2 to within 5 mHa of exact.
        This is the canonical quantum chemistry benchmark.
        Skipped if qiskit-nature or pyscf are not installed.
        """
        pytest.importorskip("qiskit_nature")
        pytest.importorskip("pyscf")

        from src.vqe.runner import VQERunner

        runner = VQERunner(
            backend="aer_statevector",
            ansatz_type="efficient_su2",
            ansatz_reps=2,
            optimizer_name="cobyla",
            max_iterations=500,
            convergence_threshold=1e-6,
            num_restarts=2,
            seed=42,
        )
        result = runner.run(h2_problem)

        assert result.ground_energy < 0, "Ground energy should be negative"
        assert result.num_qubits == 4, "H2 with JW mapping should use 4 qubits"
        assert abs(result.ground_energy - H2_EXACT_ENERGY) < H2_TOLERANCE_HA, (
            f"VQE energy {result.ground_energy:.6f} Ha deviates more than "
            f"{H2_TOLERANCE_HA * 1000:.1f} mHa from exact {H2_EXACT_ENERGY:.6f} Ha"
        )
        assert result.runtime_seconds > 0

    def test_energy_history_populated(self, h2_problem):
        """VQE should record an energy value at each iteration."""
        pytest.importorskip("qiskit_nature")
        pytest.importorskip("pyscf")

        from src.vqe.runner import VQERunner
        runner = VQERunner(
            backend="aer_statevector",
            ansatz_type="efficient_su2",
            ansatz_reps=1,
            optimizer_name="cobyla",
            max_iterations=50,
            num_restarts=1,
            seed=0,
        )
        result = runner.run(h2_problem)
        assert len(result.energy_history) > 0
        assert result.energy_history[-1] <= result.energy_history[0], (
            "Energy should generally decrease (or at least not increase monotonically)"
        )

    def test_uccsd_ansatz_h2(self, h2_problem):
        """UCCSD should achieve chemical accuracy on H2 (< 1 mHa error)."""
        pytest.importorskip("qiskit_nature")
        pytest.importorskip("pyscf")

        from src.vqe.runner import VQERunner
        runner = VQERunner(
            backend="aer_statevector",
            ansatz_type="uccsd",
            optimizer_name="lbfgsb",
            max_iterations=200,
            num_restarts=1,
            seed=42,
        )
        result = runner.run(h2_problem)
        # UCCSD should achieve chemical accuracy: < 1 mHa
        assert abs(result.ground_energy - H2_EXACT_ENERGY) < 0.001, (
            f"UCCSD energy {result.ground_energy:.6f} Ha should be within 1 mHa of exact"
        )

    def test_multiple_restarts_improves_result(self, h2_problem):
        """More restarts should find an equal or better energy."""
        pytest.importorskip("qiskit_nature")
        pytest.importorskip("pyscf")

        from src.vqe.runner import VQERunner

        runner_1 = VQERunner(
            backend="aer_statevector",
            ansatz_type="efficient_su2",
            ansatz_reps=1,
            optimizer_name="cobyla",
            max_iterations=100,
            num_restarts=1,
            seed=99,
        )
        runner_3 = VQERunner(
            backend="aer_statevector",
            ansatz_type="efficient_su2",
            ansatz_reps=1,
            optimizer_name="cobyla",
            max_iterations=100,
            num_restarts=3,
            seed=99,
        )
        result_1 = runner_1.run(h2_problem)
        result_3 = runner_3.run(h2_problem)

        # 3 restarts should find equal or better energy
        assert result_3.ground_energy <= result_1.ground_energy + 1e-6

    def test_convergence_plot_runs_without_error(self, h2_problem, tmp_path):
        """plot_convergence should save a file without raising exceptions."""
        pytest.importorskip("matplotlib")
        pytest.importorskip("qiskit_nature")
        pytest.importorskip("pyscf")

        from src.vqe.runner import VQERunner
        runner = VQERunner(
            backend="aer_statevector",
            ansatz_type="efficient_su2",
            ansatz_reps=1,
            optimizer_name="cobyla",
            max_iterations=30,
            num_restarts=1,
        )
        runner.run(h2_problem)
        save_path = str(tmp_path / "convergence.png")
        runner.plot_convergence(save_path=save_path)
        assert os.path.exists(save_path)
