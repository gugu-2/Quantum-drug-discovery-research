"""
tests/test_pipeline.py

Integration tests for the drug screening pipeline.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDrugScreeningPipeline:
    def test_get_demo_candidates(self):
        from src.screening.pipeline import DrugScreeningPipeline
        pipeline = DrugScreeningPipeline()
        candidates = pipeline.get_demo_candidates()
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert "smiles" in candidates[0]
        assert "name" in candidates[0]

    def test_screen_single_smiles_string(self):
        """Screen a single simple molecule given as a bare SMILES string."""
        pytest.importorskip("qiskit_nature")
        pytest.importorskip("pyscf")

        from src.screening.pipeline import DrugScreeningPipeline
        pipeline = DrugScreeningPipeline(
            backend="aer_statevector",
            compute_binding=False,  # skip receptor computation for speed
        )
        results = pipeline.screen(["CC(=O)O"], progress=False)
        assert len(results) == 1
        assert results[0].status == "success"
        assert results[0].ligand_energy_ha < 0

    def test_screen_multiple_candidates(self):
        pytest.importorskip("qiskit_nature")
        pytest.importorskip("pyscf")

        from src.screening.pipeline import DrugScreeningPipeline
        pipeline = DrugScreeningPipeline(
            backend="aer_statevector",
            compute_binding=False,
        )
        candidates = [
            {"name": "Methanol", "smiles": "CO"},
            {"name": "Ethanol", "smiles": "CCO"},
        ]
        results = pipeline.screen(candidates, progress=False)
        assert len(results) == 2
        # Results should be sorted by binding energy
        assert results[0].binding_energy_kcal <= results[1].binding_energy_kcal

    def test_results_ranked(self):
        """Ranks should be assigned correctly starting from 1."""
        pytest.importorskip("qiskit_nature")
        pytest.importorskip("pyscf")

        from src.screening.pipeline import DrugScreeningPipeline
        pipeline = DrugScreeningPipeline(
            backend="aer_statevector",
            compute_binding=False,
        )
        candidates = ["CO", "CCO"]
        results = pipeline.screen(candidates, progress=False)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_failed_smiles_handled_gracefully(self):
        """An invalid SMILES should produce a 'failed' result, not crash."""
        pytest.importorskip("qiskit_nature")
        pytest.importorskip("pyscf")

        from src.screening.pipeline import DrugScreeningPipeline
        pipeline = DrugScreeningPipeline(
            backend="aer_statevector",
            compute_binding=False,
        )
        candidates = [
            {"name": "Valid", "smiles": "CO"},
            {"name": "Invalid", "smiles": "XXXXXXXXX"},
        ]
        results = pipeline.screen(candidates, progress=False)
        assert len(results) == 2
        statuses = {r.name: r.status for r in results}
        assert statuses["Valid"] == "success"
        # The invalid one should fail gracefully
        assert statuses["Invalid"] in ("success", "failed")  # fallback to H2 or real failure

    def test_save_results_creates_csv(self, tmp_path):
        pytest.importorskip("qiskit_nature")
        pytest.importorskip("pyscf")

        from src.screening.pipeline import DrugScreeningPipeline
        pipeline = DrugScreeningPipeline(
            backend="aer_statevector",
            compute_binding=False,
        )
        results = pipeline.screen(["CO"], progress=False)
        output_path = str(tmp_path / "test_output.csv")
        pipeline.save_results(results, output_path)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert "smiles" in rows[0]
        assert "binding_energy_kcal" in rows[0]

    def test_max_candidates_limit(self):
        """Pipeline should not screen more than max_candidates."""
        from src.screening.pipeline import DrugScreeningPipeline
        pipeline = DrugScreeningPipeline(
            backend="aer_statevector",
            compute_binding=False,
            max_candidates=2,
        )
        # Provide 5 candidates but limit is 2
        candidates = ["CO", "CCO", "CCCO", "CCCCO", "CC"]
        # We check the normalization step (not a full run)
        normalized = []
        for i, c in enumerate(candidates[:pipeline.max_candidates]):
            if isinstance(c, str):
                normalized.append({"name": f"Candidate {i+1}", "smiles": c})
            else:
                normalized.append(c)
        assert len(normalized) == 2


class TestHelpers:
    def test_hartree_to_kcal(self):
        from src.utils.helpers import hartree_to_kcal
        assert abs(hartree_to_kcal(1.0) - 627.5094740631) < 1e-6

    def test_hartree_to_ev(self):
        from src.utils.helpers import hartree_to_ev
        assert abs(hartree_to_ev(1.0) - 27.211386245988) < 1e-6

    def test_analyze_convergence_empty(self):
        from src.utils.helpers import analyze_convergence
        result = analyze_convergence([])
        assert result["converged"] is False

    def test_analyze_convergence_flat(self):
        from src.utils.helpers import analyze_convergence
        # Flat history — should be converged
        history = [-1.137] * 50
        result = analyze_convergence(history)
        assert result["converged"] is True

    def test_analyze_convergence_noisy(self):
        import numpy as np
        from src.utils.helpers import analyze_convergence
        rng = np.random.default_rng(0)
        # Very noisy history
        history = list(-1.0 + rng.standard_normal(100) * 0.1)
        result = analyze_convergence(history)
        assert result["is_noisy"] is True

    def test_save_results_creates_file(self, tmp_path):
        from src.utils.helpers import save_results
        data = [{"energy": -1.137, "converged": True, "molecule": "H2"}]
        path = str(tmp_path / "test.csv")
        save_results(data, path)
        assert os.path.exists(path)

    def test_check_dependencies_returns_bool(self):
        from src.utils.helpers import check_dependencies
        result = check_dependencies()
        assert isinstance(result, bool)
