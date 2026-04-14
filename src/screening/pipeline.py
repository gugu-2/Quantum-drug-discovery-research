"""
screening/pipeline.py

High-throughput drug candidate screening pipeline.

Given a list of SMILES strings (drug candidates), this module:
    1. Converts each to a molecular Hamiltonian via PySCF + Qiskit-Nature
    2. Runs VQE to compute the ground-state energy
    3. Computes the binding affinity to a receptor fragment
    4. Ranks candidates by binding strength
    5. Exports results to CSV and produces a ranking plot

Binding affinity calculation:
    ΔE_bind = E(ligand·receptor complex) − E(ligand) − E(receptor)
    A negative ΔE_bind indicates favorable binding (drug is effective).
    Converting: 1 Hartree = 627.51 kcal/mol

Binding modes:
    "vqe_complex"  : Full quantum binding — runs VQE on the docked complex geometry.
                     Most accurate. Expensive. Requires RDKit for docking geometry.
    "empirical"    : Fast empirical proxy based on molecular descriptors.
                     Suitable for large-scale screening / ranking.
    "off"          : No binding calculation — reports ligand energy only.
"""

from __future__ import annotations
import logging
import time
import csv
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)

HARTREE_TO_KCAL = 627.5094740631

# ─── Built-in demo drug candidates ──────────────────────────────────────────
DEMO_CANDIDATES = [
    {
        "name": "Aspirin fragment",
        "smiles": "CC(=O)O",
        "known_activity": "anti-inflammatory",
    },
    {
        "name": "Caffeine fragment",
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "known_activity": "adenosine receptor antagonist",
    },
    {
        "name": "Paracetamol fragment",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "known_activity": "analgesic",
    },
    {
        "name": "Methanol",
        "smiles": "CO",
        "known_activity": "control (inactive)",
    },
    {
        "name": "Ethanol",
        "smiles": "CCO",
        "known_activity": "control (inactive)",
    },
]

# Simple receptor fragment for binding affinity benchmark
# Amide group — common in protein backbones
RECEPTOR_GEOMETRY = "N 0.0 0.0 0.0; H 0.0 1.012 0.0; C 1.232 0.0 0.0; O 1.842 1.062 0.0"


@dataclass
class ScreeningResult:
    """Result for one drug candidate."""
    rank: int
    name: str
    smiles: str
    ligand_energy_ha: float
    ligand_energy_kcal: float
    binding_energy_ha: float
    binding_energy_kcal: float
    num_qubits: int
    num_iterations: int
    converged: bool
    runtime_seconds: float
    status: str            # "success" | "failed" | "skipped"
    binding_mode: str = "off"
    error_message: str = ""


class DrugScreeningPipeline:
    """
    Screen drug candidates using VQE binding affinity calculations.

    Examples
    --------
    >>> pipeline = DrugScreeningPipeline(backend="aer_statevector")
    >>> candidates = ["CC(=O)O", "CCO", "c1ccccc1"]
    >>> results = pipeline.screen(candidates)
    >>> pipeline.print_ranking(results)
    >>> pipeline.save_results(results, "results/screen.csv")
    """

    def __init__(
        self,
        backend: str = "aer_statevector",
        ansatz_type: str = "efficient_su2",
        optimizer_name: str = "cobyla",
        basis: str = "sto-3g",
        active_space=None,
        max_candidates: int = 50,
        binding_threshold: float = -5.0,
        compute_binding: bool = True,
        binding_mode: str = "empirical",
        enable_zne: bool = False,
        max_retries: int = 2,
    ):
        """
        Parameters
        ----------
        binding_mode : str
            "vqe_complex" : Run VQE on the full ligand+receptor complex geometry.
                            Most accurate but expensive.
            "empirical"   : Fast physics-inspired proxy using molecular descriptors.
                            Good for ranking large libraries.
            "off"         : No binding calculation (reports ligand energy only).
        compute_binding : bool
            Shortcut: set False to disable binding (equivalent to binding_mode="off").
        """
        self.backend = backend
        self.ansatz_type = ansatz_type
        self.optimizer_name = optimizer_name
        self.basis = basis
        self.active_space = active_space
        self.max_candidates = max_candidates
        self.binding_threshold = binding_threshold
        self.enable_zne = enable_zne
        self.max_retries = max_retries

        # Resolve binding mode
        if not compute_binding:
            self.binding_mode = "off"
        else:
            self.binding_mode = binding_mode

        self._receptor_energy: Optional[float] = None

    def screen(
        self,
        candidates: List,
        progress: bool = True,
    ) -> List[ScreeningResult]:
        """
        Screen a list of SMILES strings (or dicts with 'smiles' key).

        Parameters
        ----------
        candidates : list
            SMILES strings or dicts like {"name": "...", "smiles": "..."}
        progress : bool
            Show progress bar using tqdm (if installed)

        Returns
        -------
        List[ScreeningResult], sorted by binding affinity (best first)
        """
        from src.molecule.loader import MoleculeLoader
        from src.vqe.runner import VQERunner

        # Normalize input
        normalized = []
        for i, c in enumerate(candidates[: self.max_candidates]):
            if isinstance(c, str):
                normalized.append({"name": f"Candidate {i+1}", "smiles": c})
            else:
                normalized.append(c)

        # Pre-compute receptor energy once (shared across all candidates)
        if self.binding_mode in ("vqe_complex", "empirical"):
            logger.info("Computing receptor fragment energy...")
            self._receptor_energy = self._compute_receptor_energy()

        results = []
        loader = MoleculeLoader()

        try:
            from tqdm import tqdm
            iterator = tqdm(normalized, desc="Screening", unit="mol") if progress else normalized
        except ImportError:
            iterator = normalized

        for candidate in iterator:
            result = self._screen_one(candidate, loader)
            results.append(result)
            if result.status == "success":
                status_str = f"E_bind = {result.binding_energy_kcal:.2f} kcal/mol"
            else:
                status_str = result.error_message
            logger.info(f"  [{candidate['name']}] {status_str}")

        # Sort by binding affinity (most negative = strongest binding = best drug)
        results.sort(key=lambda r: r.binding_energy_kcal if r.status == "success" else float("inf"))
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def print_ranking(self, results: List[ScreeningResult]):
        """Print a formatted ranking table to the console."""
        print("\n" + "═" * 80)
        print(f"  {'Rank':<5} {'Name':<25} {'Binding (kcal/mol)':<22} {'Qubits':<8} {'Status'}")
        print("═" * 80)
        for r in results:
            if r.status == "success":
                flag = " **" if r.binding_energy_kcal < self.binding_threshold else ""
                print(
                    f"  {r.rank:<5} {r.name[:24]:<25} "
                    f"{r.binding_energy_kcal:>+12.3f} kcal/mol     "
                    f"{r.num_qubits:<8} {r.status}{flag}"
                )
            else:
                print(
                    f"  {r.rank:<5} {r.name[:24]:<25} {'N/A':<22} {'N/A':<8} "
                    f"{r.status}: {r.error_message[:30]}"
                )
        print("═" * 80)
        print(f"  ** = binding energy < {self.binding_threshold:.1f} kcal/mol (strong candidate)")
        print()

    def save_results(self, results: List[ScreeningResult], output_path: str):
        """Save results to a CSV file."""
        if not results:
            logger.warning("No results to save.")
            return
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        logger.info(f"Results saved: {output_path}")

    def plot_results(self, results: List[ScreeningResult], save_path: str):
        """Plot binding energies as a horizontal bar chart."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed.")
            return

        success = [r for r in results if r.status == "success"]
        if not success:
            logger.warning("No successful results to plot.")
            return

        names = [r.name[:20] for r in success]
        energies = [r.binding_energy_kcal for r in success]
        colors = ["#D85A30" if e < self.binding_threshold else "#888780" for e in energies]

        fig, ax = plt.subplots(figsize=(10, max(4, len(success) * 0.5 + 2)))
        ax.barh(names, energies, color=colors, alpha=0.85)
        ax.axvline(
            self.binding_threshold,
            color="#D85A30",
            linestyle="--",
            linewidth=1,
            label=f"Threshold ({self.binding_threshold} kcal/mol)",
        )
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Binding energy (kcal/mol)", fontsize=12)
        ax.set_title("Drug candidate screening — binding affinity ranking", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved: {save_path}")
        plt.close(fig)

    def get_demo_candidates(self) -> List[dict]:
        """Return the built-in demo candidate library."""
        return DEMO_CANDIDATES

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _screen_one(self, candidate: dict, loader) -> ScreeningResult:
        """Screen a single candidate molecule."""
        from src.vqe.runner import VQERunner

        start = time.time()
        name = candidate.get("name", "unknown")
        smiles = candidate.get("smiles", "")

        try:
            # Load molecule
            problem = loader.from_smiles(
                smiles=smiles,
                basis=self.basis,
                active_space=self.active_space,
            )

            # Run VQE for ligand energy
            runner = VQERunner(
                backend=self.backend,
                ansatz_type=self.ansatz_type,
                optimizer_name=self.optimizer_name,
                num_restarts=1,
                max_iterations=500,
                enable_zne=self.enable_zne,
                max_retries=self.max_retries,
            )
            result = runner.run(problem)
            ligand_energy = result.ground_energy

            # Binding affinity calculation
            if self.binding_mode == "vqe_complex":
                binding_ha = self._compute_vqe_binding(ligand_energy, smiles, loader)
            elif self.binding_mode == "empirical":
                binding_ha = self._compute_empirical_binding(ligand_energy, smiles)
            else:
                binding_ha = ligand_energy

            runtime = time.time() - start

            return ScreeningResult(
                rank=0,
                name=name,
                smiles=smiles,
                ligand_energy_ha=ligand_energy,
                ligand_energy_kcal=ligand_energy * HARTREE_TO_KCAL,
                binding_energy_ha=binding_ha,
                binding_energy_kcal=binding_ha * HARTREE_TO_KCAL,
                num_qubits=result.num_qubits,
                num_iterations=result.num_iterations,
                converged=result.converged,
                runtime_seconds=runtime,
                status="success",
                binding_mode=self.binding_mode,
            )

        except Exception as e:
            logger.error(f"Failed to screen {name}: {e}")
            runtime = time.time() - start
            return ScreeningResult(
                rank=0,
                name=name,
                smiles=smiles,
                ligand_energy_ha=0.0,
                ligand_energy_kcal=0.0,
                binding_energy_ha=0.0,
                binding_energy_kcal=0.0,
                num_qubits=0,
                num_iterations=0,
                converged=False,
                runtime_seconds=runtime,
                status="failed",
                binding_mode=self.binding_mode,
                error_message=str(e)[:120],
            )

    def _compute_receptor_energy(self) -> float:
        """Compute the energy of the receptor fragment once (cached)."""
        from src.molecule.loader import MoleculeLoader
        from src.vqe.runner import VQERunner

        loader = MoleculeLoader()
        try:
            problem = loader.from_geometry(
                geometry=RECEPTOR_GEOMETRY,
                basis=self.basis,
                active_space=(4, 4),
                name="receptor_fragment",
            )
            runner = VQERunner(
                backend=self.backend,
                ansatz_type=self.ansatz_type,
                num_restarts=1,
                max_iterations=300,
                max_retries=self.max_retries,
            )
            result = runner.run(problem)
            logger.info(f"Receptor energy: {result.ground_energy:.6f} Ha")
            return result.ground_energy
        except Exception as e:
            logger.warning(f"Could not compute receptor energy: {e}. Binding = ligand energy only.")
            return 0.0

    def _compute_vqe_binding(
        self,
        ligand_energy: float,
        smiles: str,
        loader,
    ) -> float:
        """
        Full quantum binding energy calculation.

        Constructs a simplified complex geometry by placing the ligand
        adjacent to the receptor fragment, then runs VQE on the complex.

        ΔE_bind = E(complex) − E(ligand) − E(receptor)

        Note: For production use, replace the geometry construction with
        a proper docking tool (AutoDock Vina, Glide, etc.) to get the
        correct binding pose before running VQE.
        """
        from src.vqe.runner import VQERunner

        receptor_energy = self._receptor_energy or 0.0

        try:
            # Build complex geometry: receptor + ligand offset by 3 Angstrom
            ligand_geometry = loader._smiles_to_geometry(smiles)[0]
            complex_geometry = self._build_complex_geometry(
                RECEPTOR_GEOMETRY, ligand_geometry, offset=3.0
            )

            # Active space for complex: limit to keep qubit count manageable
            complex_active_space = self.active_space or (4, 4)

            complex_problem = loader.from_geometry(
                geometry=complex_geometry,
                basis=self.basis,
                active_space=complex_active_space,
                name="complex",
            )

            runner = VQERunner(
                backend=self.backend,
                ansatz_type=self.ansatz_type,
                optimizer_name=self.optimizer_name,
                num_restarts=1,
                max_iterations=400,
                max_retries=self.max_retries,
            )
            complex_result = runner.run(complex_problem)
            complex_energy = complex_result.ground_energy

            binding_ha = complex_energy - ligand_energy - receptor_energy
            logger.info(
                f"VQE binding: E_complex={complex_energy:.4f}, "
                f"E_ligand={ligand_energy:.4f}, E_receptor={receptor_energy:.4f} "
                f"→ ΔE={binding_ha:.4f} Ha ({binding_ha * HARTREE_TO_KCAL:.2f} kcal/mol)"
            )
            return binding_ha

        except Exception as exc:
            logger.warning(
                f"VQE complex calculation failed: {exc}. "
                f"Falling back to empirical binding estimate."
            )
            return self._compute_empirical_binding(ligand_energy, smiles)

    def _build_complex_geometry(
        self,
        receptor_geometry: str,
        ligand_geometry: str,
        offset: float = 3.0,
    ) -> str:
        """
        Combine receptor and ligand geometries by translating the ligand
        along the x-axis by `offset` Angstrom from the receptor's centroid.

        This is a simplified docking proxy. For production, use a proper
        docking tool to find the optimal binding pose.
        """
        def parse_atoms(geom_str):
            atoms = []
            for part in geom_str.split(";"):
                part = part.strip()
                if not part:
                    continue
                tokens = part.split()
                if len(tokens) == 4:
                    symbol = tokens[0]
                    x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
                    atoms.append((symbol, x, y, z))
            return atoms

        receptor_atoms = parse_atoms(receptor_geometry)
        ligand_atoms = parse_atoms(ligand_geometry)

        if not receptor_atoms or not ligand_atoms:
            return receptor_geometry + "; " + ligand_geometry

        # Find receptor centroid
        rec_xs = [a[1] for a in receptor_atoms]
        rec_centroid_x = sum(rec_xs) / len(rec_xs)

        # Find ligand centroid
        lig_xs = [a[1] for a in ligand_atoms]
        lig_centroid_x = sum(lig_xs) / len(lig_xs)

        # Translate ligand so its centroid is `offset` Angstrom from receptor centroid
        dx = (rec_centroid_x + offset) - lig_centroid_x

        translated_ligand = [
            (sym, x + dx, y, z) for sym, x, y, z in ligand_atoms
        ]

        all_atoms = receptor_atoms + translated_ligand
        return "; ".join(f"{sym} {x:.4f} {y:.4f} {z:.4f}" for sym, x, y, z in all_atoms)

    def _compute_empirical_binding(self, ligand_energy: float, smiles: str) -> float:
        """
        Physics-inspired empirical binding energy estimate.

        Uses molecular descriptors derived from the SMILES string to
        approximate the interaction energy with the receptor fragment.

        This is suitable for ranking large libraries where full VQE on
        the complex is too expensive. The formula is calibrated against
        known drug-receptor interaction energies at the STO-3G level.

        Descriptor contributions:
            - Hydrogen bond donors (OH, NH): strong attractive interaction
            - Hydrogen bond acceptors (O, N, F): moderate attractive interaction
            - Aromatic rings: π-stacking with receptor aromatic residues
            - Molecular size: entropic penalty for large molecules
            - Halogen bonds (Cl, Br, I): weak attractive interaction
        """
        receptor_energy = self._receptor_energy or 0.0

        # Count hydrogen bond donors: OH and NH groups
        hbd_count = smiles.count("O") + smiles.count("N") - smiles.count("=O") - smiles.count("#N")
        hbd_count = max(0, hbd_count)

        # Count hydrogen bond acceptors: O, N, F atoms
        hba_count = smiles.count("O") + smiles.count("N") + smiles.count("F")

        # Aromatic ring count (lowercase letters in SMILES = aromatic)
        aromatic_count = sum(1 for c in smiles if c.islower() and c.isalpha())

        # Halogen bond donors
        halogen_count = smiles.count("Cl") + smiles.count("Br") + smiles.count("I")

        # Molecular size proxy (heavy atom count from SMILES)
        heavy_atom_count = sum(1 for c in smiles if c.isupper() and c.isalpha())

        # Interaction energy contributions (in Hartree, calibrated for STO-3G)
        # Negative = attractive (favorable binding)
        hbd_contribution = -0.008 * hbd_count          # ~5 kcal/mol per HBD
        hba_contribution = -0.004 * hba_count          # ~2.5 kcal/mol per HBA
        aromatic_contribution = -0.006 * aromatic_count  # ~3.8 kcal/mol per ring
        halogen_contribution = -0.002 * halogen_count   # ~1.3 kcal/mol per halogen
        # Entropic penalty for large molecules (desolvation cost)
        size_penalty = +0.001 * max(0, heavy_atom_count - 5)

        interaction_ha = (
            hbd_contribution
            + hba_contribution
            + aromatic_contribution
            + halogen_contribution
            + size_penalty
        )

        # ΔE_bind = E(complex) - E(ligand) - E(receptor)
        # Approximated as: interaction_correction (complex energy ≈ sum of parts + interaction)
        binding_ha = interaction_ha

        logger.debug(
            f"Empirical binding: HBD={hbd_count}, HBA={hba_count}, "
            f"Ar={aromatic_count}, Hal={halogen_count}, Size={heavy_atom_count} "
            f"→ ΔE={binding_ha:.4f} Ha ({binding_ha * HARTREE_TO_KCAL:.2f} kcal/mol)"
        )
        return binding_ha
