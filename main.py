"""
main.py — Entry point for the quantum drug discovery pipeline.

Usage:
    python main.py --mode benchmark
    python main.py --mode screen --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python main.py --mode full_screen --library data/candidates.txt
    python main.py --mode benchmark --backend ibm_real
    python main.py --mode full_screen --binding-mode vqe_complex
    python main.py --mode benchmark --zne
"""

import click
import sys
import os

# Make sure src is on the path
sys.path.insert(0, os.path.dirname(__file__))

# Load .env before importing config
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.utils.helpers import setup_logging, print_banner, save_results
from src.molecule.loader import MoleculeLoader
from src.vqe.runner import VQERunner
from src.screening.pipeline import DrugScreeningPipeline
import config


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["benchmark", "screen", "full_screen"]),
    default="benchmark",
    help="Run mode: benchmark (H2/LiH validation), screen (single molecule), full_screen (library)",
)
@click.option(
    "--smiles",
    default=None,
    help="SMILES string of the molecule to screen (used with --mode screen)",
)
@click.option(
    "--library",
    default=None,
    type=click.Path(exists=False),
    help="Path to text file of SMILES strings, one per line (used with --mode full_screen)",
)
@click.option(
    "--backend",
    type=click.Choice(["aer_statevector", "aer_qasm", "aer_noise", "ibm_real"]),
    default=None,
    help="Override the backend set in config.py",
)
@click.option(
    "--ansatz",
    type=click.Choice(["efficient_su2", "uccsd", "real_amplitudes", "two_local"]),
    default=None,
    help="Override the ansatz type set in config.py",
)
@click.option(
    "--binding-mode",
    type=click.Choice(["vqe_complex", "empirical", "off"]),
    default=None,
    help="Binding energy calculation mode for full_screen",
)
@click.option(
    "--zne",
    is_flag=True,
    default=False,
    help="Enable Zero Noise Extrapolation post-processing",
)
@click.option(
    "--output",
    default="results/output.csv",
    help="Path for the output CSV file",
)
def main(mode, smiles, library, backend, ansatz, binding_mode, zne, output):
    setup_logging(config.LOG_LEVEL)
    print_banner()

    # Apply CLI overrides to config
    if backend:
        config.DEFAULT_BACKEND = backend
    if ansatz:
        config.ANSATZ_TYPE = ansatz
    if zne:
        config.ENABLE_ZNE = True

    # Ensure results directory exists
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if mode == "benchmark":
        run_benchmark()

    elif mode == "screen":
        if not smiles:
            click.echo("Error: --smiles is required for --mode screen", err=True)
            sys.exit(1)
        run_single_screen(smiles, output)

    elif mode == "full_screen":
        run_full_screen(library, output, binding_mode=binding_mode)


def run_benchmark():
    """Validate the pipeline against known exact energies for H2, LiH, BeH2."""
    from src.utils.helpers import logger

    benchmark_molecules = [
        {
            "name": "H2",
            "geometry": "H 0.0 0.0 0.0; H 0.0 0.0 0.735",
            "exact_energy_ha": -1.137270,
            "active_space": None,
        },
        {
            "name": "LiH",
            "geometry": "Li 0.0 0.0 0.0; H 0.0 0.0 1.596",
            "exact_energy_ha": -7.882173,
            "active_space": (2, 5),
        },
        {
            "name": "BeH2",
            "geometry": "Be 0.0 0.0 0.0; H 0.0 0.0 1.300; H 0.0 0.0 -1.300",
            "exact_energy_ha": -15.595240,
            "active_space": (4, 5),
        },
    ]

    loader = MoleculeLoader()
    results = []

    for mol_data in benchmark_molecules:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {mol_data['name']}")
        logger.info(f"{'='*60}")

        problem = loader.from_geometry(
            geometry=mol_data["geometry"],
            basis=config.BASIS_SET,
            active_space=mol_data["active_space"],
        )

        runner = VQERunner(
            backend=config.DEFAULT_BACKEND,
            ansatz_type=config.ANSATZ_TYPE,
            ansatz_reps=config.ANSATZ_REPS,
            optimizer_name=config.OPTIMIZER_NAME,
            max_iterations=config.MAX_ITERATIONS,
            convergence_threshold=config.CONVERGENCE_THRESHOLD,
            num_restarts=config.NUM_RESTARTS,
            enable_zne=getattr(config, "ENABLE_ZNE", False),
            max_retries=getattr(config, "MAX_RETRIES", 2),
        )

        result = runner.run(problem)

        error_ha = abs(result.ground_energy - mol_data["exact_energy_ha"])
        error_pct = error_ha / abs(mol_data["exact_energy_ha"]) * 100

        logger.info(f"Ground-state energy : {result.ground_energy:.6f} Ha")
        logger.info(f"Exact FCI energy    : {mol_data['exact_energy_ha']:.6f} Ha")
        logger.info(f"Absolute error      : {error_ha:.6f} Ha ({error_pct:.4f}%)")
        logger.info(f"Iterations          : {result.num_iterations}")
        logger.info(f"Converged           : {result.converged}")
        if result.zne_energy is not None:
            zne_error = abs(result.zne_energy - mol_data["exact_energy_ha"])
            logger.info(f"ZNE energy          : {result.zne_energy:.6f} Ha (error: {zne_error:.6f} Ha)")

        results.append(
            {
                "molecule": mol_data["name"],
                "vqe_energy_ha": result.ground_energy,
                "exact_energy_ha": mol_data["exact_energy_ha"],
                "error_ha": error_ha,
                "error_pct": error_pct,
                "zne_energy_ha": result.zne_energy,
                "iterations": result.num_iterations,
                "converged": result.converged,
                "num_qubits": result.num_qubits,
                "num_parameters": result.num_parameters,
            }
        )

        if config.SAVE_PLOTS:
            runner.plot_convergence(
                save_path=f"{config.RESULTS_DIR}/{mol_data['name']}_convergence.png"
            )

    save_results(results, f"{config.RESULTS_DIR}/benchmark_results.csv")
    logger.info(f"\nBenchmark complete. Results saved to {config.RESULTS_DIR}/benchmark_results.csv")


def run_single_screen(smiles: str, output: str):
    """Screen a single molecule given as a SMILES string."""
    from src.utils.helpers import logger

    loader = MoleculeLoader()
    problem = loader.from_smiles(
        smiles=smiles,
        basis=config.BASIS_SET,
        active_space=config.ACTIVE_SPACE,
    )

    runner = VQERunner(
        backend=config.DEFAULT_BACKEND,
        ansatz_type=config.ANSATZ_TYPE,
        ansatz_reps=config.ANSATZ_REPS,
        optimizer_name=config.OPTIMIZER_NAME,
        max_iterations=config.MAX_ITERATIONS,
        convergence_threshold=config.CONVERGENCE_THRESHOLD,
        num_restarts=config.NUM_RESTARTS,
        enable_zne=getattr(config, "ENABLE_ZNE", False),
        max_retries=getattr(config, "MAX_RETRIES", 2),
    )

    result = runner.run(problem)

    logger.info(f"\nSMILES          : {smiles}")
    logger.info(f"Num qubits      : {result.num_qubits}")
    logger.info(f"Ground energy   : {result.ground_energy:.6f} Ha")
    logger.info(f"                : {result.ground_energy * config.HARTREE_TO_KCAL:.2f} kcal/mol")
    logger.info(f"Converged       : {result.converged}")
    if result.zne_energy is not None:
        logger.info(f"ZNE energy      : {result.zne_energy:.6f} Ha")

    save_results(
        [
            {
                "smiles": smiles,
                "ground_energy_ha": result.ground_energy,
                "ground_energy_kcal": result.ground_energy * config.HARTREE_TO_KCAL,
                "zne_energy_ha": result.zne_energy,
                "num_qubits": result.num_qubits,
                "converged": result.converged,
            }
        ],
        output,
    )


def run_full_screen(library_path: str, output: str, binding_mode: str = None):
    """Screen a full library of molecules and rank by binding affinity."""
    from src.utils.helpers import logger

    effective_binding_mode = binding_mode or "empirical"

    pipeline = DrugScreeningPipeline(
        backend=config.DEFAULT_BACKEND,
        ansatz_type=config.ANSATZ_TYPE,
        optimizer_name=config.OPTIMIZER_NAME,
        max_candidates=config.MAX_CANDIDATES,
        binding_threshold=config.BINDING_THRESHOLD_KCAL,
        binding_mode=effective_binding_mode,
        enable_zne=getattr(config, "ENABLE_ZNE", False),
        max_retries=getattr(config, "MAX_RETRIES", 2),
    )

    if library_path and os.path.exists(library_path):
        with open(library_path) as f:
            candidates = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        logger.info(f"Loaded {len(candidates)} candidates from {library_path}")
    else:
        logger.info("No library path provided — using built-in demo candidates.")
        candidates = pipeline.get_demo_candidates()

    results = pipeline.screen(candidates)
    pipeline.print_ranking(results)
    pipeline.save_results(results, output)

    if config.SAVE_PLOTS:
        pipeline.plot_results(results, f"{config.RESULTS_DIR}/screening_results.png")

    logger.info(f"\nScreening complete. {len(results)} molecules evaluated.")
    logger.info(f"Results saved to {output}")


if __name__ == "__main__":
    main()
