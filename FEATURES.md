# FEATURES — Quantum Drug Discovery (VQE Pipeline)

This document summarizes the major features and capabilities present in this codebase.

## Overview

- End-to-end Variational Quantum Eigensolver (VQE) pipeline for quantum chemistry and drug discovery.
- Built with Qiskit and PySCF; supports simulator and IBM Quantum runtime backends.
- Designed for benchmarks, single-molecule screening, and high-throughput library screening.

## CLI & Entry Point

- `main.py`: Click-based CLI with modes: `benchmark`, `screen`, `full_screen`.
  - Options to override backend, ansatz, binding mode, enable ZNE, and output paths.
  - Quickstart examples for H2 benchmark, single-molecule screen, and full library screening.

## Configuration

- `config.py` centralizes all settings:
  - Backend selection (`aer_statevector`, `aer_qasm`, `aer_noise`, `ibm_real`).
  - VQE hyperparameters: ansatz type, reps, optimizer, max iterations, convergence threshold.
  - Quantum-chemistry settings: basis set, active-space configuration, fermion-to-qubit mapping.
  - Noise mitigation toggles, auto-active-space threshold, retries, output directories.

## Molecule Handling

- `src/molecule/loader.py`:
  - Load molecules from SMILES, geometry strings, PDB files, or built-in library.
  - Convert SMILES → 3D geometry using RDKit (with graceful fallback to H2 if RDKit missing).
  - Build Qiskit-Nature `ElectronicStructureProblem` via PySCF driver and optional active-space transformer.
  - `MoleculeProblem` dataclass stores problem metadata and the underlying Qiskit object.

- `src/molecule/encoder.py`:
  - Encode a molecular problem into a qubit Hamiltonian (SparsePauliOp) via Qiskit-Nature.
  - Supports fermion-to-qubit mappings: Jordan-Wigner, Bravyi-Kitaev, Parity (with parity reduction).
  - Utilities: `EncodedHamiltonian` summary, Pauli-term inspection.

## VQE Components

- `src/vqe/ansatz.py`:
  - `AnsatzBuilder` factory for hardware-efficient ansätze (`efficient_su2`, `real_amplitudes`, `two_local`) and UCCSD (chemistry ansatz).
  - Utilities for initial parameter generation and circuit drawing.

- `src/vqe/optimizer.py`:
  - `OptimizerFactory` with common classical optimizers: COBYLA, SPSA, L-BFGS-B, Nelder-Mead, ADAM, SLSQP.
  - Backend-aware recommendations (e.g., SPSA for noisy hardware).

- `src/vqe/runner.py`:
  - `VQERunner` orchestrates the full VQE workflow:
    - Encoding → Ansatz build → Estimator/backend selection → Classical optimization loop.
    - Multiple restarts to avoid local minima; convergence checking via moving-window statistics.
    - Automatic retry logic and graceful fallbacks (e.g., to statevector simulator).
    - Auto active-space truncation when qubit count exceeds threshold.
    - Optional Zero Noise Extrapolation (ZNE) integration.
    - Produces `VQEResult` objects with full metadata and plotting utilities.

## Noise Mitigation

- `src/noise/mitigation.py`:
  - `ZeroNoiseExtrapolator` — gate-folding ZNE with multiple extrapolation methods (linear, quadratic, exponential).
  - `ReadoutMitigator` — readout calibration matrix, pseudo-inverse correction to measurement counts.
  - Helpers to configure IBM Runtime mitigation/resilience levels (readout, ZNE, PEC).

## Screening Pipeline

- `src/screening/pipeline.py`:
  - `DrugScreeningPipeline` for high-throughput screening of SMILES libraries.
  - Modes for binding affinity calculation:
    - `vqe_complex`: full quantum VQE on ligand+receptor complex (most accurate).
    - `empirical`: fast proxy using molecular descriptors for large libraries.
    - `off`: report ligand energy only.
  - Ranking, CSV export, and plotting of results; built-in demo candidates and receptor fragment geometry.

## Utilities & Helpers

- `src/utils/helpers.py`:
  - Logging setup, startup banner, result CSV export.
  - Unit conversions (Hartree ↔ kcal/mol, eV conversions).
  - Dependency checker that prints missing packages and install hints.
  - Convergence analysis and plotting helpers (energy landscape, convergence curves).

## Tests & Examples

- `tests/` contains unit tests for encoder, VQE runner, and pipeline (`test_encoder.py`, `test_vqe.py`, `test_pipeline.py`).
- `notebooks/tutorial.ipynb` demonstrates usage and walkthrough examples.

## Outputs & Artifacts

- `results/` directory stores output CSVs, convergence plots, screening plots, and saved circuits.
- The pipeline can save circuits and plots (`SAVE_CIRCUITS`, `SAVE_PLOTS` in `config.py`).

## Dependencies

- Full `requirements.txt` lists the core Qiskit stack, PySCF, RDKit, OpenFermion, PennyLane (optional), scientific packages (`numpy`, `scipy`, `matplotlib`, `pandas`) and utilities (`click`, `tqdm`, `rich`).

## Robustness & Design Decisions

- Designed for flexibility: simulator-first defaults for easy testing, with straightforward switches to run on IBM hardware.
- Graceful degradation and fallbacks when optional libraries (RDKit, qiskit-ibm-runtime, qiskit-aer) are missing.
- Emphasis on reproducibility: configuration centralized, multiple-restart strategy, optional ZNE, and active-space controls.

## License

- MIT License (see `LICENSE.md`).

---

If you want this file committed and pushed to `origin/main`, I can create a Git commit and push it for you.