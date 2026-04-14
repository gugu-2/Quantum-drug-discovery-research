"""
notebooks/tutorial.py

Step-by-step tutorial for the Quantum Drug Discovery pipeline.
Convert to a Jupyter notebook with:
    jupytext --to notebook tutorial.py

Or run directly: python tutorial.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Setup and dependency check
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import setup_logging, check_dependencies, hartree_to_kcal

setup_logging("INFO")
print("Checking dependencies...")
all_ok = check_dependencies()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Load your first molecule — H2
#
# The hydrogen molecule is the canonical VQE benchmark.
# - 2 hydrogen atoms separated by 0.735 Angstrom (equilibrium bond length)
# - STO-3G basis: minimal basis set, fastest computation
# - Full orbital space: 2 spatial orbitals → 4 spin-orbitals → 4 qubits (JW)
# ─────────────────────────────────────────────────────────────────────────────

from src.molecule.loader import MoleculeLoader

loader = MoleculeLoader()

print("\n" + "═"*60)
print("  Step 1: Loading H2 molecule")
print("═"*60)

h2_problem = loader.from_name("h2", basis="sto-3g")

print(f"  Name         : {h2_problem.name}")
print(f"  Geometry     : {h2_problem.geometry}")
print(f"  Basis set    : {h2_problem.basis}")
print(f"  Electrons    : {h2_problem.num_electrons}")
print(f"  Orbitals     : {h2_problem.num_orbitals} spatial")
print(f"  Qubits (full): {h2_problem.num_qubits_full}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Encode the molecule as a qubit Hamiltonian
#
# Jordan-Wigner mapping converts each spin-orbital into one qubit.
# The result is a SparsePauliOp — a sum of tensor products of Pauli matrices.
# For H2 in STO-3G: ~15 Pauli terms on 4 qubits.
# ─────────────────────────────────────────────────────────────────────────────

from src.molecule.encoder import MoleculeEncoder

print("\n" + "═"*60)
print("  Step 2: Encoding H2 as qubit Hamiltonian (Jordan-Wigner)")
print("═"*60)

encoder = MoleculeEncoder(mapping="jordan_wigner")
encoded = encoder.encode(h2_problem)

print(f"  Qubits required : {encoded.num_qubits}")
print(f"  Pauli terms     : {len(encoded.qubit_op)}")
print(f"  HF energy       : {encoded.hartree_fock_energy:.6f} Ha (upper bound)")
print(f"  HF energy       : {hartree_to_kcal(encoded.hartree_fock_energy):.2f} kcal/mol")

print("\n  Top 5 Pauli terms (sorted by |coefficient|):")
terms = sorted(encoded.qubit_op.to_list(), key=lambda x: abs(x[1]), reverse=True)
for pauli, coeff in terms[:5]:
    print(f"    {pauli}  {coeff.real:+.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Build the ansatz circuit
#
# The ansatz |ψ(θ)⟩ is a parameterized quantum circuit that VQE optimizes.
# EfficientSU2 is hardware-efficient: alternating Ry/Rz rotations + CNOTs.
# ─────────────────────────────────────────────────────────────────────────────

from src.vqe.ansatz import AnsatzBuilder

print("\n" + "═"*60)
print("  Step 3: Building the EfficientSU2 ansatz")
print("═"*60)

builder = AnsatzBuilder()
ansatz = builder.build(
    ansatz_type="efficient_su2",
    num_qubits=encoded.num_qubits,
    reps=2,
)

print(f"  Qubits      : {ansatz.num_qubits}")
print(f"  Parameters  : {ansatz.num_parameters}")
print(f"  Depth       : {ansatz.depth()}")

print("\n  Circuit diagram:")
print(builder.draw(ansatz, output="text", fold=60))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Run VQE
#
# The VQE loop:
#   1. Bind current parameters θ to the ansatz
#   2. Measure ⟨ψ(θ)|H|ψ(θ)⟩ on the quantum backend
#   3. Classical optimizer updates θ to minimize energy
#   4. Repeat until convergence
# ─────────────────────────────────────────────────────────────────────────────

from src.vqe.runner import VQERunner

print("\n" + "═"*60)
print("  Step 4: Running VQE on H2")
print("═"*60)

runner = VQERunner(
    backend="aer_statevector",      # exact statevector simulation
    ansatz_type="efficient_su2",
    ansatz_reps=2,
    optimizer_name="cobyla",
    max_iterations=500,
    convergence_threshold=1e-6,
    num_restarts=3,                 # 3 random starts to escape local minima
    seed=42,
)

result = runner.run(h2_problem)

print(f"\n  VQE ground energy : {result.ground_energy:.6f} Ha")
print(f"  Exact FCI energy  : -1.137270 Ha")
print(f"  Absolute error    : {abs(result.ground_energy - (-1.137270)) * 1000:.3f} mHa")
print(f"  Converged         : {result.converged}")
print(f"  Iterations        : {result.num_iterations}")
print(f"  Runtime           : {result.runtime_seconds:.1f} s")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: UCCSD — chemistry-accurate ansatz
#
# UCCSD (Unitary Coupled Cluster Singles & Doubles) is the chemistry-standard
# ansatz. It's derived from excitation operators and achieves near-exact
# accuracy. The tradeoff: much deeper circuit than EfficientSU2.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*60)
print("  Step 5: UCCSD ansatz (chemistry accuracy)")
print("═"*60)

runner_uccsd = VQERunner(
    backend="aer_statevector",
    ansatz_type="uccsd",
    optimizer_name="lbfgsb",       # gradient-based: faster for UCCSD
    max_iterations=200,
    num_restarts=1,
    seed=42,
)

result_uccsd = runner_uccsd.run(h2_problem)

print(f"\n  UCCSD ground energy : {result_uccsd.ground_energy:.6f} Ha")
print(f"  Exact FCI energy    : -1.137270 Ha")
print(f"  Absolute error      : {abs(result_uccsd.ground_energy - (-1.137270)) * 1000:.4f} mHa")
print(f"  (Chemical accuracy  : < 1 mHa)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Plot the convergence curve
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*60)
print("  Step 6: Saving convergence plot")
print("═"*60)

os.makedirs("../results", exist_ok=True)
runner.plot_convergence(save_path="../results/h2_convergence.png")
print("  Saved: results/h2_convergence.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: LiH — scaling up to a bigger molecule
#
# LiH is the next step after H2. With active space (2 electrons, 5 orbitals)
# we keep the qubit count manageable: 10 qubits instead of 24.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*60)
print("  Step 7: LiH with active space truncation")
print("═"*60)

lih_problem = loader.from_name(
    "lih",
    basis="sto-3g",
    active_space=(2, 5),   # 2 active electrons, 5 active orbitals → 10 qubits
)

runner_lih = VQERunner(
    backend="aer_statevector",
    ansatz_type="efficient_su2",
    ansatz_reps=3,
    optimizer_name="cobyla",
    max_iterations=600,
    num_restarts=3,
    seed=42,
)

result_lih = runner_lih.run(lih_problem)

print(f"\n  LiH VQE energy  : {result_lih.ground_energy:.6f} Ha")
print(f"  LiH exact (FCI) : -7.882173 Ha")
print(f"  Error           : {abs(result_lih.ground_energy - (-7.882173)) * 1000:.2f} mHa")
print(f"  Qubits used     : {result_lih.num_qubits}")

runner_lih.plot_convergence(save_path="../results/lih_convergence.png")
print("  Saved: results/lih_convergence.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: Drug candidate screening (mini demo)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*60)
print("  Step 8: Drug candidate screening (3 molecules)")
print("═"*60)

from src.screening.pipeline import DrugScreeningPipeline

pipeline = DrugScreeningPipeline(
    backend="aer_statevector",
    ansatz_type="efficient_su2",
    optimizer_name="cobyla",
    compute_binding=True,
)

candidates = [
    {"name": "Methanol",  "smiles": "CO"},
    {"name": "Ethanol",   "smiles": "CCO"},
    {"name": "Acetic acid", "smiles": "CC(=O)O"},
]

results = pipeline.screen(candidates, progress=False)
pipeline.print_ranking(results)
pipeline.save_results(results, "../results/mini_screen.csv")
pipeline.plot_results(results, "../results/mini_screen.png")

print("\n  All outputs saved in results/")
print("\n  Tutorial complete! Next steps:")
print("  1. Add your IBM Quantum token to config.py")
print("  2. Change DEFAULT_BACKEND to 'ibm_real' and run on real hardware")
print("  3. Increase ACTIVE_SPACE to handle larger drug molecules")
print("  4. Swap ANSATZ_TYPE to 'uccsd' for higher chemical accuracy")
print("  5. Add your own SMILES strings to candidates and run full_screen mode")
