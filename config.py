"""
config.py — All project settings in one place.

IBM credentials are loaded from environment variables or a .env file.
NEVER hard-code tokens here. Use:
    export IBM_QUANTUM_TOKEN="your_token"
or create a .env file:
    IBM_QUANTUM_TOKEN=your_token
"""

import os

# Load .env file if present (requires python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── IBM Quantum credentials ────────────────────────────────────────────────
# Set via environment variable: export IBM_QUANTUM_TOKEN="your_token"
IBM_QUANTUM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN", "")
IBM_QUANTUM_INSTANCE = os.environ.get("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")

# ─── Simulation backend ──────────────────────────────────────────────────────
# Options: "aer_statevector"  → exact (slow for >25 qubits)
#          "aer_qasm"         → shot-based, faster, noisier
#          "aer_noise"        → fake noise model for testing mitigation
#          "ibm_real"         → actual IBM Quantum hardware (needs token)
DEFAULT_BACKEND = "aer_statevector"

# Number of measurement shots (only used for aer_qasm and ibm_real)
SHOTS = 8192

# ─── VQE settings ────────────────────────────────────────────────────────────
# Ansatz type: "efficient_su2" | "uccsd" | "real_amplitudes" | "two_local"
ANSATZ_TYPE = "efficient_su2"

# Circuit depth (reps). Higher = more expressive but harder to optimize.
ANSATZ_REPS = 2

# Classical optimizer: "cobyla" | "spsa" | "lbfgsb" | "neldermead"
# - cobyla:    gradient-free, robust, good for simulator
# - spsa:      gradient-free, noise-robust, best for real hardware
# - lbfgsb:   gradient-based, fastest on simulator (no noise)
OPTIMIZER_NAME = "cobyla"

# Maximum optimizer iterations
MAX_ITERATIONS = 1000

# Energy convergence threshold (Hartree)
CONVERGENCE_THRESHOLD = 1e-6

# Initial parameter strategy: "random" | "zero" | "small"
INITIAL_PARAMS = "random"

# Number of random restarts to escape local minima
NUM_RESTARTS = 3

# ─── Quantum chemistry settings ───────────────────────────────────────────────
# Basis set for molecular integrals
# "sto-3g"  → minimal, fast (use for testing)
# "6-31g"   → better accuracy (standard for drug discovery screening)
# "cc-pvdz" → high accuracy (use for final results)
BASIS_SET = "sto-3g"

# Fermion-to-qubit mapping: "jordan_wigner" | "bravyi_kitaev" | "parity"
QUBIT_MAPPING = "jordan_wigner"

# Active space: (num_electrons, num_orbitals) or None for full space
# None = use full orbital space (expensive for large molecules!)
# (4, 4) = 4 electrons in 4 orbitals → 8 qubits with spin
ACTIVE_SPACE = None

# Auto-select active space when qubit count exceeds this threshold
# Set to None to disable auto active space selection
AUTO_ACTIVE_SPACE_MAX_QUBITS = 12

# ─── Noise mitigation ─────────────────────────────────────────────────────────
# Enable Zero Noise Extrapolation after VQE (adds overhead but improves accuracy)
ENABLE_ZNE = False

# Resilience level for IBM Runtime (0=none, 1=readout, 2=ZNE, 3=PEC)
RESILIENCE_LEVEL = 2

# Zero Noise Extrapolation noise factors
ZNE_NOISE_FACTORS = [1, 2, 3]

# ─── Robustness / retry settings ──────────────────────────────────────────────
# Number of times to retry a failed VQE run (useful for hardware flakiness)
MAX_RETRIES = 2

# Timeout per VQE run in seconds (None = no timeout)
VQE_TIMEOUT_SECONDS = None

# ─── Screening pipeline ───────────────────────────────────────────────────────
# Maximum number of candidates to screen
MAX_CANDIDATES = 50

# Binding energy threshold (kcal/mol) — candidates below this are flagged
BINDING_THRESHOLD_KCAL = -5.0

# Hartree to kcal/mol conversion
HARTREE_TO_KCAL = 627.5094740631

# ─── Output settings ──────────────────────────────────────────────────────────
RESULTS_DIR = "results"
SAVE_CIRCUITS = True
SAVE_PLOTS = True
VERBOSE = True

# Log level: "DEBUG" | "INFO" | "WARNING" | "ERROR"
LOG_LEVEL = "INFO"
