"""
molecule/loader.py

Loads molecular data from various sources and prepares an
ElectronicStructureProblem ready for Hamiltonian generation.

Supported inputs:
    - Geometry string  : "H 0 0 0; H 0 0 0.735"
    - SMILES string    : "CC(=O)O"   (converted to geometry via RDKit + PySCF)
    - PDB file path    : "/path/to/protein_fragment.pdb"
    - Common name      : "water", "methane", "aspirin" (built-in library)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Built-in molecule library ────────────────────────────────────────────────

MOLECULE_LIBRARY = {
    "h2": {
        "geometry": "H 0.0 0.0 0.0; H 0.0 0.0 0.735",
        "description": "Hydrogen molecule — simplest benchmark",
    },
    "lih": {
        "geometry": "Li 0.0 0.0 0.0; H 0.0 0.0 1.596",
        "description": "Lithium hydride — standard VQE benchmark",
    },
    "beh2": {
        "geometry": "Be 0.0 0.0 0.0; H 0.0 0.0 1.300; H 0.0 0.0 -1.300",
        "description": "Beryllium hydride — 14-qubit benchmark",
    },
    "water": {
        "geometry": "O 0.0 0.0 0.0; H 0.757 0.586 0.0; H -0.757 0.586 0.0",
        "description": "Water molecule",
    },
    "nh3": {
        "geometry": (
            "N  0.0000  0.0000  0.1173; "
            "H  0.0000  0.9349 -0.2735; "
            "H  0.8095 -0.4674 -0.2735; "
            "H -0.8095 -0.4674 -0.2735"
        ),
        "description": "Ammonia — drug-relevant amine group",
    },
    "ch4": {
        "geometry": (
            "C  0.0000  0.0000  0.0000; "
            "H  0.6276  0.6276  0.6276; "
            "H -0.6276 -0.6276  0.6276; "
            "H -0.6276  0.6276 -0.6276; "
            "H  0.6276 -0.6276 -0.6276"
        ),
        "description": "Methane — simplest organic molecule",
    },
    "co2": {
        "geometry": "C 0.0 0.0 0.0; O 0.0 0.0 1.160; O 0.0 0.0 -1.160",
        "description": "Carbon dioxide",
    },
    "hf": {
        "geometry": "H 0.0 0.0 0.0; F 0.0 0.0 0.917",
        "description": "Hydrogen fluoride — polar bond benchmark",
    },
}


@dataclass
class MoleculeProblem:
    """Container for a fully prepared molecular problem."""
    name: str
    geometry: str
    basis: str
    charge: int
    spin: int
    active_space: Optional[Tuple[int, int]]
    # The actual PySCF / Qiskit-Nature problem object
    qiskit_problem: object = None
    num_electrons: int = 0
    num_orbitals: int = 0
    num_qubits_full: int = 0


class MoleculeLoader:
    """
    Load and prepare molecular problems for VQE.

    Examples
    --------
    >>> loader = MoleculeLoader()

    # From built-in library
    >>> problem = loader.from_name("h2")

    # From geometry string
    >>> problem = loader.from_geometry("H 0 0 0; H 0 0 0.735", basis="sto-3g")

    # From SMILES (requires RDKit + PySCF)
    >>> problem = loader.from_smiles("CC(=O)O", basis="sto-3g", active_space=(4,4))
    """

    def from_name(
        self,
        name: str,
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
        active_space: Optional[Tuple[int, int]] = None,
    ) -> MoleculeProblem:
        """Load a molecule by its common name from the built-in library."""
        key = name.lower().replace(" ", "")
        if key not in MOLECULE_LIBRARY:
            raise ValueError(
                f"Molecule '{name}' not in built-in library. "
                f"Available: {list(MOLECULE_LIBRARY.keys())}. "
                f"Use from_geometry() or from_smiles() for custom molecules."
            )
        mol = MOLECULE_LIBRARY[key]
        logger.info(f"Loading '{name}': {mol['description']}")
        return self.from_geometry(
            geometry=mol["geometry"],
            basis=basis,
            charge=charge,
            spin=spin,
            active_space=active_space,
            name=name.upper(),
        )

    def from_geometry(
        self,
        geometry: str,
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
        active_space: Optional[Tuple[int, int]] = None,
        name: str = "molecule",
    ) -> MoleculeProblem:
        """
        Load a molecule from an atom geometry string.

        Parameters
        ----------
        geometry : str
            Atoms and coordinates, e.g. "H 0 0 0; H 0 0 0.735"
            Units are Angstrom.
        basis : str
            Basis set for molecular integrals (sto-3g, 6-31g, cc-pvdz, ...)
        charge : int
            Net molecular charge (0 = neutral)
        spin : int
            Number of unpaired electrons (0 = singlet)
        active_space : tuple or None
            (num_electrons, num_spatial_orbitals) for active space truncation.
            None = use full orbital space.
        """
        try:
            from qiskit_nature.second_q.drivers import PySCFDriver
            from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
            from qiskit_nature.units import DistanceUnit
        except ImportError:
            raise ImportError(
                "qiskit-nature and pyscf are required. "
                "Run: pip install qiskit-nature pyscf"
            )

        logger.info(f"Building problem for {name} | basis={basis} | geometry={geometry[:60]}...")

        driver = PySCFDriver(
            atom=geometry,
            basis=basis,
            charge=charge,
            spin=spin,
            unit=DistanceUnit.ANGSTROM,
        )

        problem = driver.run()
        num_electrons = problem.num_particles
        num_orbitals = problem.num_spatial_orbitals
        num_qubits_full = num_orbitals * 2  # each orbital = 2 spin-orbitals = 2 qubits (JW)

        logger.info(f"  Electrons    : {num_electrons}")
        logger.info(f"  Orbitals     : {num_orbitals} spatial ({num_qubits_full} spin-orbitals)")
        logger.info(f"  Qubits (full): {num_qubits_full}")

        if active_space is not None:
            n_elec, n_orb = active_space
            # Validate active space parameters
            total_electrons = (
                sum(num_electrons) if isinstance(num_electrons, tuple) else num_electrons
            )
            if n_elec > total_electrons:
                logger.warning(
                    f"Active space electrons ({n_elec}) > total electrons ({total_electrons}). "
                    f"Clamping to {total_electrons}."
                )
                n_elec = total_electrons
            if n_orb > num_orbitals:
                logger.warning(
                    f"Active space orbitals ({n_orb}) > total orbitals ({num_orbitals}). "
                    f"Clamping to {num_orbitals}."
                )
                n_orb = num_orbitals

            logger.info(f"  Active space : {n_elec} electrons, {n_orb} orbitals → {n_orb * 2} qubits")
            transformer = ActiveSpaceTransformer(
                num_electrons=n_elec,
                num_spatial_orbitals=n_orb,
            )
            problem = transformer.transform(problem)

        return MoleculeProblem(
            name=name,
            geometry=geometry,
            basis=basis,
            charge=charge,
            spin=spin,
            active_space=active_space,
            qiskit_problem=problem,
            num_electrons=num_electrons if isinstance(num_electrons, int) else sum(num_electrons),
            num_orbitals=num_orbitals,
            num_qubits_full=num_qubits_full,
        )

    def from_smiles(
        self,
        smiles: str,
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
        active_space: Optional[Tuple[int, int]] = None,
        fallback_to_h2: bool = True,
    ) -> MoleculeProblem:
        """
        Convert a SMILES string to a molecular geometry and load the problem.

        Requires: rdkit, pyscf

        Parameters
        ----------
        smiles : str
            SMILES notation, e.g. "CC(=O)O" for acetic acid
        fallback_to_h2 : bool
            If True and RDKit is unavailable, fall back to H2 geometry with a warning.
            If False, raise ImportError when RDKit is missing.
        """
        geometry, used_fallback = self._smiles_to_geometry(smiles, fallback_to_h2)
        mol_name = f"SMILES({smiles[:20]})" if not used_fallback else "H2(fallback)"
        return self.from_geometry(
            geometry=geometry,
            basis=basis,
            charge=charge,
            spin=spin,
            active_space=active_space,
            name=mol_name,
        )

    def from_pdb(
        self,
        pdb_path: str,
        basis: str = "sto-3g",
        active_space: Optional[Tuple[int, int]] = None,
        hetatm_only: bool = True,
    ) -> MoleculeProblem:
        """
        Load a molecule from a PDB file.

        Parameters
        ----------
        pdb_path : str
            Path to the PDB file.
        hetatm_only : bool
            If True, only parse HETATM records (ligand atoms).
            If False, also parse ATOM records (protein backbone).
        """
        geometry = self._pdb_to_geometry(pdb_path, hetatm_only=hetatm_only)
        name = pdb_path.split("/")[-1].replace(".pdb", "")
        return self.from_geometry(
            geometry=geometry,
            basis=basis,
            active_space=active_space,
            name=name,
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _smiles_to_geometry(self, smiles: str, fallback_to_h2: bool = True) -> Tuple[str, bool]:
        """
        Use RDKit to generate a 3D geometry from a SMILES string.

        Returns
        -------
        (geometry_string, used_fallback)
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: '{smiles}'")

            mol = Chem.AddHs(mol)

            # Try ETKDGv3 first, fall back to ETKDG
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if result == -1:
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if result == -1:
                raise RuntimeError(
                    f"RDKit failed to generate 3D coordinates for SMILES '{smiles}'. "
                    f"The molecule may be too complex or have unusual valence."
                )

            # Optimize geometry with MMFF94 force field
            ff_result = AllChem.MMFFOptimizeMolecule(mol)
            if ff_result == -1:
                # MMFF failed — try UFF
                AllChem.UFFOptimizeMolecule(mol)

            conf = mol.GetConformer()
            atoms = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                symbol = atom.GetSymbol()
                atoms.append(f"{symbol} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}")

            geometry = "; ".join(atoms)
            logger.info(f"RDKit geometry: {len(atoms)} atoms from SMILES '{smiles}'")
            return geometry, False

        except ImportError:
            msg = (
                "RDKit not installed — cannot convert SMILES to geometry. "
                "Install with: pip install rdkit"
            )
            if fallback_to_h2:
                logger.warning(f"{msg}. Using H2 geometry as fallback.")
                return "H 0.0 0.0 0.0; H 0.0 0.0 0.735", True
            else:
                raise ImportError(msg)

        except ValueError as exc:
            # Invalid SMILES — re-raise with clear message
            raise ValueError(str(exc)) from exc

        except RuntimeError as exc:
            if fallback_to_h2:
                logger.warning(f"{exc}. Using H2 geometry as fallback.")
                return "H 0.0 0.0 0.0; H 0.0 0.0 0.735", True
            raise

    def _pdb_to_geometry(self, pdb_path: str, hetatm_only: bool = True) -> str:
        """
        Parse atom records from a PDB file into a geometry string.

        Handles:
        - HETATM records (ligand/small molecule atoms)
        - ATOM records (protein backbone, if hetatm_only=False)
        - Element symbol from columns 77-78 (standard) or inferred from atom name
        - Skips hydrogen atoms (too many for quantum chemistry)
        """
        import os
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        atoms = []
        record_types = ("HETATM",) if hetatm_only else ("HETATM", "ATOM")

        with open(pdb_path) as f:
            for line in f:
                if not line.startswith(record_types):
                    continue

                # Element symbol: columns 77-78 (PDB standard)
                element = line[76:78].strip()
                if not element:
                    # Fall back to first letter of atom name (columns 13-16)
                    atom_name = line[12:16].strip()
                    element = "".join(c for c in atom_name if c.isalpha())[:1]

                if not element:
                    continue

                # Skip hydrogens — too many for quantum chemistry
                if element.upper() in ("H", "D"):
                    continue

                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    logger.warning(f"Could not parse coordinates in PDB line: {line.rstrip()}")
                    continue

                atoms.append(f"{element} {x:.4f} {y:.4f} {z:.4f}")

        if not atoms:
            raise ValueError(
                f"No heavy atoms found in PDB file: {pdb_path}. "
                f"Check that the file contains HETATM or ATOM records."
            )

        logger.info(f"PDB loaded: {len(atoms)} heavy atoms from {pdb_path}")
        return "; ".join(atoms)
