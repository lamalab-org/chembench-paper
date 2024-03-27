# code taken from https://github.com/boskovicgroup/bottchercomplexity/blob/main/BottcherComplexity.py
import math
import os
import sys
from rdkit import Chem
from rdkit.Chem import RDConfig
from functools import lru_cache

sys.path.append(os.path.join(RDConfig.RDContribDir, "ChiralPairs"))
import ChiralDescriptors
import signal
import functools


def timeout(seconds=30, default=None):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            def handle_timeout(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            result = func(*args, **kwargs)

            signal.alarm(0)

            return result

        return wrapper

    return decorator


# D
#
# Current failures: Does not distinguish between cyclopentyl and pentyl (etc.)
#                   and so unfairly underestimates complexity.
def GetChemicalNonequivs(atom, themol):
    num_unique_substituents = 0
    substituents = [[], [], [], []]
    for item, key in enumerate(
        ChiralDescriptors.determineAtomSubstituents(
            atom.GetIdx(), themol, Chem.GetDistanceMatrix(themol)
        )[0]
    ):
        for subatom in ChiralDescriptors.determineAtomSubstituents(
            atom.GetIdx(), themol, Chem.GetDistanceMatrix(themol)
        )[0][key]:
            substituents[item].append(themol.GetAtomWithIdx(subatom).GetSymbol())
            num_unique_substituents = len(
                set(
                    tuple(
                        tuple(substituent)
                        for substituent in substituents
                        if substituent
                    )
                )
            )
            #
            # Logic to determine e.g. whether repeats of CCCCC are cyclopentyl and pentyl or two of either
            #
    return num_unique_substituents


# E
#
# The number of different non-hydrogen elements or isotopes (including deuterium
# and tritium) in the atom's microenvironment.
#
# CH4 - the carbon has e_i of 1
# Carbonyl carbon of an amide e.g. CC(=O)N e_i = 3
#     while N and O have e_i = 2
#
def GetBottcherLocalDiversity(atom):
    neighbors = []
    for neighbor in atom.GetNeighbors():
        neighbors.append(str(neighbor.GetSymbol()))
    if atom.GetSymbol() in set(neighbors):
        return len(set(neighbors))
    else:
        return len(set(neighbors)) + 1


# S
#
# RDKit marks atoms where there is potential for isomerization with a tag
# called _CIPCode. If it exists for an atom, note that S = 2, otherwise 1.
def GetNumIsomericPossibilities(atom):
    try:
        if atom.GetProp("_CIPCode"):
            return 2
    except:
        return 1


# V
#
# The number of valence electrons the atom would have if it were unbonded and
# neutral
# TODO: Move this dictionary somewhere else.
def GetNumValenceElectrons(atom):
    valence = {
        1: ["H", "Li", "Na", "K", "Rb", "Cs", "Fr"],  # Alkali Metals
        2: ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"],  # Alkali Earth Metals
        # transition metals???
        3: ["B", "Al", "Ga", "In", "Tl", "Nh"],  #
        4: ["C", "Si", "Ge", "Sn", "Pb", "Fl"],
        5: ["N", "P", "As", "Sb", "Bi", "Mc"],  # Pnictogens
        6: ["O", "S", "Se", "Te", "Po", "Lv"],  # Chalcogens
        7: ["F", "Cl", "Br", "I", "At", "Ts"],  # Halogens
        8: ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"],
    }  # Noble Gases
    for k in valence:
        if atom.GetSymbol() in valence[k]:
            return k
    return 0


# B
#
# Represents the total number of bonds to other atoms with V_i*b_i > 1, so
# basically bonds to atoms other than Hydrogen
#
# Here we can leverage the fact that RDKit does not even report Hydrogens by
# default to simply loop over the bonds. We will have to account for molecules
# that have hydrogens turned on before we can submit this code as a patch
# though.
#
# TODO: Create a dictionary for atom-B value pairs for use when AROMATIC is detected in bonds.
def GetBottcherBondIndex(atom):
    b_sub_i_ranking = 0
    bonds = []
    for bond in atom.GetBonds():
        bonds.append(str(bond.GetBondType()))
    for bond in bonds:
        if bond == "SINGLE":
            b_sub_i_ranking += 1
        if bond == "DOUBLE":
            b_sub_i_ranking += 2
        if bond == "TRIPLE":
            b_sub_i_ranking += 3
    if "AROMATIC" in bonds:
        # This list can be expanded as errors arise.
        if atom.GetSymbol() == "C":
            b_sub_i_ranking += 3
        elif atom.GetSymbol() == "N":
            b_sub_i_ranking += 2
    return b_sub_i_ranking


@timeout(15)
def GetBottcherComplexity(themol, debug=False):
    complexity = 0
    Chem.AssignStereochemistry(
        themol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )
    atoms = themol.GetAtoms()
    atom_stereo_classes = []
    atoms_corrected_for_symmetry = []
    for atom in atoms:
        if atom.GetProp("_CIPRank") in atom_stereo_classes:
            continue
        else:
            atoms_corrected_for_symmetry.append(atom)
            atom_stereo_classes.append(atom.GetProp("_CIPRank"))
    for atom in atoms_corrected_for_symmetry:
        d = GetChemicalNonequivs(atom, themol)
        e = GetBottcherLocalDiversity(atom)
        s = GetNumIsomericPossibilities(atom)
        V = GetNumValenceElectrons(atom)
        b = GetBottcherBondIndex(atom)
        complexity += d * e * s * math.log(V * b, 2)
        if debug:
            print(str(atom.GetSymbol()))
            print("\tSymmetry Class: " + str(atom.GetProp("_CIPRank")))
            print("\tNeighbors: ")
            print("\tBonds: ")
            print("\tCurrent Parameter Values:")
            print("\t\td_sub_i: " + str(d))
            print("\t\te_sub_i: " + str(e))
            print("\t\ts_sub_i: " + str(s))
            print("\t\tV_sub_i: " + str(V))
            print("\t\tb_sub_i: " + str(b))
    if debug:
        print("Current Complexity Score: " + str(complexity))
        return
    return complexity


@lru_cache(maxsize=None)
def complexity_from_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return GetBottcherComplexity(mol)
    except Exception:
        return None
