
import numpy as np

from surfator import StructureGroupAnalysis

def no_agreement_required(atoms):
    return np.full(shape = len(atoms),
                   fill_value = StructureGroupAnalysis.AGREE_GROUP_NONE)

def all_atoms_agree(atoms):
    return np.full(shape = len(atoms),
                   fill_value = 0,
                   dtype = np.int)
