
import numpy as np

from ase.neighborlist import NewPrimitiveNeighborList

from sitator.util.progress import tqdm

# TODO: compute coords at specificed SITES

def calculate_coord_numbers(traj, atoms, cutoff, skin = 0, mask = None):
    """Compute the coordination numbers for `mask` atoms at all times in `traj`.

    Args:
        - traj (ndarray n_frames x n_atoms x 3)
        - mask (ndarray bool n_atoms)
        - atoms (ase.Atoms)
        - cutoff (float, distance units)
        - skin (float, distance units, default: 0)
    Returns:
        ndarray of int, n_frames x n_atoms
    """
    n_atoms = len(atoms)
    if mask is None:
        mask = np.ones(shape = n_atoms, dtype = np.bool)
    mask_idexes = np.where(mask)[0]

    out = np.full(shape = (len(traj), np.sum(mask)), fill_value = -1, dtype = np.int)

    nl = NewPrimitiveNeighborList(cutoffs = np.full(n_atoms, cutoff * 0.5),
                                  skin = skin,
                                  self_interaction = False,
                                  bothways = True)

    nl.build(pbc = atoms.pbc,
             cell = atoms.cell,
             positions = atoms.get_positions())

    for f_idex, frame in enumerate(tqdm(traj)):
        nl.update(pbc = atoms.pbc,
                  cell = atoms.cell,
                  positions = frame)

        for out_i, atom_i in enumerate(mask_idexes):
            neighbor_idex, neighbor_offset = nl.get_neighbors(atom_i)
            out[f_idex, out_i] = len(neighbor_idex)

    assert np.min(out) >= 0

    return out
