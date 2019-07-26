
import numpy as np

from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.mdamath import triclinic_box

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

    our_triclinic_cell = triclinic_box(
        atoms.cell[0],
        atoms.cell[1],
        atoms.cell[2],
    )

    for f_idex, frame in enumerate(tqdm(traj)):
        neighbors = FastNS(
            cutoff = cutoff,
            coords = frame,
            box = our_triclinic_cell,
            pbc = True
        ).self_search().get_indices()

        for out_i, atom_i in enumerate(mask_idexes):
            out[f_idex, out_i] = len(neighbors[atom_i])

    assert np.min(out) >= 0

    return out
