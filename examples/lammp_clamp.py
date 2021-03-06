import numpy as np

import os
import sys
from pathlib import Path
import pickle

import ase
import ase.io
from ase.visualize import view

from surfator import StructureGroupAnalysis, STRUCTURE_GROUP_ATTRIBUTE
import surfator.agreement_groups
from surfator.analysis import calculate_coord_numbers
from surfator.util.layers import get_layer_heights_kmeans
from surfator.structure_groups import ClosePackedReferenceSites

from sitator import SiteNetwork
from sitator.misc import GenerateClampedTrajectory
from sitator.dynamics import SmoothSiteTrajectory

from tqdm.autonotebook import tqdm

import logging


def read_refstruct(fpath):
    """Read a reference structure generated by David's code

    Returns:
        surface_normal
        positions (n x 3 ndarray float),
        layers (n ndarray int),
        site_type (n ndarray object)
    """
    skiprows = 2
    with open(fpath) as f:
        normarr = f.readline().strip().split()
        assert normarr[:2] == ["Normal",  "vector"]
        normal = np.array(normarr[-3:], dtype = np.float)
        header = f.readline()
        assert header.strip().split("\t") == ['x', 'y', 'z', 'Layer #', 'Site type']
    positions = np.loadtxt(fpath, skiprows = skiprows, usecols = (0, 1, 2))
    layers = np.loadtxt(fpath, skiprows = skiprows, usecols = (3), dtype = np.int)
    layer_types =  np.loadtxt(fpath, skiprows = skiprows, usecols = (4), dtype = np.object)
    return normal, positions, layers, layer_types


# ---- FROM ASE https://gitlab.com/ase/ase/blob/master/ase/io/lammpsrun.py ----
# Based on ASE's code for loading lammpstrj (which we don't use because its
# performance is very bad)
def construct_cell(celldata):
    """
    ASE's original docs:

    Help function to create an ASE-cell with displacement vector from
    the lammps coordination system parameters.

    :param diagdisp: cell dimension convoluted with the displacement vector
    :param offdiag: off-diagonal cell elements
    :returns: cell and cell displacement vector
    :rtype: tuple
    """
    diagdisp = celldata[:, :2].reshape(6, 1).flatten()
    offdiag = celldata[:, 2]

    xlo, xhi, ylo, yhi, zlo, zhi = diagdisp
    xy, xz, yz = offdiag

    # create ase-cell from lammps-box
    xhilo = (xhi - xlo) - abs(xy) - abs(xz)
    yhilo = (yhi - ylo) - abs(yz)
    zhilo = zhi - zlo
    celldispx = xlo - min(0, xy) - min(0, xz)
    celldispy = ylo - min(0, yz)
    celldispz = zlo
    cell = np.array([[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]])
    celldisp = np.array([celldispx, celldispy, celldispz])

    return cell, celldisp


def read_lammpstraj(path):
    """
    Assumes unchancing cell/box bounds and unchanging number of atoms.

    !!! ASSUMES SORTED OUTPUT !!!

    Returns:
        - traj
        - frame_times
        - atoms
        - cellstr
    """

    frames = []
    frame_times = []
    n_atoms = None
    cell = None

    with open(path) as f:
        # f will raise StopIteration
        try:
            first_frame = True
            while(True):
                assert next(f).startswith("ITEM: TIMESTEP")
                processed_full_frame = False
                frame_times.append(int(next(f)))
                assert next(f).startswith("ITEM: NUMBER OF ATOMS")
                n_atoms = int(next(f))
                assert next(f).startswith("ITEM: BOX BOUNDS xy xz yz pp pp ff")
                cellstr = [next(f) for _ in range(3)]
                celldata = np.loadtxt(cellstr)
                cellstr = "ITEM: BOX BOUNDS xy xz yz pp pp ff\n" + "".join(cellstr).strip()
                cell, _ = construct_cell(celldata)
                atomhdr = next(f)
                # FIXME deal with headers...
                #assert atomhdr.startswith("ITEM: ATOMS id type xu yu z") or atomhdr.startswith("ITEM: ATOMS id c_xyu[1] c_xyu[2] z")
                assert atomhdr.startswith("ITEM: ATOMS id") or atomhdr.startswith("ITEM: ATOMS id")
                atomhdr = atomhdr.split()[2:]
                ididx = 0
                try:
                    typeidx = atomhdr.index('type')
                except ValueError:
                    typeidx = None
                xidx = None
                for x in ("x", "xu", "c_xyu[1]"):
                    try:
                        xidx = atomhdr.index(x)
                        break
                    except ValueError:
                        continue
                if xidx is None:
                    raise ValueError("no x!")
                assert atomhdr[xidx + 1] == 'y' or atomhdr[xidx + 1] == 'yu' or atomhdr[xidx + 1] == 'c_xyu[2]'
                assert atomhdr[xidx + 2] == 'z' or atomhdr[xidx + 2] == 'zu' or atomhdr[xidx + 2] == 'c_xyu[3]'
                frame = np.empty(shape = (n_atoms, 3))
                if first_frame:
                    types = np.empty(shape = n_atoms, dtype = np.int)
                    ids = np.empty(shape = n_atoms, dtype = np.int)
                for atom_i in range(n_atoms):
                    lsplit = next(f).split()
                    frame[atom_i] = lsplit[xidx:xidx + 3]
                    if typeidx is not None:
                        types[atom_i] = lsplit[typeidx]
                    else:
                        types[atom_i] = 0
                    ids[atom_i] = lsplit[ididx]
                frames.append(frame)
                processed_full_frame = True
                first_frame = False
        except StopIteration:
            pass

    assert processed_full_frame

    frames = np.asarray(frames)
    frame_times = np.asarray(frame_times)

    # Make atoms
    atoms = ase.Atoms(positions = frames[0], cell = cell, pbc = True)
    atoms.set_atomic_numbers(types)
    atoms.set_tags(ids)

    return frames, frame_times, atoms, cellstr


def write_lammpstraj(path, cellstr, traj, atoms, coords = None, timesteps = None):
    """WARNING: this function is NOT generic AT ALL

    JUST FOR SOME REALLY SPECIFIC TRAJECTORY FILES
    """
    if timesteps is None:
        timesteps = np.arange(len(traj))

    n_atoms = traj.shape[1]
    assert traj.shape[2] == 3

    if coords is None:
        atoms_header = "ITEM: ATOMS id type x y z"
        atom_format = "{} {} {:05f} {:05f} {:05f}"
    else:
        atoms_header = "ITEM: ATOMS id type xu yu z c_cn "
        atom_format = "{} {} {:05f} {:05f} {:05f} {:d}"

    ids = atoms.get_tags()
    types = atoms.get_atomic_numbers()

    with open(path, 'w') as f:
        for f_idex, frame in enumerate(traj):
            print("ITEM: TIMESTEP", file = f)
            print(timesteps[f_idex], file = f)
            print("ITEM: NUMBER OF ATOMS", file = f)
            print(n_atoms, file = f)
            print(cellstr, file = f)
            print(atoms_header, file = f)
            for atom_i in range(n_atoms):
                print(atom_format.format(ids[atom_i], types[atom_i], frame[atom_i, 0], frame[atom_i, 1], frame[atom_i, 2], (None if coords is None else coords[f_idex, atom_i])), file = f)

    return


def main(traj_path,
         ref_path,
         out_path,
         n = None,
         trajslice = None,
         cutoff = 3,
         min_layer_dist = 1.0,
         runoff_votes_weight = 0.6,
         winner_bias = 0.5,
         assign_cutoff = None,
         agreegrp_cutoff = None,
         kmeans_for_heights = False,
         min_winner_percentage = 0.50001):
    """
    Args:
        - traj (ndarray n_frames x n_atoms x 3)
        - ref_structure (ASE atoms len(.) = n_atoms)
        - cutoff (float, Angstrom): For computing coordination number
    """
    fh = logging.FileHandler(os.path.join(out_path, 'surfator.log'), mode = 'w')
    surfator_log = logging.getLogger("surfator")
    surfator_log.setLevel(logging.INFO)
    surfator_log.addHandler(fh)
    sitator_log = logging.getLogger("sitator")
    sitator_log.setLevel(logging.INFO)
    sitator_log.addHandler(fh)
    logger = logging.getLogger("lammp_clamp")
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stderr))

    logger.info("Loading trajectory and reference structure...")
    traj, timesteps, atoms, cellstr = read_lammpstraj(traj_path)

    logger.info("Writing out Python trajectory...")
    ase.io.write(os.path.join(out_path, "atoms.cif"), atoms, parallel = False)
    np.save(os.path.join(out_path, "orig_trj.npy"), traj)

    surface_normal, ref_positions, layer_labels, ref_site_types = read_refstruct(ref_path)
    ref_sn = ClosePackedReferenceSites(
        reference_positions = ref_positions,
        layer_labels = layer_labels,
        layer_type_labels = ref_site_types,
        structure = atoms,
        static_mask = np.zeros(len(atoms), dtype = np.bool),
        mobile_mask = np.ones(len(atoms), dtype = np.bool),
    )

    if trajslice is not None:
        trajslice = slice(*(None if e == '' else int(e) for e in trajslice.split(":")))
        traj = traj[trajslice]
        timesteps = timesteps[trajslice]

    logger.info("Cell: \n%s" % atoms.cell)

    logger.info("Determining layers...")
    assert n is not None
    if kmeans_for_heights:
        heights_kmeans_stride = max(len(traj) // 300, 1)  # Why not? 300 frames of heights sounds reasonable
        layers = get_layer_heights_kmeans(traj[::heights_kmeans_stride], atoms.cell, n, surface_normal = surface_normal)
    else:
        # We have the heights from the refstruct
        assert set(layer_labels) == set(range(1, n + 1))
        layers = np.zeros(shape = n)
        heights = np.dot(ref_positions, surface_normal.T)
        for i in range(n):
            layers[i] = np.mean(heights[layer_labels == i + 1])

    logger.info("Layer heights: %s" % layers)

    logger.info("Assigning to reference sites...")
    if assign_cutoff is None:
        assign_cutoff = cutoff

    # layerfunc = surfator.agreement_groups.layers.agree_within_layers_kmeans(
    #     initial_layer_heights = layers,
    #     surface_normal = surface_normal,
    #     min_layer_dist = min_layer_dist
    # )
    layerfunc = surfator.agreement_groups.layers.agree_within_layers(
        layer_heights = layers,
        surface_normal = surface_normal,
        #cutoff_above_top = assign_cutoff # Be a little more generous on top.
    )
    if agreegrp_cutoff is None:
        agreegrp_cutoff = 1.5 * cutoff
    agreefunc = surfator.agreement_groups.layers.agree_within_components_of_groups(
        layerfunc,
        cutoff = agreegrp_cutoff
    )

    sga = StructureGroupAnalysis(
        min_winner_percentage = min_winner_percentage,
        runoff_votes_weight = runoff_votes_weight,
        winner_bias = winner_bias,
        error_on_no_majority = False
    )
    st, agreegrp_assign, structgrp_assign = sga.run(
        ref_sn = ref_sn,
        traj = traj,
        cutoff = assign_cutoff,
        agreement_group_function = agreefunc,
        return_assignments = True,
    )
    np.save(os.path.join(out_path, "agreegrp-assignments.npy"), agreegrp_assign)
    np.save(os.path.join(out_path, "structgrp-assignments.npy"), structgrp_assign)
    logger.info("    Percent unassigned: %.2f%%" % (100 * st.percent_unassigned))
    logger.info("    Average majority: %i%%; minimum majority %i%%" % (100 * sga.average_majority, 100 * sga.minimum_majority))
    st.compute_site_occupancies()
    occs = st.site_network.occupancies
    logger.info("    Min occupancy: %.2f; avg. occupancy: %.2f; max occupancy: %.2f" % (np.min(occs), np.mean(occs), np.max(occs)))
    n_multiple_assign, _ = st.check_multiple_occupancy(max_mobile_per_site = 2)
    logger.info("    n multiple assignment: %i" % n_multiple_assign)

    logger.info("Removing short jumps...")
    sst = SmoothSiteTrajectory(
        window_threshold_factor = 3/2,
        set_unassigned_under_threshold = False # Maintain short transitions, only eliminate failed attempts
    )
    st = sst.run(st, threshold = 2)
    logger.info("    Percent unassigned: %.2f%%" % (100 * st.percent_unassigned))
    n_multiple_assign, _ = st.check_multiple_occupancy(max_mobile_per_site = 2)
    logger.info("    n multiple assignment: %i" % n_multiple_assign)

    logger.info("Clamping trajectory...")
    gct = GenerateClampedTrajectory(wrap = False, pass_through_unassigned = True)
    clamped_traj = gct.run(st)

    logger.info("Computing new coordination numbers...")
    # Now get coordination numbers
    coords = calculate_coord_numbers(traj = clamped_traj,
                                     atoms = atoms,
                                     cutoff = cutoff)
    nums, counts = np.unique(coords, return_counts = True)
    maxcount = np.max(counts)
    width = 50

    chist_out = []
    for n, c in zip(nums, counts):
        chist_out.append(("  {:3d}: {:%is}    (x{:8d})" % width).format(n, "#" * int(width * c / maxcount), c))
    logger.info("Coordination histogram:\n%s" % "\n".join(chist_out))

    np.save(os.path.join(out_path, "cn_clamped.npy"), coords)

    #print("Flagging events...")
    #flag_events(st, cutoff)

    logger.info("Writing trajectory out...")
    np.save(os.path.join(out_path, "site_traj.npy"), st.traj)
    np.save(os.path.join(out_path, "clamped_trj.npy"), clamped_traj)
    #write_lammpstraj(os.path.join(out_path, "clamped-vmd.out"), traj = clamped_traj, atoms = atoms, timesteps = timesteps, cellstr = cellstr)
    write_lammpstraj(os.path.join(out_path, "clamped.lammpstrj"), traj = clamped_traj, atoms = atoms, coords = coords, timesteps = timesteps, cellstr = cellstr)
    logger.info("Done.")


# def flag_events(st, max_dist_to_aux):
#     sn = st.site_network
#     pbcc = PBCCalculator(sn.structure.cell)
#     site_distances = pbcc.pairwise_distances(sn.centers)
#     for frame_number, mob_that_jumped, from_sites, to_sites in st.jumps():
#         changed_layer = sn.layer[from_sites] != sn.layer[to_sites]
#         # Propagate layers: nearest auxiliary (by MIC) gets it's layer relative to
#         # the z-mobile atom; nearest next aux to that first aux (by MIC) gets its
#         # layer relative to that (as MIC), etc.


if __name__ == "__main__":
    argv = sys.argv

    if len(argv) == 5:
        import json
        kwargs = argv[3]
        if kwargs[0] == '{':
            kwargs = json.loads(kwargs)
        else: # It's a path
            with open(kwargs) as f:
                kwargs = json.load(f)
    elif len(argv) == 4:
        kwargs = {}
    else:
        print("lammp-clamp.py traj_path ref_path [\"json-kwargs-str\"|/path/to/kwargs.json] out_path")
        sys.exit(-1)

    main(traj_path = argv[1],
         ref_path = argv[2],
         out_path = argv[-1],
         **kwargs)
