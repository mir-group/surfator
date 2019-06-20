
import numpy as np
import math

import ase

from scipy.sparse.csgraph import connected_components

from sitator.util import PBCCalculator

import logging
logger = logging.getLogger(__name__)


def agree_within_layers(layer_heights, surface_normal = np.array([0, 0, 1])):
    """Assign mobile atoms to agreement groups based on their layer in the material.

    In a system with fixed layers, like a surface slab (especially one with a
    fixed bottom layer that constrains up-and-down movement), this simple algorithm
    assigns based on which layer "height" an atom is closest to along the surface
    normal vector.

    Args:
        - layer_heights (ndarray): The "heights" (positions in normal coordinates)
            of the layers.
        - surface_normal (3-vector): The unit normal vector to the surface.
    """
    assert math.isclose(np.linalg.norm(surface_normal), 1.), "Given `surface_normal` is not a unit vector."

    diffs = layer_heights[1:] - layer_heights[:-1]
    maxdif = np.max(diffs)
    half_dists = np.concatenate(([maxdif], diffs, [maxdif]))
    half_dists *= 0.5
    assert len(half_dists) == len(layer_heights) + 1

    def func(atoms):
        # We need to wrap coordinates so that with slanted surfaces we get the
        # right heights along the normal
        atoms.wrap()
        surf_zs = np.dot(surface_normal, atoms.get_positions().T)
        assert len(surf_zs) == len(atoms)
        tags = np.full(shape = len(pos), fill_value = -1, dtype = np.int)

        for layer_i, height in enumerate(layer_heights):
            mask = (surf_zs >= height - half_dists[layer_i]) & (surf_zs <= height + half_dists[layer_i + 1])
            tags[mask] = layer_i

        assert np.min(tags) >= 0

        return tags

    return func


def agree_within_layers_and_deposits(layerfunc, surface_layer_index = 4, cutoff = 3):
    """Define agreement groups based layers but allow surface deposits to be independent.

    Args:
        - layerfunc (callable): An agreement group function assigning mobile
            atoms to layers. Must return tags 0,1,2,... where increasing agreement
            group number indicates increasing layer. Generally, this will be
            `agree_within_layers()` with the correct parameters.
        - surface_layer_index (int): The layer to consider the "surface"; all
            mobile atoms at higher layers are considered adatoms.
        - cutoff (float, distance units): The maximum distance between two adatoms
            for them to be considered part of the same deposit.
    """
    def func(atoms, **kwargs):
        pbcc = PBCCalculator(atoms.cell)
        pos = atoms.get_positions()
        tags = layerfunc(atoms, **kwargs)

        adatom_mask = tags > surface_layer_index

        # Determine the connected (as defined by cutoff) groups of adatoms
        # We take connected groups of adatoms, since they are within a distance
        # of influencing one another, as an agreement group
        conn_mat = pbcc.pairwise_distances(pos[adatom_mask])
        conn_mat = conn_mat < radius
        n_groups, groups = connected_components(conn_mat, directed = False)
        logger.debug("Found %i adatoms in %i independent deposits" % (np.sum(adatom_mask), n_groups))

        curr_max_agreegrp = np.max(tags)
        tags[adatom_mask] = curr_max_agreegrp + groups
        return tags

    return func
