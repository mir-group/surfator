
import numpy as np
import math

import ase

from scipy.sparse.csgraph import connected_components

from sitator.util import PBCCalculator

import surfator

import logging
logger = logging.getLogger(__name__)


def agree_within_layers_kmeans(initial_layer_heights, surface_normal = np.array([0, 0, 1]), min_layer_dist = 1):
    """Assign mobile atoms to agreement groups based on their layer in the material.

    In a system with fixed layers, like a surface slab (especially one with a
    fixed bottom layer that constrains up-and-down movement), this algorithm
    assigns atoms to layers by using k-means clustering in their "z" coordinate
    (their coordinate along the surface normal).

    Args:
        - n_layers (int): The number of layers to find (k for k-means).
        - surface_normal (3-vector): The unit normal vector to the surface.
        - min_layer_dist (float, distance): The minimum height difference between
            two layers' centers.
    """
    assert math.isclose(np.linalg.norm(surface_normal), 1.), "Given `surface_normal` is not a unit vector."
    from sklearn.cluster import KMeans

    n_layers = len(initial_layer_heights)
    initial_layer_heights = initial_layer_heights.reshape(-1, 1)

    # preallocate
    clust_num_to_new_clust_num = np.empty(shape = n_layers, dtype = np.int)

    def func(atoms):
        # We need to wrap coordinates so that with slanted surfaces we get the
        # right heights along the normal
        atoms.wrap()
        surf_zs = np.dot(surface_normal, atoms.get_positions().T)
        assert len(surf_zs) == len(atoms)
        kmeans = KMeans(n_clusters = n_layers, init = initial_layer_heights, n_init = 1).fit(surf_zs.reshape(-1, 1))
        heights = kmeans.cluster_centers_.reshape(-1)
        order = np.argsort(heights) # cluster numbers
        assert np.min(heights[order][1:] - heights[order][:-1]) >= min_layer_dist, "Two layers were not far enough apart. Layer heights: %s" % heights
        clust_num_to_new_clust_num[order] = np.arange(n_layers)
        return clust_num_to_new_clust_num[kmeans.labels_]

    return func


def agree_within_layers(layer_heights, surface_normal = np.array([0, 0, 1]), cutoff_above_top = None, cutoff_below_bottom = None):
    """Assign mobile atoms to agreement groups based on their layer in the material.

    In a system with fixed layers, like a surface slab (especially one with a
    fixed bottom layer that constrains up-and-down movement), this simple algorithm
    assigns based on which layer "height" an atom is closest to along the surface
    normal vector.

    Args:
        - layer_heights (ndarray): The "heights" (positions in normal coordinates)
            of the layers. Should be sorted, ascending.
        - surface_normal (3-vector): The unit normal vector to the surface.
        - cutoff_above_top (float): The maximum distance above the topmost layer
            within which an atom can be assigned to it. If None, set to half the
            maximum interlayer distance.
        - cutoff_below_bottom (float): Same as `cutoff_above_top` but for the
            bottommost layer.
    """
    assert math.isclose(np.linalg.norm(surface_normal), 1.), "Given `surface_normal` is not a unit vector."

    assert np.all(np.argsort(layer_heights) == np.arange(len(layer_heights)))

    diffs = layer_heights[1:] - layer_heights[:-1]
    maxdif = np.max(diffs)
    half_dists = np.concatenate(([maxdif if cutoff_below_bottom is None else cutoff_below_bottom * 2],
                                 diffs,
                                 [maxdif if cutoff_above_top is None else cutoff_above_top * 2]))
    half_dists *= 0.5
    assert len(half_dists) == len(layer_heights) + 1

    def func(atoms):
        # We need to wrap coordinates so that with slanted surfaces we get the
        # right heights along the normal
        atoms.wrap()
        surf_zs = np.dot(surface_normal, atoms.get_positions().T)
        assert len(surf_zs) == len(atoms)
        tags = np.full(shape = len(surf_zs), fill_value = -1, dtype = np.int)

        for layer_i, height in enumerate(layer_heights):
            mask = (surf_zs >= height - half_dists[layer_i]) & (surf_zs <= height + half_dists[layer_i + 1])
            tags[mask] = layer_i

        assert np.min(tags) >= 0

        return tags

    return func


def agree_within_layers_and_deposits(layerfunc, surface_layer_index = 4, cutoff = 3, min_deposit_size = 3):
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
        conn_mat = conn_mat < cutoff
        n_groups, groups = connected_components(conn_mat, directed = False)
        logger.debug("Found %i adatoms in %i independent deposits" % (np.sum(adatom_mask), n_groups))
        group_trans = np.bincount(groups)
        deposits = group_trans >= min_deposit_size
        group_trans[~deposits] = surfator.AGREE_GROUP_NONE
        curr_max_agreegrp = np.max(tags)
        group_trans[deposits] = np.arange(np.sum(deposits)) + curr_max_agreegrp
        tags[adatom_mask] = group_trans[groups]
        return tags

    return func
