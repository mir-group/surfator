
import numpy as np
import math

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
    assert math.isclose(np.linalg.norm(surface_normal), 1., rel_tol = 1e-07), "Given `surface_normal` is not a unit vector."
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

    Assumes unchanging number of atoms.

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
    assert math.isclose(np.linalg.norm(surface_normal), 1., rel_tol = 1e-07), "Given `surface_normal` is not a unit vector."

    assert np.all(np.argsort(layer_heights) == np.arange(len(layer_heights)))

    diffs = layer_heights[1:] - layer_heights[:-1]
    maxdif = np.max(diffs)
    half_dists = np.concatenate(([maxdif if cutoff_below_bottom is None else cutoff_below_bottom * 2],
                                 diffs,
                                 [maxdif if cutoff_above_top is None else cutoff_above_top * 2]))
    half_dists *= 0.5
    assert len(half_dists) == len(layer_heights) + 1

    surf_zs, tags, mask = None, None, None

    def func(atoms):
        nonlocal surf_zs, tags, mask
        if surf_zs is None:
            surf_zs = np.empty(shape = len(atoms))
            tags = np.empty(shape = len(atoms), dtype = np.int)
            mask = np.empty(shape = (2, len(atoms)), dtype = np.bool)
        np.dot(surface_normal, atoms.positions.T, out = surf_zs)
        tags.fill(surfator.AGREE_GROUP_UNASSIGNED)

        for layer_i, height in enumerate(layer_heights):
            np.greater_equal(surf_zs, height - half_dists[layer_i], out = mask[0])
            np.less_equal(surf_zs, height + half_dists[layer_i + 1], out = mask[1])
            mask[0] &= mask[1]
            tags[mask[0]] = layer_i

        if np.min(tags) < 0:
            logger.warning("Couldn't assign atoms %s to layers" % np.where(tags < 0)[0])

        return tags

    return func


def agree_within_components_of_groups(groupfunc,
                                      cutoff = 3):
    """Define agreement groups for each connected part each prior agreement group.

    Wraps the agreement group function ``groupfunc``. (A typical ``groupfunc``
    might divide the cell into layers.)

    Each layer is split into its connected components (by atomic neighbors), and
    each component is a new agreement group. Agreement groups are neigbors with
    the agreement groups of any atoms that are neighbors to any of their atoms.

    Atomic "neighborness" is determined by a distance under ``cutoff``.

    Assumes unchanging number of atoms.

    Args:
        groupfunc (callable): An agreement group function assigning mobile
            atoms to groups.
        cutoff (float, distance units): The maximum distance between two adatoms
            for them to be considered part of the same deposit.
    """

    pbcc, dmat, connmat, newtags, layer_mask = None, None, None, None, None
    def func(atoms, **kwargs):
        nonlocal pbcc, dmat, connmat, newtags, layer_mask
        # preallocate buffers
        if pbcc is None:
            pbcc = PBCCalculator(atoms.cell)
            dmat = np.empty(shape = (len(atoms), len(atoms)))
            connmat = np.empty(shape = (len(atoms), len(atoms)), dtype = np.bool)
            newtags = np.empty(shape = len(atoms), dtype = np.int)
            layer_mask = np.empty(shape = len(atoms), dtype = np.bool)

        tags = groupfunc(atoms, **kwargs)
        layers = np.unique(tags)
        layers.sort()
        newtags.fill(-1)

        pbcc.pairwise_distances(atoms.positions, out = dmat)
        np.less_equal(dmat, cutoff, out = connmat)

        agreegrp_conns = []
        nexttag = 0
        for layer in layers:
            np.equal(tags, layer, out = layer_mask)
            layer_conrows = connmat[layer_mask]
            layer_conmat = layer_conrows[:, layer_mask]
            n_groups_layer, group_tags = connected_components(layer_conmat, directed = False)
            group_tags += nexttag
            newtags[layer_mask] = group_tags
            neighbor_groups = newtags[np.logical_or.reduce(layer_conrows, axis = 0)]
            agreegrp_conns.append(neighbor_groups)
            nexttag += n_groups_layer

        agreegrp_connmat = np.zeros(shape = (nexttag + 1, nexttag + 1), dtype = np.bool)
        for agreegrp, neighbors in enumerate(agreegrp_conns):
            agreegrp_connmat[agreegrp, neighbors] = True
        agreegrp_connmat = agreegrp_connmat[:-1, :-1]

        agreegrp_connmat |= agreegrp_connmat.T

        return newtags, np.arange(nexttag), agreegrp_connmat

    return func
