
import numpy as np
import math

import ase
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList, get_connectivity_matrix

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

    def func(atoms):
        # We need to wrap coordinates so that with slanted surfaces we get the
        # right heights along the normal
        atoms.wrap()
        surf_zs = np.dot(surface_normal, atoms.get_positions().T)
        assert len(surf_zs) == len(atoms)
        tags = np.full(shape = len(surf_zs), fill_value = surfator.AGREE_GROUP_UNASSIGNED, dtype = np.int)

        for layer_i, height in enumerate(layer_heights):
            mask = (surf_zs >= height - half_dists[layer_i]) & (surf_zs <= height + half_dists[layer_i + 1])
            tags[mask] = layer_i

        if np.min(tags) < 0:
            logger.warning("Couldn't assign atoms %s to layers" % np.where(tags < 0)[0])

        return tags

    return func


def agree_within_layers_and_deposits(layerfunc,
                                     surface_layer_index = 4,
                                     cutoff = 3,
                                     min_deposit_size = 3,
                                     skin = 0.1):
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
    nl = None
    pbcc = None
    def func(atoms, **kwargs):
        if nl is None:
            nl = NeighborList(
                cutoffs = np.full(shape = len(atoms), fill_value = cutoff)
                skin = skin,
                self_interaction = False,
                bothways = False,
                primitive = NewPrimitiveNeighborList
            )
            pbcc = PBCCalculator(atoms.cell)

        tags = layerfunc(atoms, **kwargs)
        layers = np.unique(tags)
        layers.sort()
        newtags = np.empty_like(tags)
        newtags.fill(-1)

        nl.update(atoms)
        connmat = get_connectivity_matrix(nl)

        agreegrp_conns = []
        nexttag = 0
        layer_mask = np.empty(shape = len(tags), dtype = np.bool)
        for layer in layers:
            np.equal(tags, layer, out = layer_mask)
            layer_conrows = connmat[layer_mask]
            layer_conmat = layer_conrows[:, layer_mask]
            n_groups_layer, group_tags = connected_components(layer_conmat, directed = False)
            group_tags += nexttag
            newtags[layer_mask] = group_tags
            nexttag += n_groups_layer
            neighbor_groups = newtags[np.logical_or.reduce(layer_conrows, axis = 0)]
            agreement_conns.append(neighbor_groups)

        agreegrp_connmat = np.zeros(shape = (nexttag + 1, nexttag + 1), dtype = np.bool)
        for agreegrp, neighbors in enumerate(agreegrp_conns):
            agreegrp_connmat[agreegrp, neighbors] = True
        agreegrp_connmat = agreegrp_connmat[:-1, :-1]
        assert np.all(agreegrp_connmat == agreegrp_connmat.T)

        return newtags, np.arange(nexttag), agreegrp_connmat

    return func
