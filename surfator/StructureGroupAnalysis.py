
import numpy as np

import numbers

from scipy.spatial import cKDTree

import ase
import ase.geometry

from sitator import SiteNetwork, SiteTrajectory
from sitator.util.progress import tqdm
from sitator.util import PBCCalculator

import surfator.grouping

import logging
logger = logging.getLogger(__name__)

from functools import wraps
def analysis_result(func):
    @property
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._has_run:
            raise ValueError("This StructureGroupAnalysis hasn't been run yet.")
        return func(self, *args, **kwargs)
    return wrapper


SITE_RADIUS_ATTRIBUTE = 'site_radius'
STRUCTURE_GROUP_ATTRIBUTE = 'structure_group'

AGREE_GROUP_NONE = -1

class StructureGroupVotingError(Exception):
    pass
class StructureGroupCompatabilityError(Exception):
    pass

class StructureGroupAnalysis(object):
    """Assign atoms from a trajectory to sites in a reference structure using "Structure Group Analysis"

    Args:
        - min_winner_percentage (float): Proportion of an agreement group's votes
            that the election winner must receive. Defaults to 50.001%; values
            smaller than half should be used with caution.
    """
    def __init__(self,
                 min_winner_percentage = 0.50001,
                 runoff_votes_weight = 0.5,
                 winner_bias = 0.5):
        self.min_winner_percentage = min_winner_percentage
        self.runoff_votes_weight = runoff_votes_weight
        self._has_run = False
        assert 0 <= winner_bias < 1
        self.winner_bias = winner_bias


    def run(self,
            ref_sn,
            traj,
            cutoff,
            agreement_group_function = surfator.grouping.all_atoms_agree,
            structure_group_compatability = None):
        """
        Args:
            - ref_sn (SiteNetwork): A `SiteNetwork` containing the sites to which
                the mobile atoms will be assigned. Can contain multiple, possibly
                exclusive, sets of sites called "structure groups." The site
                attribute `STRUCTURE_GROUP_ATTRIBUTE` indicates, for each site,
                which structure group it belongs to.
            - traj (ndarray, n_frames x n_atoms x 3):
            - cutoff (ndarray or float): Cutoff radii can be given in `ref_sn`
                for the sites with site attribute SITE_RADIUS_ATTRIBUTE. If none
                are given, they will be set to `0.5 * cutoff`. If an array is
                given for `cutoff`, radii must be provided in `ref_sn`. Radii for
                the mobile atoms can be given through `cutoff` either as a single
                float cutoff, half of which is then used as the radius for all
                mobile atoms, or as an array of radii for each mobile atom which
                are not modified. In distance units.
            - agreement_group_function (callable taking an Atoms): A function that,
                given an `Atoms` object containing the current positions of the
                mobile atoms, returns labels indicating their membership in
                "agreement groups." All atoms in an agreement group at some frame
                must agree, by a majority vote (see `min_winner_percentage`),
                on a single structure group. Defaults to
                `surfator.grouping.all_atoms_agree`, which places all mobile atoms
                into a single agreement group.
            - structure_group_compatability (matrix-like): a square, symmetric
                matrix indicating the compatability between the structure groups.
                Symmetry is assumed. The side length is `max(site_groups)`.
                Defaults to `None`, in which case all entries are assumed to be
                `True`, that is, all structure groups are compatable.
            - skin (float): Skin to use for the underlying ASE `NeighborList`.
                Defaults to 0.1.
        Returns:
            a SiteTrajectory
        """
        # -- Housekeeping --
        n_frames = traj.shape[0]
        n_ref_atoms = ref_sn.n_sites
        n_mob_atoms = ref_sn.n_mobile
        assert len(ref_sn.mobile_mask) == traj.shape[1]
        assert traj.shape[2] == 3

        if np.any(ref_sn.static_mask):
            raise ValueError("Reference structure SiteNetwork cannot have static atoms")

        structgrps = getattr(ref_sn, STRUCTURE_GROUP_ATTRIBUTE)
        assert np.min(structgrps) >= 0
        n_structgrps = np.max(structgrps) + 1
        # It's square
        assert structure_group_compatability.shape[0] == structure_group_compatability.shape[1] == n_structgrps, "`structure_group_compatability` is not square or has wrong dimensions"
        # It's symmetric
        assert np.all(structure_group_compatability == structure_group_compatability.T), "Structure group compatability must be symmetric"
        # All groups are compatable with themselves
        assert np.all(np.diagonal(structure_group_compatability)), "All structure groups must be marked as compatable with themselves."
        structgrp_incomap = ~structure_group_compatability
        ref_atoms_compatable_with = np.zeros(shape = (n_structgrps, n_ref_atoms), dtype = np.bool)
        ref_atoms_of_group = np.zeros(shape = (n_structgrps, n_ref_atoms), dtype = np.bool)
        for structgrp in range(n_structgrps):
            compat_structgrps = np.where(structure_group_compatability[structgrp])[0]
            ref_atoms_compatable_with[structgrp] = np.in1d(structgrps, compat_structgrps)
            ref_atoms_of_group[structgrp] = structgrps == structgrp

        # -- Build KDTree --
        # scipy's KDTree only supports cubic periodic boxes, so we do everything in crystal coordinates
        pbcc = PBCCalculator(ref_sn.structure.cell)
        cell_coord_traj = traj.copy()
        cell_coord_traj.shape = (-1, 3)
        pbcc.to_cell_coords(cell_coord_traj)
        cell_coord_traj.shape = traj.shape

        site_pos_cell_coords = ref_sn.centers.copy()
        pbcc.to_cell_coords(site_pos_cell_coords)

        # Different cell dimensions have different sizes. Since this is for the
        # kdtree, we're gonna be generous and then be accurately conservative
        # with the real-space distances.
        cutoff_cell_coords = np.min(np.linalg.norm(ref_sn.structure.cell, axis = 1))
        logger.debug("Magnitude of largest cell vector: %f" % cutoff_cell_coords)
        cutoff_cell_coords = cutoff / cutoff_cell_coords
        logger.debug("Cell coord. cutoff: %f" % cutoff_cell_coords)

        kdtree = cKDTree(
            data = site_pos_cell_coords,
            boxsize = 1 # Make it periodic
        )

        # Will be passed to the agreement group function
        mobile_struct = ref_sn.structure[ref_sn.mobile_mask]

        # Outputs
        average_majority = 0.
        average_majority_n = 0
        min_majority = np.inf
        site_assignments = np.full(shape = (n_frames, n_mob_atoms), fill_value = -1, dtype = np.int)

        # Buffers
        nearest_neighbors = np.empty(shape = n_mob_atoms, dtype = np.int)
        can_assign_to = np.empty(shape = n_ref_atoms + n_mob_atoms, dtype = np.bool)
        structgrps_seen = np.empty(shape = n_structgrps, dtype = np.bool)
        assigned = np.ones(shape = n_mob_atoms, dtype = np.bool)
        site_weights = np.ones(shape = n_ref_atoms, dtype = np.float)
        site_available = np.ones(shape = n_ref_atoms, dtype = np.bool)
        k_neighbor = 20
        neighbor_dist = np.empty(shape = k_neighbor, dtype = np.float)

        # -- Do structure group analysis --
        for frame_idex, frame in enumerate(tqdm(cell_coord_traj)):
            mobile_struct.positions[:] = traj[frame_idex, ref_sn.mobile_mask]

            # - (1) - Determine agreement groups
            agreegrp_labels = agreement_group_function(mobile_struct)
            if isinstance(agreegrp_labels, tuple): # An ordering was given
                agreegrp_labels, agreegrp_order = agreegrp_labels
            else:
                # Just uniq and sort
                agreegrp_order = np.unique(agreegrp_labels)
                agreegrp_order.sort()
            agreegrp_masks = [agreegrp_labels == lbl for lbl in agreegrp_order]
            logger.debug("At frame %i had %i agreegrps: %s of sizes %s" % (frame_idex, len(agreegrp_masks), agreegrp_order, [np.sum(m) for m in agreegrp_masks]))
            assert not np.any(np.logical_and.reduce(agreegrp_masks, axis = 0)), "Two or more agreement groups intersected at frame %i" % frame_idex
            assert np.all(np.logical_or.reduce(agreegrp_masks, axis = 0)), "Not all mobile atoms were assigned to an agreegrp"

            # Seen no structure groups yet
            structgrps_seen.fill(False)
            # Assume all assigned
            assigned.fill(True)
            site_available.fill(True)

            # - (2) - In order, assign the agreegrps
            for agreegrp_i, agreegrp_mask in enumerate(agreegrp_masks):
                site_weights.fill(1.0)
                can_assign_to[:n_ref_atoms] = np.logical_and.reduce(ref_atoms_compatable_with[structgrps_seen])
                can_assign_to[:n_ref_atoms] &= site_available
                can_assign_to[n_ref_atoms:] = False # Can never assign to mobile atom
                n_can_assign = np.sum(can_assign_to)
                assert n_can_assign <= n_ref_atoms
                if n_can_assign == 0:
                    raise StructureGroupCompatabilityError("At agreegrp %i, there are no structure groups compatible with the existing assignments, which are: %s" % (agreegrp_i, np.where(structgrps_seen)[0]))

                to_assign = np.where(agreegrp_mask)[0]

                for is_last_round in (False, True):
                    # If the agreegrp is AGREE_GROUP_NONE, we don't actually want to enforce agreement
                    if agreegrp_order[agreegrp_i] == AGREE_GROUP_NONE:
                        is_last_round = True

                    for mob_i in to_assign:
                        kd_neighbor_dist, neighbor_idex = kdtree.query(frame[mob_i], k = k_neighbor, distance_upper_bound = cutoff_cell_coords)
                        # Euclidean distances in cell space are rather meaningless -- recompute them in real space
                        pbcc.distances(traj[frame_idex, mob_i], ref_sn.centers[neighbor_idex], out = neighbor_dist)
                        if len(neighbor_idex) > 0:
                            # inf in kd distances indicates no neighbor, set to inf
                            neighbor_dist[np.isinf(kd_neighbor_dist)] = np.inf
                            # Apply can_assign_to
                            neighbor_dist[~can_assign_to[neighbor_idex]] = np.inf
                            # Strict distance cutoff:
                            neighbor_dist[neighbor_dist > cutoff] = np.inf
                            # Apply site weights -- Products with 0 and Inf give NaN, for which < Inf gives False, so we're OK
                            neighbor_dist *= site_weights[neighbor_idex]
                            nearest_neighbor = np.nanargmin(neighbor_dist)
                            assert neighbor_dist[nearest_neighbor] < np.inf, "Had no site options for mobile atom %i in agreegrp %i" % (mob_i, agreegrp_i)
                            nearest_neighbors[mob_i] = neighbor_idex[nearest_neighbor]
                        else:
                            # Can't assign this one
                            logger.warning("At frame %i couldn't assign mobile atom %i" % (frame_idex, mob_i))
                            nearest_neighbors[mob_i] = -1
                            assigned[mob_i] = False
                        site_assignments[frame_idex, mob_i] = nearest_neighbors[mob_i]
                        if is_last_round:
                            site_available[nearest_neighbors[mob_i]] = False

                    if is_last_round:
                        break
                    else:
                        structgrp_assignments = structgrps[nearest_neighbors[agreegrp_mask & assigned]]
                        candidates, votes = np.unique(structgrp_assignments, return_counts = True)
                        winner_idex = np.argmax(votes)
                        winner = candidates[winner_idex]

                        # Atoms that voted for something compatable with the winner
                        # can be considered to have voted for the winner, just
                        # "less"; what "less" means is quantified by
                        # `self.runoff_votes_weight`.
                        runoff_votes = structure_group_compatability[winner, candidates] * votes
                        runoff_votes[winner_idex] = 0
                        runoff_votes = self.runoff_votes_weight * np.sum(runoff_votes)
                        total_votes = votes[winner_idex] + runoff_votes

                        majority = total_votes / len(structgrp_assignments)
                        if majority < min_majority:
                            min_majority = majority
                        average_majority += majority
                        average_majority_n += 1

                        if majority < self.min_winner_percentage:
                            raise StructureGroupVotingError("Winning structure group for agreegrp %i at frame %i got (%i + %.1f runoff = %.1f)/%i = %i%% votes, which is below set threshold of %i%%" % (agreegrp_order[agreegrp_i], frame_idex, votes[winner_idex], runoff_votes, total_votes, len(structgrp_assignments), 100 * total_votes / len(structgrp_assignments), 100 * self.min_winner_percentage))
                        logger.debug("At frame %i agreegrp %i voted for structgrp %i with majority (%i + %.1f runoff = %.1f)/%i" % (frame_idex, agreegrp_i, winner, votes[winner_idex], runoff_votes, total_votes, len(structgrp_assignments)))

                        # Assign only to structure groups compatable with the winner
                        # We do &= because constraints imposed by previous agreegrps
                        # take precidence.
                        can_assign_to[:n_ref_atoms] &= ref_atoms_compatable_with[winner]
                        site_weights[ref_atoms_of_group[winner]] = 1 - self.winner_bias
                        # Though technically we only need to reprocess those that
                        # were incompatible with the winner, the neighborlist
                        # already exists and reassigning them all is easier
                        # and doesn't change the number of allocations being done.
                        # Keep track -- the winner is now "seen"
                        structgrps_seen[winner] = True
                        # Now we loop and reassign

        self.average_majority = average_majority / average_majority_n
        self.minimum_majority = min_majority

        assert np.min(site_assignments) >= 0 # Make sure all atoms assigned at all times
        out_st = SiteTrajectory(ref_sn, site_assignments)
        out_st.set_real_traj(traj)
        self._has_run = True
        return out_st
