
import numpy as np

import numbers
import itertools

from scipy.spatial import cKDTree

import ase
import ase.geometry

from sitator import SiteNetwork, SiteTrajectory
from sitator.util.progress import tqdm
from sitator.util import PBCCalculator
from sitator.visualization import plot_atoms, plot_points

import surfator.agreement_groups

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

AGREE_GROUP_NONE = -1 # No agreement needed, but assignment will still occur
AGREE_GROUP_UNASSIGNED = -2 # Will simply be marked as unassigned

N_ROUNDS = 3

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
        - runoff_votes_weight (float): Atoms in an agreement group that voted
            for groups compatible with the candidate that won the most votes
            can cast "runoff" votes for that leading candidate. The impact of
            those votes is controlled by this parameter: at a value of 0, they
            have no effect; at a value of 1 a runoff vote from a compatable
            candidate is as good as one for the leading candidate directly.
            This helps to break ties and increase majorities in uncertain votes.
        - winner_bias (float): Whether (and by how much) to bias the assignment
            to the winning structure group. All compatible structure groups to the
            winner are available for assignment, but sites belonging to the
            winning structure group can be made more appealing. A value of 0
            indicates no bias; a value of 1 ensures the choice of a winning
            structure group site (assuming one exists within the cutoff distnace).
        - error_on_no_majority (bool): If True, a majority below `min_winner_percentage`
            will result in an error. If False, that agreement group will simply
            be unassigned at that time.
        - eps_factor (float): Set eps (for KDTree queries, see scipy documentation)
            to eps_factor * cutoff. Defaults to 0.05 (5%), which gives a
            meaningful performance increase with no real cost to accuracy.
    """
    def __init__(self,
                 min_winner_percentage = 0.50001,
                 runoff_votes_weight = 0.5,
                 winner_bias = 0.5,
                 error_on_no_majority = True,
                 eps_factor = 0.1,
                 kdtree_n_jobs = 1):
        assert 0 <= min_winner_percentage <= 1
        self.min_winner_percentage = min_winner_percentage
        assert 0 <= runoff_votes_weight <= 1
        self.runoff_votes_weight = runoff_votes_weight
        self._has_run = False
        assert 0 <= winner_bias < 1
        self.winner_bias = winner_bias
        self.error_on_no_majority = error_on_no_majority
        self.eps_factor = eps_factor
        self.kdtree_n_jobs = kdtree_n_jobs

    def run(self,
            ref_sn,
            traj,
            cutoff,
            k_neighbor = 6,
            agreement_group_function = surfator.agreement_groups.all_atoms_agree,
            structure_group_compatability = None,
            return_assignments = False):
        """
        Args:
            ref_sn (SiteNetwork): A `SiteNetwork` containing the sites to which
                the mobile atoms will be assigned. Can contain multiple, possibly
                exclusive, sets of sites called "structure groups." The site
                attribute `STRUCTURE_GROUP_ATTRIBUTE` indicates, for each site,
                which structure group it belongs to.
            traj (ndarray, n_frames x n_atoms x 3):
            cutoff (float, distance): The maximum distance between a mobile atom
                and the site it is assigned to.
            k_neighbor (int): How many of the nearest sites to an atom to consider.
                If 1, site weighting is effectively disabled. Should be set high
                enough that, after eliminating incompatable sites, a site can still
                be found.
            agreement_group_function (callable taking an Atoms): A function that,
                given an `Atoms` object containing the current positions of the
                mobile atoms, returns labels indicating their membership in
                "agreement groups." All atoms in an agreement group at some frame
                must agree, by a majority vote (see `min_winner_percentage`),
                on a single structure group. Defaults to
                `surfator.agreement_groups.all_atoms_agree`, which places all mobile atoms
                into a single agreement group.
                This function also returns as its second value an agreement group
                adjacency matrix (square, n_agreegrp x n_agreegrp, bool) indicating,
                for each agreement group, which other agreement groups structure
                groups it ought to take into consideration.
            structure_group_compatability (matrix-like): a square, symmetric
                matrix indicating the compatability between the structure groups.
                Symmetry is assumed. The side length is `max(site_groups)`.
                Defaults to `None`, in which case we will attempt to call
                `get_structure_group_compatability()` on `ref_sn` (presuming
                it to be a subclass of `SiteNetwork`).
            return_assignments (bool): Whether to return which agreement
                group and which structure group each mobile atom was assigned to
                at each frame. Depending on `agreement_group_function`, the
                labels for agreement groups might change from
                frame to frame and be meaningless -- please check your
                `agreement_group_function` to see if this makes sense.
        Returns:
            a SiteTrajectory[, agreegrp_assignments, structgrp_assignments]
        """
        # -- Housekeeping --
        n_frames = traj.shape[0]
        n_ref_atoms = ref_sn.n_sites
        n_mob_atoms = ref_sn.n_mobile
        centers = ref_sn.centers
        eps = self.eps_factor * cutoff
        assert len(ref_sn.mobile_mask) == traj.shape[1]
        assert traj.shape[2] == 3

        if np.any(ref_sn.static_mask):
            raise ValueError("Reference structure SiteNetwork cannot have static atoms")

        structgrps = getattr(ref_sn, STRUCTURE_GROUP_ATTRIBUTE)
        assert np.min(structgrps) >= 0
        n_structgrps = np.max(structgrps) + 1
        if structure_group_compatability is None:
            try:
                structure_group_compatability = ref_sn.get_structure_group_compatability()
            except AttributeError:
                raise ValueError("`structure_group_compatability` is `None`, but `ref_sn` doesn't implement `get_structure_group_compatability()`")
        # It's square
        assert structure_group_compatability.shape[0] == structure_group_compatability.shape[1] == n_structgrps, "`structure_group_compatability` is not square or has wrong dimensions"
        # It's symmetric
        assert np.all(structure_group_compatability == structure_group_compatability.T), "Structure group compatability must be symmetric"
        # All groups are compatable with themselves
        assert np.all(np.diagonal(structure_group_compatability)), "All structure groups must be marked as compatable with themselves."
        structgrp_incomap = ~structure_group_compatability
        # The unknown structgrp, -1, is compatable with everything
        ref_atoms_compatable_with = np.zeros(shape = (n_structgrps + 1, n_ref_atoms), dtype = np.bool)
        ref_atoms_compatable_with[-1] = True
        ref_atoms_of_group = np.zeros(shape = (n_structgrps, n_ref_atoms), dtype = np.bool)
        for structgrp in range(n_structgrps):
            compat_structgrps = np.where(structure_group_compatability[structgrp])[0]
            ref_atoms_compatable_with[structgrp] = np.in1d(structgrps, compat_structgrps)
            ref_atoms_of_group[structgrp] = structgrps == structgrp

        # -- Build KDTree --
        pbcc = PBCCalculator(ref_sn.structure.cell)
        wrapped_traj = traj.copy()
        wrapped_traj.shape = (-1, 3)
        pbcc.wrap_points(wrapped_traj)
        wrapped_traj.shape = traj.shape

        # Build set of reference sites with periodic copies
        images = list(itertools.product(range(-1, 2), repeat = 3))
        image_000 = images.index((0, 0, 0))
        ref_site_ids = np.tile(np.arange(ref_sn.n_sites), len(images))
        all_centers = np.tile(ref_sn.centers, (len(images), 1, 1))
        for image_i, image in enumerate(images):
            all_centers[image_i] += np.dot(ref_sn.structure.cell.T, image)
        all_centers.shape = (-1, 3)

        kdtree = cKDTree(
            data = all_centers,
        )

        # Will be passed to the agreement group function
        mobile_struct = ref_sn.structure[ref_sn.mobile_mask]

        # Outputs
        average_majority = 0.
        average_majority_n = 0
        min_majority = np.inf
        site_assignments = np.full(shape = (n_frames, n_mob_atoms), fill_value = -20, dtype = np.int)
        if return_assignments:
            agreegrp_assignments = np.empty(shape = (n_frames, n_mob_atoms), dtype = np.int)

        # Buffers
        nearest_neighbors = np.empty(shape = n_mob_atoms, dtype = np.int)
        can_assign_to = np.empty(shape = n_ref_atoms, dtype = np.bool)
        assigned = np.ones(shape = n_mob_atoms, dtype = np.bool)
        site_weights = np.ones(shape = n_ref_atoms, dtype = np.float)
        site_available = np.ones(shape = n_ref_atoms, dtype = np.bool)
        site_taken_by_atom = np.empty(shape = n_ref_atoms, dtype = np.int)
        site_distance_to_atom = np.empty(shape = n_ref_atoms)
        neighbor_dist = np.empty(shape = k_neighbor)

        # -- Do structure group analysis --
        for frame_idex, frame in enumerate(tqdm(wrapped_traj)):
            mobile_struct.positions[:] = traj[frame_idex, ref_sn.mobile_mask]

            # - (1) - Determine agreement groups
            agreeret = agreement_group_function(mobile_struct)
            if len(agreeret) == 3: # An ordering was given
                agreegrp_labels, agreegrp_order, agreegrp_adjacency = agreeret
            elif len(agreeret) == 2:
                agreegrp_labels, agreegrp_adjacency = agreeret
                # Just uniq and sort
                agreegrp_order = np.unique(agreegrp_labels)
                agreegrp_order.sort()
            else:
                raise ValueError("At frame %i got bad agreement group data %s" % (frame_i, agreeret))
            assert agreegrp_adjacency.shape[0] == agreegrp_adjacency.shape[1] == len(agreegrp_order)
            assert len(agreegrp_labels) == len(frame)
            if return_assignments:
                agreegrp_assignments[frame_idex] = agreegrp_labels
            agreegrp_masks = [agreegrp_labels == lbl for lbl in agreegrp_order]
            logger.debug("At frame %i had %i agreegrps: %s of sizes %s" % (frame_idex, len(agreegrp_masks), agreegrp_order, [np.sum(m) for m in agreegrp_masks]))
            assert not np.any(np.logical_and.reduce(agreegrp_masks, axis = 0)), "Two or more agreement groups intersected at frame %i" % frame_idex
            assert np.all(np.logical_or.reduce(agreegrp_masks, axis = 0)), "Not all mobile atoms were assigned to an agreegrp"

            # Seen no structure groups yet
            agreegrp_winners = np.empty(shape = len(agreegrp_order), dtype = np.int)
            agreegrp_winners.fill(-1)
            # Assume all assigned
            assigned.fill(True)
            site_available.fill(True)
            site_taken_by_atom.fill(-1)
            site_distance_to_atom.fill(np.inf)

            # Distances don't change between rounds or agreegroups:
            all_neighbor_dists, all_neighbor_idexs = kdtree.query(
                mobile_struct.positions,
                k = k_neighbor,
                distance_upper_bound = cutoff,
                eps = eps,
                n_jobs = self.kdtree_n_jobs
            )
            all_neighbor_idexs_shape = all_neighbor_idexs.shape
            all_neighbor_idexs.shape = (-1,)
            all_neighbor_idexs = ref_site_ids[all_neighbor_idexs]
            all_neighbor_idexs.shape = all_neighbor_idexs_shape

            # - (2) - In order, assign the agreegrps
            for agreegrp_i, agreegrp_mask in enumerate(agreegrp_masks):
                # The UNASSIGNED agreegrp are those atoms that couldn't be
                # figured out, and we want that unassigned status to percolate
                # up into the SiteTrajectory
                if agreegrp_order[agreegrp_i] == AGREE_GROUP_UNASSIGNED:
                    site_assignments[frame_idex, agreegrp_mask] = SiteTrajectory.SITE_UNKNOWN
                    continue

                site_weights.fill(1.0)
                # Can only assign to those sites that are compatable with
                # adjacent agreegrp's winning candidates
                neighbor_agreegrp_winners = agreegrp_winners[agreegrp_adjacency[agreegrp_i]]
                # Can leave -1's because final element of ref_atoms_compatable_with is
                # identity (all True) for that purpose
                np.logical_and.reduce(ref_atoms_compatable_with[neighbor_agreegrp_winners], out = can_assign_to)
                del neighbor_agreegrp_winners
                n_can_assign = np.sum(can_assign_to)
                assert n_can_assign <= n_ref_atoms
                if n_can_assign == 0:
                    raise StructureGroupCompatabilityError("At agreegrp %i, there are no structure groups compatible with the existing assignments." % (agreegrp_i))

                to_assign = np.where(agreegrp_mask)[0]

                for round in range(N_ROUNDS):
                    # If the agreegrp is AGREE_GROUP_NONE, we don't actually want to enforce agreement
                    if agreegrp_order[agreegrp_i] == AGREE_GROUP_NONE:
                        # Start with assignment round, then displacement round.
                        # All options will be left open without voting round.
                        round = 1

                    # At every round, enforce the no double occupancy requirement.
                    # The voting round doesn't change occupations, so this changes nothing.
                    can_assign_to &= site_available

                    displaced_by_closer = []
                    for mob_i in to_assign:
                        neighbor_idex = all_neighbor_idexs[mob_i]
                        neighbor_dist[:] = all_neighbor_dists[mob_i]
                        # Apply site weights
                        # We do this first so we're only doing arithmetic with
                        # real floats (no infs) to avoid NaNs, to avoid the performance
                        # hit of NaN checking in `np.nanargmin`
                        neighbor_dist *= site_weights[neighbor_idex]
                        # Apply can_assign_to
                        neighbor_dist[~can_assign_to[neighbor_idex]] = np.inf
                        # Strict distance cutoff:
                        # Unneeded cause KDTree is strict
                        #neighbor_dist[neighbor_dist > cutoff] = np.inf
                        nearest_neighbor = np.argmin(neighbor_dist)
                        dist_to_nn = neighbor_dist[nearest_neighbor]
                        if dist_to_nn < np.inf:
                            assign_to_site = neighbor_idex[nearest_neighbor]
                            nearest_neighbors[mob_i] = assign_to_site
                            if round == 1:
                                if dist_to_nn <= site_distance_to_atom[assign_to_site]:
                                    # This is the closest atom to the site, assign
                                    # Displace any previously assigned atom:
                                    to_displace = site_taken_by_atom[assign_to_site]
                                    if to_displace >= 0:
                                        displaced_by_closer.append(to_displace)
                                    site_distance_to_atom[assign_to_site] = dist_to_nn
                                    site_taken_by_atom[assign_to_site] = mob_i
                                else:
                                    # This atom has already been displaced
                                    displaced_by_closer.append(mob_i)
                        else:
                            # Can't assign this one
                            logger.warning("At frame %i couldn't assign mobile atom %i" % (frame_idex, mob_i))
                            nearest_neighbors[mob_i] = -1
                            assigned[mob_i] = False

                        # These will get overwritten in the final round
                        site_assignments[frame_idex, mob_i] = nearest_neighbors[mob_i]
                        if round > 0:
                            # Assignments during the voting round don't matter
                            # for site availability
                            site_available[nearest_neighbors[mob_i]] = False

                    if round == 2: # Displaced assignment round - last round
                        break
                    elif round == 1: # Assignment round
                        # set to only assign displaced atoms
                        to_assign = displaced_by_closer
                        if len(displaced_by_closer) > 0:
                            print("Frame %i displaced: %s" % (frame_idex, displaced_by_closer))
                    elif round == 0: # Voting round
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
                            msg = "Winning structure group for agreegrp %i at frame %i got (%i + %.1f runoff = %.1f)/%i = %i%% votes, which is below set threshold of %i%%" % (agreegrp_order[agreegrp_i], frame_idex, votes[winner_idex], runoff_votes, total_votes, len(structgrp_assignments), 100 * total_votes / len(structgrp_assignments), 100 * self.min_winner_percentage)
                            if self.error_on_no_majority:
                                raise StructureGroupVotingError(msg)
                            else:
                                logger.warning(msg + "; marking them unassigned.")
                                site_assignments[frame_idex, agreegrp_mask] = SiteTrajectory.SITE_UNKNOWN
                                break # Leave the last_rount loop; takes us to the next agreegrp

                        logger.debug("At frame %i agreegrp %i voted for structgrp %i with majority (%i + %.1f runoff = %.1f)/%i" % (frame_idex, agreegrp_i, winner, votes[winner_idex], runoff_votes, total_votes, len(structgrp_assignments)))

                        # Assign only to structure groups compatable with the winner
                        # We do &= because constraints imposed by previous agreegrps
                        # take precidence.
                        can_assign_to &= ref_atoms_compatable_with[winner]
                        site_weights[ref_atoms_of_group[winner]] = 1 - self.winner_bias
                        # Keep track -- the winner is now "seen"
                        agreegrp_winners[agreegrp_i] = winner
                        # Now we loop and reassign

        self.average_majority = average_majority / average_majority_n
        self.minimum_majority = min_majority

        assert np.min(site_assignments) >= -1 # Make sure all atoms assigned or intentionally unassigned at all times
        out_st = SiteTrajectory(ref_sn, site_assignments)
        out_st.set_real_traj(traj)
        self._has_run = True
        if return_assignments:
            translation = np.concatenate((structgrps, [-1]))
            structgrp_assignments = translation[site_assignments]
            return out_st, agreegrp_assignments, structgrp_assignments
        else:
            return out_st
