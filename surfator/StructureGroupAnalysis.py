from sitator import SiteNetwork, SiteTrajectory
from sitator.util.progress import tqdm

import numbers

import ase
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
import ase.geometry

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
                 min_winner_percentage = 0.50001):
        self.min_winner_percentage = min_winner_percentage
        self._has_run = False


    def run(self,
            ref_sn,
            traj,
            cutoff,
            agreement_group_function = surfator.grouping.all_atoms_agree,
            structure_group_compatability = None,
            skin = 0.1,
            return_unwrapped_clamped_traj = False,
            unwrapped_clamped_passthrough = True):
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
                Symmetry is assumed but not checked. The side length is `max(site_groups)`.
                Can be sparse. Defaults to `None`, in which case all entries are
                assumed to be `True`, that is, all structure groups are compatable.
            - skin (float): Skin to use for the underlying ASE `NeighborList`.
                Defaults to 0.1.
            - return_unwrapped_clamped_traj (bool, default: False): A byproduct
                of this analysis is a trajectory where mobile atoms are "clamped"
                to the nearest periodic image of their site (i.e. if the trajectory
                is unwrapped, the clamped trajectory is too). If set to true,
                this trajectory is returned.
            - unwrapped_clamped_passthrough (bool, defualt: True): Relevant
                if `return_unwrapped_clamped_traj == True`. If a mobile atom
                cannot be assigned and this is set to True, its unprocessed
                position will be passed through to the unwrapped clamped
                trajectory. If this is False, its clamped position at that frame
                will be NaNs.
        Returns:
            a SiteTrajectory
        """
        # -- Housekeeping --
        n_frames = traj.shape[0]
        n_ref_atoms = ref_sn.n_sites
        n_mob_atoms = ref_sn.n_mobile
        assert len(ref_sn) == traj.shape[1]
        assert traj.shape[2] == 3

        if np.any(ref_sn.static_mask):
            raise ValueError("Reference structure SiteNetwork cannot have static atoms")

        if ref_sn.has_attribute(SITE_RADIUS_ATTRIBUTE):
            ref_radii = getattr(ref_sn, SITE_RADIUS_ATTRIBUTE)
        else:
            if isinstance(cutoff, numbers.Number):
                ref_radii = np.full(shape = len(ref_sn), fill_value = cutoff)
            else:
                raise TypeError("`ref_sn` does not provide site radii, but `cutoff` is '%r', not a single value." % cutoff)

        if isinstance(cutoff, numbers.Number):
            cutoff = np.full(shape = n_mob_atoms, fill_value = 0.5 * cutoff)

        radii = np.concatenate((ref_radii, cutoff))

        structgrps = getattr(ref_sn, STRUCTURE_GROUP_ATTRIBUTE)
        assert np.min(structgrps) >= 0
        n_structgrps = np.max(structgrps) + 1
        assert structure_group_compatability.shape[0] == structure_group_compatability.shape[1] == n_structgrps
        structgrp_incomap = ~structure_group_compatability
        ref_atoms_compatable_with = np.zeros(shape = (n_structgrps, n_ref_atoms), dtype = np.bool)
        for structgrp in range(n_structgrps):
            compat_structgrps = np.where(structure_group_compatability[structgrp])[0]
            ref_atoms_compatable_with[structgrp] = np.in1d(structgrps, compat_structgrps)

        # -- Build neighbor list --
        # Build an `Atoms` that has the reference sites as atoms, first, then
        # the mobile atoms afterwards.
        REF_ATOM = 0
        MOB_ATOM = 1
        traj_struct = ref_sn.structure.copy()
        traj_struct.set_tags(np.full(n_mob_atoms, MOB_ATOM))
        traj_struct.set_positions(traj[0])
        full_struct = ref_sn.structure.copy()
        full_struct.set_tags(np.full(n_ref_atoms, REF_ATOM))
        full_struct.extend(traj_struct)

        # Build the neighborlist
        nl = NeighborList(cutoffs = radii,
                          skin = skin,
                          self_interaction = False,
                          bothways = True,
                          primative = NewPrimitiveNeighborList) # Better performance, see docs

        # Will be passed to the agreement group function
        mobile_struct = ref_sn.structure[ref_sn.mobile_mask]

        # Outputs
        average_majority = 0.
        average_majority_n = 0
        min_majority = np.inf
        site_assignments = np.full(shape = (n_frames, n_mob_atoms), fill_value = -1, dtype = np.int)
        if return_unwrapped_clamped_traj:
            outtraj = np.full_like(traj, np.nan)

        # Buffers
        nearest_neighbors = np.empty(shape = n_mob_atoms, dtype = np.int)
        can_assign_to = np.empty(shape = n_ref_atoms + n_mob_atoms, dtype = np.bool)
        structgrps_seen = np.empty(shape = n_structgrps, dtype = np.bool)
        assigned = np.ones(shape = n_mob_atoms, dtype = np.bool)

        # -- Do structure group analysis --
        for frame_idex, frame in enumerate(tqdm(traj)):
            full_struct.get_positions()[n_ref_atoms:] = frame[ref_sn.mobile_mask]
            mobile_struct.get_positions()[:] = frame[ref_sn.mobile_mask]

            # - (1) - Determine agreement groups
            agreegrp_labels = agreement_group_function(mobile_struct)
            if isinstance(agreegrp_labels, tuple): # An ordering was given
                agreegrp_labels, agreegrp_order = agreegrp_labels
            else:
                # Just uniq and sort
                agreegrp_order = np.unique(agreegrp_labels)
                agreegrp_order.sort()
            agreegrp_masks = [agreegrp_labels == lbl for lbl in agreegrp_order]
            assert not np.any(np.logical_and.reduce(agreegrp_masks, axis = 0)), "Two or more agreement groups intersected at frame %i" % frame_idex

            # Seen no structure groups yet
            structgrps_seen.fill(False)
            # Assume all assigned
            assigned.fill(True)

            # - (2) - Update neighbor list
            nl.update(full_struct)

            # - (3) - In order, assign the agreegrps
            for agreegrp_i, agreegrp_mask in enumerate(agreegrp_masks):
                can_assign_to[:n_ref_atoms] = np.logical_and.reduce(ref_atoms_compatable_with[structgrps_seen])
                can_assign_to[n_ref_atoms:] = False # Can never assign to mobile atom
                n_can_assign = np.sum(can_assign_to)
                assert n_can_assign <= n_ref_atoms
                if n_can_assign == 0:
                    raise StructureGroupCompatabilityError("At agreegrp %i, there are no structure groups compatible with the existing assignments, which are: %s" % (agreegrp_i, np.where(structgrps_seen)[0]))

                to_assign = np.where(agreegrp)[0]

                for is_last_round in (False, True):
                    # If the agreegrp is AGREE_GROUP_NONE, we don't actually want to enforce agreement
                    if agreegrp_tags[agree_i] == AGREE_GROUP_NONE:
                        is_last_round = True

                    for mob_i in to_assign:
                        neighbor_idex, neighbor_offset = nl.get_neighbors(n_ref_atoms + mob_i)
                        if len(neighbor_idex) > 0:
                            neighbor_mic_positions = (full_struct.get_positions()[neighbor_idex] + np.dot(neighbor_offset, ref_structure.cell))
                            neighbor_dists = np.linalg.norm(full_struct.get_positions()[n_ref_atoms + mob_i] - neighbor_mic_positions, axis = 1)
                            # Take can_assign_to into account
                            neighbor_dists[~can_assign_to[neighbor_idex]] = np.inf
                            nearest_neighbor = np.argmin(neighbor_dists)
                            nn_dist = neighbor_dists[nearest_neighbor]
                            assert nn_dist < np.inf, "Had no site options for mobile atom %i in agreegrp %i" % (mob_i, agreegrp_i)
                            assert nn_dist < cutoff + 2 * skin, "Neighborlist somehow returned neighbors over its cutoff + 2 * skin. This should not happen."

                            nearest_neighbors[mob_i] = neighbor_idex[nearest_neighbor]
                            if return_unwrapped_clamped_traj:
                                outtraj[frame_idex, mob_i] = neighbor_mic_positions[nearest_neighbor]
                        else:
                            # Can't assign this one
                            logger.warning("At frame %i couldn't assign mobile atom %i" % (frame_idex, mob_i))
                            nearest_neighbors[mob_i] = -1
                            assigned[mob_i] = False
                            if return_unwrapped_clamped_traj and unwrapped_clamped_passthrough:
                                outtraj[frame_idex, mob_i] = frame[mob_i]
                        site_assignments[frame_idex, mob_i] = nearest_neighbors[mob_i]

                    if is_last_round:
                        break
                    else:
                        structgrp_assignments = structgrps[nearest_neighbors[agreegrp & assigned]]
                        candidates, votes = np.unique(structgrp_assignments, return_counts = True)
                        winner_idex = np.argmax(votes)
                        winner = candidates[winner_idex]

                        majority = votes[winner_idex] / len(structgrp_assignments)
                        if majority < min_majority:
                            min_majority = majority
                        average_majority +=
                        average_majority_n += 1

                        if votes[winner_idex] < min_winner_percentage * len(structgrp_assignments):
                            raise StructureGroupVotingError("Winning structure group got %i/%i = %i%% votes, which is below set threshold of %i%%" % (votes[winner_idex], len(structgrp_assignments), 100 * votes[winner_idex] / len(structgrp_assignments), 100 * min_winner_percentage))

                        # Assign only to structure groups compatable with the winner
                        # We do &= because constraints imposed by previous agreegrps
                        # take precidence.
                        can_assign_to[:n_ref_atoms] &= ref_atoms_compatable_with[winner]
                        # We should always have SOMETHING to assign to -- the
                        # winner is necessarily compatible with the existing
                        # assignments
                        assert np.sum(can_assign_to) != 0
                        # Though technically we only need to reprocess those that
                        # were incompatible with the winner, the neighborlist
                        # already exists and reassigning them all is easier
                        # and doesn't change the number of allocations being done.
                        # Keep track -- the winner is now "seen"
                        structgrps_seen[winner] = True
                        # Now we loop and reassign

        assert np.min(site_assignments) >= 0 # Make sure all atoms assigned at all times
        out_st = SiteTrajectory(ref_sn, site_assignments)
        out_st.set_real_traj(traj)
        self._has_run = True
        if return_unwrapped_clamped_traj:
            return out_st, outtraj
        else:
            return out_st
