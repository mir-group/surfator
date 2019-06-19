from sitator import SiteNetwork, SiteTrajectory
from sitator.util.progress import tqdm

import numbers

import ase
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
import ase.geometry

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
STRUCTURE_GROUPS_ATTRIBUTE = 'structure_groups'


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
            agreement_group_function,
            skin = 0.1):
        """

        Args:
            - ref_sn (SiteNetwork): A `SiteNetwork` containing the sites to which
                the mobile atoms will be assigned. Can contain multiple, possibly
                exclusive, sets of sites called "structure groups." The site
                attribute `STRUCTURE_GROUPS_ATTRIBUTE` indicates, for each site,
                which structure groups that site is compatible with. The value
                is an integer bitmap: the nth bit indicates whether that site is
                compatible with structure group n.
            - traj (ndarray, n_frames x n_atoms x 3):
            - cutoff (ndarray or float): Cutoff radii can be given in `ref_sn`
                for the sites with site attribute SITE_RADIUS_ATTRIBUTE. If none
                are given, they will be set to `0.5 * cutoff`. If an array is
                given for `cutoff`, radii must be provided in `ref_sn`. Radii for
                the mobile atoms can be given through `cutoff` either as a single
                float cutoff, half of which is then used as the radius for all
                mobile atoms, or as an array of radii for each mobile atom which
                are not modified. In distance units.
            - agreement_group_function (callable taking an Atoms):
            - skin (float): Skin to use for the underlying ASE `NeighborList`.
                Defaults to 0.1.
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

        structgrp_masks = getattr(ref_sn, STRUCTURE_GROUPS_ATTRIBUTE)

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

        # Buffers
        nearest_neighbors = np.empty(shape = n_mob_atoms, dtype = np.int)
        can_clamp_to = np.empty(shape = n_ref_atoms + n_mob_atoms, dtype = np.bool)

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

            # Start with all 0 bits -- seen no structure groups yet
            structure_groups_seen = 0

            # - (2) - Update neighbor list
            nl.update(full_struct)

            # - (3) - In order, assign the agreegrps
            for agreegrp_i, agreegrp_mask in enumerate(agreegrp_masks):
                # We can clamp to any site that doesn't say no to a group we've seen
                # i.e. -- doesn't have a 0 anywhere we have a 1
                # ~structgrp_masks is "what I can't deal with" and
                # structure_groups_seen is "what there is," so if their AND
                # is anything other than zero, that site is incompatible with
                # the current structure
                can_clamp_to[:n_ref_atoms] = ~(~structgrp_masks & structure_groups_seen)
                can_clamp_to[n_ref_atoms:] = False # Can never clamp to mobile atom
                assert np.sum(can_clamp_to) <= n_ref_atoms

                to_assign = np.where(agreegrp)[0]

                for is_last_round in (False, True):
                    # If the agreegrp is -1, we don't actually want to enforce agreement
                    if agreegrp_tags[agree_i] == -1:
                        is_last_round = True

                    #print("Agreegrp %i last_round %i" % (agree_i, is_last_round))
                    #view(full_struct[can_clamp_to])
                    #_ = input("Enter to cont")

                    for mob_i in to_assign:
                        neighbor_idex, neighbor_offset = nl.get_neighbors(n_ref_atoms + mob_i)
                        neighbor_mic_positions = (positions[neighbor_idex] + np.dot(neighbor_offset, ref_structure.cell))

                        neighbor_dists = np.linalg.norm(positions[n_ref_atoms + mob_i] - neighbor_mic_positions, axis = 1)
                        # Don't care if close to another trajectory mobile atom
                        neighbor_dists[~can_clamp_to[neighbor_idex]] = np.inf
                        nearest_neighbor = np.argmin(neighbor_dists)
                        nn_dist = neighbor_dists[nearest_neighbor]
                        assert nn_dist < np.inf, "What?"
                        assert nn_dist < cutoff + 2 * skin, "What? Over cutoff"

                        nearest_neighbors[mob_i] = neighbor_idex[nearest_neighbor]
                        # For unwrapped positions
                        #outtraj[f_idex, mob_i] = neighbor_mic_positions[nearest_neighbor]
                        # For wrapped positions
                        outtraj[f_idex, mob_i] = ref_structure.get_positions()[nearest_neighbors[mob_i]]

                    if is_last_round:
                        break
                    else:
                        # Get majority ref group
                        # ref groups defined by ref_structure tags
                        # Find winning alt structure group
                        assigned_to_alt = ref_structure.get_tags()[nearest_neighbors[agreegrp]]
                        candidates, votes = np.unique(assigned_to_alt, return_counts = True)
                        winner_idex = np.argmax(votes)
                        winner = candidates[winner_idex]

                        majority = votes[winner_idex] / len(assigned_to_alt)
                        if majority < min_majority:
                            min_majority = majority
                        average_majority +=
                        average_majority_n += 1

                        if votes[winner_idex] < min_winner_percentage * len(assigned_to_alt):
                            pass # TODO TODO

                        # Clamp only to it
                        can_clamp_to[:n_ref_atoms][~(ref_structure.get_tags() == winner)] = False
                        # Only need to process ones that disagreed with majority
                        to_assign = to_assign[assigned_to_alt != winner]
                        # Keep track
                        agreegrps_to_alts[agreegrp_tags[agree_i]] = winner
                        # Now we loop and reclamp

        assert not np.isnan(np.sum(site_assignments))

        self._has_run = True

        return out_st
