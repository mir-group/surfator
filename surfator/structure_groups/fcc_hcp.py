import numpy as np

from sitator import SiteNetwork

from surfator import STRUCTURE_GROUP_ATTRIBUTE

class ClosePackedReferenceSites(SiteNetwork):
    """Utility class for representing typical reference close-packed structures.

    For close-packed FCC/HCP bulks and surfaces.

    Assigns one structure group for each reference site type, per layer. No
    structure group will span multiple layers.

    Enforces intra-layer exclusivity (one structure group per layer) and that
    neighboring layers cannot have the same layer_type.

    Args:
        - (standard SiteNetwork arguments by keyword)
        - reference_positions
        - layer_labels (n_sites ndarray int): Array of labels indicating which
            layer each reference site belongs to.
        - layer_type_labels (n_sites ndarray): Array of labels indicating which
            arangement (ex. A, B, or C) of its layer each reference site belonds to.
    """
    def __init__(self,
                 reference_positions,
                 layer_labels,
                 layer_type_labels,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.centers = reference_positions
        self.add_site_attribute('layer', layer_labels)
        self.add_site_attribute('layer_type', layer_type_labels)
        self._n_layers = np.ptp(self.layer)
        self._generate_structure_groups()

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def n_structure_groups(self):
        return self._n_structgrps

    def get_structure_group_compatability(self):
        ngrps = self.n_structure_groups
        mapping = self.layer_to_structure_group
        uniq_layers = np.unique(self.layer)
        uniq_types = np.unique(self.layer_type)
        grp_to_info = {v: k for k, v in mapping.items()}
        out = np.ones(shape = (ngrps, ngrps), dtype = np.bool)
        for sgrp in range(ngrps):
            layer, layer_type = grp_to_info[sgrp]
            # incompat with others in layer
            for t in uniq_types:
                if t == layer_type:
                    continue
                out[sgrp, mapping[(layer, t)]] = False
            # incompat with same in neighboring layer
            for other_layer in [layer + 1, layer - 1]:
                if other_layer in uniq_layers:
                    out[sgrp, mapping[(other_layer, layer_type)]] = False
        return out

    def _generate_structure_groups(self):
        mapping = {}
        structgrp_i = 0
        uniq_layers = np.unique(self.layer)
        uniq_types = np.unique(self.layer_type)
        structure_groups = np.empty(shape = len(self), dtype = np.int)
        for layer in uniq_layers:
            for layer_type in uniq_types:
                mask = (self.layer == layer) & (self.layer_type == layer_type)
                structure_groups[mask] = structgrp_i
                mapping[(layer, layer_type)] = structgrp_i
                structgrp_i += 1
        self._n_structgrps = structgrp_i
        self.layer_to_structure_group = mapping
        self.add_site_attribute(STRUCTURE_GROUP_ATTRIBUTE, structure_groups)
