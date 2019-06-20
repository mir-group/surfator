
import numpy as np

from sitator.util import PBCCalculator

def get_layer_heights_kmeans(traj, cell, n, surface_normal = np.array([0, 0, 1])):
    """Find the heights of the centers of the layers, along `surface_normal`, in `traj`.

    Uses k-means over all (`surface_normal`-relative) heights in the trajectory.

    Args:
        - traj (ndarray n_frames x n_atoms x 3)
        - cell (ndarray 3x3 matrix)
        - n (int): The number of layers to identify (the k for k-means).
        - surface_normal (3-vector): A unit vector normal to the surface. Defaults
            to the z direction <0, 0, 1>.
    Returns:
        sorted ndarray of heights along surface normal
    """
    from sklearn.cluster import KMeans

    # We have to wrap first to get consistant results along the surface normal
    traj = traj.copy().reshape(-1, 3)
    pbcc = PBCCalculator(cell)
    pbcc.wrap_points(traj)

    heights = np.dot(surface_normal, traj.T)
    assert heights.shape = (traj.shape[0] * traj.shape[1],)

    kmeans = KMeans(n_clusters = n).fit(heights.reshape(-1, 1))
    heights = kmeans.cluster_centers_.reshape(-1)
    heights.sort()
    return heights
