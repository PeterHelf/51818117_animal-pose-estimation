#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

import numpy as np
from scipy.spatial import cKDTree


def find_closest_neighbors(
    query: np.ndarray, ref: np.ndarray, k: int = 3
) -> np.ndarray:
    """Greedy matching of predicted keypoints to ground truth keypoints

    Args:
        query: the query keypoints
        ref: the reference keypoints
        k: The list of k-th nearest neighbors to return.

    Returns:
        an array of shape (len(query), ) containing the index of the closest
        reference keypoint for each query keypoint
    """
    n_preds = ref.shape[0]
    tree = cKDTree(ref)
    dist, inds = tree.query(query, k=k)
    idx = np.argsort(dist[:, 0])
    neighbors = np.full(len(query), -1, dtype=int)
    picked = {tree.n}
    for i, ind in enumerate(inds[idx]):
        for j in ind:
            if j not in picked:
                picked.add(j)
                neighbors[idx[i]] = j
                break
        if len(picked) == (n_preds + 1):
            break
    return neighbors