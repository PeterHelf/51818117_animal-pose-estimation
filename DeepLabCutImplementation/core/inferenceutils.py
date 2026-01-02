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
from __future__ import annotations

import itertools

import numpy as np

def calc_object_keypoint_similarity(
    xy_pred,
    xy_true,
    sigma,
    margin=0,
    symmetric_kpts=None,
):
    visible_gt = ~np.isnan(xy_true).all(axis=1)
    if visible_gt.sum() < 2:  # At least 2 points needed to calculate scale
        return np.nan

    true = xy_true[visible_gt]
    scale_squared = np.prod(np.ptp(true, axis=0) + np.spacing(1) + margin * 2)
    if np.isclose(scale_squared, 0):
        return np.nan

    k_squared = (2 * sigma) ** 2
    denom = 2 * scale_squared * k_squared
    if isinstance(sigma, np.ndarray):
        denom = denom[visible_gt]

    if symmetric_kpts is None:
        pred = xy_pred[visible_gt]
        pred[np.isnan(pred)] = np.inf
        dist_squared = np.sum((pred - true) ** 2, axis=1)
        oks = np.exp(-dist_squared / denom)
        return np.mean(oks)
    else:
        oks = []
        xy_preds = [xy_pred]
        combos = (
            pair
            for l in range(len(symmetric_kpts))
            for pair in itertools.combinations(symmetric_kpts, l + 1)
        )
        for pairs in combos:
            # Swap corresponding keypoints
            tmp = xy_pred.copy()
            for pair in pairs:
                tmp[pair, :] = tmp[pair[::-1], :]
            xy_preds.append(tmp)
        for xy_pred in xy_preds:
            pred = xy_pred[visible_gt]
            pred[np.isnan(pred)] = np.inf
            dist_squared = np.sum((pred - true) ** 2, axis=1)
            oks.append(np.mean(np.exp(-dist_squared / denom)))
        return max(oks)