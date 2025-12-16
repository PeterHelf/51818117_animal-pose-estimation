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


import os
import pickle
import shutil
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics.cluster import contingency_matrix
from tqdm import tqdm

from DeepLabCutImplementation.core.inferenceutils import (
    _parse_ground_truth_data,
    Assembler,
    evaluate_assembly,
)


def _set_up_evaluation(data):
    params = dict()
    params["joint_names"] = data["metadata"]["all_joints_names"]
    params["num_joints"] = len(params["joint_names"])
    partaffinityfield_graph = data["metadata"]["PAFgraph"]
    params["paf"] = np.arange(len(partaffinityfield_graph))
    params["paf_graph"] = params["paf_links"] = [
        partaffinityfield_graph[l] for l in params["paf"]
    ]
    params["bpts"] = params["ibpts"] = range(params["num_joints"])
    params["imnames"] = [fn for fn in list(data) if fn != "metadata"]
    return params


def _form_original_path(path):
    root, filename = os.path.split(path)
    base, ext = os.path.splitext(filename)
    return os.path.join(root, filename.split("c")[0] + ext)


def _unsorted_unique(array):
    _, inds = np.unique(array, return_index=True)
    return np.asarray(array)[np.sort(inds)]


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


def _calc_separability(
    vals_left, vals_right, n_bins=101, metric="jeffries", max_sensitivity=False
):
    if metric not in ("jeffries", "auc"):
        raise ValueError("`metric` should be either 'jeffries' or 'auc'.")

    bins = np.linspace(0, 1, n_bins)
    hist_left = np.histogram(vals_left, bins=bins)[0]
    hist_left = hist_left / hist_left.sum()
    hist_right = np.histogram(vals_right, bins=bins)[0]
    hist_right = hist_right / hist_right.sum()
    tpr = np.cumsum(hist_right)
    if metric == "jeffries":
        sep = np.sqrt(
            2 * (1 - np.sum(np.sqrt(hist_left * hist_right)))
        )  # Jeffries-Matusita distance
    else:
        sep = np.trapz(np.cumsum(hist_left), tpr)
    if max_sensitivity:
        threshold = bins[max(1, np.argmax(tpr > 0))]
    else:
        threshold = bins[np.argmin(1 - np.cumsum(hist_left) + tpr)]
    return sep, threshold


def _calc_within_between_pafs(
    data,
    metadata,
    per_edge=True,
    train_set_only=True,
):
    data = deepcopy(data)
    train_inds = set(metadata["data"]["trainIndices"])
    graph = data["metadata"]["PAFgraph"]
    within_train = defaultdict(list)
    within_test = defaultdict(list)
    between_train = defaultdict(list)
    between_test = defaultdict(list)
    for i, (key, dict_) in enumerate(data.items()):
        if key == "metadata":
            continue

        is_train = i in train_inds
        if train_set_only and not is_train:
            continue

        df = dict_["groundtruth"][2]
        try:
            df.drop("single", level="individuals", inplace=True)
        except KeyError:
            pass
        bpts = df.index.get_level_values("bodyparts").unique().to_list()
        coords_gt = (
            df.unstack(["individuals", "coords"])
            .reindex(bpts, level="bodyparts")
            .to_numpy()
            .reshape((len(bpts), -1, 2))
        )
        if np.isnan(coords_gt).all():
            continue

        coords = dict_["prediction"]["coordinates"][0]
        # Get animal IDs and corresponding indices in the arrays of detections
        lookup = dict()
        for i, (coord, coord_gt) in enumerate(zip(coords, coords_gt)):
            inds = np.flatnonzero(np.all(~np.isnan(coord), axis=1))
            inds_gt = np.flatnonzero(np.all(~np.isnan(coord_gt), axis=1))
            if inds.size and inds_gt.size:
                neighbors = find_closest_neighbors(coord_gt[inds_gt], coord[inds], k=3)
                found = neighbors != -1
                lookup[i] = dict(zip(inds_gt[found], inds[neighbors[found]]))

        costs = dict_["prediction"]["costs"]
        for k, v in costs.items():
            paf = v["m1"]
            mask_within = np.zeros(paf.shape, dtype=bool)
            s, t = graph[k]
            if s not in lookup or t not in lookup:
                continue
            lu_s = lookup[s]
            lu_t = lookup[t]
            common_id = set(lu_s).intersection(lu_t)
            for id_ in common_id:
                mask_within[lu_s[id_], lu_t[id_]] = True
            within_vals = paf[mask_within]
            between_vals = paf[~mask_within]
            if is_train:
                within_train[k].extend(within_vals)
                between_train[k].extend(between_vals)
            else:
                within_test[k].extend(within_vals)
                between_test[k].extend(between_vals)
    if not per_edge:
        within_train = np.concatenate([*within_train.values()])
        within_test = np.concatenate([*within_test.values()])
        between_train = np.concatenate([*between_train.values()])
        between_test = np.concatenate([*between_test.values()])
    return (within_train, within_test), (between_train, between_test)




def _get_n_best_paf_graphs(
    data,
    metadata,
    full_graph,
    n_graphs=10,
    root=None,
    which="best",
    ignore_inds=None,
    metric="auc",
):
    if which not in ("best", "worst"):
        raise ValueError('`which` must be either "best" or "worst"')

    (within_train, _), (between_train, _) = _calc_within_between_pafs(
        data,
        metadata,
        train_set_only=True,
    )
    # Handle unlabeled bodyparts...
    existing_edges = set(k for k, v in within_train.items() if v)
    if ignore_inds is not None:
        existing_edges = existing_edges.difference(ignore_inds)
    existing_edges = list(existing_edges)

    if not any(between_train.values()):
        # Only 1 animal, let us return the full graph indices only
        return ([existing_edges], dict(zip(existing_edges, [0] * len(existing_edges))))

    scores, _ = zip(
        *[
            _calc_separability(between_train[n], within_train[n], metric=metric)
            for n in existing_edges
        ]
    )

    # Find minimal skeleton
    G = nx.Graph()
    for edge, score in zip(existing_edges, scores):
        if np.isfinite(score):
            G.add_edge(*full_graph[edge], weight=score)
    if which == "best":
        order = np.asarray(existing_edges)[np.argsort(scores)[::-1]]
        if root is None:
            root = []
            for edge in nx.maximum_spanning_edges(G, data=False):
                root.append(full_graph.index(sorted(edge)))
    else:
        order = np.asarray(existing_edges)[np.argsort(scores)]
        if root is None:
            root = []
            for edge in nx.minimum_spanning_edges(G, data=False):
                root.append(full_graph.index(sorted(edge)))

    n_edges = len(existing_edges) - len(root)
    lengths = np.linspace(0, n_edges, min(n_graphs, n_edges + 1), dtype=int)[1:]
    order = order[np.isin(order, root, invert=True)]
    paf_inds = [root]
    for length in lengths:
        paf_inds.append(root + list(order[:length]))
    return paf_inds, dict(zip(existing_edges, scores))


# Backwards compatibility
_find_closest_neighbors = find_closest_neighbors
