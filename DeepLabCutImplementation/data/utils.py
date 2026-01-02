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

import warnings
from collections import defaultdict
from functools import reduce, lru_cache
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt 
from matplotlib.colors import Colormap
import matplotlib.patches as patches

def bbox_from_keypoints(
    keypoints: np.ndarray,
    image_h: int,
    image_w: int,
    margin: int,
) -> np.ndarray:
    """
    Computes bounding boxes from keypoints.

    Args:
        keypoints: (..., num_keypoints, xy) the keypoints from which to get bboxes
        image_h: the height of the image
        image_w: the width of the image
        margin: the bounding box margin

    Returns:
        the bounding boxes for the keypoints, of shape (..., 4) in the xywh format
    """
    squeeze = False

    # we do not estimate bbox on keypoints that have 0 or -1 flag
    keypoints = np.copy(keypoints)
    keypoints[keypoints[..., -1] <= 0] = np.nan

    if len(keypoints.shape) == 2:
        squeeze = True
        keypoints = np.expand_dims(keypoints, axis=0)

    bboxes = np.full((keypoints.shape[0], 4), np.nan)
    with warnings.catch_warnings():  # silence warnings when all pose confidence levels are <= 0
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bboxes[:, :2] = np.nanmin(keypoints[..., :2], axis=1) - margin  # X1, Y1
        bboxes[:, 2:4] = np.nanmax(keypoints[..., :2], axis=1) + margin  # X2, Y2

    # can have NaNs if some individuals have no visible keypoints
    bboxes = np.nan_to_num(bboxes, nan=0)

    bboxes = np.clip(
        bboxes,
        a_min=[0, 0, 0, 0],
        a_max=[image_w, image_h, image_w, image_h],
    )
    bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0]  # to width
    bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1]  # to height
    if squeeze:
        return bboxes[0]

    return bboxes

def map_image_path_to_id(images: list[dict]) -> dict[str, int]:
    """
    Binds the image paths to their respective IDs.

    Args:
        images: List of dictionaries containing image data in COCO-like format.
            Each dictionary should have 'file_name' and 'id' keys.

    Returns:
        A dictionary mapping image paths to their respective IDs.

    Examples:
        images = [{"file_name": "path/to/image1.jpg", "id": 1}, ...]
    """

    return {image["file_name"]: image["id"] for image in images}


def map_id_to_annotations(annotations: list[dict]) -> dict[int, list[int]]:
    """
    Maps image IDs to their corresponding annotation indices.

    Args:
        annotations: List of dictionaries containing annotation data. Each dictionary
            should have 'image_id' key.

    Returns:
        A dictionary mapping image IDs to lists of corresponding annotation indices.

    Examples:
        annotations = [{"image_id": 1, ...}, ...]
    """

    annotation_idx_map = defaultdict(list)
    for idx, annotation in enumerate(annotations):
        annotation_idx_map[annotation["image_id"]].append(idx)

    return annotation_idx_map


def _crop_and_pad_image(
    image: np.ndarray,
    coords: tuple[tuple[int, int], tuple[int, int]],
    output_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Crop the image using the given coordinates and pad the larger dimension to change
    the aspect ratio.

    Args:
        image: Image to crop, of shape (height, width, channels).
        coords: Coordinates for cropping as [(xmin, xmax), (ymin, ymax)].
        output_size: The (output_h, output_w) that this cropped image will be resized
            to. Used to compute padding to keep aspect ratios.

    Returns:
        Cropped (and possibly padded) image
        Padding (pad_h, pad_w)
    """
    cropped_image = image[coords[1][0] : coords[1][1], coords[0][0] : coords[0][1], :]

    crop_h, crop_w, c = cropped_image.shape
    pad_h, pad_w = 0, 0
    target_ratio_h = output_size[0] / crop_h
    target_ratio_w = output_size[1] / crop_w

    if target_ratio_h != target_ratio_w:
        if crop_h < crop_w:
            # Pad the height
            new_h = int(crop_w * output_size[0] / output_size[1])
            pad_h = new_h - crop_h
            pad_image = np.zeros((new_h, crop_w, c))
            y_offset = pad_h // 2
            pad_image[y_offset : y_offset + crop_h, :] = cropped_image
        else:
            # Pad the width
            new_w = int(crop_h * output_size[1] / output_size[0])
            pad_w = new_w - crop_w
            pad_image = np.zeros((crop_h, new_w, c))
            x_offset = pad_w // 2
            pad_image[:, x_offset : x_offset + crop_w] = cropped_image
    else:
        pad_image = cropped_image

    return pad_image, (pad_h, pad_w)

def calc_bbox_overlap(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """
    Calculate the overlap between two bounding boxes

    Args:
        bbox1: the first bounding box in the format (x, y, w, h)
        bbox2: the second bounding box in the format (x, y, w, h)

    Returns:
        The overlap between
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    x_overlap = max(0, min(x1_max, x2_max) - max(x1, x2))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1, y2))

    intersection = x_overlap * y_overlap
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / union

def apply_transform(
    transform: A.BaseCompose,
    image: np.ndarray,
    keypoints: np.ndarray,
    bboxes: np.ndarray,
    class_labels: list[str],
) -> dict[str, np.ndarray]:
    """
    Applies a transformation to the provided image and keypoints.

    Args:
        transform: The transformation to apply.
        image: The input image to which the transformation will be applied.
        keypoints: List of keypoints to be transformed along with the image. Each keypoint
            is expected to be a tuple or list with at least three values,
            where the third value indicates the class label index.
        bboxes: List of bounding boxes to be transformed along with the image.
        class_labels: List of class labels corresponding to the keypoints.

    Returns:
        transformed: A dictionary containing the transformed image and keypoints.
    """

    if transform:
        oob_mask = out_of_bounds_keypoints(keypoints, image.shape)
        transformed = _apply_transform(
            transform, image, keypoints, bboxes, class_labels
        )

        transformed["keypoints"] = np.array(transformed["keypoints"])

        # out-of-bound keypoints have visibility flag 0. But we don't touch coordinates
        if np.sum(oob_mask) > 0:
            transformed["keypoints"][oob_mask, 2] = 0.0

        out_shape = transformed["image"].shape
        if len(transformed["keypoints"]) > 0:
            oob_mask = out_of_bounds_keypoints(transformed["keypoints"], out_shape)
            # out-of-bound keypoints have visibility flag 0. Don't touch coordinates
            if np.sum(oob_mask) > 0:
                transformed["keypoints"][oob_mask, 2] = 0.0

        # TODO: Check that the transformed bboxes are still within the image
        if len(transformed["bboxes"]) > 0:
            transformed["bboxes"] = np.array(transformed["bboxes"])
        else:
            transformed["bboxes"] = np.zeros(shape=(0, 4))

    else:
        transformed = {"keypoints": keypoints, "image": image}

    # do we ever need to do this if we had check_keypoints_within_bounds above?
    # np.nan_to_num(transformed["keypoints"], copy=False, nan=-1)
    return transformed


def _apply_transform(
    transform: A.BaseCompose,
    image: np.ndarray,
    keypoints: np.ndarray,
    bboxes: np.ndarray,
    class_labels: list[str],
) -> dict[str, np.ndarray]:
    """
    Applies a transformation to the provided image and keypoints.

    Args:
        image : np.array or similar image data format
            The input image to which the transformation will be applied.

        keypoints : list or similar data format
            List of keypoints to be transformed along with the image. Each keypoint
            is expected to be a tuple or list with at least three values,
            where the third value indicates the class label index.

    Returns:
        dict
            A dictionary containing the transformed image and keypoints.
    """
    transformed = transform(
        image=image,
        keypoints=keypoints,
        class_labels=class_labels,
        bboxes=bboxes,
        bbox_labels=np.arange(len(bboxes)),
    )

    bboxes_out = np.zeros(bboxes.shape)
    for bbox, bbox_id in zip(transformed["bboxes"], transformed["bbox_labels"]):
        bboxes_out[bbox_id] = bbox

    transformed["bboxes"] = bboxes_out
    return transformed


def out_of_bounds_keypoints(keypoints: np.ndarray, shape: tuple) -> np.ndarray:
    """Computes which visible keypoints are outside an image

    Args:
        keypoints: A (N, 3) shaped array where N is the number of keypoints and each
            keypoint is represented as (x, y, visibility).
        shape: A tuple representing the shape or bounds as (height, width).

    Returns:
        A boolean array of shape (N,) where each element corresponds to whether
        the respective keypoint is visible (visibility > 0) and outside the image
        bounds. This mask can be used to set the visibility bit to 0 for keypoints that
        were kicked off an image due to augmentation.
    """
    return (keypoints[..., 2] > 0) & (
        np.isnan(keypoints[..., 0])
        | np.isnan(keypoints[..., 1])
        | (keypoints[..., 0] < 0)
        | (keypoints[..., 0] > shape[1])
        | (keypoints[..., 1] < 0)
        | (keypoints[..., 1] > shape[0])
    )


def pad_to_length(data: np.array, length: int, value: float) -> np.array:
    """
    Pads the first dimension of an array with a given value

    Args:
        data: the array to pad, of shape (l, ...), where l <= length
        length: the desired length of the tensor
        value: the value to pad with

    Returns:
        the padded array of shape (length, ...)
    """
    pad_length = length - len(data)
    if pad_length == 0:
        return data
    elif pad_length > 0:
        padding = value * np.ones((pad_length, *data.shape[1:]), dtype=data.dtype)
        return np.concatenate([data, padding])

    raise ValueError(f"Cannot pad! data.shape={data.shape} > length={length}")


def safe_stack(data: list[np.ndarray], default_shape: tuple[int, ...]) -> np.ndarray:
    """
    Stacks a list of arrays if there are any, otherwise returns an array of zeros
    of a desired shape.

    Args:
        data: the list of arrays to stack
        default_shape: the shape of the array to return if the list is empty

    Returns:
        the stacked data or empty array
    """
    if len(data) == 0:
        return np.zeros(default_shape, dtype=float)

    return np.stack(data, axis=0)


def prepare_figure_axes(width, height, scale=1.0, dpi=100):
    fig = plt.figure(
        frameon=False, figsize=(width * scale / dpi, height * scale / dpi), dpi=dpi
    )
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    return fig, ax

def make_multianimal_labeled_image(
    frame: np.ndarray,
    coords_truth: np.ndarray | list,
    coords_pred: np.ndarray | list,
    probs_pred: np.ndarray | list,
    colors: Colormap,
    dotsize: float | int = 12,
    alphavalue: float = 0.7,
    pcutoff: float = 0.6,
    labels: list = ["+", ".", "x"],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plots groundtruth labels and predictions onto the matplotlib's axes, with the specified graphical parameters.

    Args:
        frame: image
        coords_truth: groundtruth labels
        coords_pred: predictions
        probs_pred: prediction probabilities
        colors: colors for poses
        dotsize: size of dot
        alphavalue: transparency for the keypoints
        pcutoff: cut-off confidence value
        labels: labels to use for ground truth, reliable predictions, and not reliable predictions (confidence below cut-off value)
        ax: matplotlib plot's axes object

    Returns:
        matplotlib Axes object with plotted labels and predictions.
    """

    if ax is None:
        h, w, _ = np.shape(frame)
        _, ax = prepare_figure_axes(w, h)
    ax.imshow(frame, "gray")

    for n, data in enumerate(zip(coords_truth, coords_pred, probs_pred)):
        color = colors(n)
        coord_gt, coord_pred, prob_pred = data

        for i, coord in enumerate(coord_pred):
            if prob_pred[i] > pcutoff:
                plt.text(coord[0]-4, coord[1]+4, str(i+1), color="red", fontsize=5)

        reliable = np.repeat(prob_pred >= pcutoff, coord_pred.shape[1], axis=1)
        
        ax.plot(
            *coord_pred[reliable[:, 0]].T,
            labels[1],
            ms=dotsize,
            alpha=alphavalue,
            color=color,
        )
        
    return ax