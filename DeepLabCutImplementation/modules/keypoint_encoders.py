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

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def out_of_bounds_keypoints(keypoints: torch.Tensor, shape: tuple) -> np.ndarray:
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
        torch.isnan(keypoints[..., 0])
        | torch.isnan(keypoints[..., 1])
        | (keypoints[..., 0] < 0)
        | (keypoints[..., 0] > shape[1])
        | (keypoints[..., 1] < 0)
        | (keypoints[..., 1] > shape[0])
    )

class BaseKeypointEncoder(ABC):
    """Encodes keypoints into heatmaps

    Modified from BUCTD/data/JointsDataset
    """

    def __init__(
        self,
        num_joints: int,
        kernel_size: tuple[int, int] = (15, 15),
        img_size: tuple[int, int] = (256, 256),
    ) -> None:
        """
        Args:
            num_joints: The number of joints to encode
            kernel_size: The Gaussian kernel size to use when blurring a heatmap
            img_size: The (height, width) of the input images
        """
        self.kernel_size = kernel_size
        self.num_joints = num_joints
        self.img_size = img_size

    @property
    def num_channels(self):
        pass

    @abstractmethod
    def __call__(self, keypoints: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """
        Args:
            keypoints: the keypoints to encode
            size: the (height, width) of the heatmap in which the keypoints should
                be encoded

        Returns:
            the encoded keypoints
        """
        raise NotImplementedError

    def blur_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Applies a Gaussian blur to a heatmap

        Taken from BUCTD/data/JointsDataset, generate_heatmap

        Args:
            heatmap: the heatmap to blur (with values in [0, 1] or [0, 255])

        Returns:
            The heatmap with a Gaussian blur, such that max(heatmap) = 255
        """
        heatmap = TF.gaussian_blur(heatmap, self.kernel_size, sigma=None)
        #am = torch.max(heatmap)
        #if am == 0:
        #    return heatmap
        #heatmap /= am / 255
        return heatmap

    # def blur_heatmap_batch(self, heatmaps: torch.tensor) -> np.ndarray:
    #     heatmaps = TF.gaussian_blur(heatmaps.permute(0,3,1,2), self.kernel_size).permute(0,2,3,1).numpy()
    #     am = np.amax(heatmaps)
    #     if am == 0:
    #         return heatmaps
    #     heatmaps /= (am / 255)
    #     return heatmaps

class StackedKeypointEncoder(BaseKeypointEncoder):
    """Encodes keypoints into heatmaps, where each

    Modified from BUCTD/data/JointsDataset, get_stacked_condition
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def num_channels(self):
        return self.num_joints

    def __call__(self, keypoints: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """
        Args:
            keypoints: the keypoints to encode
            size: the (height, width) of the heatmap in which the keypoints should
                be encoded

        Returns:
            the encoded keypoints
        """

        batch_size, _, _ = keypoints.shape

        kpts = keypoints.clone()
        kpts[keypoints[..., 2] <= 0] = 0

        # Mark keypoints as visible, remove NaNs
        kpts[kpts[..., 2] > 0, 2] = 2
        kpts = torch.nan_to_num(kpts)

        #oob_mask = out_of_bounds_keypoints(kpts, self.img_size)
        #oob_mask_sum = torch.sum(oob_mask)
        #if torch.gt(oob_mask_sum, torch.tensor(0)):#oob_mask_sum > 0:
        #    kpts[oob_mask] = 0
        kpts = kpts.type(torch.int)

        zero_matrix = torch.zeros((batch_size, size[0], size[1], self.num_channels))

        def _get_condition_matrix(zero_matrix, kpts):
            j = torch.tensor(0)
            for i, pose in enumerate(kpts):
                x, y, vis = pose.T
                mask = vis > 0
                x_masked, y_masked, joint_inds_masked = (
                    x[mask],
                    y[mask],
                    torch.arange(self.num_joints, dtype=torch.int64)[mask],
                )
                print(j.dtype)
                print((y_masked - 1).dtype)
                print(( x_masked - 1).dtype)
                print(joint_inds_masked.dtype)
                index = [j, (y_masked - 1).type(torch.int64), (x_masked - 1).type(torch.int64), joint_inds_masked]
                torch.index_put_(zero_matrix, index, torch.tensor(255.0))
                #zero_matrix[torch.tensor(i), y_masked - 1, x_masked - 1, joint_inds_masked] = torch.tensor(255.0)
                j += torch.tensor(1)
            return zero_matrix

        condition = _get_condition_matrix(zero_matrix, kpts)

        for i in range(batch_size):
            condition_heatmap = self.blur_heatmap(condition[i])
            condition[i] = condition_heatmap

        return condition