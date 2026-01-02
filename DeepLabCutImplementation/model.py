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

import torch
import torch.nn as nn

from DeepLabCutImplementation.backbones.base import BaseBackbone

from DeepLabCutImplementation.heads.base import BaseHead


class PoseModel(nn.Module):
    """A pose estimation model

    A pose estimation model is composed of a backbone, optionally a neck, and an
    arbitrary number of heads. Outputs are computed as follows:
    """

    def __init__(
        self,
        #cfg: dict,
        backbone: BaseBackbone,
        heads: dict[str, BaseHead],
        neck = None,
    ) -> None:
        """
        Args:
            cfg: configuration dictionary for the model.
            backbone: backbone network architecture.
            heads: the heads for the model
            neck: neck network architecture (default is None). Defaults to None.
        """
        super().__init__()
        #self.cfg = cfg
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self.neck = neck
        self.output_features = False

        self._strides = {
            name: _model_stride(self.backbone.stride, head.stride)
            for name, head in heads.items()
        }

    def forward(self, x: torch.Tensor, **backbone_kwargs) -> dict[str, dict[str, torch.Tensor]]:
        """
        Forward pass of the PoseModel.

        Args:
            x: input images

        Returns:
            Outputs of head groups
        """
        if x.dim() == 3:
            x = x[None, :]
        features = self.backbone(x, **backbone_kwargs)
        if self.neck:
            features = self.neck(features)

        outputs = {}
        if self.output_features:
            outputs["backbone"] = dict(features=features)

        for head_name, head in self.heads.items():
            outputs[head_name] = head(features)
        return outputs

    def get_loss(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        targets: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        total_losses = []
        losses: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            head_losses = head.get_loss(outputs[name], targets[name])
            total_losses.append(head_losses["total_loss"])
            for k, v in head_losses.items():
                losses[f"{name}_{k}"] = v

        # TODO: Different aggregation for multi-head loss?
        losses["total_loss"] = torch.mean(torch.stack(total_losses))
        return losses

    def get_target(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        labels: dict,
    ) -> dict[str, dict]:
        """Summary:
        Get targets for model training.

        Args:
            outputs: output of each head group
            labels: dictionary of labels

        Returns:
            targets: dict of the targets for each model head group
        """
        return {
            name: head.target_generator(self._strides[name], outputs[name], labels)
            for name, head in self.heads.items()
        }

    def get_predictions(self, outputs: dict[str, dict[str, torch.Tensor]]) -> dict:
        """Abstract method for the forward pass of the Predictor.

        Args:
            outputs: outputs of the model heads

        Returns:
            A dictionary containing the predictions of each head group
        """
        predictions = {
            name: head.predictor(self._strides[name], outputs[name])
            for name, head in self.heads.items()
        }
        if self.output_features:
            predictions["backbone"] = outputs["backbone"]

        return predictions

    def get_stride(self, head: str) -> int:
        """
        Args:
            head: The head for which to get the total stride.

        Returns:
            The total stride for the outputs of the head.

        Raises:
            ValueError: If there is no such head.
        """
        return self._strides[head]

def _model_stride(backbone_stride: int | float, head_stride: int | float) -> float:
    """Computes the model stride from a backbone and a head"""
    if head_stride > 0:
        return backbone_stride / head_stride

    return backbone_stride * -head_stride
