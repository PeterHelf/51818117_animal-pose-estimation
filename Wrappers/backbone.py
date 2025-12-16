from DeepLabCutImplementation.backbones.hrnet_coam import HRNetCoAM
from DeepLabCutImplementation.modules.keypoint_encoders import StackedKeypointEncoder

def get_HRNetCoAM_backbone(
    num_joints: int,
    att_heads: int,
    channel_att_only: bool,
    coam_modules: tuple[int],
    selfatt_coam_modules: tuple[int] = None,
    pretrained: bool = True,
    base_model_name: str = "hrnet_w48",
) -> HRNetCoAM:
    """Helper function to create an HRNet_CoAM_Backbone with given parameters.

    Args:
        num_joints: number of joints to predict
        att_heads: number of attention heads in CoAM
        channel_att_only: whether to use only channel attention block in CoAM
        coam_modules: list of stages to apply CoAM
        selfatt_coam_modules: list of stages to apply Self-Attention-CoAM
        pretrained: whether to use ImageNet pretrained weights
        base_model_name: Type of HRNet (e.g., 'hrnet_w32', 'hrnet_w48').

    Returns:
        The built HRNet_CoAM_Backbone
    """
    model = HRNetCoAM(
        kpt_encoder=StackedKeypointEncoder(
            num_joints=num_joints,
            kernel_size=(15, 15),
            img_size=(256, 256),
        ),
        base_model_name=base_model_name,
        pretrained=pretrained,
        freeze_bn_stats=False,
        freeze_bn_weights=False,
        coam_modules=coam_modules,
        selfatt_coam_modules=selfatt_coam_modules,
        channel_att_only=channel_att_only,
        att_heads=att_heads,
        img_size=(256, 256),
    )

    return model