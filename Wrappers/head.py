from DeepLabCutImplementation.heads.simple_head import HeatmapHead
from DeepLabCutImplementation.predictors.single_predictor import HeatmapPredictor
from DeepLabCutImplementation.target_generators.heatmap_targets import HeatmapGaussianGenerator
from DeepLabCutImplementation.criterions.weighted import WeightedMSECriterion, WeightedHuberCriterion
from DeepLabCutImplementation.criterions.aggregators import WeightedLossAggregator

def get_heatmap_head(
    num_joints: int,
    channels: list[int],
    kernel_sizes: list[int],
    kernel_size_final: int,
    strides: list[int],
) -> HeatmapHead:
    """Helper function to create a HeatmapHead with given parameters.

    Args:
        num_joints: number of joints to predict
        channels: list of channels for each CoAM layer
        kernel_sizes: list of kernel sizes for each CoAM layer
        kernel_size_final: kernel size for the final convolutional layer
        strides: list of strides for each CoAM  layer

    Returns:
        The built HeatmapHead
    """
    head = HeatmapHead(
            predictor=HeatmapPredictor(
                apply_sigmoid=False,
                clip_scores=True,
                location_refinement=True,
                locref_std=7.2801,
            ),
            target_generator=HeatmapGaussianGenerator(
                num_heatmaps=num_joints,
                pos_dist_thresh=17,
                generate_locref=True,
                locref_std=7.2801,
            ),
            criterion={
                "heatmap": WeightedMSECriterion(),
                "locref": WeightedHuberCriterion(),
            },
            aggregator=WeightedLossAggregator(
                weights={
                    "heatmap": 1.0,
                    "locref": 0.05,
                }
            ),
            heatmap_config={
                "channels": channels,
                "kernel_size": kernel_sizes,
                "strides": strides,
                "final_conv":{
                    "out_channels": num_joints,
                    "kernel_size": kernel_size_final,
                }
            },
            locref_config={
                "channels": channels,
                "kernel_size": kernel_sizes,
                "strides": strides,
                "final_conv":{
                    "out_channels": num_joints*2,
                    "kernel_size": kernel_size_final,
                }
            }
        )


    return head
    