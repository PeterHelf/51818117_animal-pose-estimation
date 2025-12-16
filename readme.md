# Assignment 2 - Hacking

The aim of this project was the implementation of the [BUCTD](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Rethinking_Pose_Estimation_in_Crowds_Overcoming_the_Detection_Information_Bottleneck_ICCV_2023_paper.pdf) model and the testing of it on the [AP-10k](https://openreview.net/forum?id=rH8yliN6C83) animal pose dataset.

## Intallation
Install the necessary python packages with the requirements.txt file.

The checkpoint for the best performing model can be found in the release section on github. The file needs to be placed in the root directory of the project.

## Dataset
The dataset is publicly available under this link: https://drive.google.com/file/d/1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad/view?usp=sharing

The file needs to be palced in the dataset folder of this project and unzipped. The folder structure should look like this:

```text
51818117_animal-pose-estimation
├── DeepLabCutImplementation
├── Wrappers
├── result_images
├── training.ipynb
├── inference.ipynb
├── hyperparameter_tuning.ipynb
├── requirements.txt
├── snapshot-best.pt
|── dataset
    │── ap10k
        │-- annotations
        │   │-- ap10k-train-split1.json
        │   |-- ap10k-train-split2.json
        │   |-- ap10k-train-split3.json
        │   │-- ap10k-val-split1.json
        │   |-- ap10k-val-split2.json
        │   |-- ap10k-val-split3.json
        │   |-- ap10k-test-split1.json
        │   |-- ap10k-test-split2.json
        │   |-- ap10k-test-split3.json
        │-- data
        │   │-- 000000000001.jpg
        │   │-- 000000000002.jpg
        │   │-- ...
```
## Summary
The mAP metric was chosen for the task.

mAP Target: >80 (State of the art ViTPose++ achieves mAP of 82.4)

mAP achieved by BUCTD in this project: 76.6

Planned timebudget: 35 hours | Actual time required: 48 hours

## Project documentation
### Chosen metric
It is common for pose estimation tasks to use the mean Average Precision (mAP) as the reported metric. This score averages the precision-recall scores across the different detected keypoints. By using this score the overall ability of the model to detect and localize keypoints is reported.

A state of the art pose estimation model called [ViTPose++](https://arxiv.org/abs/2212.04246) has already been applied to the AP-10k dataset and achieves an mAP of 82.4. Thus, the aim of this project is to achieve a similar high score of >80, or in the best case surpass the score of ViTPose++.

### End-to-end implementation | Timebudget: 20 hours | Actual required time: 30 hours
Unlike popular state of the art models BUCTD does not have any easily usable implementations provided by PyTorch. Due to the complexity of the architecture and training process of BUCTD it was not feasible to implement it completely from scratch.

Because of this it was initially attempted to use the original implementation, which is available on [github](https://github.com/amathislab/BUCTD) as basis for the implementation in this project. However, the codebase is not compatible with new python and PyTorch versions. The code is also quite complex and difficult to follow. It was possible to get the inference running, but training kept returning errors. After about 18 hours of attempts it was decided to use a different implementation.

The BUCTD model is implemented in the [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) toolbox. This toolbox implements various pose estimation models and is mostly focused on pose estiamtion for animals. Because it is focused on multiple different models, the implementation is also quite complex. However, the codebase of DeepLabCut is much better documented and easier to follow. Because of this it was possible to extract the relevant code required for the BUCTD implementation. Modifications were made where necessary to run the model without the remainder of the toolbox. This, in combination the setup of a first training of the model, required an additional 12 hours.

### Model architecture modifications | Timebudget: 15 hours | Actual required time: 15 hours
After the training of an initial model which confirmed that the implementation was working, modifcations to the model architecture were attempted.

The architecture consists of two main parts, the backbone and the prediction head. The chosen backbone is based on [HRNet-W48](https://arxiv.org/pdf/1908.07919), but BUCTD adds Conditional Attention Module (CoAM) and Self-Attention-CoAM to the architecture. For the testing of architecture modifications the number and position of the CoAM modules was changed. The number of attention heads of the modules was also modified.

The prediction head is used to predict the heatmaps which are then further refined into actual keypoints. The channel count, number of layers, as well as the kernel size and stride of the convolutional layers in the the prediction head were modified.

In total three different backbone settings and three different prediction head settings were created. All possible combination of backbone and prediction head were then created to build a total of nine unique model architectures.

The nine model architectures were then trained for a limited number of 10 epochs to identify the best performing variation.

### Final training and testing of the model | Timebudget: Not specified in project proposal | Actual required time: 3 hours

The best performing model architecture identifed in the previous step was trainined for a longer period of 50 epochs to achieve better results. The model was then tested on the test set of the AP-10k dataset achieving a mAP of 76.6. This does not quiet meet the hoped mAP of >80, but is still an impressive performance when looking at the result images. Visualizations were created on images of the testset to illustrate the performance. The first image is the reference image provided by the AP-10k dataset to visualize the keypoints. The remaining images were labelled with the BUCTD model. The model does not work on images with more than one animal as it only detects each keypoint once.

!["Keypoint definition"](/result_images/keypointDef.jpg "Keypoint definition")

!["Keypoint detection result 1"](/result_images/result1.png "Result 1")

!["Keypoint detection result 2"](/result_images/result2.png "Result 2")

!["Keypoint detection result 3"](/result_images/result3.png "Result 3")

!["Keypoint detection result 4"](/result_images/result4.png "Result 4")

!["Keypoint detection result 5"](/result_images/result5.png "Result 5")

