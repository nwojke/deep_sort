# Deep SORT

## Introduction

This repository contains code for *Simple Online and Realtime Tracking with a Deep Association Metric* (Deep SORT).
We extend the original [Deep SORT](https://github.com/nwojke/deep_sort) algorithm to use YOLOv4 for object detection/
See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information.

## Dependencies

```
Keras                     2.4.0
matplotlib                3.5.1 
numpy                     1.19.5
opencv-python             4.4.0.42 
tensorflow                2.6.0 

## Installation

First, clone the repository:
```
git clone -b deep_sort_yolov4 https://github.com/nwojke/deep_sort.git
```

```

## Testing

1. Download the pre-trained YOLOv4 weights from **[here](https://github.com/Mohit-robo/Deep_Learning/tree/main/Deep_sort/data/vehicle-detector)**
2. Deep Sort model from **[here](https://github.com/Mohit-robo/Deep_Learning/tree/main/Deep_sort/model_data)**
3. YOLOv4 and YOLOv4-tiny weights are been provided in the drive folder.

The paths to video, weights, the objects to be tracked and other constants are been declared in the ./core/config.py file.
The dependency functions are been declared in the ./core/dependencies.py and ./core/Object_dependencies.py do check them out for modifications.


## Running the tracker

We assume resources have been extracted to the repository root directory.
```
python deep_track.py 

```
Check `python deep_track.py -h` for an overview of available options.
There are also scripts in the repository to visualize results, generate videos,
and evaluate the MOT challenge benchmark.

## Training the model

To train the deep association metric model we used a novel [cosine metric learning](https://github.com/nwojke/cosine_metric_learning) approach which is provided as a separate repository.

## Highlevel overview of source files

In the top-level directory are executable scripts to execute, evaluate, and
visualize the tracker. The main entry point is in `deep_track.py`.
This file runs the tracker on a MOTChallenge sequence.

In package `deep_sort` is the main tracking code:

* `detection.py`: Detection base class.
* `kalman_filter.py`: A Kalman filter implementation and concrete
   parametrization for image space filtering.
* `linear_assignment.py`: This module contains code for min cost matching and
   the matching cascade.
* `iou_matching.py`: This module contains the IOU matching metric.
* `nn_matching.py`: A module for a nearest neighbor matching metric.
* `track.py`: The track class contains single-target track data such as Kalman
  state, number of hits, misses, hit streak, associated feature vectors, etc.
* `tracker.py`: This is the multi-target tracker class.

## Citing DeepSORT

If you find this repo useful in your research, please consider citing the following papers:

    @inproceedings{Wojke2017simple,
      title={Simple Online and Realtime Tracking with a Deep Association Metric},
      author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
      booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
      year={2017},
      pages={3645--3649},
      organization={IEEE},
      doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }
