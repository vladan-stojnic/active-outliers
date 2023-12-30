# Training Ensembles with Inliers and Outliers for Semi-supervised Active Learning

This repository contains the code, data, and models from the paper Vladan Stojnić, Zakaria Laskar, Giorgos Tolias, ["Training Ensembles with Inliers and Outliers for Semi-supervised Active Learning"](https://openaccess.thecvf.com/content/WACV2024/papers/Stojnic_Training_Ensembles_With_Inliers_and_Outliers_for_Semi-Supervised_Active_Learning_WACV_2024_paper.pdf), In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024.

## Setup

This code was implemented using `Python 3.10.`, to install the required dependancies please use supplied `requirements.txt` file.

## Data

Data and splits for labeled and unlabeled datasets is available [here](http://ptak.felk.cvut.cz/personal/stojnvla/public/active_outliers/data/). Please download them and put them in `data` directory of this repository.

Splits for initial labeled and unlabeled sets are given in splits directory, pickle files given there contain dictionary with a structure:
```
- seed {0, 1, 2, 3, 4}
    - train
        - labeled
            - k_shot {1, 5, 20} # list of images in the initial labeled set, for different number of images per class
        - unlabeled
            - outlier_ratio {0, 5, 20, 50, 80, 90} # list of images in the unlabeled set for specified outlier ratio
    - val # list of images for the test set, this list is same for all seeds
```

## Models

Pre-trained self-supervised models used as initialization for all active learning methods are available [here](http://ptak.felk.cvut.cz/personal/stojnvla/public/active_outliers/models/ssl/). Please download them and put them in `models/ssl` directory of this repository.

## Running the code

To run our methods please use commands as follows:
- For the variant of our method with semi-supervision
```
python src/main.py dataset=imagenet seed=0 k_shot=20 al_rounds=11 use_outlier_class=true ensemble_len=5 budget=500 strategy=vr outlier_ratio=50 use_filtering=true use_semi=true epochs_semi=3
```
to run for other seed specify `seed` to a value from `{0,1,2,3,4}` and to run on a different outlier ratio specify `outlier_ratio` to a value from `{0,5,20,50,80,90}`.
- For the variant of our method without semi-supervision
```
python src/main.py dataset=imagenet seed=0 k_shot=20 al_rounds=11 use_outlier_class=true ensemble_len=5 budget=500 strategy=vr outlier_ratio=50 use_filtering=true use_semi=false
```
you can use the same values for `seed` and `outlier_ratio` as in the variant with semi-supervision.

To run on other datasets change `dataset` to `cifar100`, `tiny_imagenet` or `imagenet_carnivore`. For `cifar100` and `tiny_imagenet` only `outlier_ratio` of 80 is available, while for `imagenet_carnivore` only `outlier_ratio` of 50 is available. Data splits are availble for different outlier ratios as well, but pre-trained self-supervised models are not.

**For imagenet and imagenet_carnivore in `conf/dataset/imagenet.yaml` and `conf/dataset/imagenet_carnivore.yaml` set `full_imagenet_path` to the path of downloaded imagenet dataset.**

**Other active learning methods from the paper coming soon.**

## Citation
```
@InProceedings{Stojnic_2024_WACV,
    author    = {Stojni\'c, Vladan and Laskar, Zakaria and Tolias, Giorgos},
    title     = {Training Ensembles With Inliers and Outliers for Semi-Supervised Active Learning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {260-269}
}
```
