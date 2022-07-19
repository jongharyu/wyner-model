# Learning Succinct Common Representation with Wyner's Common Information

---
This repository provides a codebase for the paper "Learning Succinct Common Representation with Wyner's Common Information," (2022).

---
## Experiments
###1. MNIST Add-1 / MNIST--SVHN experiment
#### Prepare datasets
- Download this [zip file]() (`mnist-add1-svhn.zip`) and put the files under `data/mnist-add1` and `data/mnist-svhn`.
- For evaluation, download this [zip file]() (`autoencoders.zip`) and put the files under `pretrained/autoencoders`.

#### Usage
To be updated.

###2. CUB Image--Caption experiment
#### Prepare datasets
We used the [Caltech--UCSD Birds (CUB) dataset](http://www.vision.caltech.edu/datasets/cub_200_2011/) based on this [repository (MMVAE)](https://github.com/iffsid/mmvae). 
- Follow the procedure as described in https://github.com/iffsid/mmvae#cub-image-caption.
- Please restructure the folder as follows:
```
data/cub
│───img
│   │───train
│   └───test
└───sent
    │───text_testclasses.txt
    └───text_testclasses.txt
```

#### Usage
To be updated.

###3. Zero-shot sketch retrieval experiment
#### Prepare datasets
- Download the [Sketchy Extended](https://sketchy.eye.gatech.edu/) dataset by following [this repository (SEM-PCYC)](https://github.com/AnjanDutta/sem-pcyc).
- Put the dataset under `data/Sketchy` which should look like:
```
data/Sketchy
│───extended_photo
│    │───airplane
│    │    └───...jpg
│    │───alarm
│    │    └───...jpg
│    │───...
│    └───zebra
│         └───...jpg
│───photo
│   └───tx_000000000000
│        │───airplane
│        │    └───...jpg
│        │───alarm
│        │    └───...jpg
│        │───...
│        └───zebra
│             └───...jpg
│───pretrained
│   │───vgg16_photo.pth
│   └───vgg16_sketch.pth
│───sketch
│   └───tx_000000000000
│        │───airplane
│        │    └───...png
│        │───alarm
│        │    └───...png
│        │───...
│        └───zebra
│             └───...png
│───README.txt
└───test_split_eccv2018.txt
```
#### Usage
To be updated.

---
## Credits 
We have written our code largely building upon the following existing code bases:
- An overall codebase design, MNIST--SVHN dataset, and CUB Image-Caption experiment: https://github.com/iffsid/mmvae
- Frechet distance: https://github.com/mseitzer/pytorch-fid
- Zero-shot sketch retrieval experiment
  - Dataset preparation: https://github.com/AnjanDutta/sem-pcyc
  - Evaluation: https://github.com/gr8joo/IIAE/