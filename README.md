# image_lddmm_tensorflow
## Introduction
Image mapping using the LDDMM algorithm, implemented in tensorflow.  Deformable registration between well characterized atlas images and observed target images allows for labelling of anatomical parcellations to interprete data, and quantification of atrophy, growth, or shape change.  Multi modality images and missing data are handled using DR IT MD, Deformable registration with intensity transform and missing data, described in https://doi.org/10.1101/494005.


This package contains functions to run deformable image registration in python, using tensorflow to handle high performance computing issues.

Below shows an illustration of deforming an atlas human MRI to match a target human MRI.  The second row shows the error before and after the alignment.  The decrease in error is particularly noticible in the lateral ventricle.

|Atlas|Target |
|---|---|
|<img src="human_mri_example_atlas.png" alt="Human MRI atlas" width="400"/>  |  <img src="human_mri_example_target.png" alt="Human MRI target" width="400"/>|
|Initial Error|Final Error|
|<img src="human_mri_example_error_start.png" alt="Human MRI atlas" width="400"/>  |  <img src="human_mri_example_error_end.png" alt="Human MRI target" width="400"/>|
|Deformed atlas |  <img src="human_mri_example_deformed_atlas.png" alt="Human MRI target" width="400"/>|






## Examples
Please see Example*.ipynb to see various examples.  Ideally you will find one that is similar to your desired application, and you can run it on your data with minimal changes.

## Installation

This package requires tensorflow running in python 3, as well as several other pythonpackages.  These packages are numpy for working with arrays, matplotlib for generating figures, nibabel for reading neuroimages, ipython and jupyter for running interactive notebooks.

Below is an example of how to install this package in unix.  You will need python 3.4, 3.5, or 3.6, and virtualenv.  Instructions for installing these can be found on https://www.tensorflow.org/install/.  You will also need git.


```
# set up virtual environment
ENVLOCATION=~/lddmm_env
./virtualenv $ENVLOCATION

# activate virtual environment
source $LOCATION/bin/activate

# install requirements
pip3 install ipython jupyter tensorflow numpy matplotlib nibabel

# clone git repo
INSTALLLOCATION=~
cd $INSTALLLOCATION
git clone https://github.com/dtward/image_lddmm_tensorflow.git

# start jupyter notebook
jupyter-notebook

# jupyter will start in your web browser
# navigate to one of the example files, and click on kernel->restart & run all

```

Other methods (including GPU) and troubleshooting for installing tensorflow can be found here https://www.tensorflow.org/install/

## To do
1. (coding) Optimize for GPU.
1. (application) Incorporate more example model organisms, and choose optimal parameters fo reach.


