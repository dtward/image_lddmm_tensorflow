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

[Human MRI registration using images from mricloud.org](https://github.com/dtward/image_lddmm_tensorflow/blob/master/Example_Human_MRI.ipynb)

[Human MRI registration with simulated resected tissue](https://github.com/dtward/image_lddmm_tensorflow/blob/master/Example_Human_MRI_Resection.ipynb)

[Mouse serial two photon registration using the Allen atlas and a target with a brightness artifact at an injection site](https://github.com/dtward/image_lddmm_tensorflow/blob/master/Example_Mouse_Allen_to_Fluoro.ipynb)

[Mouse Nissl stain from Cold Spring harbor Laboratory](https://github.com/dtward/image_lddmm_tensorflow/blob/master/Example_Mouse_Nissl.ipynb)

[Rat Waxholm MRI atlas to iDISCO target](https://github.com/dtward/image_lddmm_tensorflow/blob/master/Example_iDISCO_rat_waxholm.ipynb)

## Installation

The human MRI with resection example can be run without any installation using [google colab here](https://colab.research.google.com/drive/1vFkEqwJJLnoRp0nTMUwHXI0MkN8FkTjE): 

This package requires tensorflow version 1 (this code was developed with 1.13.1) running in python 3, as well as several other pythonpackages.  These packages are numpy for working with arrays, matplotlib for generating figures, nibabel for reading neuroimages, ipython and jupyter for running interactive notebooks.

Below is an example of how to install this package in unix.  You will need python 3.4, 3.5, or 3.6, and virtualenv.  Instructions for installing these can be found on https://www.tensorflow.org/install/.  You will also need git.


```
# set up virtual environment
ENVLOCATION=~/lddmm_env
./virtualenv $ENVLOCATION

# activate virtual environment
source $ENVLOCATION/bin/activate

# install requirements
pip3 install ipython jupyter tensorflow==1.13.1 numpy matplotlib nibabel

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


