# image_lddmm_tensorflow
## Introduction
Image mapping using the LDDMM algorithm, implemented in tensorflow

This package contains functions to run deformable image registration in python, using tensorflow to handle high performance computing issues.


| | |
|---|---|
|<img src="human_mri_example_atlas.png" alt="Human MRI atlas" width="300"/>  |  <img src="human_mri_example_target.png" alt="Human MRI target" width="300"/>|
|<img src="human_mri_example_error_start.png" alt="Human MRI atlas" width="300"/>  |  <img src="human_mri_example_error_end.png" alt="Human MRI target" width="300"/>|






## Examples
Please see Examples.ipynb to see application to human brain and mouse brain.  More examples will be forthcoming.

## To do
1. (coding) Figure out details of running this code on GPU
1. (application) Incorporate more example model organisms, and choose optimal parameters fo reach
1. (algorithms) Incorporate moddern techniques for working with artifacts, missing data, and differences in image contrast

