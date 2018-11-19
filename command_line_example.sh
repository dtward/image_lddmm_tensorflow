#! /bin/bash

# output prefix
PREFIX=tmp/

# filenames
ATLAS_IMAGE_FNAME=average_template_50.img
TARGET_IMAGE_FNAME=180517_Downsample.img

# scale of deformation
SCALE=0.25 # about 5 pixels



# noise terms (weights in cost function)
SIGMAM=646.8 # this is the standard deviation of the example image
SIGMAR=1e0

# em algorithm
SIGMAA=6468.0
NMSTEP=5
NMSTEPAFFINE=1

# optimization
NITER=200 # iterations of gradient descent
NAFFINE=50 # steps of affine only (no deformation)
AFFINE=example_affine.txt # initialization for affine
NT=5 # timesteps in flow
EV=1e-2 # step size for deformation
ET=1e-3 # stepsize for translation
EL=2e-4 # step size for linear
POSTAFFINEREDUCE=0.1 # reduce step size for affine parameters after naffine

# run the code
# note -u means outputs will be written in real time and not buffered
python -u -m lddmm $PREFIX $ATLAS_IMAGE_FNAME $TARGET_IMAGE_FNAME $SCALE $SIGMAM $SIGMAR $NITER $EV --eL $EL --eT $ET --naffine $NAFFINE --post_affine_reduce $POSTAFFINEREDUCE --affine $AFFINE --nT $NT --sigmaA $SIGMAA --nMstep $NMSTEP --nMstep_affine $NMSTEPAFFINE --post_affine_reduce $POSTAFFINEREDUCE --pad_allen > ${PREFIX}stdout.txt
# note last argument pad allen will add an extra blank slice to the edge of allen atlas, for better boundary conditions when applying interpolation
