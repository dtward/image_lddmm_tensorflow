import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm,logm
import tensorflow as tf
import vis
import os
import argparse
import nibabel as nib
import warnings
import scipy.interpolate as spi


'''
In this file we include several basic functions that were not present natively in tensorflow,
and we implement the algorithm for image registration with intensity transformation and missing data.

The basic functions implemented are:
interp3: trilinear interpolation for deforming images and vector fields
grad3: computes the 3D gradient of a function
down: downsamples images by averaging over a rectangular neighborhood
down2: a faster downsampling version for downsampling by 2
upsample: upsample data by zero padding in the Fourier domain.  This is necessary for upsampling
    lddmm velocity fields without changing the regularization energy.
transform_data: applies the calcuated deformation fields to image data
affine_transform_data: applies an affine transformation to data with a simpler interface
orientation_to_matrix: computes transformation matrices that relate images of different orientations.
    For example 'LAS' denotes Left-Anterior-Superior, which means the first image axis
    contains data from right to left, the second from posterior to anterior, 
    and the third from inferior to superior.

Next, the lddmm algorithm is implemented in two parts in the function 'lddmm'.  
In the first part a tensorflow computatoin  graph is defined that applies 
an existing deformation to images and calculates a cost function gradient.
In the second part, these calculations are performed on data and optimization is carried out.

Finally, a basic command line interface is implemented.

'''

 
dtype = tf.float32
idtype = tf.int64
def interp3(x0,x1,x2,I,phi0,phi1,phi2,method=1,image_dtype=dtype):
    ''' 
    Linear interpolation
    Interpolate a 3D tensorflow image I
    with voxels corresponding to locations in x0, x1, x2 (1d np arrays)
    at the points phi0, phi1, phi2 (3d arrays)
        
    Note optional method:
        0 for nearest neighbor, 
        1 for trilinear (default)
    Note optional dtype:
        you may want to set it to idtype when doing nearest interpolation for label images
      
      
    Output is the image I transformed by interpolation.
    '''
    if method != 0 and method != 1:
        raise ValueError('method must be 0 (nearest neighbor) or 1 (trilinear)')
        
    I = tf.convert_to_tensor(I, dtype=image_dtype)
    phi0 = tf.convert_to_tensor(phi0, dtype=dtype)
    phi1 = tf.convert_to_tensor(phi1, dtype=dtype)
    phi2 = tf.convert_to_tensor(phi2, dtype=dtype)
    
    # get the size
    dx = [x0[1]-x0[0], x1[1]-x1[0], x2[1]-x2[0]]
    nx = [len(x0), len(x1), len(x2)]
    shape = tf.shape(phi0)
    nxout = [shape[0],shape[1],shape[2]]    
    
    #convert to index
    phi0_index = (phi0 - x0[0])/dx[0]
    phi1_index = (phi1 - x1[0])/dx[1]
    phi2_index = (phi2 - x2[0])/dx[2]
    if method == 0: # simple hack for nearest neighbor, weights should all be binary now
        phi0_index = tf.round(phi0_index)
        phi1_index = tf.round(phi1_index)
        phi2_index = tf.round(phi2_index)
        
    # take the floor to get integers
    phi0_index_floor = tf.floor(phi0_index)
    phi1_index_floor = tf.floor(phi1_index)
    phi2_index_floor = tf.floor(phi2_index)        
    
    # get the fraction to the next pixel
    phi0_p = phi0_index - phi0_index_floor
    phi1_p = phi1_index - phi1_index_floor
    phi2_p = phi2_index - phi2_index_floor
    
    # now convert to int and work with ints, otherwise I ended up with loss of precision
    phi0_index_floor = tf.cast(phi0_index_floor,dtype=idtype)
    phi1_index_floor = tf.cast(phi1_index_floor,dtype=idtype)
    phi2_index_floor = tf.cast(phi2_index_floor,dtype=idtype)
    
    # get the next samples
    phi0_index_floor_1 = phi0_index_floor+1
    phi1_index_floor_1 = phi1_index_floor+1
    phi2_index_floor_1 = phi2_index_floor+1
    
    # and apply boundary conditions
    phi0_index_floor   = tf.minimum(phi0_index_floor,nx[0]-1)
    phi0_index_floor   = tf.maximum(phi0_index_floor,0)
    phi0_index_floor_1 = tf.minimum(phi0_index_floor_1,nx[0]-1)
    phi0_index_floor_1 = tf.maximum(phi0_index_floor_1,0)
    phi1_index_floor   = tf.minimum(phi1_index_floor,nx[1]-1)
    phi1_index_floor   = tf.maximum(phi1_index_floor,0)
    phi1_index_floor_1 = tf.minimum(phi1_index_floor_1,nx[1]-1)
    phi1_index_floor_1 = tf.maximum(phi1_index_floor_1,0)
    phi2_index_floor   = tf.minimum(phi2_index_floor,nx[2]-1)
    phi2_index_floor   = tf.maximum(phi2_index_floor,0)
    phi2_index_floor_1 = tf.minimum(phi2_index_floor_1,nx[2]-1)
    phi2_index_floor_1 = tf.maximum(phi2_index_floor_1,0)
    # if I wanted to apply zero boundary conditions, I'd have to check here where they are
    # then set to zero below
    # at this point it should be impossible for any of my indices to point outside the volume
    
    # then we will need to vectorize everything to use scalar indices
    phi0_index_floor_flat = tf.reshape(phi0_index_floor,[-1])
    phi0_index_floor_flat_1 = tf.reshape(phi0_index_floor_1,[-1])
    phi1_index_floor_flat = tf.reshape(phi1_index_floor,[-1])
    phi1_index_floor_flat_1 = tf.reshape(phi1_index_floor_1,[-1])
    phi2_index_floor_flat = tf.reshape(phi2_index_floor,[-1])
    phi2_index_floor_flat_1 = tf.reshape(phi2_index_floor_1,[-1])
    I_flat = tf.reshape(I,[-1])


    # indices recall that the LAST INDEX IS CONTIGUOUS
    phi_index_floor_flat_000 = nx[2]*nx[1]*phi0_index_floor_flat   + nx[2]*phi1_index_floor_flat   + phi2_index_floor_flat
    phi_index_floor_flat_001 = nx[2]*nx[1]*phi0_index_floor_flat   + nx[2]*phi1_index_floor_flat   + phi2_index_floor_flat_1
    phi_index_floor_flat_010 = nx[2]*nx[1]*phi0_index_floor_flat   + nx[2]*phi1_index_floor_flat_1 + phi2_index_floor_flat
    phi_index_floor_flat_011 = nx[2]*nx[1]*phi0_index_floor_flat   + nx[2]*phi1_index_floor_flat_1 + phi2_index_floor_flat_1
    phi_index_floor_flat_100 = nx[2]*nx[1]*phi0_index_floor_flat_1 + nx[2]*phi1_index_floor_flat   + phi2_index_floor_flat
    phi_index_floor_flat_101 = nx[2]*nx[1]*phi0_index_floor_flat_1 + nx[2]*phi1_index_floor_flat   + phi2_index_floor_flat_1
    phi_index_floor_flat_110 = nx[2]*nx[1]*phi0_index_floor_flat_1 + nx[2]*phi1_index_floor_flat_1 + phi2_index_floor_flat
    phi_index_floor_flat_111 = nx[2]*nx[1]*phi0_index_floor_flat_1 + nx[2]*phi1_index_floor_flat_1 + phi2_index_floor_flat_1
    
    # now slice the image
    I000_flat = tf.gather(I_flat, tf.cast(phi_index_floor_flat_000, dtype=idtype)) 
    I001_flat = tf.gather(I_flat, tf.cast(phi_index_floor_flat_001, dtype=idtype))
    I010_flat = tf.gather(I_flat, tf.cast(phi_index_floor_flat_010, dtype=idtype))
    I011_flat = tf.gather(I_flat, tf.cast(phi_index_floor_flat_011, dtype=idtype))
    I100_flat = tf.gather(I_flat, tf.cast(phi_index_floor_flat_100, dtype=idtype))
    I101_flat = tf.gather(I_flat, tf.cast(phi_index_floor_flat_101, dtype=idtype))
    I110_flat = tf.gather(I_flat, tf.cast(phi_index_floor_flat_110, dtype=idtype)) 
    I111_flat = tf.gather(I_flat, tf.cast(phi_index_floor_flat_111, dtype=idtype))
    
    # reshape it
    I000 = tf.reshape(I000_flat, nxout)
    I001 = tf.reshape(I001_flat, nxout)
    I010 = tf.reshape(I010_flat, nxout)
    I011 = tf.reshape(I011_flat, nxout)
    I100 = tf.reshape(I100_flat, nxout)
    I101 = tf.reshape(I101_flat, nxout)
    I110 = tf.reshape(I110_flat, nxout)
    I111 = tf.reshape(I111_flat, nxout)

    # combine them
    p000 = tf.cast((1.0-phi0_p)*(1.0-phi1_p)*(1.0-phi2_p), dtype=image_dtype)
    p001 = tf.cast((1.0-phi0_p)*(1.0-phi1_p)*(    phi2_p), dtype=image_dtype)
    p010 = tf.cast((1.0-phi0_p)*(    phi1_p)*(1.0-phi2_p), dtype=image_dtype)
    p011 = tf.cast((1.0-phi0_p)*(    phi1_p)*(    phi2_p), dtype=image_dtype)
    p100 = tf.cast((    phi0_p)*(1.0-phi1_p)*(1.0-phi2_p), dtype=image_dtype)
    p101 = tf.cast((    phi0_p)*(1.0-phi1_p)*(    phi2_p), dtype=image_dtype)
    p110 = tf.cast((    phi0_p)*(    phi1_p)*(1.0-phi2_p), dtype=image_dtype)
    p111 = tf.cast((    phi0_p)*(    phi1_p)*(    phi2_p), dtype=image_dtype)
    Il = I000*p000\
        + I001*p001\
        + I010*p010\
        + I011*p011\
        + I100*p100\
        + I101*p101\
        + I110*p110\
        + I111*p111
    
    return Il


def grad3(I,dx):
    '''
    Calculate the gradent of a 3D image
    Inputs are I, a 3D image
    and dx, a 3-tuple of voxeldimensions
    
    Outputs are each component of the gradient returned as a tuple
    '''
    
    I_0_m = (I[1,:,:] - I[0,:,:])/dx[0]
    I_0_p = (I[-1,:,:] - I[-2,:,:])/dx[0]
    I_0_0 = (I[2:,:,:]-I[:-2,:,:])/2.0/dx[0]
    I_0 = tf.concat([I_0_m[None,:,:], I_0_0, I_0_p[None,:,:]], axis=0)
    I_1_m = (I[:,1,:] - I[:,0,:])/dx[1]
    I_1_p = (I[:,-1,:] - I[:,-2,:])/dx[1]
    I_1_0 = (I[:,2:,:]-I[:,:-2,:])/2.0/dx[1]
    I_1 = tf.concat([I_1_m[:,None,:], I_1_0, I_1_p[:,None,:]], axis=1)
    I_2_m = (I[:,:,1] - I[:,:,0])/dx[2]
    I_2_p = (I[:,:,-1] - I[:,:,-2])/dx[2]
    I_2_0 = (I[:,:,2:]-I[:,:,:-2])/2.0/dx[2]
    I_2 = tf.concat([I_2_m[:,:,None], I_2_0, I_2_p[:,:,None]], axis=2)
    return I_0, I_1, I_2

def down(I,ndown):
    '''Downsample images by averaging over a rectangular neighborhood
    
    Inputs are a 3D image I
    a downsampling factor on each axis ndown = [ndown0,ndown1,ndown2]
    
    Output is the downsampled image.
    '''
    ndown = np.array(ndown)
    n0 = np.array(I.shape)
    n1 = np.array(n0)//ndown
    J = np.zeros(n1)
    factor = 1.0 / np.prod(ndown)
    for i in range(ndown[0]):
        for j in range(ndown[1]):
            for k in range(ndown[2]):
                J += I[i:n1[0]*ndown[0]:ndown[0],j:n1[1]*ndown[1]:ndown[1],k:n1[2]*ndown[2]:ndown[2]] * factor
    return J     
        
def down2(I):
    '''Downsample by a factor of 2 by averaging over a 2x2x2 neighborhood
    Input is an image I
    Output is the downsampled image
    '''
    n0 = np.array(I.shape)
    n1 = n0//2
    J = np.zeros(n1)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                J += 0.125*I[i:n1[0]*2:2,j:n1[1]*2:2,k:n1[2]*2:2]
    return J


def upsample(I,nup):
    '''Upsample by zero padding in the Fourier domain.
    Inputs are I, an image to be upsampled
    nup, the desired size of the upsampled image.
    
    Output is the upsapmled image.
    '''
    n = np.array(I.shape)
    
    # now I want to upsample by zero padding in the Fourier domain
    even = (1 - (n % 2)).astype(int)
    shift = even * n//2 + (1-even) * (n-1)//2
    
    J = np.array(I)
    
    # upsample the 0th axis
    if nup[0] > n[0]:
        Jhat = np.fft.fft(J, axis=0)
        Jhat = np.roll(Jhat,shift[0],axis=0)
        if even[0]:
            # if even, make nyquist paired
            Jhat[0,:,:] /= 2.0
            Jhat = np.pad(Jhat,pad_width=((0,1),(0,0),(0,0)),mode='edge')
            n[0] = n[0] + 1
        # now pad
        Jhat = np.pad(Jhat,pad_width=((0,nup[0]-n[0]),(0,0),(0,0)),mode='constant',constant_values=0)
        # shift it
        Jhat = np.roll(Jhat,-shift[0],axis=0)
        J = np.fft.ifft(Jhat,axis=0).real
    
    # upsample the 1th axis
    if nup[1] > n[1]:
        Jhat = np.fft.fft(J, axis=1)
        Jhat = np.roll(Jhat,shift[1],axis=1)
        if even[1]:
            # if even, make nyquist paired
            Jhat[:,0,:] /= 2.0
            Jhat = np.pad(Jhat,pad_width=((0,0),(0,1),(0,0)),mode='edge')
            n[1] = n[1] + 1
        # now pad
        Jhat = np.pad(Jhat,pad_width=((0,0),(0,nup[1]-n[1]),(0,0)),mode='constant',constant_values=0)
        # shift it
        Jhat = np.roll(Jhat,-shift[1],axis=1)
        J = np.fft.ifft(Jhat,axis=1).real
    
    # upsample the 2th axis
    if nup[2] > n[2]:
        Jhat = np.fft.fft(J, axis=2)
        Jhat = np.roll(Jhat,shift[2],axis=2)
        if even[1]:
            # if even, make nyquist paired
            Jhat[:,:,0] /= 2.0
            Jhat = np.pad(Jhat,pad_width=((0,0),(0,0),(0,1)),mode='edge')
            n[2] = n[2] + 1
        # now pad
        Jhat = np.pad(Jhat,pad_width=((0,0),(0,0),(0,nup[2]-n[2])),mode='constant',constant_values=0)
        # shift it
        Jhat = np.roll(Jhat,-shift[2],axis=2)
        J = np.fft.ifft(Jhat,axis=2).real
    
    # correct normalization
    # note inverse has 1/n
    J = J * np.prod(nup) / np.prod(n);

    return J


def transform_data(x0,x1,x2,data,tform0,tform1,tform2,
                   y0=None,y1=None,y2=None,y=None,
                   t0=None,t1=None,t2=None,t=None,
                   **kwargs):
    ''' 
    Transform data using an arbitrary position field.
    
    Inputs are:
    x0,x1,x2: scalar arrays containing location of voxels in data to be transformed
    data: a 3D image to be transformed    
    tform0,tform1,tform2: a 3 component transformation where each component is a 3D position field
        the transform need not be the same size as the input data
    
    Often this function is applied when changing resolution.  In this case also specify:
    y0,y1,y2: scalar arrays indicating the positions at which tform is resampled at
    t0,t1,t2: scalar arrays indicating the positions of the voxels in tform
    
    
    Output is the transformed data.

    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())        
        # unpack tuples
        if y is not None:
            y0,y1,y2 = y
        if t is not None:
            t0,t1,t2 = t
        
        
        
        if y0 is None:
            pass
        else:
            if t0 is None:
                raise ValueError('If you set y (locations to resample transform) you must set t (grid points of transform)')
            Y0,Y1,Y2 = np.meshgrid(y0,y1,y2,indexing='ij')
            # first we upsample the transformation
            # note that there is a problem here
            # the transform is typically not defined on the grid points x or y
            tform0 = interp3(t0,t1,t2,tform0,Y0,Y1,Y2)
            tform1 = interp3(t0,t1,t2,tform1,Y0,Y1,Y2)
            tform2 = interp3(t0,t1,t2,tform2,Y0,Y1,Y2)

        # now upsample the data
        output = interp3(x0,x1,x2,data,tform0,tform1,tform2,**kwargs).eval()
    return output


def affine_transform_data(x0,x1,x2,data,A,y0=None,y1=None,y2=None,y=None,**kwargs):
    '''
    Apply an affine transform to data using a slightly simplified interface.
    
    Inputs are:
    x0,x1,x2: scalar arrays containing location of voxels in data to be transformed
    data: a 3D image to be transformed
    
    Often this function is applied when changing resolution.  In this case also specify:
    y0,y1,y2: scalar arrays indicating the positions at which the transformed image is resampled at
    
    Output is the transformed image
    
    '''
    B = np.linalg.inv(A)
    if y is not None:
        y0,y1,y2=y
    if y0 is None:
        y0 = x0
        y1 = x1
        y2 = x2
    Y0,Y1,Y2 = np.meshgrid(y0,y1,y2,indexing='ij')
    tform0 = B[0,0]*Y0 + B[0,1]*Y1 + B[0,2]*Y2 + B[0,3]
    tform1 = B[1,0]*Y0 + B[1,1]*Y1 + B[1,2]*Y2 + B[1,3]
    tform2 = B[2,0]*Y0 + B[2,1]*Y1 + B[2,2]*Y2 + B[2,3]
    
    return transform_data(x0,x1,x2,data,tform0,tform1,tform2,**kwargs)
    


def orientation_to_matrix(input_str, desired_str):
    ''' Specify current and desired orientatoin with a sring of symbols 
    and produce am affine transformation matrix to change orientation.
    Symbols are: R/L (right/left)
                 A/P (anterior/posterior)
                 S/I (superior/inferior)

    Each symbol specifies direction from negative to positive for three axes
    e.g. RPI means x0 axis moves from left to right
                   x1 axis moves from anterior to posterior
                   x2 axis moves from superior to inferior
                   
    Inputs are:
    input_str: the orientation of an image to be transformed
    desired_str: the orientation of the desired transformed image
    '''
    if (not isinstance(input_str, str)) or (not isinstance(desired_str, str)):
        raise TypeError('inputs must be orientation strings, e.g. RPI for right posterior anterior')
    input_str = input_str.upper()
    desired_str = desired_str.upper()
    if len(input_str) != 3 or len(desired_str) != 3:
        raise ValueError('length of input strings must be 3')
    
    # the symbols tell us about permutations and flips
    # as a canonical use analyze default "transverse unflipped (LAS*)" where * is slice dimension    
    def canonical_to_observed(input_str):
        '''This function is internal because canonical means different things to different people.
        In particular, LAS is NOT right handed.'''
        A = np.eye(4)

        if 'R' in input_str:
            A = np.diag([-1,1,1,1]) @ A
        if 'P' in input_str:
            A = np.diag([1,-1,1,1]) @ A
        if 'I' in input_str:
            A = np.diag([1,1,-1,1]) @ A
    
        # now we just have to worry about permutation
        input_str = input_str.replace('R','L')
        input_str = input_str.replace('P','A')
        input_str = input_str.replace('I','S')
        
        Lind = input_str.find('L')
        Aind = input_str.find('A')
        Sind = input_str.find('S')
        
        permutation = np.array([Lind,Aind,Sind])
        zero_to_three = np.arange(4)
        P = np.array([zero_to_three == permutation[0],
                     zero_to_three == permutation[1],
                     zero_to_three == permutation[2],
                     [0,0,0,1]]).T
        A = P@A
        return A
    A = canonical_to_observed(desired_str) @ np.linalg.inv(canonical_to_observed(input_str))                
    if np.linalg.det(A) < 0:
        warnings.warn('Determinant of orientation transformation is negative.  Maybe you mixed up left and right?')
    return A


def lddmm(I,J,**kwargs):
    '''
    This function will run the lddmm algorithm to match image I to image J
    
    Assumption is that images have domain from 0 : size(image)
    Otherwise domain can be specified with xI0,xI1,xI2 or xI (a list of three)
    and xJ0, xJ1, xJ2 or xJ (a list of three)
    
    Energy operator for smoothness is of the form A = (1 - a^2 Laplacian)^(2p)
    
    Cost function is \frac{1}{2\sigma^2_R}\int \int v_t^T(x) A v_t(x) *(1/nT) dx dt
        + \frac{1}{2\sigma^2_M}\int |I(\varphi_1^{-1}(x)) - J(x)|^2 dx
        
    The deformation is formed by integrating a flow with kwargs['nt'] timesteps.
    If nt is 0, only affine transformations and not deformations are computed.
    
    If kwargs['nMstep'] is greater than 0, this will run an expectation maximization 
    algorithm for registration in the presense of anomalous data.  The "M step" maximizes
    the likelihood of transformation prameters, and the "E step" calculates the 
    probability that data is anomalous at each pixel.
    
    All options are described below in the "default parameters" section
    
    The output is a dictionary containing deformations, deformed images, 
    and several other intermediate results
    '''
    tf.reset_default_graph()

    verbose = 1
    
    ################################################################################
    # default parameters
    params = dict();
    # image size and shape
    params['x0I'] = np.arange(I.shape[0], dtype=float)
    params['x1I'] = np.arange(I.shape[1], dtype=float)
    params['x2I'] = np.arange(I.shape[2], dtype=float)
    params['x0J'] = np.arange(J.shape[0], dtype=float)
    params['x1J'] = np.arange(J.shape[1], dtype=float)
    params['x2J'] = np.arange(J.shape[2], dtype=float)
    # Transformation parameters
    params['a'] = 5.0 # spatial length scale
    params['p'] = 2 # power of laplacian (should be at least 2 in 3D to guarantee diffeomorphisms)
    params['nt'] = 5 # number of timesteps
    params['A0'] = np.eye(4) # initial guess for affine transformation
    params['rigid'] = False # rigid only versus general affine
    params['order'] = 1 # order of polynomial for predicting intensity, must be at least 1    
    # cost function parameters
    params['sigmaM'] = 1.0 # matching weight 1/2/sigma^2
    params['sigmaA'] = 10.0 # matching weight for "artifact image"
    params['sigmaR'] = 1.0 # regularization weight 1/2/sigma^2
    # optimization parameters    
    params['eV'] = 1e-1 # step size for deformation parameters
    params['eL'] = 0.0 # step size for linear part of affine
    params['eT'] = 0.0 # step size for translation part of affine    
    params['niter'] = 100 # iterations of gradient decent
    params['naffine'] = 0 # do affine only for this number
    params['post_affine_reduce'] = 0.1 # reduce affine step sizes by this much once nonrigid starts
    params['nMstep'] = 0 # number of iterations of M step each E step in EM algorithm, 0 means don't use this feature
    params['nMstep_affine'] = 0 # number of iterations of M step during affine    
    # artifact parameters    
    params['CA0'] = np.mean(J) # initial guess for value of artifact
    params['W'] = 1.0 # a fixed weight for each pixel in J, or just a number.  Can be used for known artifact locations
    
     
    if verbose: print('Set default parameters')
    
    
    ################################################################################
    # parameter setup
    # start by updating with any input arguments
    params.update(kwargs)
    # image size and shape
    if 'xI' in params:
        x0I,x1I,x2I = params['xI']
    else:
        x0I = params['x0I']
        x1I = params['x1I']
        x2I = params['x2I']
    xI = [x0I,x1I,x2I]
    if 'xJ' in params:
        x0J,x1J,x2J = params['xJ']
    else:
        x0J = params['x0J']
        x1J = params['x1J']
        x2J = params['x2J']
    xJ = [x0J,x1J,x2J]
    X0I,X1I,X2I = np.meshgrid(x0I, x1I, x2I, indexing='ij')
    X0J,X1J,X2J = np.meshgrid(x0J, x1J, x2J, indexing='ij')
    dxI = [x0I[1]-x0I[0], x1I[1]-x1I[0], x2I[1]-x2I[0]]
    dxJ = [x0J[1]-x0J[0], x1J[1]-x1J[0], x2J[1]-x2J[0]]
    nxI = I.shape
    nxJ = J.shape    
    xJp = [np.concatenate(((xJ[c][0]-dxJ[c])[None], xJ[c], (xJ[c][-1]+dxJ[c])[None]))  for c in range(3)] # padded for boundary conditions
    
    # Transformation parameters
    a = params['a'] # spatial length scale
    p = params['p'] # power of laplacian (should be at least 2 in 3D to guarantee diffeomorphisms)
    nt = params['nt'] # number of timesteps
    if 'dt' in params:
        dt = params['dt']
    else:
        if nt > 0:
            dt = 1.0/nt
        else:
            dt = 0.0
    order = params['order'] # polynomial order for predicting target image intensity from atlas
    if order < 1:
        raise ValueError('Polynomial order must be 1 (linear) or greater')
    rigid = params['rigid'] # rigid only versus general affine
    if verbose: print('Initial affine transform {}'.format(params['A0']))
    A0 = tf.convert_to_tensor(params['A0'], dtype=dtype)
    
    # cost function parameters
    sigmaM = params['sigmaM'] # matching
    sigmaM2 = sigmaM**2    
    sigmaR = params['sigmaR'] # regularization
    sigmaR2 = sigmaR**2    
    sigmaA = params['sigmaA'] # artifact
    sigmaA2 = sigmaA**2
    
    # optimization parameters
    niter = params['niter'] # gradient descent iterations
    naffine = params['naffine'] # gradient descent iterations of affinen only
    if nt == 0: # only do affine
        naffine = niter+1        
    nMstep = params['nMstep'] # number of M steps per E step in EM algorithm for artifacts
    nMstep_affine = params['nMstep_affine'] # number of M steps per E step in EM algorithm for artifacts during affine only phase 
    eV = params['eV'] # step size for deformation parameters
    eL = params['eL'] # step size for linear part of affine
    eT = params['eT'] # step size for translation part of affine    
    post_affine_reduce = params['post_affine_reduce'] # reduce affine step sizes by this much once nonrigid starts
    
    
    # Get initial guess for deformation and resample if necessary
    # initial velocity, I need the nT in order to do this
    params['vt00'] = np.zeros((I.shape[0],I.shape[1],I.shape[2],nt),dtype=np.float32) 
    params['vt10'] = np.zeros((I.shape[0],I.shape[1],I.shape[2],nt),dtype=np.float32)
    params['vt20'] = np.zeros((I.shape[0],I.shape[1],I.shape[2],nt),dtype=np.float32)
    params.update(kwargs)
    vt00 = params['vt00'].astype(np.float32)
    vt10 = params['vt10'].astype(np.float32)
    vt20 = params['vt20'].astype(np.float32)
    nt_check = vt00.shape[-1]
    if nt_check != nt:
        raise ValueError('input velocity field should be the same number of timesteps as nt parameter')        
    n0_check = vt00.shape[0]
    n1_check = vt00.shape[1]
    n2_check = vt00.shape[2]
    if n0_check != nxI[0] or n1_check != nxI[1] or n2_check != nxI[2]:
        warnings.warn('upsampling initial guess of velocity field')
        shape = np.array([I.shape[0],I.shape[1],I.shape[2],nt])
        vt00_ = vt00
        vt10_ = vt10
        vt20_ = vt20
        vt00 = np.zeros(shape)
        vt10 = np.zeros(shape)
        vt20 = np.zeros(shape)
        for t in range(nt):
            print('Upsampling velocity time {} of {}'.format(t,nt))
            vt00[:,:,:,t] = upsample(vt00_[:,:,:,t],shape[:3])
            vt10[:,:,:,t] = upsample(vt10_[:,:,:,t],shape[:3])
            vt20[:,:,:,t] = upsample(vt20_[:,:,:,t],shape[:3])
        vt00 = vt00.astype(np.float32)
        vt10 = vt10.astype(np.float32)
        vt20 = vt20.astype(np.float32)
        
            
    # gradient descent step sizes set as placeholders
    eV_ph = tf.placeholder(dtype=dtype)
    eL_ph = tf.placeholder(dtype=dtype)
    eT_ph = tf.placeholder(dtype=dtype)
    
    
    if verbose: print('Got parameters')
    
       
        
        
        
    ################################################################################
    # some initializations    
    CA = params['CA0'] # constant value for "artifact image"
    I = tf.convert_to_tensor(I, dtype=dtype)
    J = tf.convert_to_tensor(J, dtype=dtype)
    W = tf.convert_to_tensor(params['W'], dtype=dtype)
    
    # build kernels for enforcing smoothness
    f0I = np.arange(nxI[0])/dxI[0]/nxI[0]
    f1I = np.arange(nxI[1])/dxI[1]/nxI[1]
    f2I = np.arange(nxI[2])/dxI[2]/nxI[2]
    F0I,F1I,F2I = np.meshgrid(f0I, f1I, f2I, indexing='ij')
    # identity minus laplacian, in fourier domain
    # AI[i,j] = I[i,j] - alpha^2( (I[i+1,j] - 2I[i,j] + I[i-1,j])/dx^2 + (I[i,j+1] - 2I[i,j] + I[i,j-1])/dy^2  )
    Lhat = (1.0 - a**2*( (-2.0 + 2.0*np.cos(2.0*np.pi*dxI[0]*F0I))/dxI[0]**2 
        + (-2.0 + 2.0*np.cos(2.0*np.pi*dxI[1]*F1I))/dxI[1]**2
        + (-2.0 + 2.0*np.cos(2.0*np.pi*dxI[2]*F2I))/dxI[2]**2 ) )**p
    LLhat = Lhat**2
    Khat = 1.0/LLhat
    K = np.real(np.fft.ifftn(Khat))
    Khattf = tf.complex(tf.constant(Khat,dtype=dtype),tf.zeros((1),dtype=dtype)) # this should be complex because it multiplies other complex things to do smoothing
    LLhattf = tf.constant(LLhat,dtype=dtype) # this should be real because it multiplies real things to compute energy
    f = plt.figure()
    vis.imshow_slices(np.fft.ifftshift(K),x=xI,fig=f)
    f.suptitle('Smoothing kernel')
    f.canvas.draw()
    if verbose: print('Built energy operators')
        
    
    
    
    
    # initialize tensorflow variables that will be optimized
    # we need an "old" and a "new" version for our iterative algorithm
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        A = tf.get_variable('A', dtype=dtype, trainable=False, initializer=A0)
        Anew = tf.get_variable('Anew', dtype=dtype, trainable=False, initializer=A0)
        
        vt0 = tf.get_variable('vt0', dtype=dtype, trainable=False, initializer=vt00)
        vt1 = tf.get_variable('vt1', dtype=dtype, trainable=False, initializer=vt10)
        vt2 = tf.get_variable('vt2', dtype=dtype, trainable=False, initializer=vt20)

        vt0new = tf.get_variable('vt0new', dtype=dtype, trainable=False, initializer=vt00)
        vt1new = tf.get_variable('vt1new', dtype=dtype, trainable=False, initializer=vt10)
        vt2new = tf.get_variable('vt2new', dtype=dtype, trainable=False, initializer=vt20)
        
        # build initial weights WM (matching) and WA (artifact)
        # if not using weights just use 1 and 0
        npones = np.ones(nxJ)
        if nMstep>0:
            WM0 = tf.convert_to_tensor(npones*0.9, dtype=dtype)
            WA0 = tf.convert_to_tensor(npones*0.1, dtype=dtype)
        else:
            WM0 = tf.convert_to_tensor(npones, dtype=dtype)
            WA0 = tf.convert_to_tensor(npones*0.0, dtype=dtype)
        WM = tf.get_variable('WM', dtype=dtype, trainable=False, initializer=WM0)
        WA = tf.get_variable('WA', dtype=dtype, trainable=False, initializer=WA0)
        WMnew = tf.get_variable('WMnew', dtype=dtype, trainable=False, initializer=WM0)
        WAnew = tf.get_variable('WAnew', dtype=dtype, trainable=False, initializer=WA0)
        
        
    
    if verbose: print('Built tensorflow variables')
    
    
    
    ################################################################
    # define gradient calculations and updates in tensorflow graph
    # initialize time dependent flow
    It = [I]
    phiinv0 = tf.convert_to_tensor(X0I, dtype=dtype) # make sure these are tensors
    phiinv1 = tf.convert_to_tensor(X1I, dtype=dtype)
    phiinv2 = tf.convert_to_tensor(X2I, dtype=dtype)
    ERt = []
    for t in range(nt):
        # slice the velocity for convenience
        v0 = vt0[:,:,:,t]
        v1 = vt1[:,:,:,t]
        v2 = vt2[:,:,:,t]

        # points to sample at for updating diffeomorphisms
        X0s = X0I - v0*dt
        X1s = X1I - v1*dt
        X2s = X2I - v2*dt

        # update diffeomorphism with nice boundary conditions
        phiinv0 = interp3(x0I, x1I, x2I, phiinv0-X0I, X0s, X1s, X2s) + X0s
        phiinv1 = interp3(x0I, x1I, x2I, phiinv1-X1I, X0s, X1s, X2s) + X1s
        phiinv2 = interp3(x0I, x1I, x2I, phiinv2-X2I, X0s, X1s, X2s) + X2s

        # deform the image, I will need this for image gradient computations
        It.append(interp3(x0I, x1I, x2I, I, phiinv0, phiinv1, phiinv2))

        # take the Fourier transform, for computing energy directly in Fourier domain
        # note the normalizer 1/(number of elements)
        v0hat = tf.fft3d(tf.complex(v0, 0.0))
        v1hat = tf.fft3d(tf.complex(v1, 0.0))
        v2hat = tf.fft3d(tf.complex(v2, 0.0))
        
        # I changed this to reduce mean and float64 to improve numerical precision
        ER_ = tf.reduce_mean(tf.cast( ( tf.pow(tf.abs(v0hat),2) 
                                      + tf.pow(tf.abs(v1hat),2) 
                                      + tf.pow(tf.abs(v2hat),2) ) * LLhattf , dtype=tf.float64) )
        ERt.append(ER_)
    
    # now apply affine tranform
    B = tf.linalg.inv(A)
    X0s = B[0,0]*X0J + B[0,1]*X1J + B[0,2]*X2J + B[0,3]
    X1s = B[1,0]*X0J + B[1,1]*X1J + B[1,2]*X2J + B[1,3]
    X2s = B[2,0]*X0J + B[2,1]*X1J + B[2,2]*X2J + B[2,3]
    phiinvB0 = interp3(x0I, x1I, x2I, phiinv0 - X0I, X0s, X1s, X2s) + X0s
    phiinvB1 = interp3(x0I, x1I, x2I, phiinv1 - X1I, X0s, X1s, X2s) + X1s
    phiinvB2 = interp3(x0I, x1I, x2I, phiinv2 - X2I, X0s, X1s, X2s) + X2s
    AphiI = interp3(x0I, x1I, x2I, I, phiinvB0, phiinvB1, phiinvB2)
    
    
    ################################################################################
    # Calculate posterior probability weights that each pxiel is an artifact or real data
    WMsum = tf.reduce_sum(WM*W)
    WMW = WM*W
    WAW = WA*W
    CA = tf.reduce_sum(J*WAW)/(tf.reduce_sum(WAW)+1.0e-6) # avoid divide by zero possibility
    
    
    
    
    ################################################################################
    # build polynomial contrast transform
    Is = tf.reshape(AphiI,[-1])
    Js = tf.reshape(J,[-1])
    WMWs = tf.reshape(WMW,[-1])
    Basis = tf.stack( [Is**o for o in range(order+1)] ) # size O x N (order+1 by number of voxels)
    # Basis times J (size Ox1)
    BTJ = tf.reduce_mean(Basis * Js[None]*WMWs[None], axis=1)
    # get basis times basis (size OxO)
    BTB = tf.reduce_mean( Basis[:,None,:] * Basis[None,:,:] * WMWs[None,None], axis=2 )
    coeffs = tf.matrix_solve(BTB,BTJ[:,None])
    fAphiI = tf.zeros_like(Js)
    for o in range(order+1):
        fAphiI += (Is**o)*coeffs[o]    
    fAphiI = tf.reshape(fAphiI,nxJ)
    
    
    
    ################################################################################
    # now we can update weights, this is the E step of the EM algorithm
    WMnew = tf.exp( tf.pow(fAphiI - J, 2) * (-0.5/sigmaM2 ) ) * 1.0/np.sqrt(2.0*np.pi*sigmaM2)
    WAnew = tf.exp( tf.pow(CA - J, 2) * (-0.5/sigmaA2 ) ) * 1.0/np.sqrt(2.0*np.pi*sigmaA2)
    Wsum = WMnew + WAnew
    Wsum = Wsum + tf.reduce_max(Wsum)*1e-6
    WMnew = WMnew / Wsum
    WAnew = WAnew / Wsum
    
    
    
    
    ################################################################################
    # get the energy of the flow and the sum of square error matching energy
    if nt > 0:
        ER = tf.reduce_sum(tf.stack(ERt))
    else:
        ER = tf.convert_to_tensor(0.0,dtype=tf.float64)
    ER *= dt*dxI[0]*dxI[1]*dxI[2]/sigmaR2/2.0
    # typically I would also divide by nx, but since I'm using reduce mean instead of reduce sum when summing over space, I do not
    EM = tf.reduce_sum( tf.cast( tf.pow(fAphiI - J, 2)*WM*W, dtype=tf.float64) )/sigmaM2*dxI[0]*dxI[1]*dxI[2]/2.0
    # artifact
    EA = tf.reduce_sum( tf.cast( tf.pow(CA - J, 2)*WA*W, dtype=tf.float64) )/sigmaA2*dxI[0]*dxI[1]*dxI[2]/2.0
    # let's just use these two for now
    E = EM + ER
    
    
    ################################################################################
    # now we compute the gradient with respect to affine transform parameters
    # this is for right perturbations using matrix exponential parameterization
    # i.e. A \mapsto A expm( e dA)
    lambda1 = -WM*W*(fAphiI - J)/sigmaM2
    fAphiI_0, fAphiI_1, fAphiI_2 = grad3(fAphiI, dxJ)
    gradAcol = []
    for r in range(3):
        gradArow = []
        for c in range(4):
            #dA = tf.zeros(4)
            #dA[r,c] = 1.0 # tensorflow does not support this kind of assignment
            dA = np.zeros((4,4))
            dA[r,c] = 1.0
            AdAB = tf.matmul(tf.matmul(A, tf.convert_to_tensor(dA, dtype=dtype)), B)
            AdAB0 = AdAB[0,0]*X0J + AdAB[0,1]*X1J + AdAB[0,2]*X2J + AdAB[0,3];
            AdAB1 = AdAB[1,0]*X0J + AdAB[1,1]*X1J + AdAB[1,2]*X2J + AdAB[1,3];
            AdAB2 = AdAB[2,0]*X0J + AdAB[2,1]*X1J + AdAB[2,2]*X2J + AdAB[2,3];
            tmp = tf.reduce_sum( lambda1*(fAphiI_0*AdAB0 + fAphiI_1*AdAB1 + fAphiI_2*AdAB2) )
            gradArow.append(tmp)
        
        gradAcol.append(tf.stack(gradArow))
    # last row is zeros
    gradAcol.append(tf.zeros(4))
    gradA = tf.stack(gradAcol)
    gradA *= dxI[0]*dxI[1]*dxI[2]/2.0
    # now we have an affine matrix in homogeneous coordinates, with translation on the right
    if rigid:
        gradA -= tf.transpose(gradA)
        # now we also have translation on the bottom, but this will get multiplied by zero below
    
    
    ################################################################################
    # Now calculate gradient with respect to the deformation parameters
    lambda1p = tf.pad(lambda1,[[1,1],[1,1],[1,1]],'CONSTANT')
    
    # flow the error backwards
    phi1tinv0 = tf.convert_to_tensor(X0I, dtype=dtype)
    phi1tinv1 = tf.convert_to_tensor(X1I, dtype=dtype)
    phi1tinv2 = tf.convert_to_tensor(X2I, dtype=dtype)
    vt0new_ = []
    vt1new_ = []
    vt2new_ = []
    for t in range(nt-1,-1,-1):
        v0 = vt0[:,:,:,t]
        v1 = vt1[:,:,:,t]
        v2 = vt2[:,:,:,t]
        X0s = X0I + v0*dt
        X1s = X1I + v1*dt
        X2s = X2I + v2*dt
        phi1tinv0 = interp3(x0I, x1I, x2I, phi1tinv0-X0I, X0s, X1s, X2s) + X0s
        phi1tinv1 = interp3(x0I, x1I, x2I, phi1tinv1-X1I, X0s, X1s, X2s) + X1s
        phi1tinv2 = interp3(x0I, x1I, x2I, phi1tinv2-X2I, X0s, X1s, X2s) + X2s

        # compute the gradient of the image at this time
        fIt = tf.zeros_like(I)
        for o in range(order+1):
            fIt += (It[t]**o)*coeffs[o] 
        fI_0,fI_1,fI_2 = grad3(fIt, dxI)

        # compute the determinanat of jacobian
        phi1tinv0_0,phi1tinv0_1,phi1tinv0_2 = grad3(phi1tinv0, dxI)
        phi1tinv1_0,phi1tinv1_1,phi1tinv1_2 = grad3(phi1tinv1, dxI)
        phi1tinv2_0,phi1tinv2_1,phi1tinv2_2 = grad3(phi1tinv2, dxI)
        detjac = phi1tinv0_0*(phi1tinv1_1*phi1tinv2_2 - phi1tinv1_2*phi1tinv2_1)\
            - phi1tinv0_1*(phi1tinv1_0*phi1tinv2_2 - phi1tinv1_2*phi1tinv2_0)\
            + phi1tinv0_2*(phi1tinv1_0*phi1tinv2_1 - phi1tinv1_1*phi1tinv2_0)

        # get the lambda for this time, don't forget to include jacobian factors
        Aphi1tinv0 = A[0,0]*phi1tinv0 + A[0,1]*phi1tinv1 + A[0,2]*phi1tinv2 + A[0,3];
        Aphi1tinv1 = A[1,0]*phi1tinv0 + A[1,1]*phi1tinv1 + A[1,2]*phi1tinv2 + A[1,3];
        Aphi1tinv2 = A[2,0]*phi1tinv0 + A[2,1]*phi1tinv1 + A[2,2]*phi1tinv2 + A[2,3];        
        #lambda_ = interp3(x0J, x1J, x2J, lambda1, Aphi1tinv0, Aphi1tinv1, Aphi1tinv2)*detjac*tf.abs(tf.linalg.det(A))
        lambda_ = interp3(*xJp, lambda1p, Aphi1tinv0, Aphi1tinv1, Aphi1tinv2)*detjac*tf.abs(tf.linalg.det(A))

        # set up the gradient at this time        
        grad0 = lambda_*fI_0
        grad1 = lambda_*fI_1
        grad2 = lambda_*fI_2

        # smooth it        
        grad0 = tf.real(tf.ifft3d(tf.fft3d(tf.complex(grad0, 0.0))*Khattf))
        grad1 = tf.real(tf.ifft3d(tf.fft3d(tf.complex(grad1, 0.0))*Khattf))
        grad2 = tf.real(tf.ifft3d(tf.fft3d(tf.complex(grad2, 0.0))*Khattf))

        # add the regularization
        grad0 = grad0 + v0/sigmaR2
        grad1 = grad1 + v1/sigmaR2
        grad2 = grad2 + v2/sigmaR2

        # and calculate the new v
        vt0new_.append(v0 - eV_ph*grad0)
        vt1new_.append(v1 - eV_ph*grad1)
        vt2new_.append(v2 - eV_ph*grad2)

    # stack all the times
    if nt > 0:
        vt0new = tf.stack(vt0new_[::-1], axis=3)
        vt1new = tf.stack(vt1new_[::-1], axis=3)
        vt2new = tf.stack(vt2new_[::-1], axis=3)
    
    ################################################################################
    # update affine parameters
    # use gradient descent with a right perturbation on the group
    ones_linear = np.zeros((4,4))
    ones_linear[:3,:3] = 1.0
    ones_translation = np.zeros((4,4))
    ones_translation[:3,-1] = 1.0
    e = ones_linear * eL_ph + ones_translation * eT_ph    
    Anew = tf.matmul(A, tf.linalg.expm(-e*gradA) )

    
    ################################################################################
    # define a graph operation to update
    # hopefully treating step_A separately will allow it to be computed more quickly
    step = tf.group(
      A.assign(Anew),
      vt0.assign(vt0new),
      vt1.assign(vt1new),
      vt2.assign(vt2new))
    step_v = tf.group(
      vt0.assign(vt0new),
      vt1.assign(vt1new),
      vt2.assign(vt2new))
    step_A = A.assign(Anew)
    step_W = tf.group(
        WM.assign(WMnew),
        WA.assign(WAnew))
    
    if verbose: print('Computation graph defined')
    
    
    ################################################################################
    # now that that the graph is defined, we can do gradient descent optimization
    # Plotting is performed during the computations to visualize results
    # initialize lists and figures for plotting
    EMall = []
    ERall = []
    Eall = []
    Aall = []
    vmaxall = []
    f0 = plt.figure()
    f1 = plt.figure()    
    f2,ax = plt.subplots(2,2) # this figure is for plotting energy and parameters
    if nMstep > 0: # weights
        fW = plt.figure()
        fWA = plt.figure()
        
    # create a tensorflow session and perform the update steps    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # this is always required
        
        # start the gradient descent loop
        for it in range(niter):
            # take a step of gradient descent, and get some values to plot
            # for the first naffine steps, we just update affine
            # afterwards, we update both simultaneously, but we shrink the affine stepsize
            if it < naffine:
                if verbose: print('Taking affine only step')
                _, EM_, ER_, E_, Idnp, lambda1np, Anp = sess.run([step_A,EM,ER,E,fAphiI,lambda1,Anew], feed_dict={eL_ph:eL, eT_ph:eT})
            else:
                # note use smaller affine parameters
                if verbose: print('Taking affine and deformation step')
                _, EM_, ER_, E_, \
                Idnp, lambda1np, Anp, \
                vt0np, vt1np, vt2np = sess.run([step,EM,ER,E,fAphiI,lambda1,Anew,vt0,vt1,vt2], feed_dict={eL_ph:eL*post_affine_reduce, eT_ph:eT*post_affine_reduce, eV_ph:eV})
            

            # now we perform the E step of the EM algorithm for registration with artifacts
            # this occurs only every nMstep iterations, or nMstep_affine iterations if we are in 
            # the affine only optimization stage
            if (nMstep>0
                and ( (it < naffine and not it%nMstep_affine) 
                 or (it >= naffine and not it%nMstep) ) ): 
                print('Updating weights')
                _, WMnp, WAnp = sess.run([step_W,WMnew,WAnew])
                fW.clf()
                vis.imshow_slices(WMnp, x=xJ, fig=fW)
                fW.suptitle('Image Weight')
                fW.canvas.draw()
                fWA.clf()
                vis.imshow_slices(WAnp, x=xJ, fig=fWA)
                fWA.suptitle('Artifact Weight')
                fWA.canvas.draw()
                
            ################################################################################
            # create plots
            f0.clf()
            vis.imshow_slices(Idnp, x=xJ, fig=f0)
            f0.suptitle('Deformed atlas (iter {})'.format(it))
            f1.clf()
            vis.imshow_slices(lambda1np, x=xJ, fig=f1)
            f1.suptitle('Error (iter {})'.format(it))
            
            # save energy for each iteration
            EMall.append(EM_)
            ERall.append(ER_)
            Eall.append(E_)
            ax[0,0].cla()
            ax[0,0].plot(list(zip(Eall,EMall,ERall)))
            xlim = ax[0,0].get_xlim()
            ylim = ax[0,0].get_ylim()
            ax[0,0].set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))
            ax[0,0].legend(['Etot','Ematch','Ereg'])
            ax[0,0].set_title('Energy minimization')
            
            # show some parameters to visualize affine transforms
            # use the matrix log so data is close to 0 and and the scale can be seen
            # the logm function always outputs warnings abut inaccuracy, use disp=False and save error to a variable
            logmAnp,_ = logm(Anp, disp=False) 
            Aall.append(logmAnp.ravel())
            Aallnp = np.array(Aall)
            if it >= naffine:
                vmaxall.append(np.sqrt(np.max(vt0np**2 + vt1np**2 + vt2np**2)))
            else:
                vmaxall.append(0.0)
            vmaxallnp = np.array(vmaxall)
            
            ax[0,1].cla()
            ax[0,1].plot(range(it+1),Aallnp[:,3:12:4])
            xlim = ax[0,1].get_xlim()
            ylim = ax[0,1].get_ylim()
            ax[0,1].set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))
            ax[0,1].set_title('Translation')
            # linear
            ax[1,0].cla()
            ax[1,0].plot(range(it+1),Aallnp[:,0:3])
            ax[1,0].plot(range(it+1),Aallnp[:,4:7])
            ax[1,0].plot(range(it+1),Aallnp[:,8:11])
            xlim = ax[1,0].get_xlim()
            ylim = ax[1,0].get_ylim()
            ax[1,0].set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))
            ax[1,0].set_title('log-Linear')
            
            
            ax[1,1].cla()
            ax[1,1].plot(vmaxallnp)            
            xlim = ax[1,1].get_xlim()
            ylim = ax[1,1].get_ylim()
            ax[1,1].set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))
            ax[1,1].set_title('max norm velocity')
            
            # force drawing now (otherwise python will wait until code has stopped running)
            f0.canvas.draw()
            f1.canvas.draw()
            f2.canvas.draw()
            
            # uncomment the line below to save a figure at every iteration
            # this can be used to create movies
            # f0.savefig('lddmm3d_example_iteration_{:03d}.png'.format(i))
            print('Finished iteration {}, energy {:3e} (match {:3e}, reg {:3e})'.format(it, E_, EM_, ER_))
            
        # collect output variables from this tensorflow session
        Anp,\
        vt0np,vt1np,vt2np,\
        phiinv0np,phiinv1np,phiinv2np,\
        phi1tinv0np,phi1tinv1np,phi1tinv2np,\
        phiinvB0np,phiinvB1np,phiinvB2np,\
        Aphi1tinv0np,Aphi1tinv1np,Aphi1tinv2np,\
        WMnp,WAnp                       =  sess.run([A,\
                                                     vt0,vt1,vt2,\
                                                     phiinv0,phiinv1,phiinv2,\
                                                     phi1tinv0,phi1tinv1,phi1tinv2,\
                                                     phiinvB0,phiinvB1,phiinvB2,\
                                                     Aphi1tinv0,Aphi1tinv1,Aphi1tinv2,\
                                                     WMnew,WAnew])
    # use a dictionary for output
    output = {'A':Anp, # affine transform
              'vt0':vt0np, 'vt1':vt1np, 'vt2':vt2np, # velocity field for computing deformation
              'phiinv0':phiinv0np, 'phiinv1':phiinv1np, 'phiinv2':phiinv2np, # inverse transform (for deforming images)
              'phi0':phi1tinv0np, 'phi1':phi1tinv1np, 'phi2':phi1tinv2np, # forward transform (for deforming points)
              'phiinvAinv0':phiinvB0np,'phiinvAinv1':phiinvB1np,'phiinvAinv2':phiinvB2np, # inverse transform including affine
              'Aphi0':Aphi1tinv0np,'Aphi1':Aphi1tinv1np,'Aphi2':Aphi1tinv2np, # forward transform including affine
              'WM':WMnp, 'WA':WAnp, # matching and artifact posterior probabilities from EM aglorithm
              'AphiI':Idnp, # the deformed atlas image
              'f_kernel':f, # figure of smoothing kernel      
              'f_deformed':f0, # figure for deformed atlas
              'f_error':f1, # figure showing error
              'f_energy':f2 # figure showing energy
             }
    if nMstep > 0: # if performing EM for artifacts
        output['f_WM'] = fW # figure showing matching posterior probability
        output['f_WA'] = fWA # figure showing artifact posterior probability
    return output


# if run as a script from the command line
if __name__ == '__main__':
    '''
    When this module is run as a script from command line it will run LDDMM
    Required command line options are
    prefix: string to prefix all output files
    atlas: filename of atlas image
    target: filename of target image
    scale: length scale of diffeomorphism
    sigmaM: noise scale for image matching (1/2/sigmaM^2 is fidelity weight)
    sigmaR: noise scale for deformation (1/2/sigmaR^2 is regularization weight)
    
    There are many optional argument describing different parameters of the algorithm, 
    see lddmm function above and help message printed to command line.
    '''
    
    # create parser
    parser = argparse.ArgumentParser(description='Run LDDMM between an atlas and target image.')    

    # add required arguments
    parser.add_argument('prefix', type=str, help='string prefix for all outputs (can include a directory, directories should have trailing slashes)')
    parser.add_argument('atlas', type=str, help='filename for atlas image')
    parser.add_argument('target', type=str, help='filename for target image')
    parser.add_argument('a', type=float, help='spatial scale of transformation smoothness (in same units as image headers)', metavar='scale') # note this variable will be stored as a in my namespace
    parser.add_argument('sigmaM', type=float, help='std for image matching cost')
    parser.add_argument('sigmaR', type=float, help='std for deformation cost')
    parser.add_argument('niter', type=int, help='number of iterations of gradient descent for deformation')
    parser.add_argument('eV', type=float, help='gradient descent step size for deformation')
    
    # optional arguments
    parser.add_argument('--affine', type=str, help='text filename storing initial affine transform (account for differences in orientation, defaults to identity)')
    parser.add_argument('--p', type=float, help='power of Laplacian in smoothing operator', metavar='power') # stored as p in my namespace
    parser.add_argument('--niter', type=int, help='total number of iterations of gradient descent')
    parser.add_argument('--naffine', type=int, help='number of iterations of affine only optimization before deformation')    
    parser.add_argument('--post_affine_reduce', type=float, help='factor to reduce affine (improves numerical stability)')    
    parser.add_argument('--eL', type=float, help='gradient descent step size for linear part of affine')
    parser.add_argument('--eT', type=float, help='gradient descent step size for translation part of affine')
    parser.add_argument('--nT', type=int, help='number of timesteps to integrate flow')
    
        
    # missing data
    parser.add_argument('--nMstep', type=int, help='number of M steps per E step when using missing data EM algorithm.  Default is to not use this approach')
    parser.add_argument('--nMstep_affine', type=int, help='number of M steps per E step when using missing data EM algorithm during affine only alignment.  Typically a smaller number works here. Default is to not use this approach.')
    parser.add_argument('--sigmaA', type=float, help='std for artifact when using missing data EM algorithm, typically sigmaM*10')
    
    # a feature for working with allen atlas
    parser.add_argument('--pad_allen', help='add a blank slice to the allen atlas to help with boundary conditions on interpolation', action='store_true')
    
    args = parser.parse_args()    
    print(args)
    
    
    
    
    ################################################################################
    # parse output prefix
    splitpath = os.path.split(args.prefix)
    if not os.path.exists(splitpath[0]):
        print('output prefix directory "{}" does not exist, creating it.'.format(splitpath[0]))
        os.mkdir(splitpath[0])
    
    
    ################################################################################
    # Load images
    # load them with nababel
    fnames = [args.atlas,args.target]
    img = [nib.load(fname) for fname in fnames]
    # get info about domains
    # we assume for this example that we have the same voxel size and same voxel spacing for atlas and target
    if '.img' == args.atlas[-4:] and '.img' == args.target[-4:]:    
        nxI = img[0].header['dim'][1:4]
        dxI = img[0].header['pixdim'][1:4]
        nxJ = img[1].header['dim'][1:4]
        dxJ = img[1].header['pixdim'][1:4]    
    else:
        # I'm only working with analyze for now
        raise ValueError('Only Analyze images supported for now')
    xI = [np.arange(nxi)*dxi - np.mean(np.arange(nxi)*dxi) for nxi,dxi in zip(nxI,dxI)]
    xJ = [np.arange(nxi)*dxi - np.mean(np.arange(nxi)*dxi) for nxi,dxi in zip(nxJ,dxJ)]
    print('Loaded atlas image with size {} and spacing {}'.format(nxI,dxI))
    print('Loaded target image with size {} and spacing {}'.format(nxJ,dxJ))
    
    # get the images, note they also include a fourth axis for time that I don't want
    I = img[0].get_data()[:,:,:,0]
    J = img[1].get_data()[:,:,:,0]

    if args.pad_allen:
        # I would like to pad one slice of the allen atlas so that it has zero boundary conditions
        zeroslice = np.zeros((nxI[0],1,nxI[2]))
        I = np.concatenate((I,zeroslice),axis=1)
        nxI = img[0].header['dim'][1:4]
        nxI[1] += 1
        xI = [np.arange(nxi)*dxi - np.mean(np.arange(nxi)*dxi) for nxi,dxi in zip(nxI,dxI)]
    
    # display the images and write them out
    # display the atlas
    f = plt.figure()
    vis.imshow_slices(I, x=xI, fig=f)
    f.suptitle('Atlas I')
    f.savefig(args.prefix + 'atlas.png')
    plt.close(f)
    # display the target
    f = plt.figure()
    vis.imshow_slices(J,x=xJ,fig=f)
    f.suptitle('Target J')
    f.savefig(args.prefix + 'target.png')
    plt.close(f)
    
    
    
    ################################################################################
    # parse initial affine file
    if args.affine is not None:
        print('reading affine from file')
        with open(args.affine) as f:
            A0 = np.array( [ [ float(a) for a in line.strip().split()] for line in f] )
    else:
        A0 = np.eye(4)
    print('affine is {}.  Please check output images to see if it is appropriate.'.format(A0))
    
    X0,X1,X2 = np.meshgrid(xJ[0],xJ[1],xJ[2],indexing='ij')
    X0tf = tf.constant(X0,dtype=dtype)
    X1tf = tf.constant(X1,dtype=dtype)
    X2tf = tf.constant(X2,dtype=dtype)
    Itf = tf.constant(I,dtype=dtype)
    B = np.linalg.inv(A0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Xs = B[0,0]*X0tf + B[0,1]*X1tf + B[0,2]*X2tf + B[0,3]
        Ys = B[1,0]*X0tf + B[1,1]*X1tf + B[1,2]*X2tf + B[1,3]
        Zs = B[2,0]*X0tf + B[2,1]*X1tf + B[2,2]*X2tf + B[2,3]
        Id = interp3(xI[0], xI[1], xI[2], Itf, Xs, Ys, Zs)
        Idnp = Id.eval()
    f = plt.figure()
    vis.imshow_slices(Idnp,x=xJ,fig=f)
    f.suptitle('Initial affine transformation')
    f.savefig(args.prefix + 'atlas-affine.png')
    plt.close(f)
    
    

    

    ################################################################################
    # put the defaults in this dict if they needed to be computed in this file
    params = {'sigmaA':args.sigmaM*10, # std for artifacts
              'A0':A0,
              'xI':xI,
              'xJ':xJ
             }
    # get a dictionary, but leave out any Nones
    argsdict = {k:v for k,v in vars(args).items() if v is not None}
    # update the parameters with my arguments
    params.update(argsdict)
    
    # now run lddmm   
    out = lddmm(I,J, # atlas and target images                  
                **params # all other parameters
               )
    
    # write out the figures
    out['f_kernel'].savefig(args.prefix + 'kernel.png')
    out['f_deformed'].savefig(args.prefix + 'atlas-deformed.png')
    out['f_error'].savefig(args.prefix + 'error.png')
    out['f_energy'].savefig(args.prefix + 'energy.png')
    if args.nMstep > 0:
        out['f_WM'].savefig(args.prefix + 'atlas-weight.png')
        out['f_WA'].savefig(args.prefix + 'artifact-weight.png')
        
    
    # write output in numpy format
    phi = np.concatenate([out['phi0'][None], out['phi1'][None], out['phi2'][None]])
    np.save(args.prefix + "phi", phi)
    phiinv = np.concatenate([out['phiinv0'][None], out['phiinv1'][None], out['phiinv2'][None]])
    np.save(args.prefix + "phiinv", phiinv)
    XI = np.meshgrid(*xI,indexing='ij')
    np.save(args.prefix + "XI", XI)
    XJ = np.meshgrid(*xJ,indexing='ij')
    np.save(args.prefix + "XJ", XJ)
    Aphi = np.concatenate([out['Aphi0'][None], out['Aphi1'][None], out['Aphi2'][None]])
    np.save(args.prefix + "Aphi", Aphi)
    phiinvAinv = np.concatenate([out['phiinvAinv0'][None], out['phiinvAinv1'][None], out['phiinvAinv2'][None]])
    np.save(args.prefix + "phiinvAinv", phiinvAinv)
    
    # And lets do A and A inv as text files
    A = out['A']
    with open(args.prefix + 'A.txt','wt') as f:
        for r in range(4):
            for c in range(4):
                f.write(str(A[r,c])+' ')
            f.write('\n')
    B = np.linalg.inv(A)
    with open(args.prefix + 'Ainv.txt','wt') as f:
        for r in range(4):
            for c in range(4):
                f.write(str(B[r,c])+' ')
            f.write('\n')
            
    