print('Importing helper functions')

import numpy as np
import tensorflow as tf
import vis

dtype = tf.float32
idtype = tf.int64
def interp3(x0,x1,x2,I,phi0,phi1,phi2):
    ''' 
    Linear interpolation
    Interpolate a 3D tensorflow image I
    with voxels corresponding to locations in x0, x1, x2 (1d arrays)
    at the points phi0, phi1, phi2 (3d arrays)
    '''
    # get the size
    dx = [x0[1]-x0[0], x1[1]-x1[0], x2[1]-x2[0]]
    nx = [len(x0), len(x1), len(x2)]
    #convert to index
    phi0_index = (phi0 - x0[0])/dx[0]
    phi1_index = (phi1 - x1[0])/dx[1]
    phi2_index = (phi2 - x2[0])/dx[2]
    # take the floor to get integers
    phi0_index_floor = tf.floor(phi0_index)
    phi1_index_floor = tf.floor(phi1_index)
    phi2_index_floor = tf.floor(phi2_index)
    # get the fraction to the next pixel
    phi0_p = phi0_index - phi0_index_floor
    phi1_p = phi1_index - phi1_index_floor
    phi2_p = phi2_index - phi2_index_floor
    # get the next samples
    phi0_index_floor_1 = phi0_index_floor+1
    phi1_index_floor_1 = phi1_index_floor+1
    phi2_index_floor_1 = phi2_index_floor+1
    # and apply boundary conditions
    phi0_index_floor = tf.minimum(phi0_index_floor,nx[0]-1)
    phi0_index_floor = tf.maximum(phi0_index_floor,0)
    phi0_index_floor_1 = tf.minimum(phi0_index_floor_1,nx[0]-1)
    phi0_index_floor_1 = tf.maximum(phi0_index_floor_1,0)
    phi1_index_floor = tf.minimum(phi1_index_floor,nx[1]-1)
    phi1_index_floor = tf.maximum(phi1_index_floor,0)
    phi1_index_floor_1 = tf.minimum(phi1_index_floor_1,nx[1]-1)
    phi1_index_floor_1 = tf.maximum(phi1_index_floor_1,0)
    phi2_index_floor = tf.minimum(phi2_index_floor,nx[2]-1)
    phi2_index_floor = tf.maximum(phi2_index_floor,0)
    phi2_index_floor_1 = tf.minimum(phi2_index_floor_1,nx[2]-1)
    phi2_index_floor_1 = tf.maximum(phi2_index_floor_1,0)
    # then we will need to vectorize everything to use scalar indices
    phi0_index_floor_flat = tf.reshape(phi0_index_floor,[-1])
    phi0_index_floor_flat_1 = tf.reshape(phi0_index_floor_1,[-1])
    phi1_index_floor_flat = tf.reshape(phi1_index_floor,[-1])
    phi1_index_floor_flat_1 = tf.reshape(phi1_index_floor_1,[-1])
    phi2_index_floor_flat = tf.reshape(phi2_index_floor,[-1])
    phi2_index_floor_flat_1 = tf.reshape(phi2_index_floor_1,[-1])
    I_flat = tf.reshape(I,[-1])
    # indices recall that the LAST INDEX IS CONTIGUOUS
    phi_index_floor_flat_000 = nx[2]*nx[1]*phi0_index_floor_flat + nx[2]*phi1_index_floor_flat + phi2_index_floor_flat
    phi_index_floor_flat_001 = nx[2]*nx[1]*phi0_index_floor_flat + nx[2]*phi1_index_floor_flat + phi2_index_floor_flat_1
    phi_index_floor_flat_010 = nx[2]*nx[1]*phi0_index_floor_flat + nx[2]*phi1_index_floor_flat_1 + phi2_index_floor_flat
    phi_index_floor_flat_011 = nx[2]*nx[1]*phi0_index_floor_flat + nx[2]*phi1_index_floor_flat_1 + phi2_index_floor_flat_1
    phi_index_floor_flat_100 = nx[2]*nx[1]*phi0_index_floor_flat_1 + nx[2]*phi1_index_floor_flat + phi2_index_floor_flat
    phi_index_floor_flat_101 = nx[2]*nx[1]*phi0_index_floor_flat_1 + nx[2]*phi1_index_floor_flat + phi2_index_floor_flat_1
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
    I000 = tf.reshape(I000_flat,nx)
    I001 = tf.reshape(I001_flat,nx)
    I010 = tf.reshape(I010_flat,nx)
    I011 = tf.reshape(I011_flat,nx)
    I100 = tf.reshape(I100_flat,nx)
    I101 = tf.reshape(I101_flat,nx)
    I110 = tf.reshape(I110_flat,nx)
    I111 = tf.reshape(I111_flat,nx)

    # combine them!
    Il = I000*(1.0-phi0_p)*(1.0-phi1_p)*(1.0-phi2_p)\
        + I001*(1.0-phi0_p)*(1.0-phi1_p)*(    phi2_p)\
        + I010*(1.0-phi0_p)*(    phi1_p)*(1.0-phi2_p)\
        + I011*(1.0-phi0_p)*(    phi1_p)*(    phi2_p)\
        + I100*(    phi0_p)*(1.0-phi1_p)*(1.0-phi2_p)\
        + I101*(    phi0_p)*(1.0-phi1_p)*(    phi2_p)\
        + I110*(    phi0_p)*(    phi1_p)*(1.0-phi2_p)\
        + I111*(    phi0_p)*(    phi1_p)*(    phi2_p)
    return Il


def grad3(I,dx):
    #I_0 = (tf.manip.roll(I,shift=-1,axis=0) - tf.manip.roll(I,shift=1,axis=0))/2.0/dx[0]
    #I_1 = (tf.manip.roll(I,shift=-1,axis=1) - tf.manip.roll(I,shift=1,axis=1))/2.0/dx[1]
    #I_2 = (tf.manip.roll(I,shift=-1,axis=2) - tf.manip.roll(I,shift=1,axis=2))/2.0/dx[2]
    
    #out[0,:] = out[1,:]-out[0,:] # this doesn't work in tensorflow
    # generally you cannot assign to a tensor
    # problems with energy calculations are due to this gradient function
    
    # in particular the determinant of jacobian part which is very much noncircular
    # this leads to discontinuity which becomes very large regularization energy
    # enforcing boundary conditions is essential
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


def lddmm():
    '''
    This function will run the lddmm algorithm
    '''
    # parameter setup
    
    # build kernels
    
    # define gradient step
    
    # perform gradient descent