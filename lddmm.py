print('Importing helper functions')

import matplotlib.pyplot as plt
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
    I = tf.convert_to_tensor(I, dtype=dtype)
    phi0 = tf.convert_to_tensor(phi0, dtype=dtype)
    phi1 = tf.convert_to_tensor(phi1, dtype=dtype)
    phi2 = tf.convert_to_tensor(phi2, dtype=dtype)
    
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


def lddmm(I,J,**kwargs):
    '''
    This function will run the lddmm algorithm to match image I to image J
    
    Assumption is that images have domain from 0 : size(image)
    Otherwise domain can be specified with xI0,xI1,xI2 or xI (a list of three)
    and xJ0, xJ1, xJ2 or xJ (a list of three)
    
    Energy operator for smoothness is of the form A = (1 - a^2 Laplacian)^(2p)
    
    Cost function is \frac{1}{2\sigma^2_R}\int \int v_t^T(x) A v_t(x) *(1/nT) dx dt
        + \frac{1}{2\sigma^2_M}\int |I(\varphi_1^{-1}(x)) - J(x)|^2 dx
    
    '''
    tf.reset_default_graph()

    verbose = 1
    
    # default parameters
    params = dict();
    params['x0I'] = np.arange(I.shape[0], dtype=float)
    params['x1I'] = np.arange(I.shape[1], dtype=float)
    params['x2I'] = np.arange(I.shape[2], dtype=float)
    params['x0J'] = np.arange(J.shape[0], dtype=float)
    params['x1J'] = np.arange(J.shape[1], dtype=float)
    params['x2J'] = np.arange(J.shape[2], dtype=float)
    params['a'] = 5.0
    params['p'] = 2 # should be at least 2 in 3D
    params['nt'] = 5
    params['sigmaM'] = 1.0
    params['sigmaR'] = 1.0
    params['epsilon'] = 5e-3
    params['niter'] = 100
    if verbose: print('Set default parameters')
    
    
    
    # parameter setup
    # start by updating with any input arguments
    params.update(kwargs)
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
    
    a = params['a']
    p = params['p']
    
    sigmaM = params['sigmaM']
    sigmaM2 = sigmaM**2
    sigmaR = params['sigmaR']
    sigmaR2 = sigmaR**2
    
    nt = params['nt']
    if 'dt' in params:
        dt = params['dt']
    else:
        dt = 1.0/nt
    
    niter = params['niter']
    epsilon = params['epsilon']
    
    
    if verbose: print('Got parameters')
        
        
    # build kernels
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
        
    
    # initialize tensorflow variables, note that I am not using built in training
    # we can only declare these variables once
    # so if it's already been done, just load them
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        vt0 = tf.get_variable('vt0', shape=[nxI[0],nxI[1],nxI[2],nt], dtype=dtype, trainable=False, initializer=tf.zeros_initializer())
        vt1 = tf.get_variable('vt1', shape=[nxI[0],nxI[1],nxI[2],nt],dtype=dtype,trainable=False, initializer=tf.zeros_initializer())
        vt2 = tf.get_variable('vt2', shape=[nxI[0],nxI[1],nxI[2],nt], dtype=dtype, trainable=False, initializer=tf.zeros_initializer())

        vt0new = tf.get_variable('vt0new', shape=[nxI[0],nxI[1],nxI[2],nt], dtype=dtype, trainable=False, initializer=tf.zeros_initializer())
        vt1new = tf.get_variable('vt1new', shape=[nxI[0],nxI[1],nxI[2],nt], dtype=dtype, trainable=False, initializer=tf.zeros_initializer())
        vt2new = tf.get_variable('vt2new', shape=[nxI[0],nxI[1],nxI[2],nt], dtype=dtype,trainable=False, initializer=tf.zeros_initializer())
    
    if verbose: print('built tensorflow variables')
    
    # define gradient step as a graph
    It = [I]
    phiinv0 = X0I
    phiinv1 = X1I
    phiinv2 = X2I
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

        # deform the image
        It.append(interp3(x0I, x1I, x2I, I, phiinv0, phiinv1, phiinv2))

        # get regularization energy
        # this is probably the fastest way to compute energy, note the normalizer 1/(number of elemetns)
        v0hat = tf.fft3d(tf.complex(v0, 0.0))
        v1hat = tf.fft3d(tf.complex(v1, 0.0))
        v2hat = tf.fft3d(tf.complex(v2, 0.0))
        
        # I changed this to reduce mean and float64 to improve numerical precision
        ER_ = tf.reduce_mean(tf.cast( ( tf.pow(tf.abs(v0hat),2) 
                                      + tf.pow(tf.abs(v1hat),2) 
                                      + tf.pow(tf.abs(v2hat),2) ) * LLhattf , dtype=tf.float64) )
        ERt.append(ER_)
        
    # get energy
    ER = tf.reduce_sum(tf.stack(ERt))
    ER *= dt*dxI[0]*dxI[1]*dxI[2]/sigmaR2/2.0
    # typically I would also divide by nx, but since I'm using reduce mean instead of reduce sum, I do not
    EM = tf.reduce_sum( tf.cast( tf.pow(It[-1] - J, 2) , dtype=tf.float64) )/sigmaM2*dxI[0]*dxI[1]*dxI[2]/2.0
    E = EM + ER
    
    # now compute the error of the cost function with respect to the deformed image
    lambda1 = -(It[-1] - J)/sigmaM2
    
    # flow the error backwards
    phi1tinv0 = X0I
    phi1tinv1 = X1I
    phi1tinv2 = X2I
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
        I_0,I_1,I_2 = grad3(It[t], dxI)

        # compute the determinanat of jacobian
        phi1tinv0_0,phi1tinv0_1,phi1tinv0_2 = grad3(phi1tinv0, dxI)
        phi1tinv1_0,phi1tinv1_1,phi1tinv1_2 = grad3(phi1tinv1, dxI)
        phi1tinv2_0,phi1tinv2_1,phi1tinv2_2 = grad3(phi1tinv2, dxI)
        detjac = phi1tinv0_0*(phi1tinv1_1*phi1tinv2_2 - phi1tinv1_2*phi1tinv2_1)\
            - phi1tinv0_1*(phi1tinv1_0*phi1tinv2_2 - phi1tinv1_2*phi1tinv2_0)\
            + phi1tinv0_2*(phi1tinv1_0*phi1tinv2_1 - phi1tinv1_1*phi1tinv2_0)

        # get the lambda for this time
        lambda_ = interp3(x0I, x1I, x2I, lambda1, phi1tinv0, phi1tinv1, phi1tinv2)*detjac

        # set up the gradient
        grad0 = lambda_*I_0
        grad1 = lambda_*I_1
        grad2 = lambda_*I_2

        # smooth it
        grad0 = tf.real(tf.ifft3d(tf.fft3d(tf.complex(grad0, 0.0))*Khattf))
        grad1 = tf.real(tf.ifft3d(tf.fft3d(tf.complex(grad1, 0.0))*Khattf))
        grad2 = tf.real(tf.ifft3d(tf.fft3d(tf.complex(grad2, 0.0))*Khattf))

        # add the regularization
        grad0 = grad0 + v0/sigmaR2
        grad1 = grad1 + v1/sigmaR2
        grad2 = grad2 + v2/sigmaR2

        # and calculate the new v
        vt0new_.append(v0 - epsilon*grad0)
        vt1new_.append(v1 - epsilon*grad1)
        vt2new_.append(v2 - epsilon*grad2)

    # stack
    vt0new = tf.stack(vt0new_[::-1],axis=3)
    vt1new = tf.stack(vt1new_[::-1],axis=3)
    vt2new = tf.stack(vt2new_[::-1],axis=3)

    
    # define a graph operation to update everything
    step = tf.group(
      vt0.assign(vt0new),
      vt1.assign(vt1new),
      vt2.assign(vt2new))

    if verbose: print('Computation graph defined')
    
    
    
    # now that that the graph is defined, we can do gradient descent
    EMall = []
    ERall = []
    Eall = []
    f0 = plt.figure()
    f1 = plt.figure()
    f2,ax = plt.subplots()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for it in range(niter):
            # take a step of gradient descent
            # everything I evaluate should be run at once
            _,EM_,ER_,E_,Idnp,lambda1np = sess.run([step,EM,ER,E,It[-1],lambda1])    
            f0.clf()
            vis.imshow_slices(Idnp, x=xI, fig=f0)
            f0.suptitle('Deformed atlas')
            f1.clf()
            vis.imshow_slices(lambda1np, x=xI, fig=f1)
            f1.suptitle('Error')

            EMall.append(EM_)
            ERall.append(ER_)
            Eall.append(E_)
            ax.cla()
            ax.plot(list(zip(Eall,EMall,ERall)))
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))
            ax.legend(['Etot','Ematch','Ereg'])
            f2.suptitle('Energy minimization')

            f0.canvas.draw()
            f1.canvas.draw()
            f2.canvas.draw()
            #f0.savefig('lddmm3d_example_iteration_{:03d}.png'.format(i))
            print('Finished iteration {}, energy {} (match {}, reg {})'.format(it, E_, EM_, ER_))
        # output
        vt0np,vt1np,vt2np = sess.run([vt0new,vt1new,vt2new])
    return vt0np,vt1np,vt2np
