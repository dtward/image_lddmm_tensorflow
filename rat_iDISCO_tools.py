import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import vis

def nonuniformity_corrrection(xJ, J, a=1.0, p=2.0):
    
    # get nx and dx
    x0J,x1J,x2J = xJ
    nxJ = J.shape
    dxJ = (x0J[2] - x0J[1], x1J[2] - x1J[1], x2J[2] - x2J[1])
    
    # make smoothing kernel
    fJ = [np.arange(n)/n/d for n,d in zip(nxJ,dxJ)]
    
    FJ = np.meshgrid(*fJ,indexing='ij')
    Lhat = (1.0 - 2.0*a**2*(\
                         (np.cos(2.0*np.pi*FJ[0]*dxJ[0]) - 1.0)/dxJ[0]**2 \
                         + (np.cos(2.0*np.pi*FJ[1]*dxJ[1]) - 1.0)/dxJ[1]**2 \
                         + (np.cos(2.0*np.pi*FJ[2]*dxJ[2]) - 1.0)/dxJ[2]**2 \
                        ))**p
    Khat = 1.0/(Lhat**2)

    # anyway, compute histogram

    f = plt.figure()
    f2,ax2 = plt.subplots(1,2)
    nb = 100
    minlog = 100
    niter = 10
    e = 2e-1

    Jc = np.array(J)
    Jc[Jc < minlog] = minlog
    lJc = np.log(Jc)
    # normalize
    lJc = (lJc - np.mean(lJc))/np.std(lJc)
    for it in range(niter):
        binrange = np.array([np.min(lJc),np.max(lJc)])
        binrange = np.mean(binrange) + np.diff(binrange)*np.array([-1,1])/2.0*1.25
        bins = np.linspace(binrange[0],binrange[1],nb)
        hist =  np.zeros_like(bins)
        db = bins[2] - bins[1]
        width = db*1.0
        for i in range(nb):
            hist[i] = np.sum( np.exp( - (lJc.ravel() - bins[i])**2 /2.0/width**2) * np.sqrt(2.0*np.pi*width**2) )
        dhist = np.gradient(hist,db)
        ax2[0].cla()
        ax2[0].plot(bins,hist)
        ax2[1].cla()
        ax2[1].plot(bins,dhist)

        # now we sample the image  
        lJch = spi.interp1d(bins,dhist)(lJc)
        lJchs = np.sign(lJch)

        # blur it
        lJchsb = np.fft.ifftn(np.fft.fftn(lJchs) * Khat).real
        vis.imshow_slices(lJchsb,fig=f,x=xJ)

        # step

        lJc = lJc + e * lJchsb

        # normalize
        lJc = (lJc - np.mean(lJc))/np.std(lJc)
        Jc = np.exp(lJc)
        f.clf()
        vis.imshow_slices(Jc,fig=f,x=xJ)
        f.suptitle('iteration {}'.format(it))

        # update figures and end loop
        f.canvas.draw()
        f2.canvas.draw()
    return Jc
