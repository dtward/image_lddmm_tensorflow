print('importing vis')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def imshow_slices(I,x0=None,x1=None,x2=None,x=None,n=None,fig=None,clim=None,cmap=None,
                  contour0=None,contour1=None,contour2=None,contour=None,levels=None, # for contours
                  colorbar=None,colorbar_ticks=None,colorbar_ticklabels=None): # for colorbar
    ''' Draw a set of slices in 3 planes
    Mandatory argument is image I
    Optional arguments
    x0,x1,x2: sample points in direction of 0th, 1st, 2nd array index
    x is a single list of all
    n number of slices
    '''
    if n is None:
        n = 5 # default 5 slices
    if fig is None:
        fig = plt.gcf() # if no figure specified, use current
        fig.clf()
    if x is not None:
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
    if x0 is None:
        x0 = np.arange(I.shape[0])
    if x1 is None:
        x1 = np.arange(I.shape[1])
    if x2 is None:
        x2 = np.arange(I.shape[2])
    
    if clim is None:
        vmin = np.min(I)
        vmax = np.max(I)
        vmin,vmax = np.quantile(I,(0.001,0.999))
    else:
        vmin,vmax = clim
    if cmap is None:
        cmap = 'gray'
    
    # for contours
    if contour is not None:
        contour0,contour1,contour2 = contour
    
    # calculate slices
    slices0 = np.linspace(0,I.shape[0],n+2,dtype=int)[1:-1]
    slices1 = np.linspace(0,I.shape[1],n+2,dtype=int)[1:-1]
    slices2 = np.linspace(0,I.shape[2],n+2,dtype=int)[1:-1]
    axs = []
    
    # the first slice, fix the first index
    ax0 = []
    for i,s in enumerate(slices0):
        kwargs = dict()        
        if i:
            kwargs['sharex'] = ax0[0]
            kwargs['sharey'] = ax0[0]
        ax = fig.add_subplot(3,n,i+1)
        ax.imshow(I[s,:,:], extent=(x2[0],x2[-1],x1[0],x1[-1]), 
                  cmap=cmap, interpolation='none', aspect='equal', 
                  vmin=vmin, vmax=vmax)
        ax.set_xlabel('x2')
        ax.xaxis.tick_top()
        #ax.xaxis.set_label_position('top') # its better on the bottom because it can serve as a label for the next row
        if i==0:
            ax.set_ylabel('x1')
        else:
            ax.set_yticklabels([])
            
        if contour1 is not None:
            ax.contour(x2,x1,contour1[s,:,:],levels=levels,colors=['w'],linestyles=['solid'],linewidths=1,alpha=0.5)
        if contour2 is not None:
            ax.contour(x2,x1,contour2[s,:,:],levels=levels,colors=['w'],linestyles=['solid'],linewidths=1,alpha=0.5)
        ax0.append(ax)
    axs.append(ax0)
    
    # the second slice fix the second index
    ax1 = []
    for i,s in enumerate(slices1):
        kwargs = dict()        
        if i:
            kwargs['sharex'] = ax1[0]
            kwargs['sharey'] = ax1[0]
        else:
            kwargs['sharex'] = ax0[0]
        ax = fig.add_subplot(3,n,i+1+n)
        ax.imshow(I[:,s,:], extent=(x2[0],x2[-1],x0[0],x0[-1]), 
                  cmap=cmap, interpolation='none', aspect='equal', 
                  vmin=vmin, vmax=vmax)
        if contour0 is not None:
            ax.contour(x2,x0,contour0[:,s,:],levels=levels,colors=['w'],linestyles=['solid'],linewidths=1,alpha=0.5)
        if contour2 is not None:
            ax.contour(x2,x0,contour2[:,s,:],levels=levels,colors=['w'],linestyles=['solid'],linewidths=1,alpha=0.5)
        
        # no x labels necessary
        ax.set_xticklabels([])
        ax.xaxis.tick_top() # better on top so it more obviously shares the label
        
        
        if i==0:
            ax.set_ylabel('x0')
        else:
            ax.set_yticklabels([])
        
        ax1.append(ax)
    axs.append(ax1)
    
    
    # the third slice fix the third index
    ax2 = []
    for i,s in enumerate(slices2):
        kwargs = dict()        
        if i:
            kwargs['sharex'] = ax2[0]
            kwargs['sharey'] = ax2[0]
        else:
            kwargs['sharey'] = ax1[0]
        ax = fig.add_subplot(3,n,i+1+2*n)
        h = ax.imshow(I[:,:,s], extent=(x1[0],x1[-1],x0[0],x0[-1]), 
                  cmap=cmap, interpolation='none', aspect='equal', 
                  vmin=vmin, vmax=vmax)
        ax.set_xlabel('x1')        
        if i==0:
            ax.set_ylabel('x0')
        else:
            ax.set_yticklabels([])
        
        if contour0 is not None:
            ax.contour(x1,x0,contour0[:,:,s],levels=levels,colors=['w'],linestyles=['solid'],linewidths=1,alpha=0.5)
        if contour1 is not None:
            ax.contour(x1,x0,contour1[:,:,s],levels=levels,colors=['w'],linestyles=['solid'],linewidths=1,alpha=0.5)
        
        ax2.append(ax)
    axs.append(ax2)
    
    # also add colorbar
    if colorbar is not None and colorbar:
        tmp = plt.colorbar(mappable=h,ax=axs)
        if colorbar_ticks is not None:
            tmp.set_ticks(colorbar_ticks)
        if colorbar_ticklabels is not None:
            tmp.set_ticklabels(colorbar_ticklabels)
    
    
    return fig, axs
    
    