from __future__ import division
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
#import healpy as hp
from scipy.stats import binned_statistic_2d as bd2
#import healpy as hp


def gen_fast_map(l,b,title='', nside=64,cmap='viridis', norm=None):
    ip_ = hp.ang2pix(nside, l, b, lonlat=True)
    npixel  = hp.nside2npix(nside)
    map_ = np.bincount(ip_,minlength=npixel)
    map = np.log10(map_+1.)
    hp.visufunc.mollview(map,hold=True,cmap=cmap,title=title, norm=norm)


def ploth2(x=[],y=[],z=None, statistic='mean', H=None,edges=None,ax=None,bins=100,weights=None,linex=[],liney=[],func=[],xlim=None,ylim=None,xlabel=None,ylabel=None,fontsize=14,cmap='gray_r',gamma=1,invertx=False,inverty=False,interpolation='none',title=None,vmax=None,norm=None,range=None,vmin=None, vminmax_option='percentile',zero_as_blank=True,levels=None,xlogbin=False,ylogbin=False, aspect='equal'):


    if H is None:

        if range is None: range = [[np.min(x), np.max(x)], [np.min(y), np.max(y)]]
        else: range = range

        if isinstance(bins,float) or isinstance(bins,int):
            bins_t=[bins,bins]
            samebin=True
        else:
            bins_t=[bins[0],bins[1]]
            samebin=False

        bins=[[0,],[0,]]



        if xlogbin:
            bins[0]=np.logspace(np.log10(range[0][0]),np.log10(range[0][1]),bins_t[0]+1)
        else:
            bins[0]=np.linspace(range[0][0], range[0][1],bins_t[0]+1)

        if ylogbin:
            bins[1]=np.logspace(np.log10(range[1][0]),np.log10(range[1][1]),bins_t[1]+1)
        else:
            bins[1]=np.linspace(range[1][0], range[1][1],bins_t[1]+1)




        if z is None:
            sample=np.vstack([x,y]).T
            H,edges=np.histogramdd(sample,bins,weights=weights,range=range)
            xedges,yedges=edges
            extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]


        elif len(z)==len(x):
            H, xedges, yedges,_=bd2(x, y, z, statistic=statistic, bins=bins, range=range, expand_binnumbers=False)
            extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]

        else:
            raise ValueError('Z needs to be None or an array with the same length of z and y')
    else:
        H=H
        xedges,yedges=edges
        extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]

    if norm is not None:
        if norm=='max':
            H=H/np.nanmax(H)
        elif norm=='tot':
            H=H/np.nansum(H)
        elif norm=='maxrows':
            Hm=np.nanmax(H,axis=0)
            H=H/Hm
        elif norm=='totrows':
            Hm=np.nansum(H,axis=0)
            H=H/Hm
        elif norm=='maxcols':
            Hm=np.nanmax(H,axis=1)
            H=(H.T/Hm).T
        elif norm=='totcols':
            Hm=np.nansum(H,axis=1)
            H=(H.T/Hm).T
        elif norm[:10]=='percentile':
            q=float(norm[10:12])
            Hm=np.nanpercentile(H,q=q)
            H=H/Hm
        else: raise ValueError('norm option %s not recognised (valide values: max, tot, maxcols, maxrows, totcols, totrows)'%str(norm))

    if zero_as_blank:
        H=np.where(H==0,np.nan,H)

    if ax is not None:

        if vminmax_option=='percentile':
            if vmax is None: vmaxM=np.nanpercentile(H,q=95)
            else: vmaxM=np.nanpercentile(H,q=vmax)
            if vmin is None: vminM=np.nanpercentile(H,q=5)
            else: vminM=np.nanpercentile(H,q=vmin)
        elif vminmax_option=='absolute':
            if vmax is None: vmaxM=np.nanmax(H)
            else: vmaxM=vmax
            if vmin is None: vminM=np.nanmin(H)
            else: vminM=vmin
        #X,Y=np.meshgrid(xedges,yedges)


        if gamma==0: im=ax.imshow(H.T,origin='low',extent=extent, aspect=aspect,cmap=cmap,norm=LogNorm(),interpolation=interpolation,vmax=vmaxM,vmin=vminM)
        else: im=ax.imshow(H.T,origin='low',extent=extent, aspect=aspect,cmap=cmap,norm=PowerNorm(gamma=gamma),interpolation=interpolation,vmax=vmaxM,vmin=vminM)
        #if gamma==0: im=ax.pcolormesh(X,Y,H.T, cmap=cmap,norm=LogNorm(),vmax=vmax,vmin=vmin)
        #else: im=ax.pcolormesh(X,Y,H.T, cmap=cmap,norm=PowerNorm(gamma=gamma),vmax=vmax,vmin=vmin)

        if len(linex)>0:
            for c in linex:
                ax.plot([c,c],[yedges[0],yedges[-1]],c='blue',lw=2,zorder=1000)

        if len(liney)>0:
            for c in liney:
                ax.plot([xedges[0],xedges[-1]],[c,c],c='blue',lw=2,zorder=1000)

        if len(func)>0:
            if xlim is not None: xf=np.linspace(xlim[0],xlim[1])
            else: xf=np.linspace(range[0][0],range[0][1])
            for f in func:
                ax.plot(xf,f(xf),c='blue',lw=2,zorder=1000)

        if levels is not None:
            ax.contour(H.T,origin='lower',extent=extent,levels=levels)


        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)

        if invertx: ax.invert_xaxis()
        if inverty: ax.invert_yaxis()

        if xlabel is not None: ax.set_xlabel(xlabel,fontsize=fontsize,labelpad=2)
        if ylabel is not None: ax.set_ylabel(ylabel,fontsize=fontsize,labelpad=2)
        if title is not None: ax.set_title(str(title),fontsize=fontsize)
    else:
        im=None

    edges=(xedges,yedges)
    return H,edges,im

def ploth1(x=[],ax=None,bins=100,weights=None,linex=[],liney=[],xlim=None,ylim=None,xlabel=None,ylabel=None,fontsize=14,cmap='gray_r',invertx=False,inverty=False,title=None,norm=None,range=None,label='',mode='curve',cumulative=False):

    bins=bins



    if range is None:
        extent=[[np.min(x),np.max(x)]]
    else:
        range=range



    if weights is None: weights=np.ones_like(x)



    H,edges=np.histogram(x,bins,weights=weights,range=range)



    if norm is not None:
        if norm=='max': H=H/np.max(H)
        elif norm=='tot': H=H/np.sum(H)
        else: pass

    if ax is not None:

        if cumulative:
            xplot=np.sort(x)
            yplot=np.arange(len(xplot))/len(xplot)
            im=ax.plot(xplot,yplot,label=label)
        else:
            if mode=='curve':
                xplot=0.5*(edges[:-1]+edges[1:])
                im=ax.plot(xplot,H,label=label)
            elif mode=='step':
                im=ax.step(edges[:-1],H,label=label)

        if len(linex)>0:
            for c in linex:
                ax.plot([c,c],[yedges[0],yedges[-1]],c='blue',lw=2,zorder=1000)

        if len(liney)>0:
            for c in liney:
                ax.plot([xedges[0],xedges[-1]],[c,c],c='blue',lw=2,zorder=1000)

        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)

        if invertx: ax.invert_xaxis()
        if inverty: ax.invert_yaxis()

        if xlabel is not None: ax.set_xlabel(xlabel,fontsize=fontsize,labelpad=2)
        if ylabel is not None: ax.set_ylabel(ylabel,fontsize=fontsize,labelpad=2)
        if title is not None: ax.set_title(str(title),fontsize=fontsize)
    else:
        im=None

    return H,edges,im
