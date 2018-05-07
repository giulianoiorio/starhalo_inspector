from __future__ import division
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import healpy as hp

def gen_fast_map(l,b,title='', nside=64,cmap='viridis'):
    ip_ = hp.ang2pix(nside, l, b, lonlat=True)
    npixel  = hp.nside2npix(nside)
    map_ = np.bincount(ip_,minlength=npixel)
    map = np.log10(map_+1.)
    hp.visufunc.mollview(map,hold=True,cmap=cmap,title=title)


def ploth2(x=[],y=[],H=None,edges=None,ax=None,bins=100,weights=None,linex=[],liney=[],func=[],xlim=None,ylim=None,xlabel=None,ylabel=None,fontsize=14,cmap='gray_r',gamma=1,invertx=False,inverty=False,interpolation='none',title=None,vmax=None,norm=None,range=None,vmin=None):
	
	if H is None:
	
		sample=np.vstack([x,y]).T
		
		if isinstance(bins,float) or isinstance(bins,int): bins=bins
		else:
			bins=[bins[0],bins[1]]
		
	
		if range is None: range=[[np.min(x),np.max(x)],[np.min(y),np.max(y)]]
		else: range=range

		H,edges=np.histogramdd(sample,bins,weights=weights,range=range)
		xedges,yedges=edges
		extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]
	else:
		H=H
		xedges,yedges=edges
		extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]
	if norm is not None: 
		if norm=='max': H=H/np.max(H)
		elif norm=='tot': H=H/np.sum(H)
		else: pass
	
	
	if ax is not None:
		
		if vmax is None: vmax=np.max(H)
		if vmin is None: vmin=np.min(H)
		

		if gamma==0: norm=LogNorm()
		else: norm=PowerNorm(gamma=gamma)
		

		if gamma==0: im=ax.imshow(H.T,origin='low',extent=extent, aspect='auto',cmap=cmap,norm=LogNorm(),interpolation=interpolation,vmax=vmax)
		else: im=ax.imshow(H.T,origin='low',extent=extent, aspect='auto',cmap=cmap,norm=PowerNorm(gamma=gamma),interpolation=interpolation,vmax=vmax)
	
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
