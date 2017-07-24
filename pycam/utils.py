from __future__ import division
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.io.fits as ft
import numpy as np
from time import gmtime, strftime
import os
from math import cos,sin
from pycam.cutils import calc_m
import matplotlib.pyplot as  plt
from pycam.plot import ploth2
from scipy.ndimage import median_filter
import os 
import shutil
import roteasy as rs 

def qfunc_exp(m,q0,qinf,rq,eta=1):

	ep=(1-np.sqrt(m*m+rq*rq)/rq)

	return qinf - (qinf-q0)*np.exp(ep)

def qfunc_tan(m,q0=1,qinf=1,rq=1,eta=1):

	tm = np.tanh((m-rq)/eta)
	t0 = np.tanh((-rq)/eta)
	C = (qinf-q0)/(1-t0)


	return qinf+C*(tm-1)


def mad(arr,axis=None):
	""" Median Absolute Deviation: a "Robust" version of standard deviation.
	   Indices variabililty of the sample.
	    https://en.wikipedia.org/wiki/Median_absolute_deviation
	"""
	med = np.median(arr,axis=axis)
	return med,1.4826*np.median(np.abs(arr - med),axis=axis)

def cartesian(*arrays):
	"""
	Make a cartesian combined arrays from different arrays e.g.
	al=np.linspace(0.2,5,50)
	ql=np.linspace(0.1,2,50)
	par=cartesian(al,ql)
	:param arrays:
	:return:
	"""
	mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
	dim = len(mesh)  # number of dimensions
	elements = mesh[0].size  # number of elements, any index will do
	flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
	reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
	return reshape

def skysep(x,y,xr,yr,unit='deg'):
	"""
	Calculate the angular separation between a list of point on the sky with respect to a reference point.
	The calculation is made taking into account the spherical geometry.
	@x: X (e.g. ra,l) positions in unit specified by unit.
	@y: Y (e.g. dec,b) positions in unit specified by unit.
	@xr: X (e.g. ra,l) position of the reference point in unit specified by unit.
	@yr: Y (e.g. dec,b) position of the reference point in unit specified by unit.
	@unit: Unit (deg for degree; rad for radians) of the input and output angles 
	"""
	x=np.array(x)
	y=np.array(y)
	
	if unit[:1]=='d': cost=2*np.pi/360.
	elif unit[:1]=='r': cost=1
	
	siny_ref=np.sin(cost*yr)
	cosy_ref=np.cos(cost*yr)
	
	siny=np.sin(cost*y)
	cosy=np.cos(cost*y)
	
	cosdx=np.cos( cost*(x-xr))
	
	res=np.arccos(siny_ref*siny+cosy_ref*cosy*cosdx)
	
	return res/cost

def gerr(g,case=0):
	"""
	Estimate gmag error following http://www.cosmos.esa.int/web/gaia/photometric-performance-straylight-dependence
	@g: g mag observed (NB not corrected for the extinction)
	@case: from 0 to 5, 0 is minimum estimate of the uncertainty, 5 the maximum
	"""
	cost=10**(0.4*(12-15))
	
	z=10**(0.4*(g-15))
	z[z<cost]=cost
	
	if case==0: poly_coeff=[0.01235,1.8631,0.0002230]
	elif case==5: poly_coeff=[0.18974,1.8631,0.0002294]
	elif case==1: poly_coeff=[0.03273,1.8624,0.0003082]
	elif case==2: poly_coeff=[0.04895,1.86633,0.0001985]
	elif case==3: poly_coeff=[0.08130,1.8636,0.0001658]
	elif case==4: poly_coeff=[0.13459,1.8639,0.0001403]
	
	p=np.poly1d(poly_coeff)
	
	return 1e-3*p(z)

def filter(tab,index,cols=()):
	"""
	Define a new fits table filtered with the index:
	@tab: a astropy fits table object
	@index: Boolean numpy array
	@cols: name of the cols to transfer to the new table, default all the columns.
	"""

	if len(cols)>=1: cols=cols
	else: cols=tab.columns.names


	#check dimension
	if len(index)!=len(tab.data[cols[0]]): raise ValueError
	
	i=0
	newcols=[]
	for name in cols:
		colformat=tab.columns.formats[i]
		col_tmp=ft.Column(name=name,array=tab.data[name][index],format=colformat)
		newcols.append(col_tmp)
		i+=1
	
	new_colsdef=ft.ColDefs(newcols)
	
	tabo=ft.BinTableHDU.from_columns(new_colsdef)
	
	
	tabo.header['COMMENT']='Filtered the %s'%strftime("%Y-%m-%d %H:%M:%S", gmtime())
	
	return tabo

def mkdir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

def arg_filter2(tab,cut={},extracut=[],struct=None,lneg=False):

	idx=np.array([True,]*len(tab['ra']))

	for key in cut:
		cutbox=cut[key]
		if cutbox[0] is not None: idx*=tab[key]>=cutbox[0]
		if cutbox[1] is not None: idx*=tab[key]<=cutbox[1]
	
	for s in extracut:
		tab,boxin,boxsup=s
		if boxin is not None: idx*=tab>=boxin
		if boxsup is not None: idx*=tab<=boxsup
		
	if struct is None: pass
	elif struct=='all':
		ltmp=np.where(tab['l']<0,360+tab['l'],tab['l'])
		btmp=tab['b']
		idxtpm=np.array([False,]*len(btmp))
		for s in struct_list:
				l_lim=struct_list[s][0]
				b_lim=struct_list[s][1]
				idxtpm+=(ltmp>=l_lim[0])&(ltmp<=l_lim[1])&(btmp>=b_lim[0])&(btmp<=b_lim[1])
		idx*=np.logical_not(idxtpm)
	else:
		if lneg: ltmp=np.where(tab['l']<0,360+tab['l'],tab['l'])
		else: ltmp=tab['l']
		btmp=tab['b']
		idxtpm=np.array([False,]*len(btmp))
		for s in struct:
			if isinstance(s,str):
					if s in struct_list: st=struct_list[s]
					elif s in struct_list_object: st=struct_list_object[s]
					else: pass
			else: st=s
			if len(st)==3:
				lmg,bmg,rmc=st
				skysep_dmg=skysep(ltmp,btmp,lmg,bmg)
				idxtpm+=skysep_dmg<rmc
			elif len(st)==4:
				lmin,lmax,bmin,bmax=st
				idxtpm+=(ltmp>lmin)&(ltmp<lmax)&(btmp>bmin)&(btmp<bmax)
		idx*=np.logical_not(idxtpm)
	return idx

def arg_filter(tab,ampcut=[None,None],gcut=[None,None],colcut=[None,None],ecut=[None,None],ncut=[None,None],pmcut=[None,None],dcut=[None,None],bcut=[None,None],lcut=[None,None],racut=[None,None],deccut=[None,None],bgcut=[None,None],zcut=[None,None],rlmc=0,rsmc=0,rm3=0,rm5=0,Mg=0.55,footprint=None,lneg=True,struct=None,xsun=8):
	"""
	Find the index of a tab that are between the selecting box selected:
	@tab; an open fits table
	@ampcut: Cut on the amplitude 'amp', 
	@gcut: Cut in the corrected g magnitude 'g'
	@colcut: Cut in color J-G (corrected), key 'jg
	@ecut: Cut in excess noise 'extraerr'
	@nobs: cut in number of observations 'nobs'
	@rlmc: cut the large magellanic cloud within this radius in degree
	@slmc: cut the small magellanic cloud within this radius in degree
	@rm3: cut m3 within this radius in degree
	@rm5: cut m5 within this radius in degree
	@lneg: if True l goes from -180 to 180, otherwise it goes from 0 to 360
	: return a boolean numpy array
	"""
	idx=np.array([True,]*len(tab['ra']))
	

	if footprint is not None: 
		if lneg: idx*=footprint( tab['l']  ,tab['b'])
		else: idx*=footprint( np.where(tab['l']>180.,tab['l']-360.,tab['l'])  ,tab['b'])
		
	if (ampcut[0] is None) and (ampcut[1] is None) and (gcut[0] is None) and (gcut[1] is None) and (colcut[0] is None) and (colcut[1] is None) and (ecut[0] is None) and (ecut[1] is None) and (ncut[0] is None) and (ncut[1] is None) and (pmcut[0] is None) and (pmcut[1] is None) and (dcut[0] is None) and (dcut[1] is None) and (bcut[0] is None) and (bcut[1] is None) and (lcut[0] is None) and (lcut[1] is None) and (racut[0] is None) and (racut[1] is None) and (deccut[0] is None) and (deccut[1] is None):
		pass
	else:
		if ampcut[0] is not None: idx*=tab['amp']>=ampcut[0]
		if ampcut[1] is not None: idx*=tab['amp']<=ampcut[1]
		#print('ampcut',np.sum(idx))
		if gcut[0] is not None: idx*=tab['g']>=gcut[0]
		if gcut[1] is not None: idx*=tab['g']<=gcut[1]
		#print('gcut',np.sum(idx))
		if colcut[0] is not None: idx*=tab['jg']>=colcut[0]
		if colcut[1] is not None: idx*=tab['jg']<=colcut[1]
		#print('colcut',np.sum(idx))
		if ecut[0] is not None: idx*=tab['extraerr']>=ecut[0]
		if ecut[1] is not None: idx*=tab['extraerr']<=ecut[1]
		#print('ecut',np.sum(idx))
		if ncut[0] is not None: idx*=tab['nobs']>=ncut[0]
		if ncut[1] is not None: idx*=tab['nobs']<=ncut[1]
		#print('ncut',np.sum(idx))
		if ncut[0] is not None: idx*=tab['nobs']>=ncut[0]
		if ncut[1] is not None: idx*=tab['nobs']<=ncut[1]
		#print('ncut',np.sum(idx))
		if pmcut[0] is not None: idx*= np.sqrt( (tab['pmra'])**2 + tab['pmdec']**2    )>=pmcut[0]
		if pmcut[1] is not None: idx*=np.sqrt( (tab['pmra'])**2 + tab['pmdec']**2    )<=pmcut[1]
		#print('pmcut',np.sum(idx))
		if dcut[0] is not None: 
			try: 
				distance=10**(0.2*(tab['g']-Mg+5) )/1000.
			except:
				distance=tab['distance']
			idx*= distance >=dcut[0]
		if dcut[1] is not None: 
			try: 
				distance=10**(0.2*(tab['g']-Mg+5) )/1000.
			except:
				distance=tab['distance']
			idx*= distance <=dcut[1]
		if bcut[0] is not None: idx*=np.abs(tab['b'])>=bcut[0]
		if bcut[1] is not None: idx*=np.abs(tab['b'])<=bcut[1]
		if lcut[0] is not None:
			l_tmp=np.where(tab['l']<0,tab['l']+360,tab['l'])
			idx*=l_tmp>=lcut[0]
		if lcut[1] is not None:
			l_tmp=np.where(tab['l']<0,tab['l']+360,tab['l'])
			idx*=l_tmp<=lcut[1]
		if racut[0] is not None:
			idx*=tab['ra']>=racut[0]
		if racut[1] is not None:
			idx*=tab['ra']<=racut[1]
		if deccut[0] is not None: idx*=tab['dec']>=deccut[0]
		if deccut[1] is not None: idx*=tab['dec']<=deccut[1]
		
		if (bgcut[0] is None) and (bgcut[1] is None) and (zcut[0] is None) and (zcut[1] is None): pass
		else:
			rg,zg=obs_to_cyl(tab['g'],tab['l'],tab['b'],Mg=Mg,xsun=xsun)
			zg_abs=np.abs(zg)
			theta_abs=np.arctan(zg_abs/rg)*(360/(2*np.pi))
			if bgcut[0] is not None: idx*=theta_abs>=bgcut[0]
			if bgcut[1] is not None: idx*=theta_abs<=bgcut[1]
			if zcut[0] is not None: idx*=zg_abs>=zcut[0]
			if zcut[1] is not None: idx*=zg_abs<=zcut[1]

	if rlmc>0: 
		if lneg: lmg=280.4653-360.
		else: lmg=280.4653
		bmg=-32.8883
		l=tab['l']
		b=tab['b']
		skysep_dmg=skysep(l,b,lmg,bmg)
		idx*=skysep_dmg>rlmc
	if rsmc>0: 
		if lneg: lmg=302.7969-360.
		else: lmg=302.7969
		bmg=-44.2992 
		l=tab['l']
		b=tab['b']
		skysep_dmg=skysep(l,b,lmg,bmg)
		idx*=skysep_dmg>rsmc
	if rm3>0:
		rm3dist=10.4
		lmg=42.2
		bmg=78.7
		l=tab['l']
		b=tab['b']
		skysep_dmg=skysep(l,b,lmg,bmg)
		idx*=skysep_dmg>rm3
	if rm5>0:
		lmg=3.87
		bmg=46.80
		l=tab['l']
		b=tab['b']
		skysep_dmg=skysep(l,b,lmg,bmg)
		idx*=skysep_dmg>rm5


	if struct is None: pass
	elif struct=='all':
		ltmp=np.where(tab['l']<0,360+tab['l'],tab['l'])
		btmp=tab['b']
		idxtpm=np.array([False,]*len(btmp))
		for s in struct_list:
				l_lim=struct_list[s][0]
				b_lim=struct_list[s][1]
				idxtpm+=(ltmp>=l_lim[0])&(ltmp<=l_lim[1])&(btmp>=b_lim[0])&(btmp<=b_lim[1])
		idx*=np.logical_not(idxtpm)
	else:
		if lneg: ltmp=np.where(tab['l']<0,360+tab['l'],tab['l'])
		else: ltmp=tab['l']
		btmp=tab['b']
		idxtpm=np.array([False,]*len(btmp))
		for s in struct:
			if isinstance(s,str):
					if s in struct_list: st=struct_list[s]
					elif s in struct_list_object: st=struct_list_object[s]
					else: pass
			else: st=s
			if len(st)==3:
				lmg,bmg,rmc=st
				skysep_dmg=skysep(ltmp,btmp,lmg,bmg)
				idxtpm+=skysep_dmg<rmc
			elif len(st)==4:
				lmin,lmax,bmin,bmax=st
				idxtpm+=(ltmp>lmin)&(ltmp<lmax)&(btmp>bmin)&(btmp<bmax)
		idx*=np.logical_not(idxtpm)
	return idx
	

def make_fits(dict,outname=None,header_key={}):
	'''
	Make a fits table from a numpy array
	args must be dictionary containing the type  and the columnf of the table, e.g.
	{'l':(col1,'D'),'b':(col2,'D')}
	'''
	
	col=[]
	for field in dict:
		if len(dict[field])==2:
			format=dict[field][1]
			array=dict[field][0]
		else:
			format='D'
			array=dict[field]
		
		col.append(ft.Column(name=field,format=format,array=array))
		
	cols = ft.ColDefs(col)
	tab = ft.BinTableHDU.from_columns(cols)
	for key in header_key:
		item=header_key[key]
		if item is None: tab.header[key]=str(item)
		else: tab.header[key]=item

		
	if outname is not None: tab.writeto(outname,clobber=True)
		
	return tab

def addcol_fits(fitsfile,newcols=({},),idtable=1,outname=None):
	"""
	fitsfile: name of fitsfile or table hdu
	newxols: a tuole of dics with keyword 'name', 'format' and 'array'
	idtable: the id of the table to modify
	outname: if not None the name of the outputted fits file
	"""
	
	if idtable is not None:
	
		try: 
			orig_table = ft.open(fitsfile)[idtable].data
		except:
			orig_table = fitsfile[idtable].data
	
	else:
		
		try: 
			orig_table = ft.open(fitsfile).data
		except:
			orig_table = fitsfile.data
	
	orig_cols = orig_table.columns
	
	col_list=[]
	for dic in newcols:
		coltmp=ft.Column(name=dic['name'], format=dic['format'], array=dic['array'])
		col_list.append(coltmp)
	new_cols=ft.ColDefs(col_list)
	hdu = ft.BinTableHDU.from_columns(orig_cols + new_cols)
	
	if outname is not None: hdu.writeto(outname,clobber=True)
	
	return hdu
	
	
def dist_to_g(dist,Mg):
	"""
	Transform distance in kpc to a observed g mangniture for a star of absolute magnitude Mg
	:param dist:  Distance in kpc
	:param Mg:  Aboslute magnitude or a distribution of absolute magnitudes
	:return: mg
	"""

	addend=np.log10(dist)+2
	g=Mg+5*addend


	return g

def mag_to_dist(mag,Mg):
	"""
	Transform observed magnitude  to distance in kpc  for a star of absolute magnitude Mg
	:param mag:  magnitude
	:param Mg:  Aboslute magnitude
	:return: dist [kpc]
	"""
	dmod=mag-Mg+5
	return 10**(0.2*dmod-3)

'''
def obs_to_m(mag,l,b,Mg,xsun=8,q=1,p=1,i=0):
	"""
    Return the m-value of an ellipsoid from the observ magnitude and galactic coordinate.
    if q=1 and p=1, the ellipsoid is indeed a sphere and m=r
	:param mag: observed magnitude.
	:param l: galactic longitude.
	:param b: galactic latitude.
	:param Mg: Absolute magnitude.
	:param xsun: Distance of the sun from the galactic centre.
	:param q: Flattening along the z-direction, q=1 no flattening.
	:param p: Flattening along the y-direction, p=1 no flattening.
	:return: the m-value for an ellipsoid m^2=x^2+(y/p)^2+(z/q)^2.
	"""
	cost=0.017453292519943295769 #From degree to rad
	d=mag_to_dist(mag,Mg=Mg)
	b=b*cost
	l=l*cost
	cb=np.cos(b)
	sb=np.sin(b)
	cl=np.cos(l)
	sl=np.sin(l)
	x=(xsun-d*cb*cl)
	y=(d*cb*sl)
	z=(d*sb)

	if i!=0:
		cord=rotate(np.array([x,y,z]).T,beta=i)
		x=cord[:,0]
		y=cord[:,1]
		z=cord[:,2]

	y=y/p
	z=z/q

	return np.sqrt(x*x+y*y+z*z)
'''

def obs_to_m(mag,l,b,Mg,xsun=8,q=1.0,qinf=1.0,rq=10.0,p=1.0,alpha=0,beta=0,gamma=0,ax='zyx'):
	"""
    Return the m-value of an ellipsoid from the observ magnitude and galactic coordinate.
    if q=1 and p=1, the ellipsoid is indeed a sphere and m=r
	:param mag: observed magnitude.
	:param l: galactic longitude.
	:param b: galactic latitude.
	:param Mg: Absolute magnitude.
	:param xsun: Distance of the sun from the galactic centre.
	:param q: Flattening along the z-direction, q=1 no flattening.
	:param p: Flattening along the y-direction, p=1 no flattening.
	:return: the m-value for an ellipsoid m^2=x^2+(y/p)^2+(z/q)^2.
	"""
	cost=0.017453292519943295769 #From degree to rad
	d=mag_to_dist(mag,Mg=Mg)
	b=b*cost
	l=l*cost
	cb=np.cos(b)
	sb=np.sin(b)
	cl=np.cos(l)
	sl=np.sin(l)
	x=(xsun-d*cb*cl)
	y=(d*cb*sl)
	z=(d*sb)
	
	i=np.abs(alpha)+np.abs(beta)+np.abs(gamma)
	if i!=0:
		cord=rotate(cord=np.array([x,y,z]).T, angles=(alpha,beta,gamma), axes=ax, reference='lh' )
		x=cord[:,0]
		y=cord[:,1]
		z=cord[:,2]

	

	if q==qinf:
		y=y/p
		z=z/q
		m=np.sqrt(x*x+y*y+z*z)
	else:
		m=np.array(calc_m(x,y,z, q, qinf, rq, p))


	return m

def obs_to_cyl(mag,l,b,Mg,xsun=8,negative_r=False):
	"""
    Return the R,Z coordinate from the mag, l and b
	:param mag: observed magnitude.
	:param l: galactic longitude.
	:param b: galactic latitude.
	:param Mg: Absolute magnitude.
	:param xsun: Distance of the sun from the galactic centre.
	:return: the r on the plane of the galaxy and z.
	"""
	cost=0.017453292519943295769 #From degree to rad
	d=mag_to_dist(mag,Mg=Mg)
	b=b*cost
	l=l*cost
	cb=np.cos(b)
	sb=np.sin(b)
	cl=np.cos(l)
	sl=np.sin(l)
	x=(xsun-d*cb*cl)
	y=(d*cb*sl)
	z=(d*sb)

	if negative_r:
		sign=np.where(x==0,1,np.sign(x))
		return np.sqrt(x*x+y*y)*sign,z
	else:
		return np.sqrt(x*x+y*y),z

def obs_to_xyz(mag,l,b,Mg,xsun=8):
	"""
    Return the R,Z coordinate from the mag, l and b
	:param mag: observed magnitude.
	:param l: galactic longitude.
	:param b: galactic latitude.
	:param Mg: Absolute magnitude.
	:param xsun: Distance of the sun from the galactic centre.
	:return: the x,y,z cord in the galactic plane
	"""
	cost=0.017453292519943295769 #From degree to rad
	d=mag_to_dist(mag,Mg=Mg)
	b=b*cost
	l=l*cost
	cb=np.cos(b)
	sb=np.sin(b)
	cl=np.cos(l)
	sl=np.sin(l)
	x=(xsun-d*cb*cl)
	y=(d*cb*sl)
	z=(d*sb)

	return x,y,z


def rotate(cord,alpha=0,beta=0,gamma=0,system='rh'):
	"""
    Rotata a frame of reference following the zyz formalism. The first rotation is around the z
    axis with angle alpha, the second rotation is around the new y axis with angle beta, the third rotation
    is around the new z axis with angle gamma. Positive angle means anti-clockwise rotation, Negative clockwise.
    :param cord: Nx3 array with the coordinates
    :param alpha: First rotation around z axis
    :param beta: Second rotation around (new) y axis
    :param gamma: Third rotation around (new) z axis
    :param system: rh for right-hand frame of reference, lh for left-hand.
    :return: Nx3 array with the rotated coordinates.
    """
	cost1=2*np.pi/360.
	if system=='rh': cost2=1
	elif system=='lf': cost2=-1
	else: raise ValueError('Wrong frame formalism')

	a=alpha*cost1*cost2
	b=beta*cost1*cost2
	g=gamma*cost1*cost2
	cg=np.cos(g)
	sg=np.sin(g)
	ca=np.cos(a)
	sa=np.sin(a)
	cb=np.cos(b)
	sb=np.sin(b)
	cord_n=np.zeros_like(cord)
	cord_n[:,0]=(cg*cb*ca-sg*sa)*cord[:,0] + (cg*cb*sa+sg*ca)*cord[:,1] - (cg*sb)*cord[:,2]
	cord_n[:,1]=-(sg*cb*ca+cg*sa)*cord[:,0] + (-sg*cb*sa+cg*ca)*cord[:,1] + (sg*sb)*cord[:,2]
	cord_n[:,2]=(sb*ca)*cord[:,0] + (sb*sa)*cord[:,1] + (cb)*cord[:,2]
	return cord_n


def rotate_xyz_old(cord,alpha=0,beta=0,gamma=0,xoff=0,yoff=0,zoff=0,system='rh'):
	"""
    Rotata a frame of reference following the xyz formalism. The first rotation is around the x
    axis with angle alpha, the second rotation is around the new y axis with angle beta, the third rotation
    is around the new z axis with angle gamma. Positive angle means anti-clockwise rotation, Negative clockwise.
	:param cord: Nx3 array with the coordinates
	:param alpha:  First rotation around x axis
	:param beta:  Second rotation around (new) y axis
	:param gamma: Third rotation around (new) z axis
	:param xoff:  X offset
	:param yoff:  Y offset
	:param zoff:  Z offset
	:param system: rh (for right-hand) or lh (for left-hand)
	:return:
	"""
	cost1=2*np.pi/360.
	if system=='rh': cost2=1
	elif system=='lh': cost2=-1
	else: raise ValueError('Wrong frame formalism')

	a=alpha*cost1*cost2
	b=beta*cost1*cost2
	g=gamma*cost1*cost2
	ca=cos(a)
	sa=sin(a)
	cb=cos(b)
	sb=sin(b)
	cg=cos(g)
	sg=sin(g)
	cord_n=np.zeros_like(cord)
	
	#Questo qui è vecchio e sbagliato, (facevo le moltiplicazioni per righa e non per colornna)
	#cord_n[:,0]=cb*cg*cord[:,0]-cb*sg*cord[:,1]+sb*cord[:,2]-xoff
	#cord_n[:,1]=(sa*sb*cg+ca*sg)*cord[:,0]+(ca*cg-sa*sb*sg)*cord[:,1]-sa*cb*cord[:,2]-yoff
	#cord_n[:,2]=(sa*sg-ca*sb*cg)*cord[:,0]+(ca*sb*sg+sa*cg)*cord[:,1]+ca*cb*cord[:,2]-zoff
	
	cord_n[:,0]=cb*cg*cord[:,0]+(sa*sb*cg+ca*sg)*cord[:,1]+(sa*sg-ca*sb*cg)*cord[:,2]-xoff
	cord_n[:,1]=-cb*sg*cord[:,0]+(ca*cg-sa*sb*sg)*cord[:,1]-(ca*sb*sg+sa*cg)*cord[:,2]-yoff
	cord_n[:,2]=sb*cord[:,0]-sa*cb*cord[:,1]+ca*cb*cord[:,2]-zoff
	
	return cord_n

def rot_x(a):
	"""
	Rotaiton matrix around the x axis
	:param a: angle of rotation in radiant
	"""
	#a rad
	
	ca=cos(a)
	sa=sin(a)
	
	return np.array([[1,0,0],[0,ca,sa],[0,-sa,ca]])

def rot_y(b):
	"""
	Rotaiton matrix around the y axis
	:param b: angle of rotation in radiant
	"""
	cb=cos(b)
	sb=sin(b)
	
	return np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]])

def rot_z(g):
	"""
	Rotaiton matrix around the z axis
	:param g: angle of rotation in radiant
	"""
	cg=cos(g)
	sg=sin(g)
	
	return np.array([[cg,sg,0],[-sg,cg,0],[0,0,1]])

dic_rot={'x':rot_x,'y':rot_y,'z':rot_z}

def rotate_xyz(cord,alpha=0,beta=0,gamma=0,xoff=0,yoff=0,zoff=0,system='rh',kind='zyz'):
	"""
    Rotata a frame of reference following the row formalism (x',y',z')=(x,y,z) R_rot. 
	The order od rotation is given by kind, so if kind=xyz
	the first rotation is around x axis with angle alpha,
	the second rotation is around the y axis with angle beta,
	the tird rotation is aroud z axis with angle beta
	Positive angle means anti-clockwise rotation, Negative clockwise.
	:param cord: Nx3 array with the coordinates
	:param alpha:  First rotation angle
	:param beta:  Second rotation angle around (new) axis
	:param gamma: Third rotation around  around (new)  axis
	:param xoff:  X offset
	:param yoff:  Y offset
	:param zoff:  Z offset
	:param system: rh (for right-hand) or lh (for left-hand)
	:return:
	"""
	cost1=2*np.pi/360.
	if system=='rh': cost2=1
	elif system=='lh': cost2=-1
	else: raise ValueError('Wrong frame formalism')

	a=alpha*cost1*cost2
	b=beta*cost1*cost2
	g=gamma*cost1*cost2
	
	mf=np.dot(dic_rot[kind[0]](a),dic_rot[kind[1]](b))
	mff=np.dot(mf,dic_rot[kind[2]](g))
	
	cord_n=np.zeros_like(cord,dtype=float)
	
	cord_n[:,0]=mff[0,0]*cord[:,0]+mff[1,0]*cord[:,1]+mff[2,0]*cord[:,2]-xoff
	cord_n[:,1]=mff[0,1]*cord[:,0]+mff[1,1]*cord[:,1]+mff[2,1]*cord[:,2]-yoff
	cord_n[:,2]=mff[0,2]*cord[:,0]+mff[1,2]*cord[:,1]+mff[2,2]*cord[:,2]-zoff
	
	return cord_n
	
	
def rotate_xy(cord,alpha=0,beta=0,xoff=0,yoff=0,zoff=0,system='rh'):
	"""
    Rotata a frame of reference following the xy formalism. The first rotation is around the x
    axis with angle alpha, the second rotation is around the new y axis with angle beta. Positive angle means anti-clockwise rotation, Negative clockwise.
	:param cord: Nx3 array with the coordinates
	:param alpha:  First rotation around x axis
	:param beta:  Second rotation around (new) y axis
	:param xoff:  X offset
	:param yoff:  Y offset
	:param zoff:  Z offset
	:param system: rh (for right-hand) or lh (for left-hand)
	:return:
	"""
	cost1=2*np.pi/360.
	if system=='rh': cost2=1
	elif system=='lh': cost2=-1
	else: raise ValueError('Wrong frame formalism')

	a=alpha*cost1*cost2
	b=beta*cost1*cost2
	ca=cos(a)
	sa=sin(a)
	cb=cos(b)
	sb=sin(b)
	cord_n=np.zeros_like(cord)
	cord_n[:,0]=cb*cord[:,0]+sb*cord[:,2]-xoff
	cord_n[:,1]=sa*sb*cord[:,0]+ca*cord[:,1]-sa*cb*cord[:,2]-yoff
	cord_n[:,2]=-ca*sb*cord[:,0]+sa*cord[:,1]+ca*cb*cord[:,2]-zoff

	return cord_n

def offset(cord,xoff=0,yoff=0,zoff=0):
	"""
	Move to a new frame of reference with a certain offset
	:param xoff: X offset
	:param yoff: Y offset
	:param zoff: Z offset
	:return:
	"""
	cord_n=np.zeros_like(cord)
	cord_n[:,0]-=xoff
	cord_n[:,1]-=yoff
	cord_n[:,2]-=zoff

	return cord_n


def arg_filter_hotpixel(tab,anglebin=0.5,window_size=3,toll=10,plot=True,tab_optional=None,outdir=''):
	"""
	Find hot pixel with the median filter technique
	:tab: fits tab
	:anglebin: Binning angle in degree
	:window_size: window size of the median filter in pixel
	:toll: tollerance to find the hot pixels
	"""

	idx=np.array([False,]*len(tab['ra']))

	binl=360./anglebin
	binb=180./anglebin
	bins=(binl,binb)

	l=tab['l']
	b= tab['b']


	H,edge,_=ploth2(x=tab['l'],y=tab['b'],range=((0,360),(-90,90)),bins=bins)
	xedges,yedges=edge


	blurred = median_filter(H, size=window_size)
	difference = H - blurred
	il,jb=difference.shape


	if plot:
		dir=outdir+'/hotpixels_plots'
		if tab_optional is not None:
			lc=tab_optional['l']
			bc=tab_optional['b']
		if not os.path.exists(dir):
			os.mkdir(dir)
		else:
			shutil.rmtree(dir)
			os.mkdir(dir)

	hpcord=[]
	cont=0

	for i in range(il):
		for j in range(jb):
			if difference[i,j]>=toll:
				hpcord.append([xedges[i],xedges[i+1],yedges[j],yedges[j+1]])
				linf,lsup,binf,bsup=hpcord[-1]
				idxstars=(l>=linf)&(l<=lsup)&(b>=binf)&(b<=bsup)
				idx+=idxstars
				if plot:
					fig=plt.figure()
					axot=fig.add_subplot(1,1,1)
					axot.scatter(l[idxstars],b[idxstars],c='red',zorder=1000)
					if tab_optional is not None:
						idxstarsc=(lc>=linf)&(lc<=lsup)&(bc>=binf)&(bc<=bsup)
						axot.scatter(lc[idxstarsc],bc[idxstarsc],c='blue')
					fig.savefig(dir+'/hp_%i.png'%cont)
				cont+=1
	
	idx=np.logical_not(idx)
	
	if plot:
		####Plot
		fig2=plt.figure()
		ax1bb=fig2.add_subplot(2,1,1)
		ax2bb=fig2.add_subplot(2,1,2)
		
		H,edge,_=ploth2(x=tab['l'],y=tab['b'],range=((0,360),(-90,90)),bins=50)
		xedges,yedges=edge
		extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]
		ax1bb.imshow(np.log10(H.T),origin='lower',extent=extent,interpolation='nearest')


		H,edge,_=ploth2(x=tab['l'][idx],y=tab['b'][idx],range=((0,360),(-90,90)),bins=50)
		xedges,yedges=edge
		extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]
		ax2bb.imshow(np.log10(H.T),origin='lower',extent=extent,interpolation='nearest')
		fig2.savefig(outdir+'/hp_compare.png')

		###

	return idx,hpcord


#Calc volume fraction

#Aux
def _cartesian_grid(Rmax,ngrid):
	"""
	Produce a x,y,z cartesian grid
	:param Rmax: The value of the coordinate goes from 0 to Rmax
	:param ngrid:
	:return:
	"""

	cart_1D_grid=np.linspace(-Rmax,Rmax,ngrid)

	list_cord=[cart_1D_grid,]*3

	grid=cartesian(*list_cord)

	return grid

def _galactic_grid(grid,q=1,p=1,alpha=0,beta=0,gamma=0,ax='zyx'):
	"""
	Calculate r,R,z,l,b,m with respect to the Galactic center starting from x,y,z
	:param grid: carteian grid with x,y,z as cols
	:param q: z flattening
	:param p: y flattening
	:return:
	"""
	
	grid=rs.rotate(grid,angles=(alpha,beta,gamma),axes='zyx',reference='l')
	
	z=grid[:,2]
	R=np.sqrt((grid[:,0])**2+grid[:,1]**2)
	r=np.sqrt((grid[:,0])**2+grid[:,1]**2+z**2)
	m=np.sqrt((grid[:,0])**2+(grid[:,1]/p)**2+(z/q)**2)
	l=(np.arctan2(grid[:,1],grid[:,0]))*(360./(2*np.pi))
	b=(360*np.arcsin(z/r))/(2*np.pi)

	return r,R,z,l,b,m

def _solar_grid(grid,xsun,q=1,p=1,alpha=0,beta=0,gamma=0,ax='zyx'):
	"""
	Calculate r,R,z,l,b,m with respect to the Sun starting from x,y,z (lhf system centered on the Galactic center)
	:param grid: carteian grid with x,y,z as cols
	:param q: z flattening
	:param p: y flattening
	:return:
	"""
	
	grid=rs.rotate(grid,angles=(alpha,beta,gamma),axes='zyx',reference='l')
	
	z=grid[:,2]
	xnew=xsun-grid[:,0]
	R=np.sqrt(xnew**2+grid[:,1]**2)
	r=np.sqrt(xnew**2+grid[:,1]**2+z**2)
	m=np.sqrt(xnew**2+(grid[:,1]/p)**2+(z/q)**2)
	l=(np.arctan2(grid[:,1],xnew)*(360./(2*np.pi)))
	b=(360*np.arcsin(z/r))/(2*np.pi)


	return r,R,z,l,b,m

def _calculate_idx(rsun,lsun,bsun,xsun=8,rmin=None,rmax=None,bmin=None,bmax=None,lmin=None,lmax=None,thetamin=None,thetamax=None,phimin=None,phimax=None,zgmin=None,zgmax=None,struct=None):
	"""
	Calculate the index of the gridd cells included in the sample Volume
	:param rsun: X position of the wrt the Galactic center
	:param lsun: Gal. longitude wrt to the Sun
	:param bsun: Gal. latitude wrt to the Sun
	:param rmin: Minimum distance observed from the Sun
	:param rmax:  Maximum distance observed from the Sun
	:param bmin:  Minimum Gal. latitude of the Sample
	:param bmax:  Maximum Gal. latitude of the Sample
	:param lmin:  Minimum Gal. longtitude of the Sample
	:param lmax:  Maximum Gal. longitude of the Sample
	:param thetamin:  Minimum galactic longitude centered on the Galaxy in deg
	:param thetamax:  Minimum galactic longitude centered on the Galaxy in deg
	:param phimin:  Minimum galactic longitude centered on the Galaxy in deg
	:param phimax:  Minimum galactic longitude centered on the Galaxy in deg
	:param struct: Struct to consider
	:return:
	"""
	idx=np.array([True,]*len(rsun))
	if rmin is not None: idx*=rsun>=rmin
	if rmax is not None: idx*=rsun<=rmax
	if bmin is not None: idx*=np.abs(bsun)>=bmin
	if bmax is not None: idx*=np.abs(bsun)<=bmax
	if lmin is not None: idx*=lsun>=lmin
	if lmax is not None: idx*=lsun<=lmax
	
	if (phimin is not None) or (phimax is not None):
		
		cb=np.cos(bsun)
		cl=np.cos(lsun)
		sl=np.sin(lsun)
		x=(xsun-rsun*cb*cl)
		y=(rsun*cb*sl)
		phi=(np.arctan2(y,x)*(360./(2*np.pi)))
		
		if phimin is not None: idx*=phi>=phimin
		if phimax is not None: idx*=phi<=phimax
		


	if (thetamin is not None) or (thetamax is not None) or (zgmin is not None) or (zgmax is not None):
		
		cb=np.cos(bsun)
		sb=np.sin(bsun)
		cl=np.cos(lsun)
		sl=np.sin(lsun)
		xg=(xsun-rsun*cb*cl)
		yg=(rsun*cb*sl)
		zg=rsun*sb
		
		zg_abs=np.abs(zg)
		rg=np.sqrt(xg*xg+yg*yg)
		
		theta_abs=np.arctan(zg_abs/rg)*(360/(2*np.pi))
		
		
		if thetamin is not None: idx*=theta_abs>=thetamin
		if thetamax is not None: idx*=theta_abs<=thetamax
		if zgmin is not None: idx*=zg_abs>=zgmin
		if zgmax is not None: idx*=zg_abs<=zgmax

	#Struct
	if struct is not None:
		for s in struct:
			if isinstance(s,str):
				if s in struct_list: st=struct_list[s]
				elif s in struct_list_object: st=struct_list_object[s]
				else: pass
			else: st=s
			if len(st)==3:
				lmg,bmg,rmc=st
				skysep_dmg=skysep(lsun,bsun,lmg,bmg)
				idx*=skysep_dmg>rmc
			if len(st)==4:
				lmin_str,lmax_str,bmin_str,bmax_str=st
				idx*=(lsun>lmax_str)|(lsun<lmin_str)|(bsun<bmin_str)|(bsun>bmax_str)

	return idx

def _cylindrical_fraction(R,z,idx,rangem=None,bins=30):

	H,edges,_=ploth2(R,z,range=rangem,bins=bins)
	xedges,yedges=edges
	Hn,edges,_=ploth2(R[idx],z[idx],range=((np.min(xedges),np.max(xedges)),(np.min(yedges),np.max(yedges))),bins=bins)

	frac=np.where(H==0,0,Hn/H)

	return frac,edges

def _ellipsoidal_fraction2D(m,b,idx,rangem=None,bins=30):

	Hn,edges,_=ploth2(m[idx],b[idx],range=rangem,bins=bins)

	xedges,yedges=edges

	if len(bins)==2:
		mbins,bbins=bins
	else:
		mbins=bins
		bbins=bins


	H,_=np.histogram(m,range=(np.min(xedges),np.max(xedges)),bins=mbins)

	Hf=np.zeros_like(Hn)

	for i in range(bbins):
		#if H==0: Hf[:,i]=1
		#else: Hf[:,i]=Hn[:,i]/H
		Hf[:,i]=np.where(H==0,0,Hn[:,i]/H)
	#Hf=Hn

	return Hf,edges

def _ellipsoidal_fraction1D(m,idx,medges):

	Hn,edges=np.histogram(m[idx],bins=medges) #Calc the cells at certain m with the cut

	H,_=np.histogram(m,bins=medges) #Calc the total number of cells at certain m

	Hf=np.where(H==0,0,Hn/H) #Fraction
	
	print('M',Hn,H)

	return Hf,edges

def _volume_ellipsodal2D(medges,bedges,q=1,p=1,rmin=None,rmax=21,bmin=None,bmax=None,lmin=None,lmax=None,thetamin=None,thetamax=None,phimin=None,phimax=None,zgmin=None,zgmax=None,grid=None,xsun=8,ngrid=300,struct=None,alpha=0,beta=0,gamma=0):
	
	binsA=len(medges)-1
	binsB=len(bedges)-1
	arr=np.zeros((4,binsA,binsB))
	for i in range(binsA):
		for j in range(binsB):
			arr[0,i,j]=medges[i]
			arr[1,i,j]=medges[i+1]
			arr[2,i,j]=bedges[j]
			arr[3,i,j]=bedges[j+1]
	vol_original=(4./3.)*q*p*np.pi*(arr[1,:,:]**3-arr[0,:,:]**3)
	bins=(binsA,binsB)



	if (rmin is None) and (rmax is None) and (bmin is None) and (bmax is None) and (lmin is None) and (lmax is None) and (thetamin is None) and (thetamax is None) and (phimin is None) and (phimax is None) and (zgmin is None) and (zgmax is None) and (bmax is None) and (struct is None):
		Hfrac=np.ones((binsA,binsB))
		#Vol=Hfrac*vol_original
	else:
		if grid is None: grid=_cartesian_grid(medges[-1],ngrid)
		r,R,z,l,b,m=_galactic_grid(grid,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,ax='zyx')
		rsun,Rsun,z,lsun,bsun,msun=_solar_grid(grid,xsun,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,ax='zyx')
		idx=_calculate_idx(rsun,lsun,bsun,xsun=xsun,rmin=rmin,rmax=rmax,bmin=bmin,bmax=bmax,lmin=lmin,lmax=lmax,thetamin=thetamin,thetamax=thetamax,phimin=phimin,phimax=phimax,zgmin=zgmin,zgmax=zgmax,struct=struct)


		range_ell=((medges[0],medges[-1]),(bedges[0],bedges[-1]))

		Hfrac,edges=_ellipsoidal_fraction2D(m,b,idx,rangem=range_ell,bins=bins)


	Vol=Hfrac*vol_original


	return np.where(Vol==0,np.nan,Vol),Hfrac

def _volume_ellipsodal1D(medges,q=1,p=1,alpha=0,beta=0,gamma=0,ax='zyx',rmin=None,rmax=21,bmin=None,bmax=None,lmin=None,lmax=None,thetamin=None,thetamax=None,phimin=None,phimax=None,zgmin=None,zgmax=None,grid=None,xsun=8,ngrid=300,struct=None):

	bins=len(medges)-1
	vol_original=(4./3.)*q*p*np.pi*(medges[1:]**3-medges[:-1]**3)



	if (rmin is None) and (rmax is None) and (bmin is None) and (bmax is None) and (lmin is None) and (lmax is None) and (thetamin is None) and (thetamax is None) and (phimin is None) and (zgmin is None) and (zgmax is None) and (phimax is None) and (bmax is None) and (struct is None):
		Hfrac=np.ones(bins)
		#Vol=Hfrac*vol_original
	else:
		if grid is None: grid=_cartesian_grid(medges[-1],ngrid)
		r,R,z,l,b,m=_galactic_grid(grid,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,ax='zyx')
		rsun,Rsun,z,lsun,bsun,msun=_solar_grid(grid,xsun,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,ax='zyx')
		idx=_calculate_idx(rsun,lsun,bsun,xsun=xsun,rmin=rmin,rmax=rmax,bmin=bmin,bmax=bmax,lmin=lmin,lmax=lmax,thetamin=thetamin,thetamax=thetamax,phimin=phimin,phimax=phimax,zgmin=zgmin,zgmax=zgmax,struct=struct)


		range_ell=((medges[0],medges[-1]))

		Hfrac,edges=_ellipsoidal_fraction1D(m,idx,medges)


	Vol=Hfrac*vol_original
	
		
	return np.where(Vol==0,np.nan,Vol),Hfrac

def volume_ellipsoidal(medges,bedges=None,q=1,p=1,rmin=None,rmax=21,bmin=None,bmax=None,lmin=None,lmax=None,thetamin=None,thetamax=None,phimin=None,phimax=None,zgmin=None,zgmax=None,grid=None,xsun=8,ngrid=300,struct=None):
	"""
	Calcualte the effective Volume for each cells in the plane m-b or only along m (if bedges?=None)
		THe Volume is 4/3*pi*q*p*(m(i+i)^3-m(i)^3) *Vol_frac, where Vol_frac
	is the fraction of the Volume at each cells sampled by the user sample.
	:param medges:
	:param bedges:
	:param q:
	:param p:
	:param rmin:
	:param rmax:
	:param bmin:
	:param bmax:
	:param grid:
	:param xsun:
	:param ngrid:
	:param struct:
	:return:
	"""
	if bedges is None: return _volume_ellipsodal1D(medges,q,p,rmin,rmax=rmax,bmin=bmin,bmax=bmax,lmin=lmin,lmax=lmax,thetamin=thetamin,thetamax=thetamax,phimin=phimin,phimax=phimax,zgmin=zgmin,zgmax=zgmax,grid=grid,xsun=xsun,ngrid=ngrid,struct=struct)
	else: return _volume_ellipsodal2D(medges,bedges,q,p,rmin,rmax=rmax,bmin=bmin,bmax=bmax,lmin=lmin,lmax=lmax,thetamin=thetamin,thetamax=thetamax,phimin=phimin,phimax=phimax,zgmin=zgmin,zgmax=zgmax,grid=grid,xsun=xsun,ngrid=ngrid,struct=struct)

def volume_cylindrical(Redges,zedges,rmin=None,rmax=21,bmin=None,bmax=None,lmin=None,lmax=None,thetamin=None,thetamax=None,phimin=None,phimax=None,zgmin=None,zgmax=None,grid=None,xsun=8,ngrid=300,struct=None):
	"""
	Calcualte the effective Volume for each cells in the plane R-z.
	THe Volume is pi (R(i+i)^2-R(i)^2) * (z(i+1)-z(i-1)) *Vol_frac, where Vol_frac
	is the fraction of the Volume at each cells sampled by the user sample.
	:param Redges:
	:param zedges:
	:param rmin:
	:param rmax:
	:param bmin:
	:param bmax:
	:param grid:
	:param xsun:
	:param ngrid:
	:param struct:
	:return:
	"""
	binsA=len(Redges)-1
	binsB=len(zedges)-1
	arr=np.zeros((4,binsA,binsB))
	for i in range(binsA):
		for j in range(binsB):
			arr[0,i,j]=Redges[i]
			arr[1,i,j]=Redges[i+1]
			arr[2,i,j]=zedges[j]
			arr[3,i,j]=zedges[j+1]
	vol_original=np.pi*(arr[1,:,:]**2-arr[0,:,:]**2)*(arr[3,:,:]-arr[2,:,:])

	bins=(binsA,binsB)

	if (rmin is None) and (rmax is None) and (bmin is None) and (bmax is None) and (lmin is None) and (lmax is None) and (thetamin is None) and (thetamax is None) and (phimin is None) and (phimax is None) and (zgmin is None) and (zgmax is None) and (bmax is None) and (struct is None):

		Hfrac=np.ones((binsA,binsB))
		#Vol=Hfrac*vol_original

	else:

		if grid is None: grid=_cartesian_grid(np.max(redges[-1],zedges[-1]),ngrid)
		r,R,z,l,b,_=_galactic_grid(grid)
		rsun,Rsun,z,lsun,bsun,_=_solar_grid(grid,xsun)
		idx=_calculate_idx(rsun,lsun,bsun,xsun=xsun,rmin=rmin,rmax=rmax,bmin=bmin,bmax=bmax,lmin=lmin,lmax=lmax,thetamin=thetamin,thetamax=thetamax,phimin=phimin,phimax=phimax,zgmin=zgmin,zgmax=zgmax,struct=struct)

		range_cyl=((Redges[0],Redges[-1]),(zedges[0],zedges[-1]))
		Hfrac,edges=_cylindrical_fraction(R,z,idx,rangem=range_cyl,bins=bins)

		Hfrac=np.where(Hfrac==0,np.nan,Hfrac)
	Vol=Hfrac*vol_original

	return np.where(Vol==0,np.nan,Vol),Hfrac

struct_list={'C1':(10,29,15,15.4),'C2':(3.74,3.98,46.6,47.0),'C3':(15.3,16.1,15.10,15.22),
        'C4':(1.0,3.2,-25,-18.5),'C5':(52.1,54.1,-26.6,-25.2),
        'C6':(59.0,62,10,12.5),'C7':(99,119,10,19),
        'C8':(166,178,16,23.5),'C9':(162,195,62,75),
        'C10':(156.5,175,45.5,61),'C11':(172,206,44,64),
        'C12':(214,215.8,41.2,42),'C13':(221,236,54,69),
        'C14':(205,232,21,33),'C15':(220,230,10,13),
        'C16':(286,292,13,16.5),'C17':(308,309.5,14.7,15.3),
        'C18':(350.8,351.1,15.8,16.1),'C19':(352.2,353,14.7,15.8),
        'C20':(344.6,345.6,36.0,37.0),'C21':(276,284.5,30,70),
        'C22':(40.8,43.2,78.4,79),'C23':(41.8,42.4,73.3,73.7),
        'C24':(64.9,65.1,-27.4,-27.2),'C25':(133.1,133.9,-32.5,-30.5),
        'C26':(274,286,-38,-28),'C27':(299,307,-47,-41),
        'C28':(224,263,-40,-15),'C29':(332.4,333.5,79.5,80),
        'C30':(68,77,10,15),'A1':(167,179,16,22),'A2':(160,190,63,73)}

struct_list_object={'LMC':(280.4653,-32.8883,9),'SMC':(302.7969,-44.2992,7)}