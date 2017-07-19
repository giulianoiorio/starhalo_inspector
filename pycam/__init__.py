from __future__ import division
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np


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
	