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
	
def mad_old(arr,axis=None):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
       Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    med = np.median(arr,axis=axis)
    return 1.4826*np.median(np.abs(arr - med),axis=axis)

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

def arg_filter(tab,ampcut=[None,None],gcut=[None,None],colcut=[None,None],ecut=[None,None],ncut=[None,None],pmcut=[None,None],dcut=[None,None],bcut=[None,None],lcut=[None,None],racut=[None,None],deccut=[None,None],bgcut=[None,None],zcut=[None,None],parcut=[None,None],extra={}, distkey='distance', magkey='gc', colkey='grc',  rlmc=0,rsmc=0,rm3=0,rm5=0,Mg=0.55,footprint=None,lneg=True,babs=True,bgabs=True,zabs=True,struct=None,xsun=8,struct_id_column=None):
    """
    Find the index of a tab that are between the given selection box:
    @tab; an open fits table (if f=fits.open(file), tab is f[1].data)
    @ampcut: Cut on the amplitude 'amp',
    @gcut: Cut in  magnitude (corrected for extinction), the key of this value is stored in magkey
    @colcut: Cut in color (corrected for extinction), the key of this value is stored in colkey
    @ecut: Cut in astrometric_excess_noise, the key is 'extraerr'
    @nobs: cut in number of observations (phot_n_gobs), the key is  'nobs'
    @pmcut: cut in number of observations (phot_n_gobs), the key is  'nobs'
    @dcut: cut in distance, if the key distance exists in tab this is used as key, while
    the distance is estimated (in kpc) as  10^( 0.2*(mag-Mg-5)), see below for Mg.
    @bcut: cut in galactic latitude, the key is 'b'.
    @lcut: cut in galactic longitude, the key is 'l'.
    @racut: cut in right ascension, the key is 'ra'.
    @deccut: cut in declination, the key is 'dec'.
    @bcut: Galactic b with respect to the Galactic centre (theta angle)
    it is estimated as Zg/Rg, using as distance 10^( 0.2*(mag-Mg-5)) and then corrected for the position of the sun.
    @zcut: cut in galactic Z, it is estimated as Zg/Rg, using as distance 10^( 0.2*(mag-Mg-5)) and then corrected for the position of the sun.
    @parcut: cut in 'parallax', the key is 'parallax'.
    @extra: extra cut: if is a dictionary with {key:value}, value can be a list of len2 with
    lower limit and upper limit, or a int,float or str. In this last case the filter is tab[key]==value,
    otherwise value[0]<tab[key]<value[1]
    @magkey: the key to use for gcut
    @colkey: the key to use for colcut
    @rlmc: cut the large magellanic cloud within this radius in degree
    @slmc: cut the small magellanic cloud within this radius in degree
    @rm3: cut m3 within this radius in degree
    @rm5: cut m5 within this radius in degree
    @lneg: if True l goes from -180 to 180, otherwise it goes from 0 to 360
    @babs: if True consider the absolute value of b
    @bgabs: if True consider the absolute value of bg
    @zbas: if True consider the absolute value of z
    @Mg: abosolute magnitude to use to estimate the distance.
    @footprint: footprint to consider
    @struct: structure to consider
    @xsun: Distance of the Sun wrt the Galactic centre.
    @struct_id_column: If one wants to add the name of cut structure in the table, this is the column of the table to be filled. This option is disabled using None
    : return a boolean numpy array
    """
    idx=np.array([True,]*len(tab['ra']))

    try:
       distance = tab[distkey]
    except:
       distance=10**(0.2*(tab[magkey]-Mg+5) )/1000.


    if footprint is not None:
        if lneg: idx*=footprint( tab['l']  ,tab['b'])
        else: idx*=footprint( np.where(tab['l']>180.,tab['l']-360.,tab['l'])  ,tab['b'])

    if (ampcut[0] is None) and (ampcut[1] is None) and (gcut[0] is None) and (gcut[1] is None) and (colcut[0] is None) and (colcut[1] is None) and (ecut[0] is None) and (ecut[1] is None) and (ncut[0] is None) and (ncut[1] is None) and (pmcut[0] is None) and (pmcut[1] is None) and (dcut[0] is None) and (dcut[1] is None) and (bcut[0] is None) and (bcut[1] is None) and (lcut[0] is None) and (lcut[1] is None) and (racut[0] is None) and (racut[1] is None) and (deccut[0] is None) and (deccut[1] is None)  and extra=={}:
        pass
    else:
        if ampcut[0] is not None: idx*=tab['amp']>=ampcut[0]
        if ampcut[1] is not None: idx*=tab['amp']<=ampcut[1]
        #print('ampcut',np.sum(idx))
        if gcut[0] is not None: idx*=tab[magkey]>=gcut[0]
        if gcut[1] is not None: idx*=tab[magkey]<=gcut[1]
        #print('gcut',np.sum(idx))
        if colcut[0] is not None: idx*=tab[colkey]>=colcut[0]
        if colcut[1] is not None: idx*=tab[colkey]<=colcut[1]
        #print('colcut',np.sum(idx))
        try:
            if ecut[0] is not None: idx*=tab['extraerr']>=ecut[0]
            if ecut[1] is not None: idx*=tab['extraerr']<=ecut[1]
        except:
            if ecut[0] is not None: idx*=tab['astrometric_excess_noise']>=ecut[0]
            if ecut[1] is not None: idx*=tab['astrometric_excess_noise']<=ecut[1]
        #print('ncut',np.sum(idx))
        try:
            if ncut[0] is not None: idx*=tab['nobs']>=ncut[0]
            if ncut[1] is not None: idx*=tab['nobs']<=ncut[1]
        except:
            if ncut[0] is not None: idx *= tab['phot_g_n_obs'] >= ncut[0]
            if ncut[1] is not None: idx *= tab['phot_g_n_obs'] <= ncut[1]
        #print('ncut',np.sum(idx))
        if pmcut[0] is not None: idx*= np.sqrt( (tab['pmra'])**2 + tab['pmdec']**2    )>=pmcut[0]
        if pmcut[1] is not None: idx*=np.sqrt( (tab['pmra'])**2 + tab['pmdec']**2    )<=pmcut[1]
        if parcut[0] is not None: idx*=tab['parallax']>=parcut[0]
        if parcut[1] is not None: idx*=tab['parallax']<parcut[1]

        for key in extra:

            valkey=extra[key]


            if isinstance(valkey,int) or isinstance(valkey,float) or isinstance(valkey,str):
                idx *= tab[key] == extra[key][0]
            elif len(valkey)==2:
                if extra[key][0] is not None: idx*=tab[key] >= extra[key][0]
                if extra[key][1] is not None: idx*=tab[key] <= extra[key][1]
            else:
                raise ValueError('Error in value of extra argument %s. It needs to be a float, int, str or a iterable with len(2)'%key)

        if dcut[0] is not None:
            try:
                distance = tab['distance']
            except:
                distance=10**(0.2*(tab[magkey]-Mg+5) )/1000.
            idx*= distance >=dcut[0]
        if dcut[1] is not None:
            try:
                distance = tab['distance']
            except:
                distance=10**(0.2*(tab[magkey]-Mg+5) )/1000.
            idx*= distance <=dcut[1]
        if bcut[0] is not None:
            if babs: bb = np.abs(tab['b'])
            else: bb = tab['b']
            idx*=bb>=bcut[0]
        if bcut[1] is not None:
            if babs: bb = np.abs(tab['b'])
            else: bb = tab['b']
            idx*=bb<=bcut[1]
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
            theta = np.arctan(zg / rg) * (360 / (2 * np.pi))
            if zabs: zg=np.abs(zg)
            if bgabs: theta=np.abs(theta)

            if bgcut[0] is not None: idx*=theta>=bgcut[0]
            if bgcut[1] is not None: idx*=theta<=bgcut[1]
            if zcut[0] is not None: idx*=zg>=zcut[0]
            if zcut[1] is not None: idx*=zg<=zcut[1]

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
                _idxtpm=(ltmp>=l_lim[0])&(ltmp<=l_lim[1])&(btmp>=b_lim[0])&(btmp<=b_lim[1])
                if struct_id_column is not None:
                    tab[struct_id_column][_idxtpm]=s
                idxtpm+=_idxtpm
        idx*=np.logical_not(idxtpm)
    else:
        if lneg: ltmp=np.where(tab['l']<0,360+tab['l'],tab['l'])
        else: ltmp=tab['l']
        btmp=tab['b']
        idxtpm=np.array([False,]*len(btmp))
        if struct=='allobjects': struct=struct_list_object
        for s in struct:
            sname="Unknown"
            if isinstance(s,str):
                    sname=s
                    if s in struct_list: st=struct_list[s]
                    elif s in struct_list_object: st=struct_list_object[s]
                    else: pass
            else: st=s
            if len(st)==3:
                lmg,bmg,rmc=st
                skysep_dmg=skysep(ltmp,btmp,lmg,bmg)
                _idxtpm = skysep_dmg<rmc
                if struct_id_column is not None:
                    tab[struct_id_column][_idxtpm]=s
                idxtpm +=_idxtpm
            elif len(st)==4:
                lmg,bmg,rmc, dmc=st
                skysep_dmg=skysep(ltmp,btmp,lmg,bmg)
                idx_ob = (skysep_dmg<rmc)&(distance<1.25*dmc)&(distance>0.75*dmc)
                if struct_id_column is not None:
                    tab[struct_id_column][idx_ob]=s
                idxtpm += idx_ob
            #elif len(st)==4:
            #    lmin,lmax,bmin,bmax=st
            #    idxtpm+=(ltmp>lmin)&(ltmp<lmax)&(btmp>bmin)&(btmp<bmax)
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
        cord=rs.rotate_frame(cord=np.array([x,y,z]).T, angles=(alpha,beta,gamma), axes=ax, reference='lh' )
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


def xyz_to_m(x,y,z,q=1.0,qinf=1.0,rq=10.0,p=1.0,alpha=0,beta=0,gamma=0,ax='zyx'):
    """
    Return the m-value of an ellipsoid from the observ magnitude and galactic coordinate.
    if q=1 and p=1, the ellipsoid is indeed a sphere and m=r
    :param x: Galactic x (lhf, sun is a x~8)
    :param y: Galactic y
    :param z: Galactic z
    :param q: Flattening along the z-direction, q=1 no flattening.
    :param p: Flattening along the y-direction, p=1 no flattening.
    :return: the m-value for an ellipsoid m^2=x^2+(y/p)^2+(z/q)^2.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    i=np.abs(alpha)+np.abs(beta)+np.abs(gamma)
    if i!=0:
        cord=rs.rotate_frame(cord=np.array([x,y,z]).T, angles=(alpha,beta,gamma), axes=ax, reference='lh' )
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

def obs_to_cyl(mag,l,b,Mg,xsun=8,negative_r=False, parallax=False):
    """
    Return the R,Z coordinate from the mag, l and b
    :param mag: observed magnitude  or parallax in mas.
    :param l: galactic longitude.
    :param b: galactic latitude.
    :param Mg: Absolute magnitude.
    :param xsun: Distance of the sun from the galactic centre.
    :return: the r on the plane of the galaxy and z.
    """
    cost=0.017453292519943295769 #From degree to rad

    if parallax: d=parallax_to_distance(mag)
    else: d=mag_to_dist(mag,Mg=Mg)

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

def obs_to_xyz(mag,l,b,Mg,xsun=8, parallax=False):
    """
    Return the R,Z coordinate from the mag, l and b
    :param mag: observed magnitude or parallax in mas.
    :param l: galactic longitude.
    :param b: galactic latitude.
    :param Mg: Absolute magnitude.
    :param xsun: Distance of the sun from the galactic centre.
    :return: the x,y,z cord in the galactic plane
    """
    cost=0.017453292519943295769 #From degree to rad
    if parallax: d=parallax_to_distance(mag)
    else: d=mag_to_dist(mag,Mg=Mg)
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

def _calculate_idx(rsun,lsun,bsun,xsun=8, rmin=None,rmax=None,Rmin=None, Rmax=None ,bmin=None,bmax=None,lmin=None,lmax=None,thetamin=None,thetamax=None,phimin=None,phimax=None,zgmin=None,zgmax=None,struct=None):
    """
    Calculate the index of the gridd cells included in the sample Volume
    :param rsun: X position of the wrt the Galactic center
    :param lsun: Gal. longitude wrt to the Sun
    :param bsun: Gal. latitude wrt to the Sun
    :param rmin: Minimum distance observed from the Sun
    :param rmax:  Maximum distance observed from the Sun
    :param Rmin: Minimum  cylindrical distance observed from the Sun
    :param Rmax:  Maximum cylyndrical distance observed from the Sun
    :param bmin:  Minimum Gal. latitude of the Sample
    :param bmax:  Maximum Gal. latitude of the Sample
    :param lmin:  Minimum Gal. longtitude of the Sample
    :param lmax:  Maximum Gal. longitude of the Sample
    :param thetamin:  Minimum galactic longitude centered on the Galaxy in deg
    :param thetamax:  Minimum galactic longitude centered on the Galaxy in deg
    :param phimin:  Minimum galactic longitude centered on the Galaxy in deg
    :param phimax:  Minimum galactic longitude centered on the Galaxy in deg
    :param zgmin: Minimum  absolute value of z
    :param zgmax:  Maximum  absolute value of z
    :param struct: Struct to consider
    :return:
    """

    dtr=180/np.pi #deg to rad

    idx=np.array([True,]*len(rsun))
    if rmin is not None: idx*=rsun>=rmin
    if rmax is not None: idx*=rsun<=rmax
    if bmin is not None: idx*=np.abs(bsun)>=bmin
    if bmax is not None: idx*=np.abs(bsun)<=bmax
    if lmin is not None: idx*=lsun>=lmin
    if lmax is not None: idx*=lsun<=lmax

    if (Rmin is not None) or (Rmax is not None):
        Rsun=rsun*np.sin(bsun*dtr)

        if Rmin is not None: idx*=Rsun>=Rmin
        if Rmax is not None: idx*=Rsun<=Rmax

    if (phimin is not None) or (phimax is not None):

        cb=np.cos(bsun*dtr)
        cl=np.cos(lsun*dtr)
        sl=np.sin(lsun*dtr)
        x=(xsun-rsun*cb*cl)
        y=(rsun*cb*sl)
        phi=(np.arctan2(y,x)*(360./(2*np.pi)))

        if phimin is not None: idx*=phi>=phimin
        if phimax is not None: idx*=phi<=phimax



    if (thetamin is not None) or (thetamax is not None) or (zgmin is not None) or (zgmax is not None):

        cb=np.cos(bsun*dtr)
        sb=np.sin(bsun*dtr)
        cl=np.cos(lsun*dtr)
        sl=np.sin(lsun*dtr)
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
            elif len(st)==4:
                lmin,lmax,bmin,bmax=st
                idxtpm+=(ltmp>lmin)&(ltmp<lmax)&(btmp>bmin)&(btmp<bmax)

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


def comp_cont(sample,compare_sample,idkey='source_id',ampcut=((None,None),(None,None)),gcut=((None,None),(None,None)),colcut=((None,None),(None,None)),ecut=((None,None),(None,None)),ncut=((None,None),(None,None)),pmcut=((None,None),(None,None)),racut=(None,None),deccut=(None,None),dcut=(None,None),bcut=(None,None),lcut=(None,None),bgcut=(None,None),parcut=((None,None),(None,None)),extra=({},{}), magkey=('gc','gc'), colkey=('grc','grc'), Mg=0.55,footprint=None,lneg=True,babs=True,struct=None,xsun=8):

    idx_sample = arg_filter(sample, ampcut=ampcut[0], gcut=gcut[0], colcut=colcut[0], ncut=ncut[0], pmcut=pmcut[0], ecut=ecut[0],
                               parcut=parcut[0], Mg=Mg, struct=struct,
                               bcut=bcut, lcut=lcut, lneg=lneg, bgcut=bgcut, magkey=magkey[0], colkey=colkey[0],
                               xsun=xsun, extra=extra[0],babs=babs,racut=racut,deccut=deccut)

    idx_check = arg_filter(compare_sample, ampcut=ampcut[1], gcut=gcut[1], colcut=colcut[1], ncut=ncut[1], pmcut=pmcut[1], ecut=ecut[1],
                               parcut=parcut[1], Mg=Mg, struct=struct,
                               bcut=bcut, lcut=lcut, lneg=lneg, bgcut=bgcut, magkey=magkey[1], colkey=colkey[1],
                               xsun=xsun, extra=extra[1],babs=babs,dcut=dcut,deccut=deccut)

    id_sample=sample[idkey][idx_sample]
    id_compare=compare_sample[idkey][idx_check]
    id_match=np.isin(id_compare,id_sample)
    N_match=np.sum(id_match)
    N_sample=len(id_sample)
    N_compare=len(id_compare)



    completeness=N_match/N_compare
    ecompleteness=completeness*np.sqrt(1/N_match+1/N_compare)
    contamination=1-N_match/N_sample
    econtamination=np.sqrt(  ( (N_match*N_match)/(N_sample*N_sample*N_sample) ) + (N_match/(N_sample*N_sample))  )

    return (completeness,ecompleteness), (contamination,econtamination)


def parallax_to_distance(parallax, eparallax=None):
    '''

    :param parallax:  in mas
    :param eparallax:  in mas
    :return:  distance in kpc
    '''
    parallax=parallax/1000
    dist = 1 / parallax

    if eparallax is not None:

        return dist/1000, (dist * dist * eparallax)/1000

    else:

        return dist/1000

def obs_to_align(l,b,l_pole,b_pole,l_ref=0,b_ref=0):
	"""
	Creat a new sky frame of reference with the north pole pointing toward the direction given by l_pole and b_pole, then aling the longitude toward the direction defined by l_ref=0 and b_ref=0.
	:param l:
	:param b:
	:param l_pole:
	:param b_pole:
	:param l_ref:
	:param b_ref:
	:return 
	"""
	
	cost=np.pi/180.
	
	#Set coordinate array
	#idx=0 contains l_pole and b_pole, idx=1 contains l_ref and b_ref
	ll=np.zeros(len(l)+2)
	bb=np.zeros(len(b)+2)
	ll[0]=l_pole
	bb[0]=b_pole
	ll[1]=l_ref
	bb[1]=b_ref
	ll[2:]=l
	bb[2:]=b
	
	
	#To deg
	ll*=cost
	bb*=cost


	#Trigo
	cl=np.cos(ll)
	cb=np.cos(bb)
	sl=np.sin(ll)
	sb=np.sin(bb)

	
	#Coordinate of the versors
	x=cb*cl
	y=cb*sl
	z=sb
	
	#print('xx',x[2:],y[2:],z[2:])
	
	#Pole versor
	pole_versor=(x[0],y[0],z[0])
	
	#Verson on the great circle plane Pole
	#Vp1 Asse x nel nuovo piano
	#Vp2 Asse y nel nuovo piano
	vp1=np.cross(pole_versor,(0,0,1))
	vp1_norm=np.sqrt(vp1[0]*vp1[0]+vp1[1]*vp1[1]+vp1[2]*vp1[2])
	vp1=vp1/vp1_norm
	vp2=np.cross(vp1,pole_versor)
	vp2_norm=np.sqrt(vp2[0]*vp2[0]+vp2[1]*vp2[1]+vp2[2]*vp2[2])
	vp2=vp2/vp2_norm
	
	#Project on the two vectors, new x new y
	xn=x*vp1[0]+y*vp1[1]+z*vp1[2]
	yn=x*vp2[0]+y*vp2[1]+z*vp2[2]
	
	#Project along the great circle
	ln=np.arctan2(yn,xn)/cost
	
	#Project onto the pole
	zn=pole_versor[0]*x+pole_versor[1]*y+pole_versor[2]*z
	bn=np.arcsin(zn)/cost
	
	
	#print('x',xn,yn,zn)
	#print('xxn',xn[2:],yn[2:],zn[2:])
	#print('ln',ln[2:],bn[2:])
	#print('ln3',ln,bn)
	
	
	##Align longitude toward lref
	ln=ln-ln[1]
	
	#Make sure l is between 0 and 460
	lret=np.where(ln[2:]<0,360+ln[2:],ln[2:])
	
	return lret, bn[2:]


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


#https://heasarc.gsfc.nasa.gov/W3Browse/all/globclust.html
#struct_GC={'2MASS-GC01':(10.48,0.11,0.30), '2MASS-GC02':(9.79,-0.61,0.25), 'Arp2':(8.55,-20.79,0.45), 'BH176':(328.41,4.34,0.32), 'BH261':(3.36,-5.27,0.20), 'Djorg1':(356.69,-2.47,0.79), 'Djorg2':(2.77,-2.50,0.52), 'E3':(292.27,-19.02,0.53), 'Eridanus':(218.10,-41.33,0.16), 'FSR1735':(339.19,-1.85,0.07), 'HP1':(357.44,2.12,0.47), 'IC1257':(16.54,15.15,0.44), 'IC1276':(21.83,5.67,1.08), 'IC4499':(307.35,-20.47,0.68), 'Kosopov1':(260.99,70.75,0.05), 'Kosopov2':(195.12,25.54,0.04), 'Liller1':(354.84,-0.16,0.60), 'Lynga7':(328.77,-2.80,0.40), 'NGC104':(305.89,-44.89,2.11), 'NGC1261':(270.54,-52.12,0.25), 'NGC1851':(244.51,-35.03,0.33), 'NGC1904':(227.23,-29.35,0.40), 'NGC2298':(245.63,-16.00,0.37), 'NGC2419':(180.37,25.24,0.38), 'NGC2808':(282.19,-11.25,0.45), 'NGC288':(152.30,-89.38,0.66), 'NGC3201':(277.23,8.64,1.27), 'NGC362':(301.53,-46.25,0.52), 'NGC4147':(252.85,77.19,0.30), 'NGC4372':(300.99,-9.88,1.75), 'NGC4590':(299.63,36.05,0.75), 'NGC4833':(303.60,-8.02,0.89), 'NGC5024':(332.96,79.76,0.92), 'NGC5053':(335.70,78.95,0.57), 'NGC5139':(309.10,14.97,2.42), 'NGC5272':(42.22,78.71,1.44), 'NGC5286':(311.61,10.57,0.36), 'NGC5466':(42.15,73.59,0.78), 'NGC5634':(342.21,49.26,0.53), 'NGC5694':(331.06,30.36,0.23), 'NGC5824':(332.56,22.07,0.29), 'NGC5897':(342.95,30.29,0.51), 'NGC5904':(3.86,46.80,1.18), 'NGC5927':(326.60,4.86,0.84), 'NGC5946':(327.58,4.19,1.27), 'NGC5986':(337.02,13.27,0.40), 'NGC6093':(352.67,19.46,0.36), 'NGC6101':(317.74,-15.82,0.31), 'NGC6121':(350.97,15.97,2.59), 'NGC6139':(342.37,6.94,0.54), 'NGC6144':(351.93,15.70,1.67), 'NGC6171':(3.37,23.01,0.95), 'NGC6205':(59.01,40.91,1.05), 'NGC6218':(15.72,26.31,0.86), 'NGC6229':(73.64,40.31,0.19), 'NGC6235':(358.92,13.52,0.56), 'NGC6254':(15.14,23.08,0.92), 'NGC6256':(347.79,3.31,0.32), 'NGC6266':(353.57,7.32,0.56), 'NGC6273':(356.87,9.38,0.73), 'NGC6284':(358.35,9.94,1.11), 'NGC6287':(0.13,11.02,0.35), 'NGC6293':(357.62,7.83,0.79), 'NGC6304':(355.83,5.38,0.66), 'NGC6316':(357.18,5.76,0.38), 'NGC6325':(0.97,8.00,0.47), 'NGC6333':(5.54,10.71,0.40), 'NGC6341':(68.34,34.86,0.62), 'NGC6342':(4.90,9.72,0.79), 'NGC6352':(341.42,-7.17,0.52), 'NGC6355':(359.59,5.43,0.79), 'NGC6356':(6.72,10.22,0.47), 'NGC6362':(325.55,-17.57,0.70), 'NGC6366':(18.41,16.04,0.60), 'NGC6380':(350.18,-3.42,0.60), 'NGC6388':(345.56,-6.74,0.34), 'NGC6397':(338.17,-11.96,0.79), 'NGC6401':(3.45,3.98,0.61), 'NGC6402':(21.32,14.81,0.39), 'NGC6426':(28.09,16.23,0.65), 'NGC6440':(7.73,3.80,0.29), 'NGC6441':(353.53,-5.01,0.36), 'NGC6453':(355.72,-3.87,0.79), 'NGC6496':(348.03,-10.01,0.24), 'NGC6517':(19.23,6.76,0.20), 'NGC6522':(1.02,-3.93,0.79), 'NGC6528':(1.14,-4.17,0.21), 'NGC6535':(27.18,10.44,0.39), 'NGC6539':(20.80,6.78,1.04), 'NGC6540':(3.29,-3.31,0.47), 'NGC6541':(349.29,-11.19,0.65), 'NGC6544':(5.84,-2.20,0.11), 'NGC6553':(5.26,-3.03,0.38), 'NGC6558':(0.20,-6.02,0.47), 'NGC6569':(0.48,-6.68,0.36), 'NGC6584':(342.14,-16.41,0.38), 'NGC6624':(2.79,-7.91,0.95), 'NGC6626':(7.80,-5.58,0.56), 'NGC6637':(1.72,-10.27,0.40), 'NGC6638':(7.90,-7.15,0.24), 'NGC6642':(9.81,-6.44,0.49), 'NGC6652':(1.53,-11.38,0.32), 'NGC6656':(9.89,-7.55,1.59), 'NGC6681':(2.85,-12.51,0.47), 'NGC6712':(25.35,-4.32,0.43), 'NGC6715':(5.61,-14.09,0.49), 'NGC6717':(12.88,-10.90,0.47), 'NGC6723':(0.07,-17.30,0.53), 'NGC6749':(36.20,-2.21,0.19), 'NGC6752':(336.49,-25.63,2.69), 'NGC6760':(36.11,-3.92,0.76), 'NGC6779':(62.66,8.34,0.53), 'NGC6809':(8.79,-23.27,0.77), 'NGC6838':(56.75,-4.56,0.45), 'NGC6864':(20.30,-25.75,0.28), 'NGC6934':(52.10,-18.89,0.37), 'NGC6981':(35.16,-32.68,0.37), 'NGC7006':(63.77,-19.41,0.22), 'NGC7078':(65.01,-27.31,1.36), 'NGC7089':(53.37,-35.77,0.62), 'NGC7099':(27.18,-46.84,0.95), 'NGC7492':(53.39,-63.48,0.23), 'Pal1':(130.06,19.03,0.19), 'Pal10':(52.43,2.72,0.15), 'Pal11':(31.81,-15.57,0.22), 'Pal12':(30.51,-47.68,0.96), 'Pal13':(87.10,-42.70,0.11), 'Pal14':(28.74,42.19,0.26), 'Pal15':(18.88,24.30,0.24), 'Pal2':(170.53,-9.07,0.29), 'Pal3':(240.15,41.86,0.20), 'Pal4':(202.31,71.80,0.14), 'Pal5':(0.85,45.86,0.38), 'Pal6':(2.10,1.78,0.42), 'Pal8':(14.11,-6.79,0.95), 'Pyxis':(261.32,7.00,0.00), 'Rup106':(300.88,11.67,0.25), 'Terzan1':(357.57,1.00,0.63), 'Terzan10':(4.49,-1.99,0.25), 'Terzan12':(8.37,-2.10,0.15), 'Terzan2':(356.32,2.30,0.47), 'Terzan3':(345.08,9.19,0.30), 'Terzan4':(356.02,1.31,0.36), 'Terzan5':(3.84,1.69,0.33), 'Terzan6':(358.57,-2.16,0.79), 'Terzan7':(3.39,-20.07,0.21), 'Terzan8':(5.76,-24.56,0.20), 'Terzan9':(3.61,-1.99,0.47), 'Ton2':(350.80,-3.42,0.54), 'UKS1':(5.13,0.76,0.94), 'Whiting1':(161.22,-60.76,0.04)}
#struct_Galaxy={'LMC':(280.4653,-32.8883,5),'SMC':(302.7969,-44.2992,3),'Sculptor':(287.5,-83.2,1), 'Carina':(260.1,-22.2,0.5), 'Fornax':(237.1,-65.7,1),'Draco':(86.4,34.7,1),'Uminor':(105.0,44.8,1), 'Hercules':(28.7,36.9,0.5),'SextansA':(246.1,39.9,0.5),'SextansB':(233.2,43.8,0.5),'Andromeda':(121.2,-21.6,0.5),'M32':(121.1,-22,0.5)}
#struct_Galaxy={'LMC':(280.4653,-32.8883,5),'SMC':(302.7969,-44.2992,5),'Sculptor':(287.5,-83.2,1), 'Carina':(260.1,-22.2,0.5), 'Fornax':(237.1,-65.7,1.2),'Draco':(86.4,34.7,1),'Uminor':(105.0,44.8,0.5), 'Hercules':(28.7,36.9,0.5),'Sextans':(243.5,42.3,1),'LeoI':(226,49.1,0.3),'LeoII':(220.2,67.2,0.2)}
#struct_Galaxy={'Sculptor':(287.5,-83.2,1), 'Carina':(260.1,-22.2,0.5), 'Fornax':(237.1,-65.7,1.2),'Draco':(86.4,34.7,1),'Uminor':(105.0,44.8,0.5), 'Hercules':(28.7,36.9,0.5),'Sextans':(243.5,42.3,1),'LeoI':(226,49.1,0.3),'LeoII':(220.2,67.2,0.2)}
#struct_Galaxy={'Dra':(86.40,34.70,3.33), 'UMi':(105.00,44.80,2.73), 'Scl':(287.50,-83.20,3.77), 'SexI':(243.50,42.30,9.27), 'UMaI':(159.40,54.40,3.77), 'Car':(260.10,-22.20,2.73), 'Her':(28.70,36.90,2.87), 'For':(237.10,-65.70,5.53), 'SegueI':(220.50,50.40,1.47), 'UMaII':(152.50,37.40,5.33), 'BooII':(353.70,68.90,1.40), 'SegueII':(149.40,-38.10,1.13), 'Wil1':(158.60,56.80,0.77), 'CmB':(241.90,83.60,2.00), 'BooI':(358.10,69.60,4.20), 'LeoIV':(264.40,57.40,1.53), 'CnVII':(113.60,82.70,0.53), 'LeoV':(261.90,58.50,0.87), 'PisII':(79.20,-47.10,0.37), 'CnVI':(74.30,79.80,2.97), 'LeoII':(220.20,67.20,0.87), 'LeoI':(226.00,49.10,1.13), 'Col1':(231.60,-28.90,0.63), 'Grs2':(351.10,-51.90,2.00), 'Grs1':(338.70,-58.20,0.59), 'Hor1':(271.40,-54.70,0.44), 'Hor2':(262.50,-54.10,0.70), 'Ind2':(354.00,-37.40,0.97), 'Phe2':(323.70,-59.70,0.36), 'Pic1':(257.30,-40.60,0.29), 'Ret2':(266.30,-49.70,1.21), 'Tuc2':(328.10,-52.30,3.28), 'Ret3':(273.90,-45.60,0.80), 'Tuc3':(315.40,-56.20,2.00), 'Tuc4':(313.30,-55.30,3.03), 'EriII':(249.78,-51.65,0.60), 'EriIII':(274.95,-59.60,0.14), 'IndI':(347.16,-42.07,0.20), 'PegIII':(69.84,-41.83,0.28), 'HydII':(295.61,30.46,0.57), 'Aqu2':(55.11,-53.01,1.70), 'VirI':(276.94,59.58,0.50), 'CetusIII':(163.81,-61.16,0.41), 'CarI':(269.98,-17.14,2.90), 'CarII':(270.01,-16.85,1.25), 'Cra':(274.80,47.80,0.20), 'Cra2':(282.91,42.03,10.40), 'DES1':(310.52,-67.83,0.13), 'Pic2':(269.63,-24.05,1.20), 'TriII':(140.90,-23.80,1.30), 'Lae3':(63.60,-21.20,0.13), 'DraII':(98.30,42.90,0.90), 'SgrII':(18.90,-22.90,0.67)}
#struct_Galaxy={'Carina':(260.1,-22.2,0.5), 'Fornax':(237.1,-65.7,1)}


#(l, b, 15*rh [deg], dist [kpc])
struct_Galaxy={'Dra':(86.40,34.70,2.50,76.00), 'UMi':(105.00,44.80,2.05,76.00), 'Scl':(287.50,-83.20,2.83,86.00), 'SexI':(243.50,42.30,6.95,86.00), 'UMaI':(159.40,54.40,2.83,97.00), 'Car':(260.10,-22.20,2.05,105.00), 'Her':(28.70,36.90,2.15,132.00), 'For':(237.10,-65.70,4.15,147.00), 'SegueI':(220.50,50.40,1.10,23.00), 'UMaII':(152.50,37.40,4.00,32.00), 'BooII':(353.70,68.90,1.05,42.00), 'SegueII':(149.40,-38.10,0.85,35.00), 'Wil1':(158.60,56.80,0.57,38.00), 'CmB':(241.90,83.60,1.50,44.00), 'BooI':(358.10,69.60,3.15,66.00), 'LeoIV':(264.40,57.40,1.15,154.00), 'CnVII':(113.60,82.70,0.40,160.00), 'LeoV':(261.90,58.50,0.65,178.00), 'PisII':(79.20,-47.10,0.28,182.00), 'CnVI':(74.30,79.80,2.23,218.00), 'LeoII':(220.20,67.20,0.65,233.00), 'LeoI':(226.00,49.10,0.85,254.00), 'Col1':(231.60,-28.90,0.47,182.00), 'Grs2':(351.10,-51.90,1.50,53.00), 'Grs1':(338.70,-58.20,0.44,120.00), 'Hor1':(271.40,-54.70,0.33,79.00), 'Hor2':(262.50,-54.10,0.52,78.00), 'Ind2':(354.00,-37.40,0.72,214.00), 'Phe2':(323.70,-59.70,0.27,83.00), 'Pic1':(257.30,-40.60,0.22,115.00), 'Ret2':(266.30,-49.70,0.91,30.00), 'Tuc2':(328.10,-52.30,2.46,58.00), 'Ret3':(273.90,-45.60,0.60,92.00), 'Tuc3':(315.40,-56.20,1.50,25.00), 'Tuc4':(313.30,-55.30,2.27,48.00), 'EriII':(249.78,-51.65,0.45,330.00), 'EriIII':(274.95,-59.60,0.10,95.00), 'IndI':(347.16,-42.07,0.15,69.00), 'PegIII':(69.84,-41.83,0.21,215.00), 'HydII':(295.61,30.46,0.42,134.00), 'Aqu2':(55.11,-53.01,1.27,107.90), 'VirI':(276.94,59.58,0.38,87.00), 'CetusIII':(163.81,-61.16,0.31,251.00), 'CarI':(269.98,-17.14,2.17,36.20), 'CarII':(270.01,-16.85,0.94,27.80), 'Cra':(274.80,47.80,0.15,170.00), 'Cra2':(282.91,42.03,7.80,117.50), 'DES1':(310.52,-67.83,0.10,87.10), 'Pic2':(269.63,-24.05,0.90,45.00), 'TriII':(140.90,-23.80,0.97,30.00), 'Lae3':(63.60,-21.20,0.10,67.00), 'DraII':(98.30,42.90,0.68,20.00), 'SgrII':(18.90,-22.90,0.50,67.00)}

#(l, b, 2*Rt(o 10*rh) [deg], dist [kpc])
struct_GC={'2MASS-GC01':(10.48,0.11,0.20,3.60), '2MASS-GC02':(9.79,-0.61,0.16,4.90), 'Arp2':(8.55,-20.79,0.30,28.60), 'BH176':(328.41,4.34,0.21,18.90), 'BH261':(3.36,-5.27,0.13,6.50), 'Djorg1':(356.69,-2.47,0.53,13.70), 'Djorg2':(2.77,-2.50,0.35,6.30), 'E3':(292.27,-19.02,0.35,8.10), 'Eridanus':(218.10,-41.33,0.10,90.10), 'FSR1735':(339.19,-1.85,0.05,9.80), 'HP1':(357.44,2.12,0.32,8.20), 'IC1257':(16.54,15.15,0.30,25.00), 'IC1276':(21.83,5.67,0.72,5.40), 'IC4499':(307.35,-20.47,0.45,18.80), 'Kosopov1':(260.99,70.75,0.03,48.30), 'Kosopov2':(195.12,25.54,0.03,34.70), 'Liller1':(354.84,-0.16,0.40,8.20), 'Lynga7':(328.77,-2.80,0.27,8.00), 'NGC104':(305.89,-44.89,1.41,4.50), 'NGC1261':(270.54,-52.12,0.17,16.30), 'NGC1851':(244.51,-35.03,0.22,12.10), 'NGC1904':(227.23,-29.35,0.27,12.90), 'NGC2298':(245.63,-16.00,0.25,10.80), 'NGC2419':(180.37,25.24,0.25,82.60), 'NGC2808':(282.19,-11.25,0.30,9.60), 'NGC288':(152.30,-89.38,0.44,8.90), 'NGC3201':(277.23,8.64,0.85,4.90), 'NGC362':(301.53,-46.25,0.35,8.60), 'NGC4147':(252.85,77.19,0.20,19.30), 'NGC4372':(300.99,-9.88,1.16,5.80), 'NGC4590':(299.63,36.05,0.50,10.30), 'NGC4833':(303.60,-8.02,0.59,6.60), 'NGC5024':(332.96,79.76,0.61,17.90), 'NGC5053':(335.70,78.95,0.38,17.40), 'NGC5139':(309.10,14.97,1.61,5.20), 'NGC5272':(42.22,78.71,0.96,10.20), 'NGC5286':(311.61,10.57,0.24,11.70), 'NGC5466':(42.15,73.59,0.52,16.00), 'NGC5634':(342.21,49.26,0.35,25.20), 'NGC5694':(331.06,30.36,0.16,35.00), 'NGC5824':(332.56,22.07,0.19,32.10), 'NGC5897':(342.95,30.29,0.34,12.50), 'NGC5904':(3.86,46.80,0.79,7.50), 'NGC5927':(326.60,4.86,0.56,7.70), 'NGC5946':(327.58,4.19,0.84,10.60), 'NGC5986':(337.02,13.27,0.27,10.40), 'NGC6093':(352.67,19.46,0.24,10.00), 'NGC6101':(317.74,-15.82,0.20,15.40), 'NGC6121':(350.97,15.97,1.73,2.20), 'NGC6139':(342.37,6.94,0.36,10.10), 'NGC6144':(351.93,15.70,1.11,8.90), 'NGC6171':(3.37,23.01,0.63,6.40), 'NGC6205':(59.01,40.91,0.70,7.10), 'NGC6218':(15.72,26.31,0.58,4.80), 'NGC6229':(73.64,40.31,0.13,30.50), 'NGC6235':(358.92,13.52,0.37,11.50), 'NGC6254':(15.14,23.08,0.62,4.40), 'NGC6256':(347.79,3.31,0.21,10.30), 'NGC6266':(353.57,7.32,0.38,6.80), 'NGC6273':(356.87,9.38,0.49,8.80), 'NGC6284':(358.35,9.94,0.74,15.30), 'NGC6287':(0.13,11.02,0.23,9.40), 'NGC6293':(357.62,7.83,0.53,9.50), 'NGC6304':(355.83,5.38,0.44,5.90), 'NGC6316':(357.18,5.76,0.25,10.40), 'NGC6325':(0.97,8.00,0.32,7.80), 'NGC6333':(5.54,10.71,0.27,7.90), 'NGC6341':(68.34,34.86,0.41,8.30), 'NGC6342':(4.90,9.72,0.53,8.50), 'NGC6352':(341.42,-7.17,0.35,5.60), 'NGC6355':(359.59,5.43,0.53,9.20), 'NGC6356':(6.72,10.22,0.31,15.10), 'NGC6362':(325.55,-17.57,0.46,7.60), 'NGC6366':(18.41,16.04,0.40,3.50), 'NGC6380':(350.18,-3.42,0.40,10.90), 'NGC6388':(345.56,-6.74,0.23,9.90), 'NGC6397':(338.17,-11.96,0.53,2.30), 'NGC6401':(3.45,3.98,0.41,10.60), 'NGC6402':(21.32,14.81,0.26,9.30), 'NGC6426':(28.09,16.23,0.43,20.60), 'NGC6440':(7.73,3.80,0.19,8.50), 'NGC6441':(353.53,-5.01,0.24,11.60), 'NGC6453':(355.72,-3.87,0.53,11.60), 'NGC6496':(348.03,-10.01,0.16,11.30), 'NGC6517':(19.23,6.76,0.13,10.60), 'NGC6522':(1.02,-3.93,0.53,7.70), 'NGC6528':(1.14,-4.17,0.14,7.90), 'NGC6535':(27.18,10.44,0.26,6.80), 'NGC6539':(20.80,6.78,0.70,7.80), 'NGC6540':(3.29,-3.31,0.32,5.30), 'NGC6541':(349.29,-11.19,0.43,7.50), 'NGC6544':(5.84,-2.20,0.07,3.00), 'NGC6553':(5.26,-3.03,0.26,6.00), 'NGC6558':(0.20,-6.02,0.32,7.40), 'NGC6569':(0.48,-6.68,0.24,10.90), 'NGC6584':(342.14,-16.41,0.26,13.50), 'NGC6624':(2.79,-7.91,0.63,7.90), 'NGC6626':(7.80,-5.58,0.37,5.50), 'NGC6637':(1.72,-10.27,0.26,8.80), 'NGC6638':(7.90,-7.15,0.16,9.40), 'NGC6642':(9.81,-6.44,0.33,8.10), 'NGC6652':(1.53,-11.38,0.21,10.00), 'NGC6656':(9.89,-7.55,1.06,3.20), 'NGC6681':(2.85,-12.51,0.32,9.00), 'NGC6712':(25.35,-4.32,0.28,6.90), 'NGC6715':(5.61,-14.09,0.33,26.50), 'NGC6717':(12.88,-10.90,0.31,7.10), 'NGC6723':(0.07,-17.30,0.36,8.70), 'NGC6749':(36.20,-2.21,0.13,7.90), 'NGC6752':(336.49,-25.63,1.79,4.00), 'NGC6760':(36.11,-3.92,0.51,7.40), 'NGC6779':(62.66,8.34,0.35,9.40), 'NGC6809':(8.79,-23.27,0.51,5.40), 'NGC6838':(56.75,-4.56,0.30,4.00), 'NGC6864':(20.30,-25.75,0.19,20.90), 'NGC6934':(52.10,-18.89,0.25,15.60), 'NGC6981':(35.16,-32.68,0.25,17.00), 'NGC7006':(63.77,-19.41,0.15,41.20), 'NGC7078':(65.01,-27.31,0.91,10.40), 'NGC7089':(53.37,-35.77,0.41,11.50), 'NGC7099':(27.18,-46.84,0.63,8.10), 'NGC7492':(53.39,-63.48,0.15,26.30), 'Pal1':(130.06,19.03,0.12,11.10), 'Pal10':(52.43,2.72,0.10,5.90), 'Pal11':(31.81,-15.57,0.15,13.40), 'Pal12':(30.51,-47.68,0.64,19.00), 'Pal13':(87.10,-42.70,0.07,26.00), 'Pal14':(28.74,42.19,0.17,76.50), 'Pal15':(18.88,24.30,0.16,45.10), 'Pal2':(170.53,-9.07,0.19,27.20), 'Pal3':(240.15,41.86,0.13,92.50), 'Pal4':(202.31,71.80,0.09,108.70), 'Pal5':(0.85,45.86,0.25,23.20), 'Pal6':(2.10,1.78,0.28,5.80), 'Pal8':(14.11,-6.79,0.63,12.80), 'Pyxis':(261.32,7.00,0.00,39.40), 'Rup106':(300.88,11.67,0.17,21.20), 'Terzan1':(357.57,1.00,0.42,6.70), 'Terzan10':(4.49,-1.99,0.17,5.80), 'Terzan12':(8.37,-2.10,0.10,4.80), 'Terzan2':(356.32,2.30,0.32,7.50), 'Terzan3':(345.08,9.19,0.20,8.20), 'Terzan4':(356.02,1.31,0.24,7.20), 'Terzan5':(3.84,1.69,0.22,6.90), 'Terzan6':(358.57,-2.16,0.53,6.80), 'Terzan7':(3.39,-20.07,0.14,22.80), 'Terzan8':(5.76,-24.56,0.13,26.30), 'Terzan9':(3.61,-1.99,0.32,7.10), 'Ton2':(350.80,-3.42,0.36,8.20), 'UKS1':(5.13,0.76,0.63,7.80), 'Whiting1':(161.22,-60.76,0.03,30.10)}




struct_list_object={}
#test={'Scl':(287.50,-83.20,3.77,86.00),}
#struct_list_object.update(test)
struct_list_object.update(struct_GC)
struct_list_object.update(struct_Galaxy)



