from __future__ import division
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.io.fits as ft
import numpy as np
from time import gmtime, strftime
import matplotlib.pyplot as plt
import OpOp.Model as md
from roteasy import rotate
import OpOp.grid as gd
import OpOp.analysis as an
from functools import partial
from scipy.interpolate import UnivariateSpline
import pycam.utils as ut
import emcee as em
from pycam.cutils import newton,calc_m
from OpOp.particle import Particles

def power_law(rnorm,beta,rs=1,rt=None):
    """
    Simplet Power law  d=(r/rs)^-beta=rnorm^-beta
    :param rnorm: Normalized radii (r/rs)
    :param beta: PL exponent
    :param rs: scale radius
    :param rt: truncation radius
    :return:
    """

    if rt is None: fade=1
    else:  fade=(np.exp(-(rnorm*rnorm*rs*rs)/(rt*rt)))

    return fade*(rnorm**-beta)

def power_law_break(rnorm,beta,gamma,rb,rs=1,rt=None):
    """
    Broken Power law  d=(r/rs)^-gamma if r<rb; d=(r/rs)^-beta if r>rb

    :param rnorm: Normalized radii (r/rs)
    :param beta: PL exponent in the outer parts
    :param beta: PL exponent in the inner parts
    :param rs: scale radius
    :param rb: Break radius (not normalized!)
    :param rt: truncation radius
    :return:
    """
    if rt is None: fade=1
    else: fade=(np.exp(-(rnorm*rnorm*rs*rs)/(rt*rt)))
    rb_norm=rb/rs
    ret=np.where(rnorm<=rb_norm,rnorm**-gamma,(rb_norm**(beta-gamma))*(rnorm**-beta))

    return fade*ret

def double_power_law(rnorm,rc,beta,gamma=None,rt=None):
    """
    rnorm is r/rc
    """
    if rt is None: fade=1
    else: fade=(np.exp(-(rnorm*rnorm*rc*rc)/(rt*rt)))

    if gamma is None: inner_dens=1
    else: inner_dens=rnorm**(-gamma)

    outer_dens=(1+rnorm)**(-beta)

    return fade*inner_dens*outer_dens


def einasto(rnorm,n,rs=1,rt=None):
    """
    Einasto profiles
    dn=3n-173.+0.0079/n only for n>0.5 (Graham et al. 2006).
    :param rnorm: Normalized radius
    :param n: Index n, n about 6 for DM cuspy profiles
    :param rs: Reff
    :param rt: Truncation radius
    :return:
    """
    if rt is None: fade=1
    else:  fade=(np.exp(-(rnorm*rnorm*rs*rs)/(rt*rt)))

    dn=3*n-1/3.+0.0079/n

    exp=dn*((rnorm)**(1/n) - 1)

    return np.exp(-exp)

def qfunc_exp(m,q0,qinf,rq):
    """

    :param m: The ellipsoidal m
    :param q0: q at m=0
    :param qinf:  q at m=inf
    :param rq:  scale radius
    :return:
    """
    ep=1-np.sqrt(rq*rq+m*m)/rq

    return qinf - (qinf-q0)*np.exp(ep)


def qfunc_tanh(m, q0=1, qinf=1, rq=1, eta=1):
    """

    :param m: The ellipsoidal m
    :param q0: q at m=0
    :param qinf:  q at m=inf
    :param rq:  scale radius
    :param eta:  regulates the region of transition, if eta=1, it is the classical tanh for eta>1 the transition is slower, for eta<1 it is faster.
    :return:
    """
    tm = np.tanh((m - rq) / eta)
    t0 = np.tanh((-rq) / eta)

    C = (qinf - q0) / (1 - t0)

    ret = qinf + C * (tm - 1)

    return ret


def logl_part(theta,dens_func,rc,q,qinf,rq,rmax,rmin):
    """

	:param theta:
	:param dens_func: func of m/rc
	:param rc:
	:param q:
	:param qinf:
	:param rq:
	:param rmax:  in func of rc
	:param rmin:  in func of rc
	:return:
	"""
    x,y,z=theta
    r=np.sqrt(x*x+y*y+z*z)
    if q==qinf: m=np.sqrt(x*x+y*y+(z*z)/(q*q))
    else: m=newton(r,x,y,z,q,qinf,rq)
    mnorm=m/rc

    if mnorm>rmax or mnorm<rmin: return -np.inf

    return np.log(dens_func(mnorm))

def generate_particle_mc(logl_func,arg_list,p0lim=1,nwalker=300,nstep=333,burn=1000):
    """

    :param logl_func: Loglike function
    :param arg_list:  arg list to pass to the loglike function
    :param p0lim:  The inital walkers will be randomly distribuited between -p0lim and p0lim
    :param nwalker:
    :param nstep:
    :param burn:
    :return:
    """
    dim=3
    #Initialise sampler
    sampler = em.EnsembleSampler(nwalker, dim, logl_func, args=arg_list)
    p0 = np.random.uniform(low=-p0lim, high=p0lim, size=(nwalker, dim))
    p0, prob, state = sampler.run_mcmc(p0, burn)
    sampler.reset()
    sampler.run_mcmc(p0, nstep)
    samples = sampler.chain[:, :, :].reshape((-1, dim))

    pp=Particles(N=nstep*nwalker)
    pp.Pos=samples
    pp.Type[:]=np.ones_like(pp.Type)
    pp.Mass[:]=np.ones_like(pp.Mass)
    pp.setrad()

    return pp


def halo_model(ainn=0,aout=0,rc=1,rt=None,rb=None,neinasto=None,rmin=4e-5,rmax=400,npart=1e5,q=1,qinf=None,eta=None,rq=1,p=1,mode='dist'):
    """
    Build a nbody realisation of a certain density function.
    @a: Exponent of the power law.
    @aini: Exponent of the inner power law.
    @rc:  Scale radius, if aini=0 e rc != 0 is the core radius.
    @rt: If not not None the radius of the truncation.
    @rb: If not None and aini not None and rc=None, the break radius.
    @rmin: Minimus radius where extract the particles (It is in units of rc, if rc is None it is non-rnomalized r)
    @rmax: Max radius where extract the particles (It is in units of rc, if rc is None it is non-rnomalized r)
    @npart: number of particles to extract
    @q: z-flattening of the model or z-flattening at m=0 i qinf is not None
    @qinf: asymptotic value of the z-flattenig.
    @rq: Scale radius for the z-flattening variation.
    @eta: defines the slope of the z-flattening variation.
    return Cord: array with [X,Y,Z] cord.
    Can realize the following density law:
    1-Power law: d propto r/rc**-ainn (rb=None, rt=None aout=0)
    2-Truncated power law: d propto (r/rc**-ainn)*exp(-(r/rt)**2)  (rb=None, aout=0)
    3-Break power law: d propto  r/rc**-ainn if r<rb; r/rc**-aout if r>rb (rt=None)
    4-Truncated Break power law: d propto  exp(-(r/rt)**2)*r/rc**-ainn  if r<rb; exp(-(r/rt)**2)*r/rc**-aout if r>rb
    5-Cored profile: d propto (1+r/rc)**-aout (ainn=0,rt=None,rb=None)
    6-Truncated cored profile: d propto exp(-(r/rt)**2)*(1+r/rc)**-a (ainn=0rb=None)
    7-Double Power law model: ((r/rc)**-ainn)*(1+r/rc)**-aout (rb=None,rt=None)
    8-Truncated Double Power law model: exp(-(r/rt)**2)*((r/rc)**-ainn)*(1+r/rc)**-aout (rb=None)
    About the flattening, p is the flattening along the y-axis and it could be only a single value,
    q is the flattening along the z-axis  and it could have three difference functional form:
        -cost: if qinf is None the q is everywhere constant to the value q
        -exp: qinf - (qinf-q0)*exp(1+sqrt(rq^2+m^2)/rq) if qinf is not note and eta is None
        -tanh: qinf + (qinf-q0)/(1-tanh(-rq/eta))*(tanh( (m-rq)/eta)-1) is qinf is not None and eta is not None
    """

    if qinf is None:
        qinf=q
        qfunc=partial(qfunc_exp,q0=q,qinf=qinf,rq=rq)
    elif eta is None:
        qfunc = partial(qfunc_exp, q0=q, qinf=qinf, rq=rq)
    else:
        qfunc = partial(qfunc_tanh, q0=q, qinf=qinf, rq=rq,eta=eta)


    '''
    #Set q and p function
    if isinstance(q,int) or isinstance(q,float):
        def qfunc(r):
            if isinstance(r,int) or isinstance(r,float): return q
            else: return  np.where(r==0,q,q*r/r)
    else:
        qfunc=q
    '''

    if isinstance(p,int) or isinstance(p,float):
        def pfunc(r):
            if isinstance(r,int) or isinstance(r,float): return p
            else: return  np.where(r==0,p,p*r/r)
    else:
        pfunc=p

    gamma=ainn
    beta=aout

    if rb is not None: fdens=partial(power_law_break,rs=rc,beta=beta,gamma=gamma,rb=rb,rt=rt)
    elif neinasto is not None: fdens=partial(einasto,n=neinasto,rs=rc,rt=rt)
    else: fdens=partial(double_power_law,rc=rc,beta=beta,gamma=gamma,rt=rt)


    rini=rmin
    rfin=rmax

    if mode=='dist':

        if rc*400<200: rfin=400/rc
        if rc*4e-5<4e-5: rini=4e-5

        R=np.logspace(np.log10(rini+0.01),np.log10(rfin+0.01),512)-0.01
        mod=md.GeneralModel(R,fdens,q=1,rc=rc)
        nb=md.NbodyModel([{'type':1,'model':mod,'npart':int(npart)}],xmin=rini,xmax=rfin)

        particles=nb.generate(set_vel=False)

        #The generated particle are spherical
        r=np.sqrt(particles.Pos[:,0]**2+particles.Pos[:,1]**2+particles.Pos[:,2]**2)

        p=pfunc(r)
        q=qfunc(r)

        cord=np.zeros_like(particles.Pos)
        cord[:,0]=particles.Pos[:,0]
        cord[:,1]=particles.Pos[:,1]*p
        cord[:,2]=particles.Pos[:,2]*q

    if mode=='ar':
        #Acceptance-Rejection method. We use as envelope function
        #the spherical power-law, since (x^2+y^2+(z)^2)^(-a)>(x^2+y^2+(z/q)^2)^(-a)
        #for q<1.

        if q>1 or qinf>1 or (isinstance(p,int)==False and isinstance(p,float)==False): raise ValueError('Halo model error: q(m)>1 not allowed')
        if rc*400<200: rfin=400/rc
        if rc*4e-5<4e-5: rini=4e-5
        #print(rini,rfin,rc)
        #input()
        R=np.logspace(np.log10(rini+0.01),np.log10(rfin+0.01),512)-0.01
        mod=md.GeneralModel(R,fdens,q=1,rc=rc)
        nb=md.NbodyModel([{'type':1,'model':mod,'npart':int(npart)}],xmin=rini,xmax=rfin)
        particles_tmp=nb.generate(set_vel=False)
        x=particles_tmp.Pos[:,0]
        y=particles_tmp.Pos[:,1]*p
        z=particles_tmp.Pos[:,2]
        r=np.sqrt(x*x+y*y+z*z) #spherical radius with p


        m=np.array(calc_m(x,y,z,q,qinf,rq,p=p,eta=eta))


        #Remeber the argument of dens should be divided by rc
        g=fdens(r/rc) #The envelope function for the rejection method
        f=fdens(m/rc) #The real function




        uprob=np.random.uniform(0,1,len(m))*g
        idx=(uprob<=f)#&(m/rc<=rfin) #Questa seconda condizione è un pò azzardata! Ricordalo!


        #The idx particles all alredy in a ellipsoidal configuration, we do not need to modify particles
        cord=np.zeros_like(particles_tmp.Pos[idx])
        cord[:,0]=particles_tmp.Pos[idx,0]
        cord[:,1]=particles_tmp.Pos[idx,1]*p
        cord[:,2]=particles_tmp.Pos[idx,2]


        #To be consistent with the return calculate the spherical r
        r=np.array(calc_m(cord[:,0],cord[:,1],cord[:,2],q,qinf,rq,p=p,eta=eta))
        p=pfunc(r)
        q=qfunc(r)

        #Generate particles for output
        particles=Particles(N=np.sum(idx))
        particles.Pos[:,0]=particles_tmp.Pos[idx,0]
        particles.Pos[:,1]=particles_tmp.Pos[idx,1]*p[0]
        particles.Pos[:,2]=particles_tmp.Pos[idx,2]
        particles.Type[:]=particles_tmp.Type[idx]
        particles.Mass[:]=particles_tmp.Mass[idx]
        #Put in the particles the spherical positions
        particles.Pos[:,2]/=q
        particles.Pos[:,1] /= p
        particles.setrad()

    if mode=='mc':

        print('Generating particles in mc mode',flush=True)

        #Raise error if p!=1 (we need to implement it)
        if isinstance(p,int) or isinstance(p,float):
            if p!=1: raise NotImplementedError('p!=1 not implemented for Particle geneartio model mc')
        else: raise NotImplementedError('p!=1 not implemented for Particle generation model mc')

        nwalker=300
        nstep=int(npart/nwalker)
        alist=(fdens,rc,q,qinf,rq,rmax,rmin)
        particles=generate_particle_mc(logl_part,arg_list=alist,p0lim=rc,nwalker=nwalker,nstep=nstep,burn=3000)

        #The particles all alrady in a ellipsoidal configuration, we do not need to modify particles
        cord=np.zeros_like(particles.Pos)
        cord[:,0]=particles.Pos[:,0]
        cord[:,1]=particles.Pos[:,1]
        cord[:,2]=particles.Pos[:,2]

        #To be consistent with the return calculate the spherical r
        r=np.array(calc_m(cord[:,0],cord[:,1],cord[:,2],q,qinf,rq,eta=eta))
        p=pfunc(r)
        q=qfunc(r)
        #Put in the particles the spherical positions
        particles.Pos[:,2]=particles.Pos[:,2]/q
        particles.setrad()

        print('Done',flush=True)

    return cord, particles,fdens,r,q,p

def halo_observe(cord, xsun=8):
    """
    Take the X,Y,Z coordinate in the Galactic reference and return the l,b distance from the sun assuming xsun as distance of the sun from the Galactic centre
    :param cord: X,Y,Z in galactic frame of reference, (left-hand X toward the sun. Y toward the sun rotation)
    :param xsun: position of the sun in kpc
    :return: l,b galactic coordinate, distance from the sun
    """
    cost=360./(2*np.pi)
    x_s=cord[:,0]
    y_s=cord[:,1]
    z_s=cord[:,2]
    x_s=xsun-x_s
    rad_s=np.sqrt(x_s*x_s+y_s*y_s+z_s*z_s)
    l=np.arctan2(y_s,x_s)*cost
    l[l<0]+=360
    b=np.arcsin(z_s/rad_s)*cost

    return l,b,rad_s

def halo_mask(l,b,gmag,bmin=None,gmagmin=None,gmagmax=None,bmax=None,thetamin=None,thetamax=None,zgmin=None,zgmax=None,struct=(),xsun=8,Mg=0.525):
    """
    Take the l and b  and gmang of a certain distriution of star and returtn the filter version
    following:
    :param bmin: Minimum galactic latitude to take into account
    :param gmin: Minimumum g mag to take into account
    :param gmax: Max g maf to take into account
    :param struct: Mask part of the sky, each member of struct need to be an array-like with the following
    structure: [lmin,lmax,bmin,bmax] oppure [lc,bc,rc] and in this case the function mask all the star that have
    a distance less than rc from a sky point of coordinate (lc,bc)
    :return:
    """
    if (bmin is None) and (bmax is None) and (gmagmin is None) and (gmagmax is None) and (len(struct)==0) and (thetamin is None) and (thetamax is None) and (zgmin is None) and (zgmax is None): return l,b,gmag,np.array([True,]*len(l))
    else:
        idx=np.array([True,]*len(l))
        #general
        if bmin is not None: idx*=np.abs(b)>=bmin
        if gmagmin is not None: idx*=gmag>=gmagmin
        if gmagmax is not None: idx*=gmag<=gmagmax
        if bmax is not None: idx*=np.abs(b)<=bmax

        if (thetamin is not None) or (thetamax is not None) or (zgmin is not None) or (zgmax is not None):

            rg,zg=ut.obs_to_cyl(gmag,l,b,Mg=Mg,xsun=xsun)
            zg_abs=np.abs(zg)
            theta_abs=np.arctan(zg_abs/rg)*(360./(2*np.pi))

            if thetamin is not None: idx*=theta_abs>=thetamin
            if thetamax is not None: idx*= theta_abs<= thetamax
            if zgmin is not None: idx*=zg_abs>=zgmin
            if zgmax is not None: idx*=zg_abs<=zgmax

        #struct
        for st in struct:
            if len(st)==3:
                lmg,bmg,rmc=st
                skysep_dmg=ut.skysep(l,b,lmg,bmg)
                idx*=skysep_dmg>rmc
            elif len(st)==4:
                lmin,lmax,bmin,bmax=st
                idx*=(l<lmin)|(l>lmax)|(b<bmin)|(b>bmax)
            else: raise NotImplementedError('Halo mask function-> The struct need containt 3 o 4 items')

    return l[idx],b[idx],gmag[idx],idx

def _exponential_radial_cumulative_mass(x):
    return 1-(1+x)*np.exp(-x)

def disc_model(rd=1,zd=1,npart=1e5):
    """
    Generate position of a double exponential disc model
    :param rd:  Radial scale-length
    :param zd: Vertical scale-length
    :param npart:  Number of particle to generate
    :return:
    """
    npart=int(npart)
    cord=np.zeros(shape=(npart,3))
    #Invert cumulative mass distribution
    x=np.logspace(np.log10(0+0.01),np.log10(20+0.01),10000)-0.01
    um=_exponential_radial_cumulative_mass(x)
    um=um/um[-1] #To normalize to 1
    fint=UnivariateSpline(um,x,s=0)

    #Generate radial sampling
    ur=np.random.uniform(0,1,size=npart)
    R=rd*fint(ur)
    #Generate |z| sampling
    uz=np.random.uniform(0,1,size=npart)
    zabs=-zd*np.log(1-uz)
    #Generate x and y
    uphi=np.random.uniform(0,2*np.pi,size=npart)
    cord[:,0]=R*np.cos(uphi)
    cord[:,1]=R*np.sin(uphi)
    #Generate z
    cord[:,2]=zabs*np.random.choice([-1,1],size=npart)



    return cord

#Tested
def disc_mag(isofile,dist,agecut=(None,None),fecut=(None,None),gcut=(None,None),colcut=(None,None)):
    """
    Return the g mag given the distance of the star and a given distribution of Mg taken from the isochrones
    :param isofile: Fits file with the isochrones
    :param dist:  Distance of the stars in kpc
    :param agecut:  cut in log(age)
    :param fecut:  cut in metallicity
    :param gcut:  cut in g abs magnitude
    :param colcut:  cut in color j-g
    :return: observed g magnitude.
    """
    tf=ft.open(isofile)
    t=tf[1].data

    idx=np.array([True,]*len(t['g']))
    if agecut[0] is not None: idx*=t['age']>=agecut[0]
    if agecut[1] is not None: idx*=t['age']<=agecut[1]
    if fecut[0] is not None: idx*=t['feh']>=fecut[0]
    if fecut[1] is not None: idx*=t['feh']<=fecut[1]
    if gcut[0] is not None: idx*=t['g']>=gcut[0]
    if gcut[1] is not None: idx*=t['g']<=gcut[1]
    if colcut[0] is not None: idx*=(t['j']-t['g'])>=colcut[0]
    if colcut[1] is not None: idx*=(t['j']-t['g'])<=colcut[1]

    amg_list=t['g'][idx]
    x=np.sort(amg_list)
    cum=np.arange(0,len(x))/len(x)

    f=UnivariateSpline(cum,x,s=0)
    amg=f(np.random.uniform(0,1,len(dist)))

    mg=amg+5*(2+np.log10(dist))

    return mg,amg,amg_list

def gau_mag(dist,Mg0=0,Mgs=1):

    amg=np.random.normal(Mg0,Mgs,len(dist))
    mg=amg+5*(2+np.log10(dist))

    return mg,amg

def uni_mag(dist,range=(-2,5)):


    amg=np.random.uniform(range[0],range[1],len(dist))
    mg=amg+5*(2+np.log10(dist))

    return mg,amg


def make_model(aout=0,ainn=0,rc=1,rt=None,rb=None,neinasto=None,q=1,qinf=None,rq=1,p=1,wd=0,rd=2.682,zd=0.196,alpha=0,beta=0,gamma=0,xoff=0,yoff=0,zoff=0,rot_ax='zyx',mode='dist',bmin=10,gmin=10,gmax=19,n=20000,nt=100,nini=5e4,Mgh=0.525,Mgch=0.525,Mgsh=0.096,Mgd=0.55,Mgcd=0,Mgsd=1,Mgud=(-2,5),mask=None,name='model',diagnostic=True,agecut=(None,None),fecut=(None,None),colcut=(-1.0,-0.4),Mgcut=(-3,6),output=True,outdir='halo_model',thetamin=None,thetamax=None,zgmin=None,zgmax=None,eta=None):
    """

    :param aout:
    :param ainn:
    :param rc:
    :param rt:
    :param rb:
    :param neinasto:
    :param q:
    :param qinf:
    :param rq:
    :param p:
    :param wd:
    :param rd:
    :param zd:
    :param alpha:
    :param beta:
    :param gamma:
    :param xoff:
    :param yoff:
    :param zoff:
    :param rot_ax:  order of rotation
    :param mode:
    :param bmin:
    :param gmin:
    :param gmax:
    :param n:
    :param nt:
    :param nini:
    :param Mgh:
    :param Mgch:
    :param Mgsh:
    :param Mgd:
    :param Mgcd:
    :param Mgsd:
    :param Mgud:
    :param mask:
    :param name:
    :param diagnostic:
    :param agecut:
    :param fecut:
    :param colcut:
    :param Mgcut:
    :param output:
    :param outdir:
    :param thetamin:
    :param thetamax:
    :param zgmin:
    :param zgmax:
    :param eta:
    :return:
    """
	
    npart_ini=int(nini)
    npartsh=0
    npartsd=0
	
	
	
    d_offset=np.array((xoff,yoff,zoff))

    if name=='': outdir=outdir
    else: outdir+='_'+name
    ut.mkdir(outdir)

    #Set q and p function
    if qinf is None:
        qinf=q
        qfunc=partial(qfunc_exp,q0=q,qinf=qinf,rq=rq)
    elif eta is None:
        qfunc = partial(qfunc_exp, q0=q, qinf=qinf, rq=rq)
    else:
        qfunc = partial(qfunc_tanh, q0=q, qinf=qinf, rq=rq,eta=eta)


    if isinstance(p,int) or isinstance(p,float):
        def pfunc(r):
            if isinstance(r,int) or isinstance(r,float): return p
            else: return  np.where(r==0,p,p*r/r)
    else: pfunc=p

    if mode=='dist': phalo=pfunc
    else: phalo=p

    #Halo Mg distribution
    if isinstance(Mgh,float) or isinstance(Mgh,int): Mgh=float(Mgh)
    elif Mgh[0].lower()=='g': Mgh='g'
    elif Mgh[:2].lower()=='dg': Mgh='dg'
    else: raise NotImplementedError('%s not implemenented for Mgh'%Mgh)

    #Disc Mg distribution
    if isinstance(Mgd,float) or isinstance(Mgd,int): Mgd=float(Mgd)
    elif Mgd[0].lower()=='f': Mgd=Mgd[1:] #from file
    elif Mgd[0].lower()=='g':Mgd='g' #gau
    elif Mgd[0].lower()=='u':Mgd='u' #uniform
    else: raise NotImplementedError('%s not implemented for Mgd'%Mgd)

    struct_l=[]
    print('struct_l',struct_l)
    print('mask',mask)
    if mask is not None:
        if mask[0]=='all':
            for item in ut.struct_list: struct_l.append(ut.struct_list[item.upper()])
        else:
            for item in mask:
                if isinstance(item,str):
                    if item.upper() in ut.struct_list: struct_l.append(ut.struct_list[item.upper()])
                    elif item.upper() in ut.struct_list_object: struct_l.append(ut.struct_list_object[item.upper()])
                    else: struct_l.append(tuple(float(i) for i in item.split(',')))
                else:
                    struct_l.append(item)


    #Halo
    if wd!=1:
        ntargeth=int(n*(1-wd))
        tollh=int(nt*(1-wd))
        print('Start- First Run Halo')
        cordh,parth,fdens_input,rtrue,qtrue,ptrue=halo_model(aout=aout,ainn=ainn,rt=rt,rb=rb,neinasto=neinasto,rc=rc,npart=npart_ini,p=phalo,q=q,qinf=qinf,rq=rq,mode=mode,eta=eta)


        print('Rotating and traslating....',end='')
        #cordh=ut.rotate_xyz(cordh,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff,system='lh')
        cordh=cordh+d_offset
        cordh=rotate(cord=cordh,angles=(alpha,beta,gamma),axes=rot_ax,reference='lh')
        print('Done')


        print('Observing....',end='')
        lh,bh,disth=halo_observe(cordh)
        if isinstance(Mgh,float):
            oldgh=ut.dist_to_g(disth,Mg=Mgh)
            agh=Mgh
        elif Mgh=='g': oldgh,agh=gau_mag(disth,Mg0=Mgch,Mgs=Mgsh)

        print('Done')
        print('Masking....',end='')
        lh,bh,gh,idxh=halo_mask(lh,bh,oldgh,bmin=bmin,gmagmin=gmin,gmagmax=gmax,struct=struct_l,thetamin=thetamin,thetamax=thetamax,zgmin=zgmin,zgmax=zgmax,xsun=8,Mg=agh)
        disth=disth[idxh]
        print('Done')
        npartsh=len(lh)
        print('Npart=%i'%npartsh)

        if (ntargeth-tollh)<=npartsh<=(ntargeth+tollh):
            print('Halo Done')
        else:
            corr_facth=ntargeth/npartsh
            npart_finalh=corr_facth*npart_ini
            if npart_finalh>7e6:
                print('Warning Maximum particles greatern than 3e6, it has ben set to 3e6')
                npart_finalh=int(7e6)
            print('Start Second Run')
            cordh,parth,fdens_inputh,rtrue,qtrue,ptrue=halo_model(aout=aout,ainn=ainn,rt=rt,rb=rb,neinasto=neinasto,rc=rc,npart=npart_finalh,p=phalo,q=q,qinf=qinf,rq=rq,mode=mode,eta=eta)
            print('Rotating and traslating....',end='')
            #cordh=ut.rotate_xyz(cordh,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff,system='lh')
            cordh = cordh + d_offset
            cordh = rotate(cord=cordh, angles=(alpha, beta, gamma), axes=rot_ax, reference='lh')
            print('Done')
            print('Observing....',end='')
            lh,bh,disth=halo_observe(cordh)
            if isinstance(Mgh,float):
                oldgh=ut.dist_to_g(disth,Mg=Mgh)
                agh=Mgh
            elif Mgh=='g': oldgh,agh=gau_mag(disth,Mg0=Mgch,Mgs=Mgsh)

            print('Done')
            print('Masking....',end='')
            lh,bh,gh,idxh=halo_mask(lh,bh,oldgh,bmin=bmin,gmagmin=gmin,gmagmax=gmax,struct=struct_l,thetamin=thetamin,thetamax=thetamax,zgmin=zgmin,zgmax=zgmax,xsun=8,Mg=agh)
            disth=disth[idxh]
            print('Halo Done')
            npartsh=len(lh)
            print('Final Npart=%i'%npartsh)

    ####Halo Diagnostic
    if diagnostic:
        print('Plotting halo density')
        prof=an.Profile(particles=parth,kind='log')
        dens_arr=prof.dens(func=False)
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        norm_x=np.searchsorted(dens_arr[:,0],2*rc)
        ax1.plot(np.log10(dens_arr[:,0]/rc),np.log10(dens_arr[:,1]/dens_arr[norm_x,1]),label='Particles',zorder=100,alpha=0.6,lw=2,c='blue')
        ax1.plot(np.log10(dens_arr[:,0]/rc),np.log10( (fdens_input(dens_arr[:,0]/rc)) / (fdens_input(dens_arr[norm_x,0]/rc)) ),label='Input',c='red')
        ax1.legend()
        ax1.set_xlabel('Log(R/Rc)')
        ax1.set_ylabel('Log(dens/(dens(2Rc)))')
        fig.savefig(outdir+'/diagnostic_halodens_spheerical.pdf')

        fig=plt.figure()
        ax1=fig.add_subplot(111)
        m_temp=np.sqrt(parth.Pos[:,0]**2+parth.Pos[:,1]**2+parth.Pos[:,2]**2)
        grid_temp=np.logspace(np.log10(np.min(m_temp)),np.log10(np.max(m_temp)),512)
        H_temp,edge_temp=np.histogram(m_temp,bins=grid_temp)
        mplot=0.5*(edge_temp[1:]+edge_temp[:-1])
        qvol_tmp=qfunc(grid_temp)
        vol_temp=(4./3.)*np.pi*(qvol_tmp[1:]*edge_temp[1:]**3 - qvol_tmp[:-1]*edge_temp[:-1]**3)
        dens_temp=H_temp/vol_temp
        norm_x=np.searchsorted(grid_temp,2*rc)
        ax1.plot(np.log10(mplot/rc),np.log10(dens_temp/dens_temp[norm_x]),label='Particles',zorder=100,alpha=0.6,lw=2,c='blue')
        ax1.plot(np.log10(mplot/rc),np.log10((fdens_input(mplot/rc))/(fdens_input(grid_temp[norm_x]/rc))),label='Input',c='red')
        ax1.legend()
        ax1.set_xlabel('Log(m/Rc)')
        ax1.set_ylabel('Log(dens/(dens(2Rc)))')
        fig.savefig(outdir+'/diagnostic_halodens_ellipsoid.pdf')

        #2
        if isinstance(Mgh,float):
            fig=plt.figure(figsize=(6,6))
            ax1=fig.add_subplot(111)
            ax1.hist(oldgh,bins=40,normed=True,alpha=0.5,label='All',color='blue',range=(5,25))
            ax1.hist(gh,bins=40,normed=True,alpha=0.5,label='Masked',color='red',range=(5,25))
            ax1.legend(loc='upper left')
            ax1.set_ylabel('N normed')
            ax1.set_xlabel('g')
            ax1.set_xlim(ax1.get_xlim()[::-1])
            ax1.set_title('Mgh:%.2f'%Mgh)
        else:
            fig=plt.figure(figsize=(12,5))
            ax1=fig.add_subplot(121)
            ax2=fig.add_subplot(122)

            ax1.hist(agh,bins=40,normed=True)
            ax2.hist(oldgh,bins=40,normed=True,alpha=0.5,label='All',color='blue',range=(5,25))
            ax2.hist(gh,bins=40,normed=True,alpha=0.5,label='Masked',color='red',range=(5,25))
            ax2.legend(loc='upper left')
            ax1.set_ylabel('N normed')
            ax1.set_xlabel('Mg')
            ax2.set_ylabel('N normed')
            ax2.set_xlabel('g')
            ax1.set_xlim(ax1.get_xlim()[::-1])
            ax2.set_xlim(ax2.get_xlim()[::-1])

        fig.savefig(outdir+'/diagnostic_Mgh.pdf')
        ##Q and p
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        rplot=np.linspace(0,100,100)
        ax1.plot(rplot,qfunc(rplot),color='red',label='q')
        ax1.plot(rplot,pfunc(rplot),color='blue',label='p')
        ax1.set_ylim(0,0.3+np.max([np.max(qfunc(rplot)),np.max(pfunc(rplot))]))
        ax1.legend()
        ax1.set_xlabel('r [kpc]')
        ax1.set_ylabel('q/p')
        fig.savefig(outdir+'/diagnostic_flattening_profile.pdf')
    ###########

    #Disc
    if wd!=0:
        ntargetd=int(n*wd)
        tolld=int(nt*wd)
        print('Start- First Run Disc')
        cordd=disc_model(rd,zd,npart=npart_ini)
        print('Observing....',end='')
        ld,bd,distd=halo_observe(cordd)
        print('Done')
        if isinstance(Mgd,float):
            oldgd=ut.dist_to_g(distd,Mg=Mgd)
            agd=Mgd
        elif Mgd=='g': oldgd,agd=gau_mag(distd,Mg0=Mgcd,Mgs=Mgsd)
        elif Mgd=='u': oldgd,agd=uni_mag(distd,range=Mgud)
        else: oldgd,agd,agd_input=disc_mag(Mgd,distd,agecut=agecut,fecut=fecut,gcut=Mgcut,colcut=colcut)
        print('Masking....',end='')
        ld,bd,gd,idxd=halo_mask(ld,bd,oldgd,bmin=bmin,gmagmin=gmin,gmagmax=gmax,struct=struct_l,thetamin=thetamin,thetamax=thetamax,zgmin=zgmin,zgmax=zgmax,xsun=8,Mg=agd)
        distd=distd[idxd]
        print('Done')
        npartsd=len(ld)
        if (ntargetd-tolld)<=npartsd<=(ntargetd+tolld):
            print('Disc Done')
        else:
            corr_factd=ntargetd/npartsd
            npart_finald=corr_factd*npart_ini
            if npart_finald>3e6:
                print('Warning Maximum particles greatern than 3e6, it has ben set to 3e6')
                npart_finald=int(3e6)
            print('Start Second Run Disc')
            cordd=disc_model(rd,zd,npart=npart_finald)
            print('Observing....',end='')
            ld,bd,distd=halo_observe(cordd)
            print('Done')
            if isinstance(Mgd,float):
                oldgd=ut.dist_to_g(distd,Mg=Mgd)
                agd=Mgd
            elif Mgd=='g': oldgd,agd=gau_mag(distd,Mg0=Mgcd,Mgs=Mgsd)
            elif Mgd=='u': oldgd,agd=uni_mag(distd,range=Mgud)
            else: oldgd,agd,agd_input=disc_mag(Mgd,distd,agecut=agecut,fecut=fecut,gcut=Mgcut,colcut=colcut)
            print('Masking....',end='')
            ld,bd,gd,idxd=halo_mask(ld,bd,oldgd,bmin=bmin,gmagmin=gmin,gmagmax=gmax,struct=struct_l,thetamin=thetamin,thetamax=thetamax,zgmin=zgmin,zgmax=zgmax,xsun=8,Mg=agd)
            distd=distd[idxd]
            print('Done')
            npartsd=len(ld)
            print('Disc Done')
            print('Final Npart=%i'%npartsd)


    ####Disc Diagnostic
        if diagnostic:
            print('Plotting disc diagnostic')
            if isinstance(Mgd,float)==False:
                fig=plt.figure(figsize=(12,5))
                ax1=fig.add_subplot(121)
                ax2=fig.add_subplot(122)
                if Mgd=='g' or Mgd=='u': ax1.hist(agd,bins=40,normed=True)
                else: ax1.hist((agd,agd_input),bins=40,normed=True)
                ax2.hist(oldgd,bins=40,normed=True,alpha=0.5,label='All',color='blue',range=(5,25))
                ax2.hist(gd,bins=40,normed=True,alpha=0.5,label='Masked',color='red',range=(5,25))
                ax2.legend(loc='upper left')
                ax1.set_ylabel('N normed')
                ax1.set_xlabel('Mg')
                ax2.set_ylabel('N normed')
                ax2.set_xlabel('g')
                ax1.set_xlim(ax1.get_xlim()[::-1])
                ax2.set_xlim(ax2.get_xlim()[::-1])
                fig.savefig(outdir+'/diagnostic_Mgd.pdf')
            else:
                fig=plt.figure(figsize=(6,6))
                ax1=fig.add_subplot(111)
                ax1.hist(oldgd,bins=40,normed=True,alpha=0.5,label='All',color='blue',range=(5,25))
                ax1.hist(gd,bins=40,normed=True,alpha=0.5,label='Masked',color='red',range=(5,25))
                ax1.legend(loc='upper left')
                ax1.set_ylabel('N normed')
                ax1.set_xlabel('g')
                ax1.set_xlim(ax1.get_xlim()[::-1])
                ax1.set_title('Mgd:%.2f'%Mgd)
                fig.savefig(outdir+'/diagnostic_Mgd.pdf')
    ###########


    ####Combined Diagnostic
    if 0<wd<1:
        if diagnostic:
            print('Plotting combined diagnostic')
            gp=np.hstack([gh,gd])
            oldgp=np.hstack([oldgh,oldgd])
            fig=plt.figure(figsize=(12,5))
            ax1=fig.add_subplot(121)
            ax2=fig.add_subplot(122)
            ax1.hist(oldgh,bins=40,normed=True,range=(5,25),label='All Halo',color='red')
            ax1.hist(oldgd,bins=40,normed=True,alpha=0.5,label='All Disc',color='blue',range=(5,25))
            ax1.hist(oldgp,bins=40,normed=True,alpha=0.5,label='All',color='green',range=(5,25))
            ax2.hist(gh,bins=40,normed=True,alpha=0.5,label='Masked Halo',color='red',range=(5,25))
            ax2.hist(gd,bins=40,normed=True,alpha=0.5,label='Masked Disc',color='blue',range=(5,25))
            ax2.hist(gp,bins=40,normed=True,alpha=0.5,label='Masked',color='green',range=(5,25))
            ax2.legend(loc='upper right')
            ax1.legend(loc='upper right')
            ax1.set_ylabel('N normed')
            ax1.set_xlabel('g')
            ax2.set_ylabel('N normed')
            ax2.set_xlabel('g')
            ax1.set_xlim(ax1.get_xlim()[::-1])
            ax2.set_xlim(ax2.get_xlim()[::-1])
            fig.savefig(outdir+'/diagnostic_Mgcomb.pdf')
    ###########

    print('Log file....',end='')
    idcode=np.random.randint(1,int(1e6))
    dmodel=''

    if wd==1: dmodel+='Double Exponential disc'
    else:
        if rt is not None: dmodel+='Truncated '
        if rb is not None: dmodel+='Broken Power Law'
        elif neinasto is not None: dmodel+='Einasto'
        else:
            if (aout!=0) and (ainn!=0): dmodel+='Double Power Law'
            elif aout!=0: dmodel+='Cored Power Law'
            else: dmodel+='Power Law'
        if qinf is None:
            dmodel+=' constant q'
        elif eta is None:
            dmodel += ' Exponential q'
        else:
            dmodel += ' Tanh q'

        if wd!=0: dmodel+=' + Double Exponential disc'


    p0='%.2f'%pfunc(0)
    p100='%.2f'%pfunc(100)

    dict_global={'Model':dmodel ,'aout':aout,'ainn':ainn,'neinasto':neinasto,'rt':rt,'rb':rb,'rc':rc,'p0':p0,'p100':p100,'q0':q,'qinf':str(qinf),'rq':rq,'eta':eta,'alpha':alpha,'beta':beta,'gamma':gamma,'xoff':xoff,'yoff':yoff,'zoff':zoff,'ntarget':n,'toll':nt,'wd':wd,'npartsh':npartsh,'npartsd':npartsd,'idcode':idcode,'Mgh':Mgh,'mode':mode}
    dict_masked={'Model':dmodel ,'aout':aout,'ainn':ainn,'neinasto':neinasto,'rt':rt,'rb':rb,'rc':rc,'p0':p0,'p100':p100,'q0':q,'qinf':str(qinf),'rq':rq,'eta':eta,'alpha':alpha,'beta':beta,'gamma':gamma,'xoff':xoff,'yoff':yoff,'zoff':zoff,'bmin':bmin,'gmin':gmin,'gmax':gmax,'thetamin':None,'thetamax':None,'Zgmin':None,'Zgmax':None,'ntarget':n,'toll':nt,'wd':wd,'npartsh':npartsh,'npartsd':npartsd,'idcode':idcode,'Mgh':Mgh,'mode':mode}
    if wd!=0:
        dict_global['rd']=rd
        dict_global['zd']=zd
        dict_masked['rd']=rd
        dict_masked['zd']=zd
        dict_global['Mgd']=Mgd
        dict_masked['Mgd']=Mgd
        if Mgd=='g':
            dict_masked['Mgcd']=Mgcd
            dict_masked['Mgsd']=Mgsd
            dict_global['Mgcd']=Mgcd
            dict_global['Mgsd']=Mgsd
        else:
            dict_masked['agecut']=str(agecut)
            dict_masked['fecut']=str(fecut)
            dict_masked['colcut']=str(colcut)
            dict_masked['Mgcut']=str(Mgcut)
            dict_global['agecut']=str(agecut)
            dict_global['fecut']=str(fecut)
            dict_global['colcut']=str(colcut)
            dict_global['Mgcut']=str(Mgcut)

    if wd!=1:
        if Mgh=='g':
            dict_masked['Mgch']=Mgch
            dict_masked['Mgsh']=Mgsh
            dict_global['Mgch']=Mgch
            dict_global['Mgsh']=Mgsh


    print()
    print('##################################################')
    print('Summary')
    if output:
        with open(outdir+'/'+name+'_logfile.txt', 'w') as out:
            for item in dict_masked:
                line=item+':'+str(dict_masked[item]) + '\n'
                print(line,end='')
                out.write(line)
    else:
        for item in dict_masked:
            line=item+':'+str(dict_masked[item]) + '\n'
            print(line,end='')
    print('name:',name)
    print('##################################################')
    print('Done....')

    print('Making fits....',end='')
    if wd==0:
        x,y,z=cordh.T
        r=np.sqrt(x*x+(y*y)+(z*z)) #rgal ora
        rtrue=rtrue #rgal for the spherical model
        qtrue=qtrue #q
        ptrue=ptrue #p
        m=np.sqrt(x*x+(y*y/(ptrue*ptrue))+(z*z/(qtrue*qtrue)))
        rcyl=np.sqrt(x*x+y*y)
        c=SkyCoord(l=lh*u.degree,b=bh*u.degree,frame='galactic')
        if isinstance(Mgh,float):
            Mghl=[Mgh,]*np.sum(idxh)
            Mghl_all=[Mgh,]*len(rcyl)
        else:
            Mghl=agh[idxh]
            Mghl_all=agh
        cir=c.icrs
        if output: tabglobal=ut.make_fits({'X':x,'Y':y,'Z':z,'m':m,'r':r,'rtrue':rtrue,'q':qtrue,'p':ptrue,'rcyl':rcyl,'idh':np.ones_like(r),'idd':np.zeros_like(r),'Mg':Mghl_all},outname=outdir+'/'+name+'_global.fits',header_key=dict_global)
        else: tabglobal=ut.make_fits({'X':x,'Y':y,'Z':z,'m':m,'r':r,'rtrue':rtrue,'q':qtrue,'p':ptrue,'rcyl':rcyl,'idh':np.ones_like(r),'idd':np.zeros_like(r),'Mg':Mghl_all},outname=None,header_key=dict_global)
        if output: tabmasked=ut.make_fits({'X':x[idxh],'Y':y[idxh],'Z':z[idxh],'m':m[idxh],'r':r[idxh],'rtrue':rtrue[idxh],'q':qtrue[idxh],'p':ptrue[idxh],'l':lh,'b':bh,'distance':disth,'g':gh,'Mg':Mghl,'ra':cir.ra.degree,'dec':cir.dec.degree,'idh':np.ones_like(lh),'idd':np.zeros_like(lh)},outname=outdir+'/'+name+'_masked.fits',header_key=dict_masked)
        else: tabmasked=ut.make_fits({'X':x[idxh],'Y':y[idxh],'Z':z[idxh],'m':m[idxh],'r':r[idxh],'rtrue':rtrue[idxh],'q':qtrue[idxh],'p':ptrue[idxh],'l':lh,'b':bh,'distance':disth,'g':gh,'Mg':Mghl,'ra':cir.ra.degree,'dec':cir.dec.degree,'idh':np.ones_like(lh),'idd':np.zeros_like(lh)},outname=None,header_key=dict_masked)

    elif wd==1:
        x,y,z=cordd.T
        r=np.sqrt(x*x+(y*y)+(z*z)) #rgal ora
        rtrue=rtrue #rgal for the spherical model
        qtrue=qtrue #q
        ptrue=ptrue #p
        m=np.sqrt(x*x+(y*y/(ptrue*ptrue))+(z*z/(qtrue*qtrue)))
        rcyl=np.sqrt(x*x+y*y)
        c=SkyCoord(l=ld*u.degree,b=bd*u.degree,frame='galactic')
        if isinstance(Mgd,float):
            Mgdl=[Mgd,]*len(ld)
            Mgdl_all=[Mgd,]*len(rcyl)
        else:
            Mgdl=agd[idxd]
            Mgdl_all=agd
        cir=c.icrs
        if output: tabglobal=ut.make_fits({'X':x,'Y':y,'Z':z,'m':m,'r':r,'rtrue':rtrue,'q':qtrue,'p':ptrue,'rcyl':rcyl,'idh':np.zeros_like(r),'idd':np.ones_like(r),'Mg':Mgdl_all},outname=outdir+'/'+name+'_global.fits',header_key=dict_global)
        else: tabglobal=ut.make_fits({'X':x,'Y':y,'Z':z,'m':m,'r':r,'rtrue':rtrue,'q':qtrue,'p':ptrue,'rcyl':rcyl,'idh':np.zeros_like(r),'idd':np.ones_like(r),'Mg':Mgdl_all},outname=None,header_key=dict_global)
        if output: tabmasked=ut.make_fits({'X':x[idxd],'Y':y[idxd],'Z':z[idxd],'m':m[idxd],'r':r[idxd],'rtrue':rtrue[idxd],'q':qtrue[idxd],'p':ptrue[idxd],'l':ld,'b':bd,'distance':distd,'g':gd,'Mg':Mgdl,'ra':cir.ra.degree,'dec':cir.dec.degree,'idh':np.zeros_like(ld),'idd':np.ones_like(ld)},outname=outdir+'/'+name+'_masked.fits',header_key=dict_masked)
        else: tabmasked=ut.make_fits({'X':x[idxd],'Y':y[idxd],'Z':z[idxd],'m':m[idxd],'r':r[idxd],'rtrue':rtrue[idxd],'q':qtrue[idxd],'p':ptrue[idxd],'l':ld,'b':bd,'distance':distd,'g':gd,'Mg':Mgdl,'ra':cir.ra.degree,'dec':cir.dec.degree,'idh':np.zeros_like(ld),'idd':np.ones_like(ld)},outname=None,header_key=dict_masked)

    else:
        #global
        x,y,z=np.vstack([cordh,cordd]).T
        r=np.sqrt(x*x+(y*y)+(z*z)) #rgal ora


        rcyl=np.sqrt(x*x+y*y)
        idh=np.zeros_like(x)
        idd=np.zeros_like(x)
        idh[:len(cordh)]=1
        idd[len(cordh):]=1

        m_h=np.sqrt(x[idh==1]*x[idh==1]+(y[idh==1]*y[idh==1]/(ptrue*ptrue))+(z[idh==1]*z[idh==1]/(qtrue*qtrue)))
        fake_d=np.zeros(len(cordd))

        rtrueo=np.hstack([rtrue,fake_d])
        qtrueo=np.hstack([qtrue,fake_d])
        ptrueo=np.hstack([ptrue,fake_d])
        m=np.hstack([m_h,fake_d])


        if isinstance(Mgh,float): Mghl_all=[Mgh,]*len(cordh)
        else: Mghl_all=agh
        if isinstance(Mgd,float): Mgdl_all=[Mgd,]*len(cordd)
        else: Mgdl_all=agd
        Mgl_all=np.hstack([Mghl_all,Mgdl_all])
        if output: tabglobal=ut.make_fits({'X':x,'Y':y,'Z':z,'m':m,'r':r,'rtrue':rtrueo,'q':qtrueo,'p':ptrueo,'rcyl':rcyl,'idh':idh,'idd':idd,'Mg':Mgl_all},outname=outdir+'/'+name+'_global.fits',header_key=dict_global)
        else: tabglobal=ut.make_fits({'X':x,'Y':y,'Z':z,'m':m,'r':r,'rtrue':rtrueo,'q':qtrueo,'p':ptrueo,'rcyl':rcyl,'idh':idh,'idd':idd,'Mg':Mgl_all},outname=None,header_key=dict_global)
        #local
        xlocal,ylocal,zlocal=np.vstack([cordh[idxh],cordd[idxd]]).T
        l=np.hstack([lh,ld])
        b=np.hstack([bh,bd])
        g=np.hstack([gh,gd])
        if isinstance(Mgh,float): Mghl=[Mgh,]*len(lh)
        else: Mghl=agh[idxh]
        if isinstance(Mgd,float): Mgdl=[Mgd,]*len(ld)
        else: Mgdl=agd[idxd]
        Mgl=np.hstack([Mghl,Mgdl])
        dist=np.hstack([disth,distd])
        r=np.hstack([r[idh==1][idxh],r[idd==1][idxd]])
        m=np.hstack([m[idh==1][idxh],m[idd==1][idxd]])
        rtt=np.hstack([rtrueo[idh==1][idxh],rtrueo[idd==1][idxd]])
        qtt=np.hstack([qtrueo[idh==1][idxh],qtrueo[idd==1][idxd]])
        ptt=np.hstack([ptrueo[idh==1][idxh],ptrueo[idd==1][idxd]])
        c=SkyCoord(l=l*u.degree,b=b*u.degree,frame='galactic')
        cir=c.icrs
        idhlocal=np.zeros_like(xlocal)
        iddlocal=np.zeros_like(xlocal)
        idhlocal[:npartsh]=1
        iddlocal[npartsh:]=1
        if output: tabmasked=ut.make_fits({'X':xlocal,'Y':ylocal,'Z':zlocal,'m':m,'r':r,'rtrue':rtt,'q':qtt,'p':ptt,'l':l,'b':b,'distance':dist,'g':g,'Mg':Mgl,'ra':cir.ra.degree,'dec':cir.dec.degree,'idh':idhlocal,'idd':iddlocal},outname=outdir+'/'+name+'_masked.fits',header_key=dict_masked)
        else: tabmasked=ut.make_fits({'X':xlocal,'Y':ylocal,'Z':zlocal,'m':m,'r':r,'rtrue':rtt,'q':qtt,'p':ptt,'l':l,'b':b,'distance':dist,'g':g,'Mg':Mgl,'ra':cir.ra.degree,'dec':cir.dec.degree,'idh':idhlocal,'idd':iddlocal},outname=None,header_key=dict_masked)

    print('Done')

    return tabmasked,tabglobal