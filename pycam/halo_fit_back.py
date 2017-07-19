from __future__ import  division
import numpy as np
import pycam.utils as ut
import pycam.model as md
from cubature.cubature import cubature
from functools import partial
import scipy.optimize as opt
import vegas
import astropy.io.fits as ft
from pycam.plot import ploth2
import emcee
import os
import sys
import time
import corner
import copy
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.pyplot as plt
import matplotlib as mpl


class Fit():

    def __init__(self,fitsfile,magkey='g',lkey='l',bkey='b',glim=(10,18),llim=(0,360),blim=(10,90)):
        """
        Load data and set initial parameters
        :param fitsfile: Name of the fitsfile containing the table
        :param magkey: Key of column containing the observe magnitude (corrected for the extinction)
        :param lkey:  Key of column containing the galactic latitude
        :param bkey:  Kye of the column containing the galactic longitude
        :return:
        """
        self.data=[]
        self.fitsfile=None
        self.nstars=None

        self.load_data(fitsfile,magkey=magkey,lkey=lkey,bkey=bkey)

        self.Mgd=[0.55,]
        self.wd=[1,]
        self.Mgd_descriptor='Delta Dirac Mg=0.55'
        self._Mgdkind='d'
        self._Mgdgrange=None
        self._Mgdurange=None
        self.Mgh=[0.55,]
        self.wh=[1,]
        self.Mgh_descriptor='Delta Dirac Mg=0.55'
        self._Mghkind='d'
        self._Mghgrange=None
        self._Mghurange=None

        #Galactic prop.
        self.rd=2.682
        self.zd=0.196
        self.xsun=8
        self.zsun=0.

        #Standard par
        self.par={'ainn':0, 'aout':0, 'rbs':1,'q':1,'p':1,'alpha':0,'beta':0,'gamma':0,'xoff':0,'yoff':0,'zoff':0,'f':1,'qinf':1,'rq':1}
        self.ainn=0
        self.aout=0
        self.rbs=1
        self.q=1
        self.qinf=1
        self.rq=1
        self.p=1
        self.alpha=0
        self.beta=0
        self.gamma=0
        self.xoff=0
        self.yoff=0
        self.zoff=0
        self.f=0

        #Integ
        self.erel=1e-6
        self.nval=100000
        self.niter=10

        ####Legati
        self.glim=glim
        self.llim=llim
        self.blim=blim

        self.xmin_nord=[self.glim[0],self.llim[0],self.blim[0]]
        self.xmax_nord=[self.glim[1],self.llim[1],self.blim[1]]
        self.xmin_sud=[self.glim[0],self.llim[0],-self.blim[1]]
        self.xmax_sud=[self.glim[1],self.llim[1],-self.blim[0]]

        self.xlim_nord_vega=[self.glim,self.llim,self.blim]
        self.xlim_sud_vega=[self.glim,self.llim,(-self.blim[1],-self.blim[0])]
        ###

        #Struct
        self.struct=[]

        #Geometrical parameter
        self.option_model={'s':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn',),self._loglike_s),
                      'qs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q'),self._loglike_sq),
                      'ps':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','p'),self._loglike_sp),
                      'pqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','p'),self._loglike_sqp),
                      'iqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('ainn','q','alpha','beta'),self._loglike_sqi),
                      'ips':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('ainn','p','alpha','gamma'),self._loglike_spi),
                      'ipqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('ainn','q','p','alpha','beta','gamma'),self._loglike_sqpi),
                      'fs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','f'),self._loglike_sf),
                      'fqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','f'),self._loglike_sqf),
                      'fps':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','p','f'),self._loglike_spf),
                      'fpqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','p','f'),self._loglike_sqpf),
                      'fiqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('ainn','q','alpha','beta','f'),self._loglike_sqif),
                      'fips':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('ainn','p','alpha','gamma','f'),self._loglike_spif),
                      'fipqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('ainn','q','p','alpha','beta','gamma','f'),self._loglike_sqpif),
                       'qvs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','qinf','rq'),self._loglike_sqv),    }




        #Prior limit
        self.ainn_lim=(0.2,5)
        self.aout_lim=(0.2,5)
        self.rbs_lim=(0.2,100)
        self.q_lim=(0.05,5)
        self.p_lim=(0.05,5)
        self.i_lim=(-90,90)
        self.off_lim=(-50,50)
        self.ffrac_lim=(0,1)
        self.prior_dict={'ainn':self.ainn_lim,'aout':self.aout_lim,'rbs':self.rbs_lim,'q':self.q_lim,'p':self.p_lim,'i':self.i_lim,'off':self.off_lim,'f':self.ffrac_lim}

        self.ainn_prior=partial(self._prior_uniform,lim=self.ainn_lim)
        self.ainn_prior_descriptor='Uniform [0.2;5]'

        self.aout_prior=partial(self._prior_uniform,lim=self.aout_lim)
        self.aout_prior_descriptor='Uniform [0.2;5]'

        self.rbs_prior=partial(self._prior_uniform,lim=self.rbs_lim)
        self.rbs_prior_descriptor='Uniform [0.2;100]'

        self.q_prior=partial(self._prior_uniform,lim=self.q_lim)
        self.q_prior_descriptor='Uniform q=[0.05;5]'

        self.p_prior=partial(self._prior_uniform,lim=self.p_lim)
        self.p_prior_descriptor='Uniform [0.05;5]'

        self.i_prior=partial(self._prior_uniform,lim=self.i_lim)
        self.i_prior_descriptor='Uniform [-90;90]'

        self.off_prior=partial(self._prior_uniform,lim=self.off_lim)
        self.off_prior_descriptor='Uniform [-50;50]'

        self.ffrac_prior=partial(self._prior_uniform,lim=self.ffrac_lim)
        self.ffrac_prior_descriptor='Uniform [0;1]'

        #Best value
        self.bestavalues={}
        self.bestlnlike={}

        #Function
        self.current_model=None
        self.discdens=None
        self.halodens=None
        self.vold=None
        self.volh=None
        self.Pdisc=None
        self.par_fit_list=None
        self.lnlike=None

        #Constant
        self.deg_to_rad=0.017453292519943295769 #From degree to rad

    #Dens functions
    def _halodens(self,m,ainn=1,aout=0,rs=1):

        inndens=(m/rs)**(-ainn)
        if aout==0: outdens=1
        else: outdens=(1+m/rs)**(-aout)

        return inndens*outdens
    def _halodens_break(self,m,ainn=1,aout=1,rb=1):

        ret=np.where(m<=rb,m**-ainn,(m**(aout-ainn))*(m**-aout))

        return ret
    def _discdens(self,rcyl,z,rd,zd):

        return np.exp(-rcyl/rd)*np.exp(-np.abs(z)/zd)

    #Window functin
    def _Wfunc(self,arr):
        l=arr[:,1]
        b=arr[:,2]
        ret=np.array([True,]*len(arr))
        for st in self.struct:
            if len(st)==3:
                lmg,bmg,rmc=st
                skysep_dmg=ut.skysep(l,b,lmg,bmg)
                ret*=skysep_dmg>rmc
            elif len(st)==4:
                lmin,lmax,bmin,bmax=st
                ret*=(l<lmin)|(l>lmax)|(b<bmin)|(b>bmax)
        return ret

    #Disc Volumes
    def _integrand_disc(self,arr,Mg,rd,zd):
        dist=ut.mag_to_dist(mag=arr[:,0],Mg=Mg)
        rcyl,z=ut.obs_to_cyl(arr[:,0],arr[:,1],arr[:,2],Mg,xsun=self.xsun)
        rhod=self.discdens(rcyl=rcyl,z=z,rd=rd,zd=zd)
        j=(dist**3)*np.cos(arr[:,2]*2*np.pi/360)

        return j*rhod
    def _integrand_disc_struct(self,arr,Mg,rd,zd):
        dist=ut.mag_to_dist(mag=arr[:,0],Mg=Mg)
        rcyl,z=ut.obs_to_cyl(arr[:,0],arr[:,1],arr[:,2],Mg,xsun=self.xsun)
        rhod=self.discdens(rcyl=rcyl,z=z,rd=rd,zd=zd)
        j=(dist**3)*np.cos(arr[:,2]*2*np.pi/360)
        Windowf=self._Wfunc(arr)

        return j*rhod*Windowf
    def _vold_symmetric(self,Mg,rd,zd):
        """
        Volume density of the disc for Z symmetric model without structure
        :param Mg:
        :param rd:
        :param zd:
        :return:
        """

        integral=2*cubature(self._integrand_disc,3,1,kwargs={'Mg':Mg,'rd':rd,'zd':zd},xmin=self.xmin_nord,xmax=self.xmax_nord,vectorized=True,abserr=0,relerr=self.erel)[0][0]

        return integral
    def _vold_asymmetric(self,Mg,rd,zd):
        """
        Volume density of the disc for Z asymmetric model without structure
        :param Mg:
        :param rd:
        :param zd:
        :return:
        """


        integral_nord=cubature(self._integrand_disc,3,1,kwargs={'Mg':Mg,'rd':rd,'zd':zd},xmin=self.xmin_nord,xmax=self.xmax_nord,vectorized=True,abserr=0,relerr=self.erel)[0][0]
        integral_sud=cubature(self._integrand_disc,3,1,kwargs={'Mg':Mg,'rd':rd,'zd':zd},xmin=self.xmin_sud,xmax=self.xmax_sud,vectorized=True,abserr=0,relerr=self.erel)[0][0]

        return integral_nord+integral_sud
    def _vold_struct(self,Mg,rd,zd):
        """
        Volume density of the disc for structure
        :param Mg:
        :param rd:
        :param zd:
        :return:
        """
        @vegas.batchintegrand
        def integrand_struct(arr):
            return self._integrand_disc_struct(arr,Mg=Mg,rd=rd,zd=zd)

        integ_s = vegas.Integrator(self.xlim_sud_vega, nhcube_batch=self.nval)
        integral_south=integ_s(integrand_struct,nitn=self.niter, neval=self.nval).mean

        integ_n = vegas.Integrator(self.xlim_nord_vega, nhcube_batch=self.nval)
        integral_nord=integ_n(integrand_struct,nitn=self.niter, neval=self.nval).mean

        return integral_south+integral_nord


    #Halo Volumes
    def _integrand_halo(self,arr,Mg,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff):
        """

        :param arr:
        :param Mg:
        :param ainn:
        :param aout:
        :param rbs: Can be rs or rb depending on the halo model
        :return:
        """
        dist=ut.mag_to_dist(arr[:,0],Mg=Mg)
        mm=self.obs_to_m(arr,Mg=Mg,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
        j=(dist**3)*np.cos(arr[:,2]*2*np.pi/360)
        rhoh=self.halodens(mm,ainn,aout,rbs)

        return j*rhoh
    def _integrand_halo_struct(self,arr,Mg,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff):
        """

        :param arr:
        :param Mg:
        :param ainn:
        :param aout:
        :param rbs: Can be rs or rb depending on the halo model
        :return:
        """
        dist=ut.mag_to_dist(arr[:,0],Mg=Mg)
        mm=self.obs_to_m(arr,Mg=Mg,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
        j=(dist**3)*np.cos(arr[:,2]*2*np.pi/360)
        rhoh=self.halodens(mm,ainn,aout,rbs)
        Windowf=self._Wfunc(arr)

        return j*rhoh*Windowf
    def _volh_symmetric(self,Mg,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff):

        integral=2*cubature(self._integrand_halo,3,1,kwargs={'Mg':Mg,'ainn':ainn,'aout':aout,'rbs':rbs,'q':q,'p':p,'alpha':alpha,'beta':beta,'gamma':gamma,'xoff':xoff,'yoff':yoff,'zoff':zoff},xmin=self.xmin_nord,xmax=self.xmax_nord,vectorized=True,abserr=0,relerr=self.erel)[0][0]
        return integral
    def _volh_asymmetric(self,Mg,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff):

        integral_nord=cubature(self._integrand_halo,3,1,kwargs={'Mg':Mg,'ainn':ainn,'aout':aout,'rbs':rbs,'q':q,'p':p,'alpha':alpha,'beta':beta,'gamma':gamma,'xoff':xoff,'yoff':yoff,'zoff':zoff},xmin=self.xmin_nord,xmax=self.xmax_nord,vectorized=True,abserr=0,relerr=self.erel)[0][0]
        integral_sud=cubature(self._integrand_halo,3,1,kwargs={'Mg':Mg,'ainn':ainn,'aout':aout,'rbs':rbs,'q':q,'p':p,'alpha':alpha,'beta':beta,'gamma':gamma,'xoff':xoff,'yoff':yoff,'zoff':zoff},xmin=self.xmin_sud,xmax=self.xmax_sud,vectorized=True,abserr=0,relerr=self.erel)[0][0]

        return integral_nord+integral_sud
    def _volh_struct(self,Mg,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff):

        @vegas.batchintegrand
        def integrand_struct(arr):
            return self._integrand_halo_struct(arr,Mg=Mg,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)

        integ_s = vegas.Integrator(self.xlim_sud_vega, nhcube_batch=self.nval)
        integral_south=integ_s(integrand_struct,nitn=self.niter, neval=self.nval).mean

        integ_n = vegas.Integrator(self.xlim_nord_vega, nhcube_batch=self.nval)
        integral_nord=integ_n(integrand_struct,nitn=self.niter, neval=self.nval).mean

        return integral_south+integral_nord

    #Vol grid
    def par_grid(self,model,range={},param={},ngrid=10,dimgrid=None):
        """
        Calculate a cartesina grid of the fit parameters of the model within the range
        given in range.
        :param model:  Halo model
        :param range:  Dict with the range to use for a particular par: e.g range={'q':(0,2)}
        If the range for a certain par is not set it is taken from self.prior_dict.
        :param param:  Dict with the fixed param, if not set they are taken from self.par
        :param ngrid:  Number of point to evaluate for each grid, it can be an int or a list of int
        with dimension equal to the number of fit param in the model.
        :param dimgrid:
        :return:
        """
        if model is not None: self.set_model(model)
        elif self.current_model is None: raise TypeError('Before of the minimization you need to set a model')
        else: pass

        #1-Make grid
        fit_list=self.par_fit_list

        if dimgrid is not None: ngrid=dimgrid**(1/len(fit_list))

        arglist={}
        for s in self.par:
            if s in fit_list: pass
            elif s in param: arglist[s]=param[s]
            else: arglist[s]=self.par[s]

        arr_fit_list=[]
        i=0
        if isinstance(ngrid,int) or isinstance(ngrid,float): ngrid=int(ngrid)
        else: raise ValueError('Ngrid with wrong dimensionality (func par_grid in class Fit)')
        for pfit in fit_list:
            if pfit=='alpha' or pfit=='beta' or pfit=='gamma': pfit_n='i'
            elif pfit=='xoff' or pfit=='yoff' or pfit=='zoff': pfit_n='off'
            else: pfit_n=pfit

            if pfit in range: llim,ulim=range[pfit]
            else: llim,ulim=self.prior_dict[pfit_n]
            arr_fit_list.append(np.linspace(llim,ulim,ngrid))

        grid=ut.cartesian(*arr_fit_list)
        self.reset_model()
        return grid, arglist



    #Prob func
    def _Pdisc(self,arr,rd,zd):

        rhoarr=np.zeros(len(arr))
        volarr=0

        for i in range(len(self.Mgd)):
            w=self.wd[i]
            Mgd=self.Mgd[i]
            dist=ut.mag_to_dist(arr[:,0],Mg=Mgd)
            rcyl,z=ut.obs_to_cyl(arr[:,0],arr[:,1],arr[:,2],Mgd,xsun=self.xsun)
            rhod=self.discdens(rcyl=rcyl,z=z,rd=rd,zd=zd)

            rhoarr+=(w*dist**3)*rhod
            volarr+=w*self.vold(Mg=Mgd,rd=rd,zd=zd)

        return rhoarr/volarr
    def _Phalo(self,arr,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff):

        rhoarr=np.zeros(len(arr))
        volarr=0

        for i in range(len(self.Mgh)):
            w=self.wh[i]
            Mgh=self.Mgh[i]
            dist=ut.mag_to_dist(arr[:,0],Mg=Mgh)
            mm=self.obs_to_m(arr,Mg=Mgh,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            rhoh=self.halodens(mm,ainn,aout,rbs)

            rhoarr+=(w*dist**3)*rhoh
            volarr+=w*self.volh(Mg=Mgh,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)

        return rhoarr/volarr

    #Lnlike
    #Single power law
    def _loglike_s(self,theta,*args):
        """
        Fit of a single power law
        :fit: ainn
        :param theta:
        :param args:
        :return:
        """

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0:
            return -np.inf

        lp=self.ainn_prior(ainn)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))


        return lp + lprob
    def _loglike_sf(self,theta,*args):
        """
        Fit of a single power law + disc
        :fit: ainn,f
        :param theta:
        :param args:
        :return:
        """

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args


        ainn,f=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or f>1 or f<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.ffrac_prior(f)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))


        return lp + lprob
    def _loglike_sq(self,theta,*args):
        """
        Fit of a single power law + disc
        :fit: ainn, q
        :param theta:
        :param args:
        :return:
        """

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,q=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.q_prior(q)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqf(self,theta,*args):
        """
        Fit of a single power law + disc
        :fit: ainn, q, f
        :param theta:
        :param args:
        :return:
        """


        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,q,f=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or f>1 or f<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))


        return lp + lprob
    def _loglike_sp(self,theta,*args):
        """
        Fit of a single power law + disc
        :fit: ainn, p
        :param theta:
        :param args:
        :return:
        """

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,p=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.p_prior(p)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_spf(self,theta,*args):
        """
        Fit of a single power law + disc
        :fit: ainn, p, f
        :param theta:
        :param args:
        :return:
        """


        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,p,f=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0 or f>1 or f<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.p_prior(p)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqp(self,theta,*args):

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,q,p=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0 or q<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqpf(self,theta,*args):

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,q,p,f=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0 or q<0 or f>1 or f<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqi(self,theta,*args):

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,q,alpha,beta=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.i_prior(alpha)+self.i_prior(beta)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqif(self,theta,*args):

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,q,alpha,beta,f=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or f<0 or f>1:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_spi(self,theta,*args):

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,p,alpha,gamma=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.p_prior(p)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_spif(self,theta,*args):

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,p,alpha,gamma,f=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0 or f>1 or f<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.p_prior(p)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqpi(self,theta,*args):

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,q,p,alpha,beta,gamma=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or p<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqpif(self,theta,*args):

        Pdisc,ainn,aout,rbs,q,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn,q,p,alpha,beta,gamma,f=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or p<0 or f>1 or f<0:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob


    #Fit
    def minimize(self,model=None,param={}):


        if model is not None: self.set_model(model)
        elif self.current_model is None: raise TypeError('Before of the minimization you need to set a model')
        else: pass


        def loglike(theta,*args):
            return -self.lnlike(theta,*args)


        #Calculate Pdisc
        Pdisc=self._Pdisc(self.data,zd=self.zd,rd=self.rd)

        #Set par lists
        args_list=self.make_arglist(Pdisc,param)
        parlist=self.par_fit_list
        x0_list=self.make_iguess(parlist,param)


        #Minimize
        r=opt.minimize(loglike,x0=x0_list,args=args_list,method='Nelder-Mead')


        #Store results
        self.bestavalues_item=copy.deepcopy(self.par)

        for item in param:
            self.bestavalues_item[item]=param[item]

        for i in range(len(parlist)):
            item=parlist[i]
            self.bestavalues_item[item]=r.x[i]

        self.bestavalues[self.current_model]=self.bestavalues_item


        self.bestlnlike[self.current_model]=loglike(r.x,*args_list)

        return r,self.current_model
    def set_mcmc_func(self,model=None):

        if model is not None: self.set_model(model)
        elif self.current_model is None: raise TypeError('Before of the minimization you need to set a model')
        else: pass

        return self.lnlike,model
    def mcmc_fit(self,loglike,model=None,nwalker=10,nstep=100,nburn=50,param={},iguess='min',nproc=4,ini_pos_gau=1e-2,plot=None):
        """
        Dato che i metodi di una classe non possono essere parallelizzati con multiprocessing, per
        usare questo facciamo unt rucchetto.
        Prima di lanciare mcmc_fit, bisogna usare set_mcmc_func e poi definire una funzione con la lnlike
        che restituisce  il metodo e utilizzarla come funzione loglike qui.
        :param loglike:
        :param model:
        :param nwalker:
        :param nstep:
        :param nburn:
        :param param:
        :param iguess:
        :param nproc:
        :param ini_pos_gau:
        :param plot:
        :return:
        """
        #Check
        model=''.join(sorted(model))
        if model!=self.current_model: raise ValueError('Different models')

        parlist=self.par_fit_list

        #Dimesion
        dim=len(parlist)

        #Calculate Pdisc
        Pdisc=self._Pdisc(self.data,zd=self.zd,rd=self.rd)
        #Set par lists
        args_list=self.make_arglist(Pdisc,param)

        if isinstance(iguess,str):
            if iguess=='min':
                r,_=self.minimize(model=model,param=param)
                x0_list=r.x
            elif iguess=='par':
                x0_list=self.make_iguess(parlist,param)
            else: raise NotImplementedError('iguess=%s not implemented'%iguess)
            #Initial position
            pos=[x0_list + ini_pos_gau*np.random.randn(dim) for i in range(nwalker)]
        else:
            if (len(iguess)==nwalker) and (len(iguess[0])==dim): pos=iguess
            else: raise ValueError()

        #Initialise sampler
        sampler = emcee.EnsembleSampler(nwalker, dim, loglike, args=args_list,threads=nproc)


        tini=time.time() #Timer
        #Burn phase
        if nburn>0:
            print('Burn')
            sys.stdout.flush()
            pos0,lnprob0,_=sampler.run_mcmc(pos,nburn)
            sampler.reset()
        else:
            pos0=pos
            lnprob0=None

        #MCMC
        print('Start MCMC chain')
        sys.stdout.flush()
        sampler.run_mcmc(pos0, nstep,lnprob0=lnprob0)
        tfin=time.time()
        print('Done in %f s'%(tfin-tini))


        samples=sampler.flatchain
        postprob=sampler.flatlnprobability


        fixed_param={}
        for item in self.par:
            if item in param:fixed_param[item]=param[item]
            else: fixed_param[item]=self.par[item]


        res=MC_result(fitobj=self,nwalker=nwalker,nstep=nstep,samples=samples,prob=postprob,parlist=parlist,fixed_par=fixed_param,stat=self.query_stat())

        maxlik_idx=np.argmax(postprob)
        best_pars=samples[maxlik_idx,:]
        best_like=postprob[maxlik_idx]

        if plot is not None: res.plot_triangle(plot+'.png',quantiles=(0.16,0.5,0.84),levels=(0.68,0.95),sigma=False)




        return res,best_pars,best_like,self.current_model

    #Utilities
    def obs_to_m(mag,l,b,Mg,xsun=8,q=1.0,qinf=1.0,rq=10.0,p=1.0,i=0):
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

    if q==qinf:
        z=z/q
        m=np.sqrt(x*x+y*y+z*z)
    else:
        m=calc_m(x,y,z, q, qinf, rq)


    return m

    '''
    def obs_to_m(self,arr,Mg,q,p,alpha,beta,gamma,xoff,yoff,zoff):
        """
        Return the m-value of an ellipsoid from the observ magnitude and galactic coordinate.
        if q=1 and p=1, the ellipsoid is indeed a sphere and m=r
        :param arr: arr wit col-0=gmag, col-1=l, col-2=b.
        :param Mg: Absolute magnitude.
        :param q: Flattening along the z-direction, q=1 no flattening.
        :param p: Flattening along the y-direction, p=1 no flattening.
        :param alpha: First rotationa around the X axis
        :param gamma: Second rotation around the Y axis
        :param beta: Third rotation around the Z axis
        :return: the m-value for an ellipsoid m^2=x^2+(y/p)^2+(z/q)^2.
        """

        mag=arr[:,0]
        l=arr[:,1]
        b=arr[:,2]
        cost=self.deg_to_rad
        xsun=self.xsun

        d=ut.mag_to_dist(mag,Mg=Mg)
        b=b*cost
        l=l*cost
        cb=np.cos(b)
        sb=np.sin(b)
        cl=np.cos(l)
        sl=np.sin(l)
        x=(xsun-d*cb*cl)
        y=(d*cb*sl)
        z=(d*sb)

        if (alpha!=0) or (gamma!=0) or (beta!=0):
            cord=ut.rotate_xyz(np.array([x,y,z]).T,alpha=alpha,beta=beta,gamma=gamma,system='lh')
            x=cord[:,0]
            y=cord[:,1]
            z=cord[:,2]

        if (xoff!=0) or (yoff!=0) or (zoff!=0):
            x-=xoff
            y-=yoff
            z-=zoff


        y=y/p
        z=z/q

        return np.sqrt(x*x+y*y+z*z)
        '''
    
    def make_arr(self,name,magkey='g',lkey='l',bkey='b'):
        tabf=ft.open(name)
        tab=tabf[1].data
        ndim=len(tab[magkey])
        out=np.zeros(shape=(ndim,3))
        out[:,0]=tab[magkey]
        out[:,1]=tab[lkey]
        out[:,2]=tab[bkey]
        del tab
        tabf.close()
        return out
    def make_arglist(self,Pdisc=None,arglist={}):

        outl=[]
        if Pdisc is not None: outl.append(Pdisc)

        aname='ainn'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])
        aname='aout'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])
        aname='rbs'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])
        aname='q'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])
        aname='p'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])
        aname='alpha'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])
        aname='beta'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])
        aname='gamma'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])
        aname='xoff'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])
        aname='yoff'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])
        aname='zoff'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])
        aname='f'
        if aname in arglist: outl.append(arglist[aname])
        else: outl.append(self.par[aname])

        return tuple(outl)
    def make_iguess(self,pars,x0_list):

        outl=[]
        for par in pars:
            if par in x0_list: outl.append(x0_list[par])
            else: outl.append(self.par[par])

        return tuple(outl)

    #Priors
    def _prior_uniform(self,value,lim):

        if lim[0]<=value<=lim[1]: return 0
        else: return -np.inf
    def _prior_gau(self,value,lim):

        y=(value-lim[0])/lim[1]

        return -0.5*y*y

    #Set
    def _set_Mg(self,Mg=[],w=None,bins=10,kind=None,urange=None,grange=None):

        if kind is not None:
            if kind=='uniform':
                Mg_o=np.linspace(urange[0],urange[1],bins)
                w_o=np.ones(len(Mg_o))/len(Mg_o)
                descriptor='Uniform Mg=[%.3f;%.3f]'%(urange[0],urange[1])
                kind_ou='u'
            elif kind=='gau':
                Mgdist=np.random.normal(grange[0],grange[1])
                w_o,edge=np.histogram(Mgdist,bins)
                Mg_o=0.5*(edge[:-1]+edge[1:])
                w_o=w_o/np.sum(w_o)
                descriptor='Gaussian gau(Mgc=%.3f, Mgs=%.3f)'%(grange[0],grange[1])
                kind_ou='g'
            else: NotImplementedError('kind %s not implemented'%str(kind))
        elif isinstance(Mg,float) or isinstance(Mg,int):
            Mg_o=np.array([Mg,])
            w_o=np.array([1,])
            descriptor='Delta Dirac Mg=%.3f'%Mg_o
            kind_ou=None
        elif len(Mg)>0:
            Mg_o=np.array(Mg)
            if w is None: w_o=np.array([1/len(Mg_o),]*len(Mg_o))
            elif np.sum(w)!=1: w_o=w_o/np.sum(w_o)
            else: w_o=w
            descriptor='Custom Mg=[%.3f;%.3f]'%(np.min(Mg_o),np.max(Mg_o))
            kind_ou=None
        else:
            raise ValueError('Invalid Mg format')

        return Mg_o,w_o,descriptor,kind_ou,grange,urange
    def set_Mgh(self,Mg=[],w=None,bins=10,kind=None,urange=None,grange=None):
        self.Mgh,self.wh,self.Mgh_descriptor,self._Mghkind,self._Mghgrange,self._Mghurange=self._set_Mg(Mg=Mg,w=w,bins=bins,kind=kind,urange=urange,grange=grange)
    def set_Mgd(self,Mg=[],w=None,bins=10,kind=None,urange=None,grange=None):
        self.Mgd,self.wd,self.Mgd_descriptor,self._Mgdkind,self._Mgdgrange,self._Mgdurange=self._set_Mg(Mg=Mg,w=w,bins=bins,kind=kind,urange=urange,grange=grange)
    def add_struct(self,*args,reset=False):

        if reset: self.reset_struct()

        for s in args:

            if isinstance(s,str):
                if s in ut.struct_list:
                    self.struct.append(ut.struct_list[s])
                elif s in ut.struct_list_object:
                    self.struct.append(ut.struct_list_object[s])
                else:
                    print('Warning: Unkonow structs %s'%s)
                    pass
            elif (len(s)==3) or (len(s)==4):
                self.struct.append(s)
            else:
                print('Warning: wrong structs',s)
                pass
    def reset_struct(self):
        self.struct=[]
    def set_int_option(self,glim=None,llim=None,blim=None,erel=None,nval=None,niter=None):
        """
        Set the integration option
        :param glim: Set the gmag limit.
        :param llim: Set the l limit.
        :param blim: Set the b limit.
        :param erel: Set the erel precision for the cubature integration.
        :param nval:  Set the number of sampling points for the MC vegas integration.
        :param niter:  Set the number of iteration for the MC vegas integration.
        :return:
        """
        if glim is not None: self.glim=glim
        if llim is not None: self.llim=llim
        if blim is not None: self.blim=blim
        if erel is not None: self.erel=erel
        if nval is not None: self.nval=nval
        if niter is not None: self.niter=niter

        #Update lim
        self.xmin_nord=[self.glim[0],self.llim[0],self.blim[0]]
        self.xmax_nord=[self.glim[1],self.llim[1],self.blim[1]]
        self.xmin_sud=[self.glim[0],self.llim[0],-self.blim[1]]
        self.xmax_sud=[self.glim[1],self.llim[1],-self.blim[0]]

        self.xlim_nord_vega=[self.glim,self.llim,self.blim]
        self.xlim_sud_vega=[self.glim,self.llim,(-self.blim[1],-self.blim[0])]
        ###






    def set_model(self,model):

        model=''.join(sorted(model))

        olist=self.option_model
        if model in olist:
            self.current_model=model
            discdens,halodens,vold,volh,parl,lnlike=olist[model]
            self.discdens=discdens
            self.halodens=halodens
            self.par_fit_list=parl
            self.lnlike=lnlike



            if len(self.struct)==0:
                self.vold=vold
                self.volh=volh
            else:
                self.vold=self._vold_struct
                self.volh=self._volh_struct

        else: raise NotImplementedError('Model %s not implemented'%model)
    def reset_model(self):
        #Function
        self.current_model=None
        self.discdens=None
        self.halodens=None
        self.vold=None
        self.volh=None
        self.Pdisc=None
        self.par_fit_list=None
        self.lnlike=None
    def set_sunposition(self,x=None,z=None):
        """
        Position of the sun in galactocentric cordinate,
        the system is lh, y point toward the motion of Sun.
        :param x: X is in the direction of Sun
        :param z: Z is the height abode the disc
        :return:
        """

        if x is not None: self.xsun=x
        if z is not None: self.zsun=z
    def set_disc(self,rd=None,zd=None):
        if rd is not None: self.rd=rd
        if zd is not None: self.zd=zd
    def set_prior(self,*args):
        """
        Usage set_prior(('ainn','u',[0,5]),('f','g',[0.6,0.005]))
        :param args:
        :return:
        """
        for opt in args:
            if len(opt)==3 and isinstance(opt[0],str) and isinstance(opt[1],str) and len(opt[2])==2:

                name,kind,rangel=opt


                if name.lower()=='ainn':
                    if kind[0].lower()=='u':
                        self.ainn_prior=partial(self._prior_uniform,lim=rangel)
                        self.ainn_prior_descriptor='Uniform  [%.3f,%.3f]'%(float(rangel[0]),float(rangel[1]))
                        self.ainn_lim=rangel
                        self.prior_dict['ainn']=rangel
                    elif kind[0].lower()=='g':
                        self.ainn_prior=partial(self._prior_gau,lim=rangel)
                        self.ainn_prior_descriptor='Gaussian  Gau(c=%.3f,s=%.3f)'%(float(rangel[0]),float(rangel[1]))
                    else:
                        raise NotImplementedError('Prior of kind %s not implemented'%kind)

                elif name.lower()=='aout':
                    if kind[0].lower()=='u':
                        self.aout_prior=partial(self._prior_uniform,lim=rangel)
                        self.aout_prior_descriptor='Uniform  [%.3f,%.3f]'%(float(rangel[0]),float(rangel[1]))
                        self.aout_lim=rangel
                        self.prior_dict['aout']=rangel
                    elif kind[0].lower()=='g':
                        self.aout_prior=partial(self._prior_gau,lim=rangel)
                        self.aout_prior_descriptor='Gaussian  Gau(c=%.3f,s=%.3f)'%(float(rangel[0]),float(rangel[1]))
                    else:
                        raise NotImplementedError('Prior of kind %s not implemented'%kind)

                elif name.lower()=='rbs':
                    if kind[0].lower()=='u':
                        self.rbs_prior=partial(self._prior_uniform,lim=rangel)
                        self.rbs_prior_descriptor='Uniform  [%.3f,%.3f]'%(float(rangel[0]),float(rangel[1]))
                        self.rbs_lim=rangel
                        self.prior_dict['rbs']=rangel
                    elif kind[0].lower()=='g':
                        self.rbs_prior=partial(self._prior_gau,lim=rangel)
                        self.rbs_prior_descriptor='Gaussian  Gau(c=%.3f,s=%.3f)'%(float(rangel[0]),float(rangel[1]))
                    else:
                        raise NotImplementedError('Prior of kind %s not implemented'%kind)

                elif name.lower()=='q':
                    if kind[0].lower()=='u':
                        self.q_prior=partial(self._prior_uniform,lim=rangel)
                        self.q_prior_descriptor='Uniform  [%.3f,%.3f]'%(float(rangel[0]),float(rangel[1]))
                        self.q_lim=rangel
                        self.prior_dict['q']=rangel
                    elif kind[0].lower()=='g':
                        self.q_prior=partial(self._prior_gau,lim=rangel)
                        self.q_prior_descriptor='Gaussian  Gau(c=%.3f,s=%.3f)'%(float(rangel[0]),float(rangel[1]))
                    else:
                        raise NotImplementedError('Prior of kind %s not implemented'%kind)

                elif name.lower()=='p':
                    if kind[0].lower()=='u':
                        self.p_prior=partial(self._prior_uniform,lim=rangel)
                        self.p_prior_descriptor='Uniform  [%.3f,%.3f]'%(float(rangel[0]),float(rangel[1]))
                        self.p_lim=rangel
                        self.prior_dict['p']=rangel
                    elif kind[0].lower()=='g':
                        self.p_prior=partial(self._prior_gau,lim=rangel)
                        self.p_prior_descriptor='Gaussian  Gau(c=%.3f,s=%.3f)'%(float(rangel[0]),float(rangel[1]))
                    else:
                        raise NotImplementedError('Prior of kind %s not implemented'%kind)

                elif name.lower()=='i':
                    if kind[0].lower()=='u':
                        self.i_prior=partial(self._prior_uniform,lim=rangel)
                        self.i_prior_descriptor='Uniform  [%.3f,%.3f]'%(float(rangel[0]),float(rangel[1]))
                        self.i_lim=rangel
                        self.prior_dict['i']=rangel
                    elif kind[0].lower()=='g':
                        self.i_prior=partial(self._prior_gau,lim=rangel)
                        self.i_prior_descriptor='Gaussian  Gau(c=%.3f,s=%.3f)'%(float(rangel[0]),floatl(range[1]))
                    else:
                        raise NotImplementedError('Prior of kind %s not implemented'%kind)

                elif name.lower()=='off':
                    if kind[0].lower()=='u':
                        self.off_prior=partial(self._prior_uniform,lim=rangel)
                        self.off_prior_descriptor='Uniform  [%.3f,%.3f]'%(float(rangel[0]),float(rangel[1]))
                        self.off_lim=rangel
                        self.prior_dict['off']=rangel
                    elif kind[0].lower()=='g':
                        self.off_prior=partial(self._prior_gau,lim=rangel)
                        self.off_prior_descriptor='Gaussian  Gau(c=%.3f,s=%.3f)'%(float(rangel[0]),floatl(range[1]))
                    else:
                        raise NotImplementedError('Prior of kind %s not implemented'%kind)

                elif name.lower()=='f':
                    if kind[0].lower()=='u':
                        self.ffrac_prior=partial(self._prior_uniform,lim=rangel)
                        self.ffrac_prior_descriptor='Uniform  [%.3f,%.3f]'%(float(rangel[0]),float(rangel[1]))
                        self.ffrac_lim=rangel
                        self.prior_dict['f']=rangel
                    elif kind[0].lower()=='g':
                        self.ffrac_prior=partial(self._prior_gau,lim=rangel)
                        self.ffrac_prior_descriptor='Gaussian  Gau(c=%.3f,s=%.3f)'%(float(rangel[0]),float(rangel[1]))
                    else:
                        raise NotImplementedError('Prior of kind %s not implemented'%kind)

                else:
                    raise NotImplementedError('Variable %s is not implemented'%name.lower())


            else:
                print('Warning wrong opt options, skypping')

    #Load
    def load_data(self,fitsfile,magkey='g',lkey='l',bkey='b'):
        self.data=self.make_arr(fitsfile,magkey=magkey,lkey=lkey,bkey=bkey)
        self.nstars=len(self.data)
        self.fitsfile=fitsfile

    #Stat
    def query_stat(self):

        h='DATA\n'
        h+='Fitsfile: %s\n'%str(self.fitsfile)

        h+='GALACTIC PARAMS\n'
        h+='Rd: %.3f Zd: %.3f\n'%(self.rd,self.zd)
        h+='Xsun: %.3f Zsun:%.3f\n'%(self.xsun,self.zsun)

        h+='INTEGRATION PARAMS\n'
        h+='glim: (%.2f,%.2f)\n'%self.glim
        h+='llim: (%.1f,%.1f)\n'%self.llim
        h+='blim: (%.1f,%.1f)\n'%self.blim
        h+='Erel: %.1e\n'%self.erel
        h+='Nval: %.1e Niter=%i\n'%(self.nval,int(self.niter))

        h+='MODEL\n'
        h+=str(self.current_model)+'\n'

        h+='PRIORS\n'
        h+='ainn: %s\n'%self.ainn_prior_descriptor
        h+='aout: %s\n'%self.aout_prior_descriptor
        h+='rbs: %s\n'%self.rbs_prior_descriptor
        h+='q: %s\n'%self.q_prior_descriptor
        h+='p: %s\n'%self.p_prior_descriptor
        h+='f: %s\n'%self.ffrac_prior_descriptor
        h+='i: %s\n'%self.i_prior_descriptor
        h+='pos_offset: %s\n'%self.off_prior_descriptor

        h+='ABSOLUTE MAGNITUDE\n'
        h+='Halo: %s   sample bins=%i\n'%(self.Mgh_descriptor,len(self.Mgh))
        h+='Disc: %s   sample bins=%i\n'%(self.Mgd_descriptor,len(self.Mgd))

        h+='STRUCT\n'
        if len(self.struct)>0:
            for s in self.struct:
                if len(s)==3:
                    h+='lc=%.2f bc=%.2f rc=%.2f\n'%s
                elif len(s)==4:
                    h+='l=[%.2f;%.2f]  b=[%.2f;%.2f]\n'%s
                else:
                    h+='Struct with wrong formats\n'
        else:
            h+='No structures\n'



        return h

    #Reset
    def reset(self):

        self.data=[]
        self.fitsfile=None


        self.Mgd=[0.55,]
        self.wd=[1,]
        self.Mgd_descriptor='Delta Dirac Mg=0.55'
        self._Mgdkind='d'
        self._Mgdgrange=None
        self._Mgdurange=None
        self.Mgh=[0.55,]
        self.wh=[1,]
        self.Mgh_descriptor='Delta Dirac Mg=0.55'
        self._Mghkind='d'
        self._Mghgrange=None
        self._Mghurange=None

        #Galactic prop.
        self.rd=2.682
        self.zd=0.196
        self.xsun=8
        self.zsun=0.

        #Standard par
        self.par={'ainn':0, 'aout':0, 'rbs':1,'q':1,'p':1,'alpha':0,'beta':0,'gamma':0,'xoff':0,'yoff':0,'zoff':0,'f':1}
        self.ainn=0
        self.aout=0
        self.rbs=1
        self.q=1
        self.p=1
        self.alpha=0
        self.beta=0
        self.gamma=0
        self.xoff=0
        self.yoff=0
        self.zoff=0
        self.f=0

        #Integ
        self.erel=1e-6
        self.nval=100000
        self.niter=10

        self.glim=(10,18)
        self.llim=(0,360)
        self.blim=(10,90)

        self.xmin_nord=[self.glim[0],self.llim[0],self.blim[0]]
        self.xmax_nord=[self.glim[1],self.llim[1],self.blim[1]]
        self.xmin_sud=[self.glim[0],self.llim[0],-self.blim[1]]
        self.xmax_sud=[self.glim[1],self.llim[1],-self.blim[0]]

        self.xlim_nord_vega=[self.glim,self.llim,self.blim]
        self.xlim_sud_vega=[self.glim,self.llim,(-self.blim[1],-self.blim[0])]

        #Struct
        self.struct=[]

        #Geometrical parameter
        self.option_model={'s':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn',),self._loglike_s),
                      'qs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q'),self._loglike_sq),
                      'ps':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','p'),self._loglike_sp),
                      'pqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','p'),self._loglike_sqp,3),
                      'iqs':(self._discdens,self._halodens,self._vold_asymmetric,self._volh_asymmetric,('ainn','q','alpha','beta'),self._loglike_sqi),
                      'ips':(self._discdens,self._halodens,self._vold_asymmetric,self._volh_asymmetric,('ainn','p','alpha','beta','gamma'),self._loglike_spi),
                      'ipqs':(self._discdens,self._halodens,self._vold_asymmetric,self._volh_asymmetric,('ainn','q','p','alpha','beta','gamma'),self._loglike_sqpi),
                      'fs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','f'),self._loglike_sf),
                      'fqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','f'),self._loglike_sqf),
                      'fps':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','p','f'),self._loglike_spf),
                      'fpqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','p','f'),self._loglike_sqpf),
                      'fiqs':(self._discdens,self._halodens,self._vold_asymmetric,self._volh_asymmetric,('ainn','q','alpha','beta','gamma','f'),self._loglike_sqif),
                      'fips':(self._discdens,self._halodens,self._vold_asymmetric,self._volh_asymmetric,('ainn','p','alpha','beta','gamma','f'),self._loglike_spif),
                      'fipqs':(self._discdens,self._halodens,self._vold_asymmetric,self._volh_asymmetric,('ainn','q','p','alpha','beta','gamma','f'),self._loglike_sqpif)}




        #Prior limit
        self.ainn_lim=(0.2,5)
        self.aout_lim=(0.2,5)
        self.rbs_lim=(0.2,100)
        self.q_lim=(0.05,5)
        self.p_lim=(0.05,5)
        self.i_lim=(-90,90)
        self.off_lim=(-50,50)
        self.ffrac_lim=(0,1)
        self.prior_dict={'ainn':self.ainn_lim,'aout':self.aout_lim,'rbs':self.rbs_lim,'q':self.q_lim,'p':self.p_lim,'i':self.i_lim,'off':self.off_lim,'f':self.ffrac_lim}

        self.ainn_prior=partial(self._prior_uniform,lim=self.ainn_lim)
        self.ainn_prior_descriptor='Uniform [0.2;5]'

        self.aout_prior=partial(self._prior_uniform,lim=self.aout_lim)
        self.aout_prior_descriptor='Uniform [0.2;5]'

        self.rbs_prior=partial(self._prior_uniform,lim=self.rbs_lim)
        self.rbs_prior_descriptor='Uniform [0.2;100]'

        self.q_prior=partial(self._prior_uniform,lim=self.q_lim)
        self.q_prior_descriptor='Uniform q=[0.05;5]'

        self.p_prior=partial(self._prior_uniform,lim=self.p_lim)
        self.p_prior_descriptor='Uniform [0.05;5]'

        self.i_prior=partial(self._prior_uniform,lim=self.i_lim)
        self.i_prior_descriptor='Uniform [-90;90]'

        self.off_prior=partial(self._prior_uniform,lim=self.off_lim)
        self.off_prior_descriptor='Uniform [-50;50]'

        self.ffrac_prior=partial(self._prior_uniform,lim=self.ffrac_lim)
        self.ffrac_prior_descriptor='Uniform [0;1]'

        #Best value
        self.bestavalues={}
        self.bestlnlike={}

        #Function
        self.current_model=None
        self.discdens=None
        self.halodens=None
        self.vold=None
        self.volh=None
        self.Pdisc=None
        self.par_fit_list=None
        self.lnlike=None

        #Constant
        self.deg_to_rad=0.017453292519943295769 #From degree to rad

    #Print
    def __str__(self):
        return self.query_stat()

class MC_result():

    def __init__(self,fitobj,nwalker,nstep,samples,prob,parlist,fixed_par,stat=''):

        idx_best=np.argmax(prob) #Index with the max Likelihood

        #Fit prop
        self.fitmodel=copy.copy(fitobj.current_model)
        self.rd=copy.copy(fitobj.rd)
        self.zd=copy.copy(fitobj.zd)
        self.xsun=copy.copy(fitobj.xsun)
        self.blim=copy.copy(fitobj.blim)
        self.glim=copy.copy(fitobj.glim)
        self.nstars=copy.copy(fitobj.nstars)
        self.struct=copy.copy(fitobj.struct)
        self.Mgh=copy.copy(fitobj.Mgh)
        self.Mgd=copy.copy(fitobj.Mgd)
        self.wh=copy.copy(fitobj.wh)
        self.wd=copy.copy(fitobj.wd)
        self.Mgdkind=copy.copy(fitobj._Mgdkind)
        self.Mghkind=copy.copy(fitobj._Mghkind)
        self.Mgdurange=copy.copy(fitobj._Mgdurange)
        self.Mgdgrange=copy.copy(fitobj._Mgdgrange)
        self.Mghurange=copy.copy(fitobj._Mghurange)
        self.Mghgrange=copy.copy(fitobj._Mghgrange)
        self.data=copy.copy(fitobj.data)


        self.Nbody_models=None #Continete il fits con la realizazione Nbody
        self.sample={} #Contiente i valori di tutte le variabili fittate
        self.best={} #Contiente i best-value (max value of the likelihood) delle variabili fittate
        self.median={} #Contiene le mediane di tutte le variabili fittate
        self.percentile={} #Contiente i percentili (0.5,2.5,16,50,84,97.5,99.5) di tutte le variabili fittate
        self.one_sigma={} #1sigma interval di tutte le variabili fittate
        self.two_sigma={} #2sigma interval di tutte le variabili fittate
        self.three_sigma={} #3sigma interval di tutte le variabili fittate
        self.bestlnlike=prob[idx_best] #Max likelihood
        self.evidence=np.log(np.sum(np.exp(prob))) #Calcolo dell'evidence (schifo)
        self.ndim=len(parlist) #Dimendion of the fitted params
        self.arr=np.zeros((nstep*nwalker,self.ndim+1)) #Array with the value of the fitted values and the resultat log likelihood
        self.param=fixed_par #Containi all the params
        self.varlist=parlist #The name of the fitted params


        for i in range(self.ndim):
            self.arr[:,i]=samples[:,i]
        self.arr[:,-1]=prob



        i=0
        for par in parlist:
            vals=samples[:,i]
            perc=tuple(np.percentile(vals,q=[0.5,2.5,16,50,84,97.5,99.5]))
            self.sample[par]=vals
            self.percentile[par]=perc
            self.one_sigma[par]=(perc[2],perc[4])
            self.two_sigma[par]=(perc[1],perc[5])
            self.three_sigma[par]=(perc[0],perc[6])
            self.best[par]=vals[idx_best]
            self.median[par]=perc[3]
            i+=1

        self.stat=stat


    def _print_result(self):

        h=self.stat

        h+='\nFIT\n'
        h+='Fixed Param:\n'
        for item in self.param:
            h+='%s: %.2f\n'%(item,float(self.param[item]))
        h+='Fitted Param:\n'

        for item in self.varlist:

            best=self.median[item]
            err_plus=self.one_sigma[item][1]-self.median[item]
            err_min=self.median[item]-self.one_sigma[item][0]
            osm,osp=self.one_sigma[item]
            tsm,tsp=self.two_sigma[item]
            thsm,thsp=self.three_sigma[item]

            h+='%s: %.2f +%.2f-%.2f  1-sigma:[%.2f;%.2f] 2-sigma:[%.2f;%.2f] 3-sigma:[%.2f;%.2f]\n'%(item,best,err_plus,err_min,osm,osp,tsm,tsp,thsm,thsp)


        h+='Best Likelihood\n'
        h+='Log(L)=%.5f\n'%self.bestlnlike
        h+='Best Likelihood params:\n'
        for item in self.best:
            h+='%s: %.2f1\n'%(item,float(self.best[item]))

        return h
    def __str__(self):

        return self._print_result()
    def plot_triangle(self,outname='Triangle_plot.png',quantiles=(0.16,0.5,0.84),levels=(0.68,0.95),sigma=False,plot_datapoints=False,fill_contours=True,show_titles=True,**kwargs):


        levels=np.array(levels)
        if sigma: levels=np.exp(-0.5*(levels*levels))


        label_list=[]
        for item in self.varlist:
            if item=='ainn': label_list.append('$\\alpha_{\mathrm{inn}}$')
            elif item=='aout': label_list.append('$\\alpha_{\mathrm{out}}$')
            elif item=='rbs':
                if 'b' in self.fitmodel: label_list.append('$\\{\mathrm{r}}_{\mathrm{b}}$')
                if 'c' in self.fitmodel: label_list.append('$\\{\mathrm{r}}_{\mathrm{c}}$')
                else: label_list.append('$\\{\mathrm{r}}_{\mathrm{s}}$')
            else: label_list.append(item)


        fig=corner.corner(self.arr[:,:-1],labels=label_list,quantiles=quantiles,levels=levels,plot_datapoints=plot_datapoints,fill_contours=fill_contours,show_titles=show_titles,**kwargs)
        fig.savefig(outname)

    def make_model(self,outdir='',bvalue='median',diagnostic=False,Mgh=None,Mgd=None,extra_struct=None,nmodel=1):
        """

        :param output:
        :param bvalue:
        :param diagnostic:
        :return:
        """

        if outdir is None:
            output_m=False
            outdir=''
            name=''
        else:
            output_m=True
            outdir=outdir
            name='Nmodel'

        tpar_list=self.param.copy()
        if bvalue[0].lower()=='b': tpar_list.update(self.best)
        elif bvalue[0].lower()=='m': tpar_list.update(self.median)
        else: raise NotImplementedError()

        if 'b' in self.fitmodel:
            rb=tpar_list['rbs']
            rc=1
        else:
            rb=None
            rc=tpar_list['rbs']



        if Mgh is None: Mgh=np.median(self.Mgh)

        Mgcd=None
        Mgsd=None
        Mgud=None

        print('Mgdud',self.Mgdurange)
        print('Mgdcd',self.Mgdgrange)

        if isinstance(Mgd,int) or isinstance(Mgd,float): pass
        elif self.Mgdkind=='d': Mgd=np.median(self.Mgd)
        elif self.Mgdkind=='u':
            Mgd='u'
            Mgud=self.Mgdurange
        elif self.Mgdkind=='g':
            Mgd='g'
            Mgcd,Mgsd=self.Mgdgrange
        else: raise ValueError('Mgd %s not allowed'%string(Mgd))


        if extra_struct is None: struct_tmp=self.struct
        else:
            struct_tmp=list(copy.copy(self.struct))
            for item in extra_struct:
                struct_tmp.append(item)
            if (len(struct_tmp)==0): struct_tmp=None

        self.Nbody_models=[]

        if nmodel==1:
            Nbody_model,_=md.make_model(aout=tpar_list['aout'],ainn=tpar_list['ainn'],rc=rc,rb=rb,q=tpar_list['q'],p=tpar_list['p'],wd=1-tpar_list['f'],rd=self.rd,zd=self.zd,alpha=tpar_list['alpha'],beta=tpar_list['beta'],gamma=tpar_list['gamma'],xoff=tpar_list['xoff'],yoff=tpar_list['yoff'],zoff=tpar_list['zoff'],bmin=self.blim[0],gmin=self.glim[0],gmax=self.glim[1],n=self.nstars,Mgh=Mgh,Mgd=Mgd,Mgcd=Mgcd,Mgsd=Mgsd,Mgud=Mgud,mask=struct_tmp,name=name,diagnostic=diagnostic,output=output_m,outdir=outdir)
            self.Nbody_models.append(Nbody_model)
        elif nmodel>=1:
            for i in range(nmodel):
                #Save in output only the first model
                if i==0: Nbody_model,_=md.make_model(aout=tpar_list['aout'],ainn=tpar_list['ainn'],rc=rc,rb=rb,q=tpar_list['q'],p=tpar_list['p'],wd=1-tpar_list['f'],rd=self.rd,zd=self.zd,alpha=tpar_list['alpha'],beta=tpar_list['beta'],gamma=tpar_list['gamma'],xoff=tpar_list['xoff'],yoff=tpar_list['yoff'],zoff=tpar_list['zoff'],bmin=self.blim[0],gmin=self.glim[0],gmax=self.glim[1],n=self.nstars,Mgh=Mgh,Mgd=Mgd,Mgcd=Mgcd,Mgsd=Mgsd,Mgud=Mgud,mask=struct_tmp,name=name,diagnostic=diagnostic,output=output_m,outdir=outdir)
                else: Nbody_model,_=md.make_model(aout=tpar_list['aout'],ainn=tpar_list['ainn'],rc=rc,rb=rb,q=tpar_list['q'],p=tpar_list['p'],wd=1-tpar_list['f'],rd=self.rd,zd=self.zd,alpha=tpar_list['alpha'],beta=tpar_list['beta'],gamma=tpar_list['gamma'],xoff=tpar_list['xoff'],yoff=tpar_list['yoff'],zoff=tpar_list['zoff'],bmin=self.blim[0],gmin=self.glim[0],gmax=self.glim[1],n=self.nstars,Mgh=Mgh,Mgd=Mgd,Mgcd=Mgcd,Mgsd=Mgsd,Mgud=Mgud,mask=struct_tmp,name=name,output=False,diagnostic=False)
                self.Nbody_models.append(Nbody_model)



    def residual(self,outname='residual',outdir=os.getcwd(),glim=(14,16),lrange=(0,360),brange=(-90,90),bins=50,Mgh=None,Mgdrange=None,model_idx=None):

        ut.mkdir(outdir) #Create directory

        if self.Nbody_models is None:
            print('Warning: Nbody model is not set, setting now with default option')
            self.make_model()

        if Mgh is None: Mgh=np.median(self.Mgh)

        lrange=(lrange[0]-180,lrange[1]-180)

        gd=self.data[:,0]
        ld=self.data[:,1]
        ld=np.where(ld>180,ld-360,ld)
        bd=self.data[:,2]



        gm=[]
        gm_halo=[]
        lm=[]
        bm=[]

        if model_idx is not None:
            nmod=self.Nbody_models[model_idx]
            gm.append(nmod.data['g'])
            lmt=nmod.data['l']
            lm.append(np.where(lmt>180,lmt-360,lmt))
            bm.append(nmod.data['b'])
            idx_halo=nmod.data['idh']==1
            gm_halo.append(nmod.data['g'][idx_halo])
        else:
            for nmod in self.Nbody_models:
                gm.append(nmod.data['g'])
                lmt=nmod.data['l']
                lm.append(np.where(lmt>180,lmt-360,lmt))
                bm.append(nmod.data['b'])
                idx_halo=nmod.data['idh']==1
                gm_halo.append(nmod.data['g'][idx_halo])

        gm=np.array(gm)
        gm_halo=np.array(gm_halo)
        lm=np.array(lm)
        bm=np.array(bm)



        label_size =25
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        fig=plt.figure(figsize=(30,20))
        ax1a=fig.add_subplot(3,4,1)
        ax1b=fig.add_subplot(3,4,5)
        ax1c=fig.add_subplot(3,4,9)
        ax2a=fig.add_subplot(3,4,2)
        ax2b=fig.add_subplot(3,4,6)
        ax2c=fig.add_subplot(3,4,10)
        ax3a=fig.add_subplot(3,4,3)
        ax3b=fig.add_subplot(3,4,7)
        ax3c=fig.add_subplot(3,4,11)
        ax4a=fig.add_subplot(3,4,4)
        ax4b=fig.add_subplot(3,4,8)
        ax4c=fig.add_subplot(3,4,12)
        axb=fig.add_axes([0.123,0.05,0.375,0.02])

        self._residual_obs2(ld,bd,gd,lm,bm,gm,axd=ax1a,axm=ax1b,axr=ax1c,axcb=axb,glim=(None,None),lrange=lrange,brange=brange,bins=bins)
        self._residual_obs2(ld,bd,gd,lm,bm,gm,axd=ax2a,axm=ax2b,axr=ax2c,glim=(None,glim[0]),lrange=lrange,brange=brange,bins=bins)
        self._residual_obs2(ld,bd,gd,lm,bm,gm,axd=ax3a,axm=ax3b,axr=ax3c,glim=(glim[0],glim[1]),lrange=lrange,brange=brange,bins=bins)
        self._residual_obs2(ld,bd,gd,lm,bm,gm,axd=ax4a,axm=ax4b,axr=ax4c,glim=(glim[1],None),lrange=lrange,brange=brange,bins=bins)

        ax1a.set_ylabel('b [deg]',fontsize=30)
        ax1b.set_ylabel('b [deg]',fontsize=30)
        ax1c.set_ylabel('b [deg]',fontsize=30)
        ax3c.set_xlabel('l [deg]',fontsize=30)
        ax4c.set_xlabel('l [deg]',fontsize=30)
        axb.set_xlabel('Residuals',fontsize=25)

        plt.figtext(0.19,0.95,'G all',fontsize=35)
        plt.figtext(0.38,0.95,'G<%.1f'%glim[0],fontsize=35)
        plt.figtext(0.56,0.95,'%.1f<G<%.1f'%glim,fontsize=35)
        plt.figtext(0.78,0.95,'G>%.1f'%glim[1],fontsize=35)


        fig.savefig(outdir+'/'+outname+'_sky.png')
        ###########

        #Zslab
        fig=plt.figure(figsize=(30,20))
        ax1a=fig.add_subplot(3,2,1)
        ax1b=fig.add_subplot(3,2,3)
        ax1c=fig.add_subplot(3,2,5)
        ax2a=fig.add_subplot(3,2,2)
        ax2b=fig.add_subplot(3,2,4)
        ax2c=fig.add_subplot(3,2,6)
        axb=fig.add_axes([0.123,0.05,0.375,0.02])
        axb2=fig.add_axes([0.123,0.95,0.375,0.02])


        _,_,_,_,vmax=self._residual_2Dslab(ld,bd,gd,lm,bm,gm,s='z',axd=ax1a,axm=ax1b,axr=ax1c,axcbr=axb,axcbdm=axb2,slim=(10,20),bins=30,xsun=self.xsun,Mg=Mgh,fontsize=20,normalize=True)
        self._residual_2Dslab(ld,bd,gd,lm,bm,gm,s='z',axd=ax2a,axm=ax2b,axr=ax2c,slim=(-20,-10),bins=30,xsun=self.xsun,Mg=Mgh,fontsize=20,normalize=True,vmax=vmax)

        fig.savefig(outdir+'/'+outname+'_zslab.png')
        ###########


        ############
        #Galactic, use the same abs magnitude for all the stars both in the data and the model to calculate the distance
        fig=plt.figure(figsize=(15,20))
        ax1a=fig.add_subplot(3,1,1)
        ax1b=fig.add_subplot(3,1,2)
        ax1c=fig.add_subplot(3,1,3)
        axb=fig.add_axes([0.123,0.02,0.78,0.02])
        axb.set_xlabel('Residuals',fontsize=25)

        ax1a.set_ylabel('z [kpc]',fontsize=30)
        ax1b.set_ylabel('z [kpc]',fontsize=30)
        ax1c.set_ylabel('z [kpc]',fontsize=30)
        ax1c.set_xlabel('R [kpc]',fontsize=30)


        plt.figtext(0.425,0.95,'Mg=%.3f'%np.median(self.Mgh),fontsize=35)

        self._residual_physic2(ld,bd,gd,lm,bm,gm,Mg=Mgh,axd=ax1a,axm=ax1b,axr=ax1c,axcb=axb,bins=bins,xsun=self.xsun)

        fig.savefig(outdir+'/'+outname+'_Galacitc.png')
        ############

        #########################
        #Mag distribution
        label_size =16
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)


        '''
        Hm=[]
        Hm_halo=[]

        if Mgdrange is None:
            for nmod in self.Nbody_models:
                gm=nmod.data['g']
                idx_halo=nmod.data['idh']==1

                Hd_f,Hm_t,Hm_halo_t,edge=self._mag_dist(gd,gm,gm_halo=gm[idx_halo],bins=10,ax=None,grange=self.glim)
                Hm.append(Hm_t/np.sum(Hm_t))
                Hm_halo.append(Hm_halo_t/np.sum(Hm_halo_t))

        else:
            for amgd in Mgdrange:
                nmod=Nbody_model.data
                gm=np.where(nmod['idh']==1,nmod['g'],ut.dist_to_g(nmod['distance'],amgd))
                idx_halo=nmod['idh']==1
                Hd_f,Hm_t,Hm_halo_t,edge=self._mag_dist(gd,gm,gm_halo=gm[idx_halo],bins=10,ax=None,grange=self.glim)
                Hm.append(Hm_t/np.sum(Hm_t))
                Hm_halo.append(Hm_halo_t/np.sum(Hm_halo_t))


        x=0.5*(edge[:-1]+edge[1:])
        Hd=Hd_f/np.sum(Hd_f)
        Hde=np.sqrt(Hd_f)/np.sum(Hd_f)
        Hm,Hme=ut.mad(Hm,axis=0)
        Hm_halo,Hme_halo=ut.mad(Hm_halo,axis=0)


        ax.errorbar(x,Hd,Hde,c='black',label='Data',fmt='o',zorder=1000)
        ax.errorbar(x,Hm,Hme,c='red',label='Model',fmt='-o',zorder=500)
        ax.errorbar(x,Hm_halo,Hme_halo,c='red',label='Model (halo only)',fmt='-s',ls='dashed')
        ax.legend(loc='upper left')
        '''

        gmin=np.min(gd)
        gmax=np.max(gd)
        self._mag_dist2(gd,gm,gm_halo=gm_halo,bins=10,ax=ax,grange=(gmin,gmax))


        ax.set_xlabel('N/Ntot',fontsize=18)
        ax.set_xlabel('g',fontsize=18)
        fig.savefig(outdir+'/'+outname+'_gdist.png')

    def _residual_obs2(self,ld,bd,gd,lm,bm,gm,axd=None,axm=None,axr=None,axcb=None,glim=(None,None),lrange=(-180,180),brange=(-90,90),bins=50,gamma=0.5):
        """

        :param ld: galactic longitude of the data
        :param bd: galactic latitude of the data
        :param gd: observed magnitude of the data
        :param lm: galactic longitude of the model
        :param bm: galactic latitude of the model
        :param gm: observed magnitude of the model
        :param axd: ax where to plot the sky-map of the data
        :param axm: ax where to plot the sky-map of the model
        :param axr: ax where to plot the sky-map of the residulas
        :param glim: The maps will be produced for G<min(glim), min(glim)<G<max(glim), G>max(glim)
        :param llim: The maps will be produced in this longitude range
        :param blim: The maps will be produced in this latitude range
        :param bins: Number of bins of the map
        :param gamma: gamma factor for the Power Normal color map
        :return:
        """

        data_max=len(ld)

        #Data
        idxd=np.array([True,]*len(ld))
        if glim[0] is not None: idxd*=gd>=glim[0]
        if glim[1] is not None: idxd*=gd<=glim[1]



        Hd,edge,_=ploth2(ld[idxd],bd[idxd],bins=bins,range=[lrange,brange])
        Hd=Hd/data_max

        Hm=[]
        #Model
        for i in range(len(lm)):

            lmm=lm[i]
            gmm=gm[i]
            bmm=bm[i]

            model_max=len(lmm)

            idxm=np.array([True,]*len(lmm))


            if glim[0] is not None: idxm*=gmm>=glim[0]
            if glim[1] is not None: idxm*=gmm<=glim[1]



            Hm_t,edge,_=ploth2(lmm[idxm],bmm[idxm],bins=bins,range=[lrange,brange])
            Hm.append(Hm_t/model_max)

        Hm=np.median(Hm,axis=0)

        Hr=(Hd-Hm)/Hm
        xedges,yedges=edge
        extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]


        if axd is not None:
            axd.imshow(Hd.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=np.max(Hm),vmin=0,norm=PowerNorm(gamma=gamma))
            axd.set_title('Data',fontsize=30)
            axd.yaxis.set_ticks((-30,-60,0,30,60))
            axd.xaxis.set_ticks((-120,-60,0,60,120))
        if axm is not None:
            axm.imshow(Hm.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=np.max(Hm),vmin=0,norm=PowerNorm(gamma=gamma))
            axm.set_title('Model',fontsize=30)
            axm.yaxis.set_ticks((-30,-60,0,30,60))
            axm.xaxis.set_ticks((-120,-60,0,60,120))
        if axr is not None:
            im=axr.imshow(Hr.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=2.5,vmin=-2.5)
            axr.yaxis.set_ticks((-30,-60,0,30,60))
            axr.xaxis.set_ticks((-120,-60,0,60,120))
            if axcb is not None:
                fig=axr.figure
                fig.colorbar(im, cax=axcb, orientation='horizontal')
            axr.set_title('Data-Model/Model',fontsize=30)

        return Hd,Hm,Hr,edge



    def _residual_obs(self,ld,bd,gd,lm,bm,gm,axd=None,axm=None,axr=None,axcb=None,glim=(None,None),lrange=(-180,180),brange=(-90,90),bins=50,gamma=0.5):
        """

        :param ld: galactic longitude of the data
        :param bd: galactic latitude of the data
        :param gd: observed magnitude of the data
        :param lm: galactic longitude of the model
        :param bm: galactic latitude of the model
        :param gm: observed magnitude of the model
        :param axd: ax where to plot the sky-map of the data
        :param axm: ax where to plot the sky-map of the model
        :param axr: ax where to plot the sky-map of the residulas
        :param glim: The maps will be produced for G<min(glim), min(glim)<G<max(glim), G>max(glim)
        :param llim: The maps will be produced in this longitude range
        :param blim: The maps will be produced in this latitude range
        :param bins: Number of bins of the map
        :param gamma: gamma factor for the Power Normal color map
        :return:
        """



        idxd=np.array([True,]*len(ld))
        idxm=np.array([True,]*len(lm))

        if glim[0] is not None:
            idxd*=gd>=glim[0]
            idxm*=gm>=glim[0]
        if glim[1] is not None:
            idxd*=gd<=glim[1]
            idxm*=gm<=glim[1]

        Hd,edge,_=ploth2(ld[idxd],bd[idxd],bins=bins,range=[lrange,brange])
        Hd=Hd/np.sum(Hd)
        Hm,edge,_=ploth2(lm[idxm],bm[idxm],bins=bins,range=[lrange,brange])
        Hm=Hm/np.sum(Hm)
        Hr=(Hd-Hm)/Hm
        xedges,yedges=edge
        extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]


        if axd is not None:
            axd.imshow(Hd.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=np.max(Hm),vmin=0,norm=PowerNorm(gamma=gamma))
            axd.set_title('Data',fontsize=30)
            axd.yaxis.set_ticks((-30,-60,0,30,60))
            axd.xaxis.set_ticks((-120,-60,0,60,120))
        if axm is not None:
            axm.imshow(Hm.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=np.max(Hm),vmin=0,norm=PowerNorm(gamma=gamma))
            axm.set_title('Model',fontsize=30)
            axm.yaxis.set_ticks((-30,-60,0,30,60))
            axm.xaxis.set_ticks((-120,-60,0,60,120))
        if axr is not None:
            im=axr.imshow(Hr.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=2.5,vmin=-2.5)
            axr.yaxis.set_ticks((-30,-60,0,30,60))
            axr.xaxis.set_ticks((-120,-60,0,60,120))
            if axcb is not None:
                fig=axr.figure
                fig.colorbar(im, cax=axcb, orientation='horizontal')
            axr.set_title('Data-Model/Model',fontsize=30)

        return Hd,Hm,Hr,edge


    def _residual_physic2(self,ld,bd,gd,lm,bm,gm,Mg,axd=None,axm=None,axr=None,axcb=None,bins=50,xsun=8,gamma=0.5):

        #Data
        rd,zd=ut.obs_to_cyl(gd,ld,bd,Mg,xsun=xsun,negative_r=True)
        rmax=np.max(rd)
        zmax=np.max(zd)
        Hd,edge,_=ploth2(rd,zd,bins=bins,range=[(-rmax+xsun,rmax),(-zmax,zmax)])
        Hd=Hd/np.sum(Hd)


        #Model
        Hm=[]
        for i in range(len(lm)):

            lmm=lm[i]
            gmm=gm[i]
            bmm=bm[i]

            rm,zm=ut.obs_to_cyl(gmm,lmm,bmm,Mg,xsun=xsun,negative_r=True)


            Hm_t,edge,_=ploth2(rm,zm,bins=bins,range=[(-rmax+xsun,rmax),(-zmax,zmax)])
            Hm.append(Hm_t/np.sum(Hm_t))


        Hm=np.median(Hm,axis=0)
        Hr=(Hd-Hm)/Hm
        xedges,yedges=edge
        extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]

        if axd is not None:
            axd.imshow(Hd.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=np.max(Hm),vmin=0,norm=PowerNorm(gamma=gamma))
            axd.set_title('Data',fontsize=30)
            #axd.yaxis.set_ticks((-30,-60,0,30,60))
            #axd.xaxis.set_ticks((-120,-60,0,60,120))
        if axm is not None:
            axm.imshow(Hm.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=np.max(Hm),vmin=0,norm=PowerNorm(gamma=gamma))
            axm.set_title('Model',fontsize=30)
            #axm.yaxis.set_ticks((-30,-60,0,30,60))
            #axm.xaxis.set_ticks((-120,-60,0,60,120))
        if axr is not None:
            im=axr.imshow(Hr.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=2.5,vmin=-2.5)
            #axr.yaxis.set_ticks((-30,-60,0,30,60))
            #axr.xaxis.set_ticks((-120,-60,0,60,120))
            if axcb is not None:
                fig=axr.figure
                fig.colorbar(im, cax=axcb, orientation='horizontal')
            axr.set_title('Data-Model/Model',fontsize=30)

        return Hd,Hm,Hr,edge


    def _residual_physic(self,ld,bd,gd,lm,bm,gm,Mg,axd=None,axm=None,axr=None,axcb=None,bins=50,xsun=8,gamma=0.5):


        rd,zd=ut.obs_to_cyl(gd,ld,bd,Mg,xsun=xsun,negative_r=True)
        rm,zm=ut.obs_to_cyl(gm,lm,bm,Mg,xsun=xsun,negative_r=True)

        #rmin=np.min((np.min(rd),np.min(rm)))
        rmax=np.max(rd)
        zmax=np.max(zd)

        Hd,edge,_=ploth2(rd,zd,bins=bins,range=[(-rmax+xsun,rmax),(-zmax,zmax)])
        Hd=Hd/np.sum(Hd)
        Hm,edge,_=ploth2(rm,zm,bins=bins,range=[(-rmax+xsun,rmax),(-zmax,zmax)])
        Hm=Hm/np.sum(Hm)
        Hr=(Hd-Hm)/Hm
        xedges,yedges=edge
        extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]

        if axd is not None:
            axd.imshow(Hd.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=np.max(Hm),vmin=0,norm=PowerNorm(gamma=gamma))
            axd.set_title('Data',fontsize=30)
            #axd.yaxis.set_ticks((-30,-60,0,30,60))
            #axd.xaxis.set_ticks((-120,-60,0,60,120))
        if axm is not None:
            axm.imshow(Hm.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=np.max(Hm),vmin=0,norm=PowerNorm(gamma=gamma))
            axm.set_title('Model',fontsize=30)
            #axm.yaxis.set_ticks((-30,-60,0,30,60))
            #axm.xaxis.set_ticks((-120,-60,0,60,120))
        if axr is not None:
            im=axr.imshow(Hr.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=2.5,vmin=-2.5)
            #axr.yaxis.set_ticks((-30,-60,0,30,60))
            #axr.xaxis.set_ticks((-120,-60,0,60,120))
            if axcb is not None:
                fig=axr.figure
                fig.colorbar(im, cax=axcb, orientation='horizontal')
            axr.set_title('Data-Model/Model',fontsize=30)

        return Hd,Hm,Hr,edge

    def _mag_dist(self,gd,gm,gm_halo=None,bins=10,ax=None,grange=(10,18)):



        Hd,edge=np.histogram(gd,bins=bins,range=grange)
        Hm,_=np.histogram(gm,bins=bins,range=grange)


        Hd_f=Hd/np.sum(Hd)
        Hde_f=np.sqrt(Hd)/np.sum(Hd)
        Hm_f=Hm/np.sum(Hm)
        Hme_f=np.sqrt(Hm)/np.sum(Hm)


        if gm_halo is not None:
            Hm_halo,_=np.histogram(gm_halo,bins=bins,range=grange)
            Hm_halo_f=Hm_halo/np.sum(Hm_halo)
            Hme_halo_f=np.sqrt(Hm_halo)/np.sum(Hm_halo)


        x=0.5*(edge[:-1]+edge[1:])

        if ax is not None:
            ax.errorbar(x,Hd_f,Hde_f,c='black',label='Data',fmt='o',zorder=1000)
            ax.errorbar(x,Hm_f,Hme_f,c='red',label='Model',fmt='-o',zorder=500)
            if gm_halo is not None: ax.errorbar(x,Hm_halo_f,Hme_halo_f,c='red',label='Model (halo only)',fmt='-s',ls='dashed')
            ax.legend(loc='upper left')
            ax.set_xlabel('g')
            ax.set_ylabel('N/Ntot')

        if gm_halo is not None: return Hd,Hm,Hm_halo,edge
        else: return Hd,Hm,edge

    def _mag_dist2(self,gd,gm,gm_halo=None,bins=10,ax=None,grange=(10,18)):

        #Data
        Hd,edge=np.histogram(gd,bins=bins,range=grange)
        Hd_f=Hd/np.sum(Hd)
        Hde_f=np.sqrt(Hd)/np.sum(Hd)

        #Model
        Hm=[]
        Hm_halo=[]
        for i in range(len(gm)):

            gmm=gm[i]
            Hm_t,_=np.histogram(gmm,bins=bins,range=grange)
            Hm.append(Hm_t/np.sum(Hm_t))

            if gm_halo is not None:
                gmm_halo=gm_halo[i]
                Hm_halo_t,_=np.histogram(gmm_halo,bins=bins,range=grange)
                Hm_halo.append(Hm_halo_t/np.sum(Hm_halo_t))

        x=0.5*(edge[:-1]+edge[1:])
        Hm,Hme=ut.mad(Hm,axis=0)
        if gm_halo is not None: Hm_halo,Hme_halo=ut.mad(Hm_halo,axis=0)

        if ax is not None:
            ax.errorbar(x,Hd_f,Hde_f,c='black',label='Data',fmt='o',zorder=1000)
            ax.errorbar(x,Hm,Hme,c='red',label='Model',fmt='-o',zorder=500)
            if gm_halo is not None: ax.errorbar(x,Hm_halo,Hme_halo,c='red',label='Model (halo only)',fmt='-s',ls='dashed')
            ax.legend(loc='upper left')
            ax.set_xlabel('g')
            ax.set_ylabel('N/Ntot')

        if gm_halo is not None: return Hd,Hm,Hm_halo,edge
        else: return Hd,Hm,np.zeros(len(Hm)),edge

    def _residual_2Dslab(self,ld,bd,gd,lm,bm,gm,Mg,s='z',slim=(None,None),axd=None,axm=None,axr=None,axcbr=None,axcbdm=None,xsun=8,bins=30,gamma=0.5,fontsize=20,vmax=None,normalize=True):
        """
        Zslab between zlim
        :param ld:
        :param bd:
        :param gd:
        :param lm:
        :param bm:
        :param gm:
        :param Mg:
        :param zlim:
        :param axd:
        :param axm:
        :param axr:
        :return:
        """

        if normalize: data_max=len(ld)
        else: data_max=1



        xd,yd,zd=ut.obs_to_xyz(mag=gd,l=ld,b=bd,Mg=Mg,xsun=xsun)

        if s=='z':
            sd=zd
            ad=xd
            bd=yd
        elif s=='x':
            sd=xd
            ad=yd
            bd=zd
        elif s=='y':
            sd=yd
            ad=xd
            bd=yd

        #Data
        idxd=np.array([True,]*len(ld))
        if slim[0] is not None: idxd*=sd>slim[0]
        if slim[1] is not None: idxd*=sd<slim[1]

        print('Len Modello',np.sum(idxd))

        ad=ad[idxd]
        bd=bd[idxd]


        xmax=np.max(np.abs(ad))
        ymax=np.max(np.abs(bd))
        xymax=np.max([xmax,ymax])

        Hd,edge,_=ploth2(ad,bd,bins=bins,range=[(-xymax+xsun,xymax),(-xymax+xsun,xymax)])
        Hd=Hd/data_max

        Hm=[]
        #Model
        for i in range(len(lm)):

            lmm=lm[i]
            gmm=gm[i]
            bmm=bm[i]

            if normalize: mod_max=len(lmm)
            else: mod_max=1

            xm,ym,zm=ut.obs_to_xyz(mag=gmm,l=lmm,b=bmm,Mg=Mg,xsun=xsun)

            if s=='z':
                sm=zm
                am=xm
                bm=ym
            elif s=='x':
                sm=xm
                am=ym
                bm=zm
            elif s=='y':
                sm=ym
                am=xm
                bm=ym

            idxm=np.array([True,]*len(lmm))


            if slim[0] is not None: idxm*=sm>slim[0]
            if slim[1] is not None: idxm*=sm<slim[1]

            print('Len Modello',np.sum(idxm))


            Hm_t,edge,_=ploth2(am[idxm],bm[idxm],bins=bins,range=[(-xymax+xsun,xymax),(-xymax+xsun,xymax)])
            Hm.append(Hm_t/mod_max)

        Hm=np.median(Hm,axis=0)

        Hr=(Hd-Hm)/Hm
        xedges,yedges=edge
        extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]

        if vmax is None: vmax=np.max([np.max(Hm),np.max(Hd)])

        if axd is not None:
            imd=axd.imshow(Hd.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=vmax,vmin=0,norm=PowerNorm(gamma=gamma))
            axd.set_title('Data',fontsize=fontsize)
        if axm is not None:
            imm=axm.imshow(Hm.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=vmax,vmin=0,norm=PowerNorm(gamma=gamma))
            axm.set_title('Model',fontsize=fontsize)
        if axr is not None:
            imr=axr.imshow(Hr.T,origin='low',extent=extent,aspect='auto',interpolation='nearest',vmax=2.5,vmin=-2.5)
            if axcbr is not None:
                fig=axr.figure
                fig.colorbar(imr, cax=axcbr, orientation='horizontal')
            axr.set_title('Data-Model/Model',fontsize=fontsize)
        if axcbdm is not None:
            fig=axm.figure
            fig.colorbar(imm, cax=axcbdm, orientation='horizontal')

        return Hd,Hm,Hr,edge,vmax