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
from roteasy import rotate_frame, rotate
import emcee
import os
import sys
import time
import corner
import copy
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.pyplot as plt
import matplotlib as mpl
from pycam.cutils import calc_m
import datetime
from multiprocessing import Pool


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

        self._set_ini()

        self.load_data(fitsfile,magkey=magkey,lkey=lkey,bkey=bkey)
        self.set_int_option(glim=glim,llim=llim,blim=blim)




        #Standard par
        self.par={'ainn':0, 'aout':0,'rbs':1,'q':1,'p':1,'alpha':0,'beta':0,'gamma':0,'xoff':0,'yoff':0,'zoff':0,'f':1,'qinf':1,'rq':1,'eta':None}
        self.ainn=0
        self.aout=0
        self.rbs=1
        self.q=1
        self.qinf=1
        self.rq=1
        self.eta=1
        self.p=1
        self.alpha=0
        self.beta=0
        self.gamma=0
        self.xoff=0
        self.yoff=0
        self.zoff=0
        self.f=1

        #Geometrical parameter
        self.option_model={'s':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn',),self._loglike_s),
                      'qs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q'),self._loglike_sq),
                      'ps':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','p'),self._loglike_sp),
                      'pqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','p'),self._loglike_sqp),
                      'iqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('ainn','q','alpha','beta'),self._loglike_sqi),
                      'is':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('ainn','alpha','beta'),self._loglike_si),
                      'iq':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('q','alpha','beta'),self._loglike_qi),
                      'ips':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','p','alpha'),self._loglike_spi),
                      'ipqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('ainn','q','p','alpha','beta','gamma'),self._loglike_sqpi),
                      'jpqs': (self._discdens, self._halodens, self._vold_symmetric, self._volh_symmetric,('ainn', 'q', 'p', 'alpha'), self._loglike_sqpj),
                      'ipqsv': (self._discdens, self._halodens, self._vold_symmetric, self._volh_asymmetric,('ainn', 'q','qinf','rq', 'p', 'alpha', 'beta', 'gamma'), self._loglike_sqvpi),
                      'jpqsv': (self._discdens, self._halodens, self._vold_symmetric, self._volh_symmetric,('ainn', 'q', 'qinf', 'rq', 'p', 'alpha'), self._loglike_sqvpj),
                      'fs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','f'),self._loglike_sf),
                      'fqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','f'),self._loglike_sqf),
                      'fps':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','p','f'),self._loglike_spf),
                      'fpqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','p','f'),self._loglike_sqpf),
                      'fiqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','alpha','beta','f'),self._loglike_sqif),
                      'fips':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('ainn','p','alpha','f'),self._loglike_spif),
                      'fipqs':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('ainn','q','p','alpha','beta','gamma','f'),self._loglike_sqpif),
                           'qsv':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','qinf','rq'),self._loglike_sqv),
                           'qv':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('q','qinf','rq'),self._loglike_qv),
                           'pqsv': (self._discdens, self._halodens, self._vold_symmetric, self._volh_symmetric,('ainn','p','q', 'qinf', 'rq'), self._loglike_spqv),
                           'qst': (self._discdens, self._halodens, self._vold_symmetric, self._volh_symmetric, ('ainn','q', 'qinf', 'rq','eta'), self._loglike_sqt),
                           'fqst': (self._discdens, self._halodens, self._vold_symmetric, self._volh_symmetric, ('ainn','q', 'qinf', 'rq', 'eta','f'), self._loglike_sqtf),
                           'qstt': (self._discdens, self._halodens, self._vold_symmetric, self._volh_symmetric,('ainn','q', 'qinf', 'rq'), self._loglike_sqtt),
                      'fqsv':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','qinf','rq','f'),self._loglike_sqvf),
                      'fqsw':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','rq','f'),self._loglike_sqwf),
                           'qsw':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','q','rq'),self._loglike_sqw),
                           'bfq':(self._discdens,self._halodens_break,self._vold_symmetric,self._volh_symmetric,('ainn','aout','rbs','q','f'),self._loglike_bfq),
                            'bq':(self._discdens,self._halodens_break,self._vold_symmetric,self._volh_symmetric,('ainn','aout','rbs','q'),self._loglike_bq),
                           'boq': (self._discdens, self._halodens_break, self._vold_symmetric, self._volh_asymmetric,('ainn', 'aout', 'rbs', 'q','xoff','yoff','zoff'), self._loglike_boq),
                           'bqv': (self._discdens, self._halodens_break, self._vold_symmetric, self._volh_symmetric,('ainn', 'aout', 'rbs', 'q','qinf','rq'), self._loglike_bqv),
                           'bpq':(self._discdens,self._halodens_break,self._vold_symmetric,self._volh_symmetric,('ainn','aout','rbs','q','p'),self._loglike_bpq),
                           'bjpq': (self._discdens, self._halodens_break, self._vold_symmetric, self._volh_symmetric,('ainn', 'aout', 'rbs', 'q', 'p','alpha'), self._loglike_bpqj),
                           'bipq': (self._discdens, self._halodens_break, self._vold_symmetric, self._volh_asymmetric,('ainn', 'aout', 'rbs', 'q', 'p','alpha','beta','gamma'), self._loglike_bpqi),
                           'b':(self._discdens,self._halodens_break,self._vold_symmetric,self._volh_symmetric,('ainn','aout','rbs'),self._loglike_b),
                           'cfq':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('aout','rbs','q','f'),self._loglike_cfq),
                           'dfq':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','aout','rbs','q'),self._loglike_dfq),
                           'c': (self._discdens, self._halodens, self._vold_symmetric, self._volh_symmetric,('aout', 'rbs'), self._loglike_c),
                           'cq': (self._discdens, self._halodens, self._vold_symmetric, self._volh_symmetric,('aout', 'rbs', 'q'), self._loglike_cq),
                           'dq': (self._discdens, self._halodens, self._vold_symmetric, self._volh_symmetric,('ainn', 'aout', 'rbs', 'q'), self._loglike_dq),
                           'cpq': (self._discdens, self._halodens, self._vold_symmetric, self._volh_symmetric,('aout', 'rbs', 'q','p'), self._loglike_cqp),
                           'dpq': (self._discdens, self._halodens, self._vold_symmetric, self._volh_symmetric,('ainn', 'aout', 'rbs', 'q', 'p'), self._loglike_dqp),
                           'd':(self._discdens,self._halodens,self._vold_symmetric,self._volh_symmetric,('ainn','aout','rbs'),self._loglike_d),
                           'o':(self._discdens,self._halodens,self._vold_symmetric,self._volh_asymmetric,('xoff','yoff','zoff'),self._loglike_o),
                           'efq':(self._discdens,self._halodens_einasto,self._vold_symmetric,self._volh_symmetric,('ainn','rbs','q','f'),self._loglike_efq),
                           'eq':(self._discdens,self._halodens_einasto,self._vold_symmetric,self._volh_symmetric,('ainn','rbs','q'),self._loglike_eq),
                           'e': (self._discdens, self._halodens_einasto, self._vold_symmetric, self._volh_symmetric,('ainn', 'rbs'), self._loglike_e),
                           'epq':(self._discdens,self._halodens_einasto,self._vold_symmetric,self._volh_symmetric,('ainn','rbs','q','p'),self._loglike_epq),
                           'foqs': (self._discdens, self._halodens, self._vold_symmetric, self._volh_asymmetric,('ainn','q','f','xoff', 'yoff', 'zoff'), self._loglike_fsqo),
                           'oqs': (self._discdens, self._halodens, self._vold_symmetric, self._volh_asymmetric,('ainn','q','xoff', 'yoff', 'zoff'), self._loglike_soq),
                           'opqs': (self._discdens, self._halodens, self._vold_symmetric, self._volh_asymmetric,('ainn','q','p','xoff', 'yoff', 'zoff'), self._loglike_sopq),
                           'jopqs': (self._discdens, self._halodens, self._vold_symmetric, self._volh_asymmetric,('ainn', 'q', 'p', 'alpha', 'xoff', 'yoff', 'zoff'), self._loglike_sopqj),
			   'opqsv': (self._discdens, self._halodens, self._vold_symmetric, self._volh_asymmetric,('ainn','q','qinf','rq','p','xoff', 'yoff', 'zoff'), self._loglike_sopqv),
			   'iopqs': (self._discdens, self._halodens, self._vold_symmetric, self._volh_asymmetric,('ainn','q','p','xoff', 'yoff', 'zoff','alpha','beta','gamma'), self._loglike_soipq),
}

    #Set and Reset
    def _set_ini(self):

        self.data=[]
        self.fitsfile=None


        self.Mgd=[0.525,]
        self.wd=[1,]
        self.Mgd_descriptor='Delta Dirac Mg=0.55'
        self._Mgdkind='d'
        self._Mgdgrange=None
        self._Mgdurange=None
        self.Mgh=[0.525,]
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

        #Integ
        self.erel=1e-6
        self.nval=100000
        self.niter=10

        self.glim=(10,18)
        self.llim=(0,360)
        self.blim=(10,90)

        self.thetalim=(0,90)


        self.xmin_nord=[self.glim[0],self.llim[0],self.blim[0]]
        self.xmax_nord=[self.glim[1],self.llim[1],self.blim[1]]
        self.xmin_sud=[self.glim[0],self.llim[0],-self.blim[1]]
        self.xmax_sud=[self.glim[1],self.llim[1],-self.blim[0]]

        self.xlim_nord_vega=[self.glim,self.llim,self.blim]
        self.xlim_sud_vega=[self.glim,self.llim,(-self.blim[1],-self.blim[0])]

        #Rot
        self._rot='zyx'

        #Struct
        self.struct=[]


        self.set_Wfunc()

        #Prior limit
        self.ainn_lim=(0.,10)
        self.aout_lim=(0.,10)
        self.rbs_lim=(0.0001,150)
        self.q_lim=(0.05,5)
        self.qinf_lim=(0.05,5)
        self.rq_lim=(0.0001,300)
        self.eta_lim=(0.5,100)
        self.p_lim=(0.05,5)
        self.i_lim=(-90,90)
        self.off_lim=(-50,50)
        self.ffrac_lim=(0,1)


        self.ainn_prior=partial(self._prior_uniform,lim=self.ainn_lim)
        self.ainn_prior_descriptor='Uniform [0.;10]'

        self.aout_prior=partial(self._prior_uniform,lim=self.aout_lim)
        self.aout_prior_descriptor='Uniform [0.;10]'


        self.rbs_prior=partial(self._prior_uniform,lim=self.rbs_lim)
        self.rbs_prior_descriptor='Uniform [0.0001;100]'

        self.q_prior=partial(self._prior_uniform,lim=self.q_lim)
        self.q_prior_descriptor='Uniform q=[0.05;5]'

        self.qinf_prior=partial(self._prior_uniform,lim=self.qinf_lim)
        self.qinf_prior_descriptor='Uniform qinf=[0.05;5]'

        self.rq_prior=partial(self._prior_uniform,lim=self.rq_lim)
        self.rq_prior_descriptor='Uniform rq=[0.0001;200]'

        self.eta_prior=partial(self._prior_uniform,lim=self.eta_lim)
        self.eta_prior_descriptor='Uniform eta=[0.5;100]'

        self.p_prior=partial(self._prior_uniform,lim=self.p_lim)
        self.p_prior_descriptor='Uniform [0.05;5]'

        self.i_prior=partial(self._prior_uniform,lim=self.i_lim)
        self.i_prior_descriptor='Uniform [-90;90]'

        self.off_prior=partial(self._prior_uniform,lim=self.off_lim)
        self.off_prior_descriptor='Uniform [-50;50]'

        self.ffrac_prior=partial(self._prior_uniform,lim=self.ffrac_lim)
        self.ffrac_prior_descriptor='Uniform [0;1]'

        self.prior_dict={'ainn':self.ainn_lim,'aout':self.aout_lim,'rbs':self.rbs_lim,'q':self.q_lim,'qinf':self.qinf_lim,'rq':self.rq_lim,'eta':self.eta_lim,'p':self.p_lim,'i':self.i_lim,'off':self.off_lim,'f':self.ffrac_lim}
        self.prior_dict_descriptor={'ainn':self.ainn_prior_descriptor,'aout':self.aout_prior_descriptor,'rbs':self.rbs_prior_descriptor,'q':self.q_prior_descriptor,'qinf':self.qinf_prior_descriptor,'rq':self.rq_prior_descriptor,'eta':self.eta_prior_descriptor,'p':self.p_prior_descriptor,'i':self.i_prior_descriptor,'off':self.off_prior_descriptor,'f':self.ffrac_prior_descriptor}


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

            #Update the model volume
            if len(self.struct)>0:
                self.vold=self._vold_struct
                self.volh=self._volh_struct
                self.set_Wfunc()

    def reset_struct(self):
        self.struct=[]
        self.set_Wfunc()
        if (thetalim[0]==0) and (thetalim[1]==90): 
                self.vold=None
                self.volh=None

    def set_thetalim(self,thetamin=0,thetamax=90):
        self.thetalim=(thetamin,thetamax)
        if (thetamin!=0) or (thetamax!=90): 
                self.vold=self._vold_struct
                self.volh=self._volh_struct
        self.set_Wfunc()

    def set_rot(self,strg):

        self._rot=strg

    def reset_thetalim(self):
        self.thetalim=(0,90)
        self.set_Wfunc()
        if len(self.struct)==0: 
                self.vold=None
                self.volh=None

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



            if (len(self.struct)==0) and (self.thetalim[0]==0) and (self.thetalim[1]==90):
                print('Set Vol nostruct')
                self.vold=vold
                self.volh=volh
            else:
                print('Set Vol struct')
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

                elif name.lower()=='qinf':
                    if kind[0].lower()=='u':
                        self.qinf_prior=partial(self._prior_uniform,lim=rangel)
                        self.qinf_prior_descriptor='Uniform  [%.3f,%.3f]'%(float(rangel[0]),float(rangel[1]))
                        self.qinf_lim=rangel
                        self.prior_dict['qinf']=rangel
                    elif kind[0].lower()=='g':
                        self.qinf_prior=partial(self._prior_gau,lim=rangel)
                        self.qinf_prior_descriptor='Gaussian  Gau(c=%.3f,s=%.3f)'%(float(rangel[0]),float(rangel[1]))
                    else:
                        raise NotImplementedError('Prior of kind %s not implemented'%kind)


                elif name.lower()=='rq':
                    if kind[0].lower()=='u':
                        self.rq_prior=partial(self._prior_uniform,lim=rangel)
                        self.rq_prior_descriptor='Uniform  [%.3f,%.3f]'%(float(rangel[0]),float(rangel[1]))
                        self.rq_lim=rangel
                        self.prior_dict['rq']=rangel
                    elif kind[0].lower()=='g':
                        self.rq_prior=partial(self._prior_gau,lim=rangel)
                        self.rq_prior_descriptor='Gaussian  Gau(c=%.3f,s=%.3f)'%(float(rangel[0]),float(rangel[1]))
                    else:
                        raise NotImplementedError('Prior of kind %s not implemented'%kind)

                elif name.lower()=='eta':
                    if kind[0].lower()=='u':
                        self.eta_prior=partial(self._prior_uniform,lim=rangel)
                        self.eta_prior_descriptor='Uniform  [%.3f,%.3f]'%(float(rangel[0]),float(rangel[1]))
                        self.eat_lim=rangel
                        self.prior_dict['eta']=rangel
                    elif kind[0].lower()=='g':
                        self.eta_prior=partial(self._prior_gau,lim=rangel)
                        self.eta_prior_descriptor='Gaussian  Gau(c=%.3f,s=%.3f)'%(float(rangel[0]),float(rangel[1]))
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
    def reset(self):
        self._set_ini()

    #Dens functions
    def _halodens(self,m,ainn=1,aout=0,rs=1):

        inndens=(m/rs)**(-ainn)
        outdens=(1+m/rs)**(-aout)

        return inndens*outdens
    def _halodens_break(self,m,ainn=1,aout=1,rb=1):


        ret=np.where(m<=rb,m**-ainn,(rb**(aout-ainn))*(m**-aout))


        return ret

    def _halodens_einasto(self,m,ainn=1,aout=1,rs=1):
        """
        ainn will be the famous n or einasto, aout does not do anytighin, but serve per manentere la simmetria
        :param m:
        :param ainn:
        :param aout:
        :param rs:
        :return:
        """

        n=ainn

        dn=3*n-1/3.+0.0079/n

        exp=dn*((m/rs)**(1/n) - 1)

        return np.exp(-exp)

    def _discdens(self,rcyl,z,rd,zd):

        return np.exp(-rcyl/rd)*np.exp(-np.abs(z)/zd)

    #Window functin
    def _Wfunc_struct(self,arr,Mg=None):
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

    #Window functin
    def _Wfunc_theta(self,arr,Mg):
        thetamin=self.thetalim[0]
        thetamax=self.thetalim[1]
        g=arr[:,0]
        l=arr[:,1]
        b=arr[:,2]
        rg,zg=ut.obs_to_cyl(g,l,b,Mg=Mg,xsun=self.xsun)
        theta_abs=np.arctan(np.abs(zg)/rg)*(360/(2*np.pi))
 

        return np.where((theta_abs>=thetamin)&(theta_abs<=thetamax),True,False)


    #Window functin
    def _Wfunc_comb(self,arr,Mg):
        ret_struct=self._Wfunc_struct(arr,Mg)
        ret_theta=self._Wfunc_theta(arr,Mg)

        return ret_struct*ret_theta

    #Window functin
    def _Wfunc_err(self,arr,Mg=None):

        raise RuntimeError('Called Wfunc, but Wfunc has not been set')

        return 0

    #Window functin
    def set_Wfunc(self):

        cond_struct=len(self.struct)>0
        cond_theta1=(self.thetalim[0]!=0)
        cond_theta2=(self.thetalim[1]!=90)
        if (cond_struct==True) and ( (cond_theta1==True) or  (cond_theta2==True)): self._Wfunc=self._Wfunc_comb
        elif (cond_struct==True): self._Wfunc=self._Wfunc_struct
        elif ((cond_theta1==True) or (cond_theta2==True) ): self._Wfunc=self._Wfunc_theta
        else: self._Wfunc=self._Wfunc_err



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
        Windowf=self._Wfunc(arr,Mg)

        return j*rhod*Windowf
    def _vold_symmetric(self,Mg,rd,zd):
        """
        Volume density of the disc for Z symmetric model without structure
        :param Mg:
        :param rd:
        :param zd:
        :return:
        """
        #print('Self',self.erel)
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

        if integral_nord<0 or integral_sud<0:
            return np.nan

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
    def _integrand_halo(self,arr,Mg,ainn,aout,rbs,q,qinf,rq,eta,p,alpha,beta,gamma,xoff,yoff,zoff):
        """

        :param arr:
        :param Mg:
        :param ainn:
        :param aout:
        :param rbs: Can be rs or rb depending on the halo model
        :return:
        """
        dist=ut.mag_to_dist(arr[:,0],Mg=Mg)
        mm=self.obs_to_m(arr,Mg=Mg,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
        j=(dist**3)*np.cos(arr[:,2]*2*np.pi/360)
        rhoh=self.halodens(mm,ainn,aout,rbs)

        #print('j',arr,np.max(mm),np.min(mm))

        return j*rhoh

    def _integrand_halo_struct(self,arr,Mg,ainn,aout,rbs,q,qinf,rq,eta,p,alpha,beta,gamma,xoff,yoff,zoff):
        """

        :param arr:
        :param Mg:
        :param ainn:
        :param aout:
        :param rbs: Can be rs or rb depending on the halo model
        :return:
        """
        dist=ut.mag_to_dist(arr[:,0],Mg=Mg)
        mm=self.obs_to_m(arr,Mg=Mg,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
        j=(dist**3)*np.cos(arr[:,2]*2*np.pi/360)
        rhoh=self.halodens(mm,ainn,aout,rbs)
        Windowf=self._Wfunc(arr,Mg)


        return j*rhoh*Windowf
    def _volh_symmetric(self,Mg,ainn,aout,rbs,q,qinf,rq,eta,p,alpha,beta,gamma,xoff,yoff,zoff):

        integral=2*cubature(self._integrand_halo,3,1,kwargs={'Mg':Mg,'ainn':ainn,'aout':aout,'rbs':rbs,'q':q,'qinf':qinf,'rq':rq,'eta':eta,'p':p,'alpha':alpha,'beta':beta,'gamma':gamma,'xoff':xoff,'yoff':yoff,'zoff':zoff},xmin=self.xmin_nord,xmax=self.xmax_nord,vectorized=True,abserr=0,relerr=self.erel,maxEval=500000)[0][0]

        if integral<0:
            return np.nan

        return integral
    def _volh_asymmetric(self,Mg,ainn,aout,rbs,q,qinf,rq,eta,p,alpha,beta,gamma,xoff,yoff,zoff):

        integral_nord=cubature(self._integrand_halo,3,1,kwargs={'Mg':Mg,'ainn':ainn,'aout':aout,'rbs':rbs,'q':q,'qinf':qinf,'rq':rq,'eta':eta,'p':p,'alpha':alpha,'beta':beta,'gamma':gamma,'xoff':xoff,'yoff':yoff,'zoff':zoff},xmin=self.xmin_nord,xmax=self.xmax_nord,vectorized=True,abserr=0,relerr=self.erel,maxEval=500000)[0][0]
        integral_sud=cubature(self._integrand_halo,3,1,kwargs={'Mg':Mg,'ainn':ainn,'aout':aout,'rbs':rbs,'q':q,'qinf':qinf,'rq':rq,'eta':eta,'p':p,'alpha':alpha,'beta':beta,'gamma':gamma,'xoff':xoff,'yoff':yoff,'zoff':zoff},xmin=self.xmin_sud,xmax=self.xmax_sud,vectorized=True,abserr=0,relerr=self.erel,maxEval=500000)[0][0]

        #print('int',integral_nord,integral_sud,self.xsun)

        if integral_sud<0 or integral_nord<0:
            return np.nan


        return integral_nord+integral_sud
    def _volh_struct(self,Mg,ainn,aout,rbs,q,qinf,rq,eta,p,alpha,beta,gamma,xoff,yoff,zoff):

        @vegas.batchintegrand
        def integrand_struct(arr):
            return self._integrand_halo_struct(arr,Mg=Mg,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)

        integ_s = vegas.Integrator(self.xlim_sud_vega, nhcube_batch=self.nval)
        integ_s_run=integ_s(integrand_struct,nitn=self.niter, neval=self.nval)
        integral_south=integ_s_run.mean
        integral_south_std=integ_s_run.sdev


        integ_n = vegas.Integrator(self.xlim_nord_vega, nhcube_batch=self.nval)
        integ_n_run = integ_n(integrand_struct,nitn=self.niter, neval=self.nval)
        integral_nord = integ_n_run.mean
        integral_nord_std = integ_n_run.sdev

        integral_std=np.sqrt(integral_nord_std*integral_nord_std + integral_south_std*integral_south_std)

        return integral_south+integral_nord
        #return integral_south+integral_nord, integral_std

    def _volh_sdss(self,Mg,ainn,aout,rbs,q,qinf,rq,eta,p,alpha,beta,gamma,xoff,yoff,zoff):

        integral=0
        #1
        xmin=(gliml,0,1)
        xmax=(glimu,0,1)
        integral+=cubature(self._integrand_halo,3,1,kwargs={'Mg':Mg,'ainn':ainn,'aout':aout,'rbs':rbs,'q':q,'qinf':qinf,'rq':rq,'eta':eta,'p':p,'alpha':alpha,'beta':beta,'gamma':gamma,'xoff':xoff,'yoff':yoff,'zoff':zoff},xmin=xmin,xmax=xmax,vectorized=True,abserr=0,relerr=self.erel)[0][0]

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

    def _Phalo(self,arr,ainn,aout,rbs,q,qinf,rq,eta,p,alpha,beta,gamma,xoff,yoff,zoff):

        rhoarr=np.zeros(len(arr))
        volarr=0

        #print('xoff in Phalo %.20f'%xoff)

        for i in range(len(self.Mgh)):
            w=self.wh[i]
            Mgh=self.Mgh[i]
            dist=ut.mag_to_dist(arr[:,0],Mg=Mgh)
            mm=self.obs_to_m(arr,Mg=Mgh,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            rhoh=self.halodens(mm,ainn,aout,rbs)

            rhoarr+=(w*dist**3)*rhoh
            volarr+=w*self.volh(Mg=Mgh,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            #volarr=1
            #print('vol',volarr)
            #print('rho',np.sum(rhoarr))

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

        Pdisc,ainn,aout,rbs,q,qinf,rq,eta,p,alpha,beta,gamma,xoff,yoff,zoff,f=args
        ainn=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
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

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args


        ainn,f=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or f>1 or f<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.ffrac_prior(f)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
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

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
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

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,f=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or f>1 or f<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
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

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,p=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.p_prior(p)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
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

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,p,f=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0 or f>1 or f<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.p_prior(p)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqp(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,p=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqpf(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,p,f=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0 or q<0 or f>1 or f<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob

    def _loglike_spqv(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,p,q,qinf,rq=theta

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or qinf<q:
            return -np.inf

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.qinf_prior(qinf)+self.rq_prior(rq)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob


    def _loglike_sqi(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,alpha,beta=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.i_prior(alpha)+self.i_prior(beta)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_si(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,alpha,beta=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.i_prior(alpha)+self.i_prior(beta)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_qi(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        q,alpha,beta=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.q_prior(q)+self.i_prior(alpha)+self.i_prior(beta)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob

    def _loglike_sqif(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,alpha,beta,f=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or f<0 or f>1:
            return -np.inf


        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_spi(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,p,alpha=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.p_prior(p)+self.i_prior(alpha)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob

    def _loglike_spif(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,p,alpha,f=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or p<0 or f>1 or f<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.p_prior(p)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqpi(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,p,alpha,beta,gamma=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or p<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob

    def _loglike_sqpj(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,p,alpha=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or p<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob


    def _loglike_sqpif(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,p,alpha,beta,gamma,f=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or p<0 or f>1 or f<0:
            return -np.inf


        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob



    def _loglike_sqpi(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,p,alpha,beta,gamma=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or p<0 or f>1 or f<0:
            return -np.inf


        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob

    def _loglike_sqvpi(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,qinf,rq,p,alpha,beta,gamma=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or p<0 or f>1 or f<0 or qinf<q:
            return -np.inf


        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)+self.qinf_prior(qinf)+self.rq_prior(rq)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob

    def _loglike_sqvpj(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,qinf,rq,p,alpha=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or p<0 or f>1 or f<0 or qinf<q:
            return -np.inf


        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)+self.qinf_prior(qinf)+self.rq_prior(rq)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob


    def _loglike_sqv(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,qinf,rq=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or qinf<q:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.qinf_prior(qinf)+self.rq_prior(rq)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob

    def _loglike_sqt(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,qinf,rq,eta=theta


        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or qinf<q:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.qinf_prior(qinf)+self.rq_prior(rq)+self.eta_prior(eta)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob

    def _loglike_sqtf(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,qinf,rq,eta,f=theta


        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or qinf<q:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.qinf_prior(qinf)+self.rq_prior(rq)+self.eta_prior(eta)+self.ffrac_prior(f)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob

    def _loglike_sqtt(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,qinf,rq=theta


        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or qinf<q:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.qinf_prior(qinf)+self.rq_prior(rq)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob

    def _loglike_qv(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        q,qinf,rq=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or qinf<q:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.q_prior(q)+self.qinf_prior(qinf)+self.rq_prior(rq)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob


    def _loglike_sqvf(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,qinf,rq,f=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or qinf<q:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.qinf_prior(qinf)+self.rq_prior(rq)+self.ffrac_prior(f)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_qv(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        q,qinf,rq=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or qinf<q:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.q_prior(q)+self.qinf_prior(qinf)+self.rq_prior(rq)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqwf(self,theta,*args):

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,rq,f=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or qinf<q:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.rq_prior(rq)+self.ffrac_prior(f)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

        return lp + lprob
    def _loglike_sqw(self,theta,*args):
        """

        :param theta:
        :param args:
        :return:
        """
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,rq=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or qinf<q:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.q_prior(q)+self.rq_prior(rq)

        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))


        return lp + lprob

    def _loglike_bfq(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs,q,f=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or aout<ainn:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)+self.ffrac_prior(f)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_bq(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs,q=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or aout<ainn:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_boq(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs,q,xoff,yoff,zoff=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or aout<ainn:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)+self.off_prior(xoff)+self.off_prior(yoff)+self.off_prior(zoff)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

            if not np.isfinite(lprob):
                return -np.inf

        return lp + lprob

    def _loglike_bqv(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs,q,qinf,rq=theta
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or aout<ainn or qinf<q:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)+self.qinf_prior(qinf)+self.rq_prior(rq)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob


    def _loglike_bpq(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs,q,p=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or aout<ainn or p<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)+self.p_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_bpqi(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs,q,p,alpha,beta,gamma=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or aout<ainn or p<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_bpqj(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs,q,p,alpha=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or aout<ainn or p<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_b(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff


        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_cfq(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        aout,rbs,q,f=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)+self.ffrac_prior(f)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_c(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        aout,rbs=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.aout_prior(aout)+self.rbs_prior(rbs)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_cq(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        aout,rbs,q=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_cqp(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        aout,rbs,q,p=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)+self.p_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob


    def _loglike_dfq(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs,q,f=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)+self.ffrac_prior(f)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_dq(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs,q=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_dqp(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs,q,p=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)+self.q_prior(q)+self.p_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob


    def _loglike_d(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,aout,rbs=theta
        qinf=q
        eta=None


        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.aout_prior(aout)+self.rbs_prior(rbs)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_o(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        xoff,yoff,zoff=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.off_prior(xoff)+self.off_prior(yoff)+self.off_prior(zoff)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
            if np.isnan(lprob): return -np.inf
            else: return lp + lprob
        return lp + lprob


    def _loglike_fsqo(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,xoff,yoff,zoff,q,f=theta
        qinf=q
        eta=None


        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.off_prior(xoff)+self.off_prior(yoff)+self.off_prior(zoff)+self.ainn_prior(ainn)+self.q_prior(q)+self.ffrac_prior(f)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_soq(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,xoff,yoff,zoff=theta
        qinf=q
        eta=None


        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #print('x prima',xoff)
        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-np.floor(xoff*1000)/1000
        yoff=-np.floor(yoff*1000)/1000
        zoff=-np.floor(zoff*1000)/1000

        #print('x %.20f'%xoff)
        #xoff=0.415
        #print('x %.20f'%xoff)

        lp=self.off_prior(xoff)+self.off_prior(yoff)+self.off_prior(zoff)+self.ainn_prior(ainn)+self.q_prior(q)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))

            if not np.isfinite(lprob):
                return -np.inf

        return lp + lprob

    def _loglike_sopq(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,p,xoff,yoff,zoff=theta
        qinf=q
        eta=None


        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.off_prior(xoff)+self.off_prior(yoff)+self.off_prior(zoff)+self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
            if np.isnan(lprob): return -np.inf
            else: return lp + lprob
        return lp + lprob

    def _loglike_sopqj(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,p,alpha,xoff,yoff,zoff=theta
        qinf=q
        eta=None


        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.off_prior(xoff)+self.off_prior(yoff)+self.off_prior(zoff)+self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
            if np.isnan(lprob): return -np.inf
            else: return lp + lprob
        return lp + lprob


    def _loglike_sopqv(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,qinf,rq,p,xoff,yoff,zoff=theta
        eta=None


        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or qinf<q:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.off_prior(xoff)+self.off_prior(yoff)+self.off_prior(zoff)+self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.qinf_prior(qinf)+self.rq_prior(rq)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_soipq(self,theta,*args):
        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,q,p,xoff,yoff,zoff,alpha,beta,gamma=theta
        qinf=q
        eta=None


        #Ulteriore controllo per evitare cose assurde
        if ainn<0 or q<0 or p<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.off_prior(xoff)+self.off_prior(yoff)+self.off_prior(zoff)+self.ainn_prior(ainn)+self.q_prior(q)+self.p_prior(p)+self.i_prior(alpha)+self.i_prior(beta)+self.i_prior(gamma)
        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
        return lp + lprob

    def _loglike_efq(self,theta,*args):
        """
        Fit of a single power law + disc
        :fit: ainn, q, f
        :param theta:
        :param args:
        :return:
        """

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,rbs,q,f=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0.05 or q<0 or f>1 or f<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.rbs_prior(rbs)+self.q_prior(q)+self.ffrac_prior(f)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
            if not np.isfinite(lprob):
                return -np.inf


        return lp + lprob

    def _loglike_e(self,theta,*args):
        """
        Fit of a single power law + disc
        :fit: ainn, q, f
        :param theta:
        :param args:
        :return:
        """

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,rbs=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0.05 or q<0 or p<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.rbs_prior(rbs)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
            if not np.isfinite(lprob):
                return -np.inf


        return lp + lprob


    def _loglike_eq(self,theta,*args):
        """
        Fit of a single power law + disc
        :fit: ainn, q, f
        :param theta:
        :param args:
        :return:
        """

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,rbs,q=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0.05 or q<0 or p<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.rbs_prior(rbs)+self.q_prior(q)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
            if not np.isfinite(lprob):
                return -np.inf


        return lp + lprob


    def _loglike_epq(self,theta,*args):
        """
        Fit of a single power law + disc
        :fit: ainn, q, f
        :param theta:
        :param args:
        :return:
        """

        Pdisc, ainn, aout, rbs, q, qinf, rq, eta, p, alpha, beta, gamma, xoff, yoff, zoff, f = args
        ainn,rbs,q,p=theta
        qinf=q
        eta=None

        #Ulteriore controllo per evitare cose assurde
        if ainn<0.05 or q<0 or p<0:
            return -np.inf

        #Questo perchè Se l'alone è rotato di 45 allora serve rotare nel senso inverso le stelle per riallinearle col il sistema galattico, vale anche per l'offset
        #alpha=-alpha
        #beta=-beta
        #gamma=-gamma
        xoff=-xoff
        yoff=-yoff
        zoff=-zoff

        lp=self.ainn_prior(ainn)+self.rbs_prior(rbs)+self.q_prior(q)+self.p_prior(p)


        if not np.isfinite(lp):
            return -np.inf
        else:
            c1=f*self._Phalo(self.data,ainn=ainn,aout=aout,rbs=rbs,q=q,qinf=qinf,rq=rq,eta=eta,p=p,alpha=alpha,beta=beta,gamma=gamma,xoff=xoff,yoff=yoff,zoff=zoff)
            c2=(1-f)*Pdisc
            lprob=np.sum(np.log(c1+c2))
            if not np.isfinite(lprob):
                return -np.inf


        return lp + lprob




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
                pos = [x0_list + ini_pos_gau * np.random.randn(dim) for i in range(nwalker)]
            elif iguess=='par':
                x0_list=self.make_iguess(parlist,param)
                pos = [x0_list + ini_pos_gau * np.random.randn(dim) for i in range(nwalker)]

            elif iguess=='prior':

                pos=self.make_iguess_fromprior(parlist,nwalker)

            else: raise NotImplementedError('iguess=%s not implemented'%iguess)
            #Initial position

        else:
            if (len(iguess)==nwalker) and (len(iguess[0])==dim): pos=iguess
            else: raise ValueError()

        #Initialise sampler
        if nproc>1: pool=Pool(nproc)
        else: pool=None
        sampler = emcee.EnsembleSampler(nwalker, dim, loglike, args=args_list,pool=pool)#threads=nproc)


        tini=time.time() #Timer


        #Burn phase
        if nburn>0:
            tiniburn=time.time()
            print('Burn',flush=True)
            sys.stdout.flush()
            pos0,lnprob0,_=sampler.run_mcmc(pos,nburn)
            tfinburn=time.time()
            sampler.reset()
            tburn=tfinburn-tiniburn
            tburnstep=tburn/nburn
            tburnstepchain=tburn/(nburn*nwalker)
            tburnforecast=tburnstep*nstep
            print('Burn end. \nTime burn: %.3s \nTime per eval:%.5s \nTime per chain:%.5s \nExpected Time for MCMC:%.5s \nExpected end: %s'%(tburn,tburnstepchain,tburnstep,tburnforecast,datetime.datetime.now()+datetime.timedelta(seconds=tburnforecast)),flush=True)
        else:
            pos0=pos
            lnprob0=None


        #MCMC
        print('Start MCMC chain')
        sys.stdout.flush()
        if int(emcee.__version__[0])>2:
                sampler.run_mcmc(pos0, nstep,log_prob0=lnprob0)
        else:
                sampler.run_mcmc(pos0, nstep,lnprob0=lnprob0)
        tfin=time.time()


        print('Done in %f s'%(tfin-tini))

        samples=sampler.flatchain
        postprob=sampler.flatlnprobability


        fixed_param={}
        for item in self.par:
            if item in param:fixed_param[item]=param[item]
            else: fixed_param[item]=self.par[item]


        res=MC_result(fitobj=self,nwalker=nwalker,nstep=nstep,samples=samples, sampler=sampler, prob=postprob,fitlist=parlist,fixed_par=fixed_param,stat=self.query_stat())

        maxlik_idx=np.argmax(postprob)
        best_pars=samples[maxlik_idx,:]
        best_like=postprob[maxlik_idx]


        if plot is not None:
            try:
                res.plot_triangle(plot+'.png',quantiles=(0.16,0.5,0.84),levels=(0.68,0.95),sigma=False)
            except:
                print('WARNING, error in plot_triangle, plot skipped.')



        return res,best_pars,best_like,self.current_model


    def obs_to_m(self,arr,Mg,q,qinf,rq,eta,p,alpha,beta,gamma,xoff,yoff,zoff):
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

        #print('xsun',xsun,xoff,yoff,zoff)

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

        #Off set always on the Galactic axes
        if (xoff!=0) or (yoff!=0) or (zoff!=0):
            x-=xoff
            y-=yoff
            z-=zoff

        if (alpha!=0) or (gamma!=0) or (beta!=0):
            cord=rotate_frame(cord=np.array([x,y,z]).T, angles=(alpha,beta,gamma), axes=self._rot, reference='lh')
            #cord=ut.rotate_xyz(np.array([x,y,z]).T,alpha=alpha,beta=beta,gamma=gamma,system='lh')
            x=cord[:,0]
            y=cord[:,1]
            z=cord[:,2]




        

        if q==qinf:
            y=y/p
            z=z/q
            m=np.sqrt(x*x+y*y+z*z)
        else:
            m=np.array(calc_m(x,y,z, q, qinf, rq, p, eta))


        return m

    
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

        aname_list=('ainn','aout','rbs','q','qinf','rq','eta','p','alpha','beta','gamma','xoff','yoff','zoff','f')

        for aname in aname_list:

            if aname in arglist: outl.append(arglist[aname])
            else:
                if aname=='qinf': outl.append(arglist['q'])
                else: outl.append(self.par[aname])


        return tuple(outl)

    def make_iguess(self,pars,x0_list):

        outl=[]
        for par in pars:
            if par in x0_list: outl.append(x0_list[par])
            else: outl.append(self.par[par])

        return tuple(outl)

    def make_iguess_fromprior(self,pars,nwalkers):

        ndim=len(pars)

        ipar=np.zeros(shape=(nwalkers,ndim))

        i=0
        for par in pars:

            if par=='alpha' or par=='beta' or par=='gamma':
                if self.prior_dict_descriptor['i'][0].lower()=='u':
                    low,up=self.prior_dict['i']
                    ipar[:,i]=np.random.uniform(low,up,size=nwalkers)
                elif self.prior_dict_descriptor['i'][0].lower() == 'g':
                    c, s = self.prior_dict['i']
                    ipar[:, i] = np.random.normal(c, s, size=nwalkers)
                else:
                    raise NotImplementedError('Error in make_iguess_fromprior')

            elif par=='xoff' or par=='yoff' or par=='zoff':
                if self.prior_dict_descriptor['off'][0].lower()=='u':
                    low,up=self.prior_dict['off']
                    ipar[:,i]=np.random.uniform(low,up,size=nwalkers)
                elif self.prior_dict_descriptor['off'][0].lower() == 'g':
                    c, s = self.prior_dict['off']
                    ipar[:, i] = np.random.normal(c, s, size=nwalkers)
                else:
                    raise NotImplementedError('Error in make_iguess_fromprior')

            else:
                if self.prior_dict_descriptor[par][0].lower() == 'u':
                    low, up = self.prior_dict[par]
                    ipar[:, i] = np.random.uniform(low, up, size=nwalkers)
                elif self.prior_dict_descriptor[par][0].lower() == 'g':
                    c, s = self.prior_dict[par]
                    ipar[:, i] = np.random.normal(c, s, size=nwalkers)
                else:
                    raise NotImplementedError('Error in make_iguess_fromprior')


            i+=1

        return ipar

    #Priors
    def _prior_uniform(self,value,lim):

        if lim[0]<=value<=lim[1]: return 0
        else: return -np.inf
    def _prior_gau(self,value,lim):

        y=(value-lim[0])/lim[1]

        return -0.5*y*y

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
        h+='thetalim: (%.1f,%.1f)\n'%self.thetalim
        h+='Erel: %.1e\n'%self.erel
        h+='Nval: %.1e Niter=%i\n'%(self.nval,int(self.niter))

        h+='MODEL\n'
        h+=str(self.current_model)+'\n'

        h+='PRIORS\n'
        h+='ainn: %s\n'%self.ainn_prior_descriptor
        h+='aout: %s\n'%self.aout_prior_descriptor
        h+='rbs: %s\n'%self.rbs_prior_descriptor
        h+='q: %s\n'%self.q_prior_descriptor
        h+='qinf: %s\n'%self.qinf_prior_descriptor
        h+='rq: %s\n'%self.rq_prior_descriptor
        h += 'eta: %s\n' % self.eta_prior_descriptor
        h+='p: %s\n'%self.p_prior_descriptor
        h+='f: %s\n'%self.ffrac_prior_descriptor
        h+='i: %s\n'%self.i_prior_descriptor
        h+='rotax: %s\n'%self._rot
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

    #Print
    def __str__(self):
        return self.query_stat()

class MC_result():

    def __init__(self,fitobj,nwalker,nstep,samples,sampler,prob,fitlist,fixed_par,stat=''):

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
        self.halodens=copy.copy(fitobj.halodens)

        self.Nbody_models=None #Continete il fits con la realizazione Nbody
        self.sample={} #Contiente i valori di tutte le variabili fittate
        self.best={} #Contiente i best-value (max value of the likelihood) delle variabili fittate
        self.median={} #Contiene le mediane di tutte le variabili fittate
        self.percentile={} #Contiente i percentili (0.5,2.5,16,50,84,97.5,99.5) di tutte le variabili fittate
        self.one_sigma={} #1sigma interval di tutte le variabili fittate
        self.two_sigma={} #2sigma interval di tutte le variabili fittate
        self.three_sigma={} #3sigma interval di tutte le variabili fittate
        self.bestlnlike=prob[idx_best] #Max likelihood
        self.ndim=len(fitlist) #Dimendion of the fitted params
        self.arr=np.zeros((nstep*nwalker,self.ndim+1)) #Array with the value of the fitted values and the resultat log likelihood
        self.param=fixed_par #Containi all the params
        self.varlist=fitlist #The name of the fitted params
        self.bic=-2*self.bestlnlike+len(self.varlist)*np.log(len(self.data)) #Calcolo del BIC (vedi Xue)

        self.accep_frac=sampler.acceptance_fraction
        try:
            self.acor=sampler.acor
        except:
            self.acor=np.array([-9999,]*self.ndim)

        for i in range(self.ndim):
            self.arr[:,i]=samples[:,i]
        self.arr[:,-1]=prob



        i=0
        for par in fitlist:
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
            if (item=='ainn') and ('e' in self.fitmodel): h+='%s: %.2f\n'%('n',float(self.param[item]))
            elif (item=='eta') and (self.param[item] is None):  h+='eta: None'
            else: h+='%s: %.2f\n'%(item,float(self.param[item]))

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
        h+='BIC=%.5f\n'%self.bic
        h+='Best Likelihood params:\n'
        for item in self.best:
            h+='%s: %.5f\n'%(item,float(self.best[item]))

        return h
    def __str__(self):

        return self._print_result()
    def plot_triangle(self,outname='Triangle_plot.png',quantiles=(0.16,0.5,0.84),levels=(0.68,0.95),sigma=False,plot_datapoints=False,fill_contours=True,show_titles=True,**kwargs):


        levels=np.array(levels)
        if sigma: levels=np.exp(-0.5*(levels*levels))


        label_list=[]
        for item in self.varlist:
            if item=='ainn':
                if 'e' in self.fitmodel: label_list.append('${\\mathrm{n}}$')
                else: label_list.append('$\\alpha_{\\mathrm{inn}}$')
            elif item=='aout': label_list.append('$\\alpha_{\\mathrm{out}}$')
            elif item=='rbs':
                if 'b' in self.fitmodel: label_list.append('${\\mathrm{r}}_{\\mathrm{b}}$')
                elif 'c' in self.fitmodel: label_list.append('${\\mathrm{r}}_{\\mathrm{c}}$')
                elif 'd' in self.fitmodel: label_list.append('${\\mathrm{r}}_{\\mathrm{s}}$')
                elif 'e' in self.fitmodel: label_list.append('${\\mathrm{r}}_{\\mathrm{eff}}$')
            elif item=='qinf': label_list.append('${\\mathrm{q}}_{\\infty}$')
            elif item=='rq': label_list.append('${\\mathrm{r}}_{\\mathrm{q}}$')
            elif item == 'eta': label_list.append('${\\eta}$')
            else: label_list.append(item)

        print('Llist',label_list)

        fig=corner.corner(self.arr[:,:-1],labels=label_list,quantiles=quantiles,levels=levels,plot_datapoints=plot_datapoints,fill_contours=fill_contours,show_titles=show_titles,**kwargs)
        fig.savefig(outname)

    def make_model(self,outdir='',bvalue='median',diagnostic=False,Mgh=None,Mgd=None,extra_struct=None,thetamin=None,thetamax=None,zgmin=None,zgmax=None,nmodel=1,mode='ar'):
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
            neinasto=None
        elif 'e' in self.fitmodel:
            rb=None
            rc=tpar_list['rbs']
            neinasto=tpar_list['ainn']
        else:
            rb=None
            neinasto=None
            rc=tpar_list['rbs']

        if 'qinf' in self.varlist:
            q0=tpar_list['q']
            qinf=tpar_list['qinf']
            rq=tpar_list['rq']
            if 'eta' in self.varlist: eta=tpar_list['eta']
            else: eta=None
        else:
            q0=tpar_list['q']
            qinf=q0
            rq=1
            eta=None


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
        else: raise ValueError('Mgd %s not allowed'%str(Mgd))


        if extra_struct is None: struct_tmp=self.struct
        else:
            struct_tmp=list(copy.copy(self.struct))
            for item in extra_struct:
                struct_tmp.append(item)
            if (len(struct_tmp)==0): struct_tmp=None

        self.Nbody_models=[]

        if nmodel==1:
            Nbody_model,_=md.make_model(xsun=self.xsun,aout=tpar_list['aout'],ainn=tpar_list['ainn'],neinasto=neinasto,rc=rc,rb=rb,q=q0,qinf=qinf,rq=rq,eta=eta,p=tpar_list['p'],wd=1-tpar_list['f'],rd=self.rd,zd=self.zd,alpha=tpar_list['alpha'],beta=tpar_list['beta'],gamma=tpar_list['gamma'],xoff=tpar_list['xoff'],yoff=tpar_list['yoff'],zoff=tpar_list['zoff'],bmin=self.blim[0],gmin=self.glim[0],gmax=self.glim[1],n=self.nstars,Mgh=Mgh,Mgd=Mgd,Mgcd=Mgcd,Mgsd=Mgsd,Mgud=Mgud,mask=struct_tmp,name=name,diagnostic=diagnostic,output=output_m,outdir=outdir,mode=mode,thetamin=thetamin,thetamax=thetamax,zgmin=zgmin,zgmax=zgmax)
            self.Nbody_models.append(Nbody_model)
        elif nmodel>=1:
            for i in range(nmodel):
                #Save in output only the first model
                if i==0: Nbody_model,_=md.make_model(xsun=self.xsun,aout=tpar_list['aout'],ainn=tpar_list['ainn'],neinasto=neinasto,rc=rc,rb=rb,q=q0,qinf=qinf,rq=rq,eta=eta,p=tpar_list['p'],wd=1-tpar_list['f'],rd=self.rd,zd=self.zd,alpha=tpar_list['alpha'],beta=tpar_list['beta'],gamma=tpar_list['gamma'],xoff=tpar_list['xoff'],yoff=tpar_list['yoff'],zoff=tpar_list['zoff'],bmin=self.blim[0],gmin=self.glim[0],gmax=self.glim[1],n=self.nstars,Mgh=Mgh,Mgd=Mgd,Mgcd=Mgcd,Mgsd=Mgsd,Mgud=Mgud,mask=struct_tmp,name=name,diagnostic=diagnostic,output=output_m,outdir=outdir,mode=mode,thetamin=thetamin,thetamax=thetamax,zgmin=zgmin,zgmax=zgmax)
                else: Nbody_model,_=md.make_model(xsun=self.xsun,aout=tpar_list['aout'],ainn=tpar_list['ainn'],neinasto=neinasto,rc=rc,rb=rb,q=q0,qinf=qinf,rq=rq,eta=eta,p=tpar_list['p'],wd=1-tpar_list['f'],rd=self.rd,zd=self.zd,alpha=tpar_list['alpha'],beta=tpar_list['beta'],gamma=tpar_list['gamma'],xoff=tpar_list['xoff'],yoff=tpar_list['yoff'],zoff=tpar_list['zoff'],bmin=self.blim[0],gmin=self.glim[0],gmax=self.glim[1],n=self.nstars,Mgh=Mgh,Mgd=Mgd,Mgcd=Mgcd,Mgsd=Mgsd,Mgud=Mgud,mask=struct_tmp,name=name,output=False,diagnostic=False,mode=mode,thetamin=thetamin,thetamax=thetamax,zgmin=zgmin,zgmax=zgmax)
                self.Nbody_models.append(Nbody_model)



    def residual(self,outname='residual',outdir=os.getcwd(),glim=(15.5,15.5),lrange=(0,360),brange=(-90,90),bins=50,Mgh=None,Mgdrange=None,model_idx=None):

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
        ax1a=fig.add_subplot(3,3,1)
        ax1b=fig.add_subplot(3,3,4)
        ax1c=fig.add_subplot(3,3,7)
        ax2a=fig.add_subplot(3,3,2)
        ax2b=fig.add_subplot(3,3,5)
        ax2c=fig.add_subplot(3,3,8)
        ax3a=fig.add_subplot(3,3,3)
        ax3b=fig.add_subplot(3,3,6)
        ax3c=fig.add_subplot(3,3,9)
        #ax4a=fig.add_subplot(3,4,4)
        #ax4b=fig.add_subplot(3,4,8)
        #ax4c=fig.add_subplot(3,4,12)
        axb=fig.add_axes([0.123,0.05,0.51,0.02])

        self._residual_obs2(ld,bd,gd,lm,bm,gm,axd=ax1a,axm=ax1b,axr=ax1c,axcb=axb,glim=(None,None),lrange=lrange,brange=brange,bins=bins)
        self._residual_obs2(ld,bd,gd,lm,bm,gm,axd=ax2a,axm=ax2b,axr=ax2c,glim=(None,glim[0]),lrange=lrange,brange=brange,bins=bins)
        self._residual_obs2(ld,bd,gd,lm,bm,gm,axd=ax3a,axm=ax3b,axr=ax3c,glim=(glim[1],None),lrange=lrange,brange=brange,bins=bins)
        #self._residual_obs2(ld,bd,gd,lm,bm,gm,axd=ax4a,axm=ax4b,axr=ax4c,glim=(glim[1],None),lrange=lrange,brange=brange,bins=bins)

        ax1a.set_ylabel('b [deg]',fontsize=30)
        ax1b.set_ylabel('b [deg]',fontsize=30)
        ax1c.set_ylabel('b [deg]',fontsize=30)
        ax3c.set_xlabel('l [deg]',fontsize=30)
        #ax4c.set_xlabel('l [deg]',fontsize=30)
        axb.set_xlabel('Residuals',fontsize=25)

        plt.figtext(0.22,0.95,'G all',fontsize=35)
        plt.figtext(0.47,0.95,'G<%.1f'%glim[0],fontsize=35)
       # plt.figtext(0.56,0.95,'%.1f<G<%.1f'%glim,fontsize=35)
        plt.figtext(0.75,0.95,'G>%.1f'%glim[1],fontsize=35)


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


        plt.figtext(0.425,0.95,'Mg=%.3f'%float(Mgh),fontsize=35)

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


        #Qvar
        if 'qinf' in self.sample:
            fig=plt.figure()
            ax=fig.add_subplot(111)
            self.qplot(ax)
            ax.set_xlabel('m',fontsize=18)
            ax.set_ylabel('q',fontsize=18)
            fig.savefig(outdir+'/'+outname+'_qplot.png')

        #Dens
        fig=plt.figure()
        ax=fig.add_subplot(111)
        self.densplot(ax)
        ax.set_xlabel('$\\rho$ [Arbitrary unit]',fontsize=18)
        ax.set_ylabel('m [kpc]',fontsize=18)
        fig.savefig(outdir+'/'+outname+'_densplot.png')

    def densplot(self,ax,rmax=60):

        if 'b' in self.fitmodel:
            ainn=self.sample['ainn']
            aout=self.sample['aout']
            rbs=self.sample['rbs']

        elif 's' in self.fitmodel:
            ainn=self.sample['ainn']
            aout=self.param['aout']
            rbs=self.param['rbs']

        elif 'd' in self.fitmodel:
            ainn=self.sample['ainn']
            aout=self.sample['aout']
            rbs=self.sample['rbs']

        elif 'c' in self.fitmodel:
            ainn=self.param['ainn']
            aout=self.sample['aout']
            rbs=self.sample['rbs']

        elif 'e' in self.fitmodel:
            ainn=self.sample['ainn']
            aout=1
            rbs=self.sample['rbs']

        m=np.linspace(0.0001,rmax,1000)
        d50=np.zeros(len(m))
        d16=np.zeros(len(m))
        d84=np.zeros(len(m))
        d97=np.zeros(len(m))
        d2=np.zeros(len(m))

        for i in range(len(m)):
            rhoh=self.halodens(i,ainn,aout,rbs)
            d2[i],d16[i],d50[i],d84[i],d97[i]=tuple(np.percentile(rhoh,q=[2.5,16,50,84,97.5]))

        ax.fill_between(m, d16, d84, facecolor='green', interpolate=True,zorder=10,label='1$\\sigma$')
        ax.fill_between(m, d2, d97, facecolor='lightgreen', interpolate=True,label='1$\\sigma$')
        ax.plot(m,d50,color='black',lw=2,label='Mean')
        ax.legend(loc='upper right')
        ax.set_yscale('log')
        ax.set_xscale('log')


    def qplot(self,ax,rmax=60):
        """

        :param q0: List of value of q0
        :param qinf:  List of value of qinf
        :param rq:  List of value of rq
        :return:
        """

        if 'v' in self.fitmodel:
            q0=self.sample['q']
            qinf=self.sample['qinf']
            rq=self.sample['rq']
            eta=None
        elif 'w' in self.fitmodel:
            q0=self.sample['q']
            qinf=1
            rq=self.sample['rq']
            eta=None
        elif 't' in self.fitmodel:
            q0 = self.sample['q']
            qinf = self.sample['qinf']
            rq = self.sample['rq']
            eta=self.sample['eta']


        m=np.linspace(0.0001,rmax,1000)
        q50=np.zeros(len(m))
        q16=np.zeros(len(m))
        q84=np.zeros(len(m))
        q97=np.zeros(len(m))
        q2=np.zeros(len(m))

        for i in range(len(m)):
            mvals=ut.qfunc_exp(m[i],q0,qinf,rq,eta)
            q2[i],q16[i],q50[i],q84[i],q97[i]=tuple(np.percentile(mvals,q=[2.5,16,50,84,97.5]))

        ax.fill_between(m, q16, q84, facecolor='green', interpolate=True,zorder=10,label='1$\\sigma$')
        ax.fill_between(m, q2, q97, facecolor='lightgreen', interpolate=True,label='1$\\sigma$')
        ax.plot(m,q50,color='black',lw=2,label='Mean')
        ax.legend(loc='upper left')

        if (np.min(q2)-0.2)<0: ymin=0
        else: ymin=np.min(q2)-0.2

        ax.set_ylim(ymin,np.max(q97)+0.2)
        ax.set_xlim(0,rmax)




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
