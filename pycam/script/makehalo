#!/usr/bin/env python3

import OpOp.Model as md
import OpOp.grid as gd
import OpOp.analysis as an
import numpy as np
import matplotlib.pyplot as plt
import corner
import pycam.utils as ut
import pycam.model as mod
from astropy.coordinates import SkyCoord
from astropy import units as u
import argparse
import sys

descrip="Make a fits with a Nbody realisation of a density law\n"
descrip+="Can realize the following density law:\n"
descrip+="1-Power law: d propto r/rc**-ainn (rb=None, rt=None aout=0) \n"
descrip+="2-Truncated power law: d propto (r/rc**-ainn)*exp(-(r/rt)**2)  (rb=None, aout=0)\n"
descrip+="3-Break power law: d propto  r/rc**-ainn if r<rb; r/rc**-aout if r>rb (rt=None)\n"
descrip+="4-Truncated Break power law: d propto  exp(-(r/rt)**2)*r/rc**-ainn  if r<rb; exp(-(r/rt)**2)*r/rc**-aout if r>rb\n"
descrip+="5-Cored profile: d propto (1+r/rc)**-aout (ainn=0,rt=None,rb=None)\n"
descrip+="6-Truncated cored profile: d propto exp(-(r/rt)**2)*(1+r/rc)**-a (ainn=0rb=None)\n"
descrip+="7-Double Power law model: ((r/rc)**-ainn)*(1+r/rc)**-aout (rb=None,rt=None)\n"
descrip+="8-Truncated Double Power law model: exp(-(r/rt)**2)*((r/rc)**-ainn)*(1+r/rc)**-aout (rb=None)\n"
descrip+="With the following flattening along q (p can be only constant for now):\n"
descrip+="1-q costant: qinf <0 (eta is not considered) \n"
descrip+="2-q exp: q=qinf - (qinf-q0)*exp(1+sqrt(rq^2+m^2)/rq)  (qinf>q, eta<0)\n"
descrip+="3-q tanh: q=qinf + (qinf-q0)/(1-tanh(-rq/eta))*(tanh( (m-rq)/eta)-1) (qinf>q, eta>0)\n"


parser = argparse.ArgumentParser(description=descrip,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-aout",default=0,type=float,help="Outer Exponent [0]")
parser.add_argument("-ainn","-ainner",default=0, type=float,help="Inner Exponent [0]")
parser.add_argument("-rt",default=-999,type=float,help="Truncation radius [None]")
parser.add_argument("-rb",default=-999,type=float,help="Break radius [None]")
parser.add_argument("-neinasto",default=-999,type=float,help="Einasto n [None]")
parser.add_argument("-rc",default=1,type=float,help="Scale/Core radius [1]")
parser.add_argument("-p",default=1.0,type=float,help="X flattening [1]")

parser.add_argument("-q",default=1.0,type=float,help="Z flattening [1]")
parser.add_argument("-qinf",default=-999,type=float,help="Z flattening at r=inf [None]")
parser.add_argument("-rq",default=1.0,type=float,help="flattening radius")
parser.add_argument("-eta",default=-999,type=float,help="slope of the change of flattening [None]")


parser.add_argument("-wd",default=0.,type=float,help="Contamination of the disc wrt to halo in the masked area [0]")
parser.add_argument("-rd",default=2.682,type=float,help="Disc radial scale-length [2.682]")
parser.add_argument("-zd",default=0.196,type=float,help="Disc vertical scale-length [0.196]")

parser.add_argument("-alpha",default=0,type=float,help="Rotation around first axis [0]")
parser.add_argument("-beta",default=0,type=float,help="Rotation around second axis [0]")
parser.add_argument("-gamma",default=0,type=float,help="Rotation around third axis [0]")
parser.add_argument("-rotax",default='zyx',type=str,help="Rotation axes (3)")

parser.add_argument("-xoff",default=0,type=float,help="Offset wrt X gal. center [0]")
parser.add_argument("-yoff",default=0,type=float,help="Offset wrt Y gal. center [0]")
parser.add_argument("-zoff",default=0,type=float,help="Offset wrt Z gal. center [0]")

parser.add_argument("-xsun",default=8,type=float,help="Xsun position [8]")
parser.add_argument("-bmin",default=10,type=float,help="Lowest galactic latitude [10]")
parser.add_argument("-gmin","-gmagmin",default=10.,type=float,help="Min g magnitude [10]")
parser.add_argument("-gmax","-gmagmax",default=17.7,type=float,help="Max g magnitude [17.7]")
parser.add_argument("-tmin",'-thetamin',default=0,type=float,help="Lowest theta galactic latitude [0]")
parser.add_argument("-tmax",'-thetamax',default=90,type=float,help="Highest theta galactic latitude [0]")
parser.add_argument("-zmin",'-zgmin',default=0,type=float,help="Lowest Z galactic [0]")
parser.add_argument("-zmax",'-zgmax',default=500,type=float,help="Highest Z galactic [500]")
parser.add_argument("-n","-ntarget",default=20000,type=int,help="Number of objects to get [17000]")
parser.add_argument("-nt","-ntoll",default=200,type=int,help="Tollerance in the number of objects [2000]")
parser.add_argument("-nini","-ninitial",default=int(5e4),type=int,help="Initial number of particles [5e4]")

parser.add_argument("-Mgh",default='0.55',type=str,help="Absolute magnitude of star in  the halo [0.55]")
parser.add_argument("-Mgch",default=0.525,type=float,help="Gaussian centroid [0.525]")
parser.add_argument("-Mgsh",default=0.096,type=float,help="Gaussian dispersion [0.096]")
parser.add_argument("-Mguh",default=[-2,5],nargs=2,type=float,help="Uniform interval [-2,5]")


parser.add_argument("-Mgd",default='0.55',type=str,help="Absolute magnitude of star in the disc [0.55]")
parser.add_argument("-Mgcd",default=0.,type=float,help="Gaussian centroid [0]")
parser.add_argument("-Mgsd",default=1,type=float,help="Gaussian dispersion [1]")
parser.add_argument("-Mgud",default=[-2,5],nargs=2,type=float,help="Uniform interval [-2,5]")

parser.add_argument("-agecut",default=[-999,-999],nargs=2,type=float,help="Age cut of the isochrones (None,None)")
parser.add_argument("-fecut",default=[-999,-999],nargs=2,type=float,help="Metallicity cut of the isochrones (None,None)")
parser.add_argument("-colcut",default=[-1.0,-0.4],nargs=2,type=float,help="j-g color cut of the isochrones (-1.0,-0.4)")
parser.add_argument("-Mgcut",default=[-3,6],nargs=2,type=float,help="Mg cut of the isochrones (-3,6)")

parser.add_argument("-mask",nargs='*',type=str,help='Coordinate or name of struct to mask. Can be a name take from struc_list or struct_list_object or a new struct written as e.g. 10,20,30,40 or 10,20,1')

parser.add_argument("-mode",type=str,default='dist',help="Particles generation mode. dist (classical) or mc (mcmc sampling, works for p=1) or ar (rejection method, works for p=1 and  q<=1)")

parser.add_argument("-o","-output",default='model',help="Name of the model [model]")
parser.add_argument("-d",help="Debug mode",action="store_true")
args = parser.parse_args()

if  args.tmin<=0: tmin=None
else: tmin=args.tmin
if args.tmax>=90: tmax=None
else: tmax=args.tmax
if args.zmin<=0: zgmin=None
else: zgmin=args.zmin
if args.zmax==500: zgmax=None
else: zgmax=args.zmax


#Rt and Rb
if args.rt==-999: rt=None
else: rt=args.rt
if args.rb==-999: rb=None
else: rb=args.rb
if args.neinasto==-999: neinasto=None
else: neinasto=args.neinasto

#Halo Mg distribution
if args.Mgh[0].lower()=='g': Mgh='g'
elif args.Mgh[:2].lower()=='dg': Mgh='dg'
elif args.Mgh[:2].lower()=='u': Mgh='dg'
else:  Mgh=float(args.Mgh)

#Disc Mg distribution
if args.Mgd[0].lower()=='f': Mgd=args.Mgd[0:]
elif args.Mgd[0].lower()=='g':Mgd='g'
elif args.Mgd[0].lower()=='u':Mgd='u'
else: Mgd=float(args.Mgd)


#Cuts
agecut=np.array(args.agecut)
agecut=np.where(agecut==-999,None,agecut)
fecut=np.array(args.fecut)
fecut=np.where(fecut==-999,None,fecut)
colcut=np.array(args.colcut)
colcut=np.where(colcut==-999,None,colcut)
Mgcut=np.array(args.Mgcut)
Mgcut=np.where(colcut==-999,None,Mgcut)

'''
if args.qinf<0:
    q=args.q
else:
    q0=args.q
    qinf=args.qinf
    rq=args.rq
    q=lambda m: qinf - (qinf-q0)*np.exp((1-np.sqrt(m*m+rq*rq)/rq))
'''
if args.qinf<0:
    q=args.q
    qinf=None
    eta=None
elif args.eta<0:
    q=args.q
    qinf=args.qinf
    eta=None
else:
    q=args.q
    qinf=args.qinf
    eta=args.eta

rq=args.rq

if args.mode[0].lower()=='d': mode='dist'
elif args.mode[0].lower()=='m': mode='mc'
elif args.mode[0].lower()=='a': mode='ar'
else: raise NotImplementedError('Generation mode %s not implementend'%args.mode)


mod.make_model(ainn=args.ainn,aout=args.aout,rc=args.rc,rt=rt,rb=rb,neinasto=neinasto,q=q,qinf=qinf,rq=rq,eta=eta,p=args.p,wd=args.wd,rd=args.rd,zd=args.zd,bmin=args.bmin,gmin=args.gmin,gmax=args.gmax,xsun=args.xsun,thetamin=tmin,thetamax=tmax,zgmin=zgmin,zgmax=zgmax,n=args.n,nt=args.nt,nini=args.nini,Mgh=Mgh,Mgch=args.Mgch,Mgsh=args.Mgsh,Mgd=Mgd,Mgcd=args.Mgcd,Mgsd=args.Mgsd,Mgud=args.Mgud,mask=args.mask,name=args.o,diagnostic=args.d,agecut=agecut,fecut=fecut,colcut=colcut,Mgcut=Mgcut,mode=mode,alpha=args.alpha,beta=args.beta,gamma=args.gamma, xoff=args.xoff, yoff=args.yoff, zoff=args.zoff, rot_ax=args.rotax)

