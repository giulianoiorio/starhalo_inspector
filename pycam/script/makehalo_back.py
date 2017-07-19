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

parser = argparse.ArgumentParser(description=descrip,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-aout",default=0,type=float,help="Outer Exponent [0]")
parser.add_argument("-ainn","-ainner",default=0, type=float,help="Inner Exponent [0]")
parser.add_argument("-rt",default=-999,type=float,help="Truncation radius [None]")
parser.add_argument("-rb",default=-999,type=float,help="Break radius [None]")
parser.add_argument("-rc",default=1,type=float,help="Scale/Core radius [1]")
parser.add_argument("-q",default=1.0,type=float,help="Z flattening [1]")
parser.add_argument("-p",default=1.0,type=float,help="X flattening [1]")

parser.add_argument("-wd",default=0.,type=float,help="Contamination of the disc wrt to halo in the masked area [0]")
parser.add_argument("-rd",default=2.682,type=float,help="Disc radial scale-length [2.682]")
parser.add_argument("-zd",default=0.196,type=float,help="Disc vertical scale-length [0.196]")

parser.add_argument("-i","-irot",default=0,type=float,help="Rotation around y-axis [0]")
parser.add_argument("-bmin",default=10,type=float,help="Lowest galactic latitude [10]")
parser.add_argument("-gmin","-gmagmin",default=10.,type=float,help="Min g magnitude [10]")
parser.add_argument("-gmax","-gmagmax",default=17.7,type=float,help="Max g magnitude [17.7]")
parser.add_argument("-n","-ntarget",default=17000,type=int,help="Number of objects to get [17000]")
parser.add_argument("-nt","-ntoll",default=2000,type=int,help="Tollerance in the number of objects [2000]")

parser.add_argument("-Mgh",default='0.55',type=str,help="Absolute magnitude of star in  the halo [0.55]")
parser.add_argument("-Mgch",default=0.525,type=float,help="Gaussian centroid [0.525]")
parser.add_argument("-Mgsh",default=0.096,type=float,help="Gaussian dispersion [0.096]")


parser.add_argument("-Mgd",default='0.55',type=str,help="Absolute magnitude of star in the disc [0.55]")
parser.add_argument("-Mgcd",default=0.,type=float,help="Gaussian centroid [0]")
parser.add_argument("-Mgsd",default=1,type=float,help="Gaussian dispersion [1]")

parser.add_argument("-agecut",default=[-999,-999],nargs=2,type=float,help="Age cut of the isochrones (None,None)")
parser.add_argument("-fecut",default=[-999,-999],nargs=2,type=float,help="Metallicity cut of the isochrones (None,None)")
parser.add_argument("-colcut",default=[-1.0,-0.4],nargs=2,type=float,help="j-g color cut of the isochrones (-1.0,-0.4)")
parser.add_argument("-Mgcut",default=[-3,6],nargs=2,type=float,help="Mg cut of the isochrones (-3,6)")

parser.add_argument("-mask",nargs='*',type=str,help='Coordinate or name of struct to mask. Can be a name take from struc_list or struct_list_object or a new struct written as e.g. 10,20,30,40 or 10,20,1')

parser.add_argument("-o","-output",default='model',help="Name of the model [model]")
parser.add_argument("-d",help="Debug mode",action="store_true")
args = parser.parse_args()


aout=args.aout
if args.rt==-999: rt=None
else: rt=args.rt
if args.rb==-999: rb=None
else: rb=args.rb

ainn=args.ainn
rc=args.rc
p=args.p
q=args.q
irot=args.i
bmin=args.bmin
gmin=args.gmin
gmax=args.gmax
name=str(args.o)

#Halo Mg distribution
if args.Mgh[0].lower()=='g': Mgh='g'
elif args.Mgh[:2].lower()=='dg': Mgh='dg'
else:  Mgh=float(args.Mgh)



#Disc Mg distribution
if args.Mgd[0].lower()=='f':
	Mgd=args.Mgd[1:]
	agecut=np.array(args.agecut)
	agecut=np.where(agecut==-999,None,agecut)
	fecut=np.array(args.fecut)
	fecut=np.where(fecut==-999,None,fecut)
	colcut=np.array(args.colcut)
	colcut=np.where(colcut==-999,None,colcut)
	Mgcut=np.array(args.Mgcut)
	Mgcut=np.where(colcut==-999,None,Mgcut)
elif args.Mgd[0].lower()=='g':Mgd='g'
else: Mgd=float(args.Mgd)


outdir='halo_model_'+name
ut.mkdir(outdir)

struc_l=[]


if args.mask is not None:
	if args.mask[0]=='all':
		for item in ut.struct_list: struc_l.append(ut.struct_list[item.upper()])
	else:
		for item in args.mask:
			if item.upper() in ut.struct_list: struc_l.append(ut.struct_list[item.upper()])
			elif item.upper() in ut.struct_list_object: struc_l.append(ut.struct_list_object[item.upper()])
			else: struc_l.append(tuple(float(i) for i in item.split(',')))

print(struc_l)



npart_ini=5e4
npartsh=0
npartsd=0


#Halo
if args.wd!=1:
	ntargeth=int(args.n*(1-args.wd))
	tollh=int(args.nt*(1-args.wd))
	print('Start- First Run Halo')
	cordh,parth,fdens_input=mod.halo_model(aout=aout,ainn=ainn,rt=rt,rb=rb,rc=rc,npart=npart_ini,p=p,q=q)

	if irot!=0:
		print('Rotating....',end='')
		cordh=ut.rotate(cordh,beta=irot)
		print('Done')


	print('Observing....',end='')
	lh,bh,disth=mod.halo_observe(cordh)
	if isinstance(Mgh,float):
		oldgh=ut.dist_to_g(disth,Mg=Mgh)
	elif Mgh=='g':
		oldgh,agh=mod.gau_mag(disth,Mg0=args.Mgch,Mgs=args.Mgsh)
	else: raise NameError('Not implemented')
	print('Done')
	print('Masking....',end='')
	lh,bh,gh,idxh=mod.halo_mask(lh,bh,oldgh,bmin=bmin,gmagmin=gmin,gmagmax=gmax,struct=struc_l)
	disth=disth[idxh]
	print('Done')
	npartsh=len(lh)
	print('Npart=%i'%npartsh)

	if (ntargeth-tollh)<=npartsh<=(ntargeth+tollh):
		print('Halo Done')
	else:
		corr_facth=ntargeth/npartsh
		npart_finalh=corr_facth*npart_ini
		if npart_finalh>3e6:
			print('Warning Maximum particles greatern than 3e6, it has ben set to 3e6')
			npart_finalh=int(3e6)
		print('Start Second Run')
		cordh,parth,fdens_inputh=mod.halo_model(aout=aout,ainn=ainn,rt=rt,rb=rb,rc=rc,npart=npart_finalh,p=p,q=q)
		if irot!=0:
			print('Rotating....',end='')
			cordh=ut.rotate(cordh,beta=irot)
			print('Done')
		print('Observing....',end='')
		lh,bh,disth=mod.halo_observe(cordh)
		if isinstance(Mgh,float):
			oldgh=ut.dist_to_g(disth,Mg=Mgh)
		elif Mgh=='g':
			oldgh,agh=mod.gau_mag(disth,Mg0=args.Mgch,Mgs=args.Mgsh)
		else: raise NameError('Not implemented')
		print('Done')
		print('Masking....',end='')
		lh,bh,gh,idxh=mod.halo_mask(lh,bh,oldgh,bmin=bmin,gmagmin=gmin,gmagmax=gmax,struct=struc_l)
		disth=disth[idxh]
		print('Halo Done')
		npartsh=len(lh)
		print('Final Npart=%i'%npartsh)

####Halo Diagnostic
	if args.d:
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
		fig.savefig(outdir+'/diagnostic_halodens.pdf')
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
###########

#Disc
if args.wd!=0:
	ntargetd=int(args.n*args.wd)
	tolld=int(args.nt*args.wd)
	print('Start- First Run Disc')
	cordd=mod.disc_model(args.rd,args.zd,npart=npart_ini)
	print('Observing....',end='')
	ld,bd,distd=mod.halo_observe(cordd)
	print('Done')
	if isinstance(Mgd,float):
		oldgd=ut.dist_to_g(distd,Mg=Mgd)
	elif Mgd=='g':
		oldgd,agd=mod.gau_mag(distd,Mg0=args.Mgcd,Mgs=args.Mgsd)
	else:
		oldgd,agd,agd_input=mod.disc_mag(Mgd,distd,agecut=agecut,fecut=fecut,gcut=Mgcut,colcut=colcut)
	print('Masking....',end='')
	ld,bd,gd,idxd=mod.halo_mask(ld,bd,oldgd,bmin=bmin,gmagmin=gmin,gmagmax=gmax,struct=struc_l)
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
		cordd=mod.disc_model(args.rd,args.zd,npart=npart_finald)
		print('Observing....',end='')
		ld,bd,distd=mod.halo_observe(cordd)
		print('Done')
		if isinstance(Mgd,float):
			oldgd=ut.dist_to_g(distd,Mg=Mgd)
		elif Mgd=='g':
			oldgd,agd=mod.gau_mag(distd,Mg0=args.Mgcd,Mgs=args.Mgsd)
		else:
			oldgd,agd,agd_input=mod.disc_mag(Mgd,distd,agecut=agecut,fecut=fecut,gcut=Mgcut,colcut=colcut)
		print('Masking....',end='')
		ld,bd,gd,idxd=mod.halo_mask(ld,bd,oldgd,bmin=bmin,gmagmin=gmin,gmagmax=gmax,struct=struc_l)
		distd=distd[idxd]
		print('Done')
		npartsd=len(ld)
		print('Disc Done')
		print('Final Npart=%i'%npartsd)

####Disc Diagnostic
	if args.d:
		print('Plotting disc diagnostic')
		if isinstance(Mgd,float)==False:
			fig=plt.figure(figsize=(12,5))
			ax1=fig.add_subplot(121)
			ax2=fig.add_subplot(122)
			if Mgd=='g': ax1.hist(agd,bins=40,normed=True)
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
if 0<args.wd<1:
	if args.d:
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

if args.wd==1: dmodel+='Double Exponential disc'
else:
	if rt is not None: dmodel+='Truncated '
	if rb is not None: dmodel+='Broken Power Law'
	else:
		if (aout!=0) and (ainn!=0): dmodel+='Double Power Law'
		elif aout!=0: dmodel+='Cored Power Law'
		else: dmodel+='Power Law'

	if args.wd!=0: dmodel+=' + Double Exponential disc'


dict_global={'Model':dmodel ,'aout':aout,'ainn':ainn,'rt':rt,'rb':rb,'rc':rc,'p':p,'q':q,'irot':irot,'ntarget':args.n,'toll':args.nt,'wd':args.wd,'npartsh':npartsh,'npartsd':npartsd,'idcode':idcode,'Mgh':args.Mgh}
dict_masked={'Model':dmodel ,'aout':aout,'ainn':ainn,'rt':rt,'rb':rb,'rc':rc,'p':p,'q':q,'irot':irot,'bmin':bmin,'gmin':gmin,'gmax':gmax,'ntarget':args.n,'toll':args.nt,'wd':args.wd,'npartsh':npartsh,'npartsd':npartsd,'idcode':idcode,'Mgh':args.Mgh}
if args.wd!=0:
	dict_global['rd']=args.rd
	dict_global['zd']=args.zd
	dict_masked['rd']=args.rd
	dict_masked['zd']=args.zd
	dict_global['Mgd']=Mgd
	dict_masked['Mgd']=Mgd
	if args.Mgd[0].lower()=='f':
		dict_masked['agecut']=str(agecut)
		dict_masked['fecut']=str(fecut)
		dict_masked['colcut']=str(colcut)
		dict_masked['Mgcut']=str(Mgcut)
		dict_global['agecut']=str(agecut)
		dict_global['fecut']=str(fecut)
		dict_global['colcut']=str(colcut)
		dict_global['Mgcut']=str(Mgcut)
	elif args.Mgd[0].lower()=='g':
		dict_masked['Mgcd']=args.Mgcd
		dict_masked['Mgsd']=args.Mgsd
		dict_global['Mgcd']=args.Mgcd
		dict_global['Mgsd']=args.Mgsd
	else: pass
if args.wd!=1:
	if Mgh=='g':
		dict_masked['Mgch']=args.Mgch
		dict_masked['Mgsh']=args.Mgsh
		dict_global['Mgch']=args.Mgch
		dict_global['Mgsh']=args.Mgsh





print()
print('##################################################')
print('Summary')
with open(outdir+'/'+name+'_logfile.txt', 'w') as out:
	for item in dict_masked:
		line=item+':'+str(dict_masked[item]) + '\n'
		print(line,end='')
		out.write(line)
print('name:',name)
print('##################################################')
print('Done....')


print('Making fits....',end='')
if args.wd==0:
	x,y,z=cordh.T
	m=np.sqrt(x*x+(y*y/(p*p))+(z*z/(q*q)))
	r=np.sqrt(x*x+(y*y)+(z*z))
	rcyl=np.sqrt(x*x+y*y)
	c=SkyCoord(l=lh*u.degree,b=bh*u.degree,frame='galactic')
	if isinstance(Mgh,float):
		Mghl=[Mgh,]*np.sum(idxh)
		Mghl_all=[Mgh,]*len(rcyl)
	else:
		Mghl=agh[idxh]
		Mghl_all=agh
	cir=c.icrs
	ra=cir.ra.degree
	dec=cir.dec.degree
	ut.make_fits({'X':x,'Y':y,'Z':z,'m':m,'r':r,'rcyl':rcyl,'idh':np.ones_like(r),'idd':np.zeros_like(r),'Mg':Mghl_all},outname=outdir+'/'+name+'_global.fits',header_key=dict_global)
	ut.make_fits({'X':x[idxh],'Y':y[idxh],'Z':z[idxh],'l':lh,'b':bh,'distance':disth,'g':gh,'Mg':Mghl,'ra':cir.ra.degree,'dec':cir.dec.degree,'idh':np.ones_like(lh),'idd':np.zeros_like(lh)},outname=outdir+'/'+name+'_masked.fits',header_key=dict_masked)

elif args.wd==1:
	x,y,z=cordd.T
	m=np.sqrt(x*x+(y*y/(p*p))+(z*z/(q*q)))
	r=np.sqrt(x*x+(y*y)+(z*z))
	rcyl=np.sqrt(x*x+y*y)
	c=SkyCoord(l=ld*u.degree,b=bd*u.degree,frame='galactic')
	if isinstance(Mgd,float):
		Mgdl=[Mgd,]*len(ld)
		Mgdl_all=[Mgd,]*len(rcyl)
	else:
		Mgdl=agd[idxd]
		Mgdl_all=agd
	cir=c.icrs
	ra=cir.ra.degree
	dec=cir.dec.degree
	ut.make_fits({'X':x,'Y':y,'Z':z,'m':m,'r':r,'rcyl':rcyl,'idh':np.zeros_like(r),'idd':np.ones_like(r),'Mg':Mgdl_all},outname=outdir+'/'+name+'_global.fits',header_key=dict_global)
	ut.make_fits({'X':x[idxd],'Y':y[idxd],'Z':z[idxd],'l':ld,'b':bd,'distance':distd,'g':gd,'Mg':Mgdl,'ra':cir.ra.degree,'dec':cir.dec.degree,'idh':np.zeros_like(ld),'idd':np.ones_like(ld)},outname=outdir+'/'+name+'_masked.fits',header_key=dict_masked)


else:
	#global
	x,y,z=np.vstack([cordh,cordd]).T
	m=np.sqrt(x*x+(y*y/(p*p))+(z*z/(q*q)))
	r=np.sqrt(x*x+(y*y)+(z*z))
	rcyl=np.sqrt(x*x+y*y)
	idh=np.zeros_like(x)
	idd=np.zeros_like(x)
	idh[:len(cordh)]=1
	idd[len(cordh):]=1
	if isinstance(Mgh,float): Mghl_all=[Mgh,]*len(cordh)
	else: Mghl_all=agh
	if isinstance(Mgd,float): Mgdl_all=[Mgd,]*len(cordd)
	else: Mgdl_all=agd
	Mgl_all=np.hstack([Mghl_all,Mgdl_all])
	ut.make_fits({'X':x,'Y':y,'Z':z,'m':m,'r':r,'rcyl':rcyl,'idh':idh,'idd':idd,'Mg':Mgl_all},outname=outdir+'/'+name+'_global.fits',header_key=dict_global)
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
	c=SkyCoord(l=l*u.degree,b=b*u.degree,frame='galactic')
	cir=c.icrs
	ra=cir.ra.degree
	dec=cir.dec.degree
	idhlocal=np.zeros_like(xlocal)
	iddlocal=np.zeros_like(xlocal)
	idhlocal[:npartsh]=1
	iddlocal[npartsh:]=1
	ut.make_fits({'X':xlocal,'Y':ylocal,'Z':zlocal,'l':l,'b':b,'distance':dist,'g':g,'Mg':Mgl,'ra':cir.ra.degree,'dec':cir.dec.degree,'idh':idhlocal,'idd':iddlocal},outname=outdir+'/'+name+'_masked.fits',header_key=dict_masked)

print('Done')








