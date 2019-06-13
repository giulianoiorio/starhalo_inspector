#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False


import numpy as np
from scipy.optimize import  newton
from libc.math cimport sqrt, exp, fabs, tanh, cosh


#Q exp########### #TESTATO FUNZIONA FIDATI!
cdef double qfunc(double m,double q0, double qinf, double rq):

    return qinf - (qinf-q0)*exp(1-sqrt(m*m+rq*rq)/rq)


cdef double qfunc_der(double m,double q0, double qinf, double rq):

    cdef double q=qfunc(m,q0,qinf,rq)
    cdef double num, den

    num=m*exp(1-sqrt(m*m+rq*rq)/rq)*(qinf-q0)
    den=rq*sqrt(rq*rq+m*m)

    return num/den


cdef double zerofunc(double m, double x, double y, double z,double q0, double qinf, double rq, double p):

    cdef double q=qfunc(m,q0,qinf,rq)

    return m*m - x*x - y*y/(p*p) - (z*z)/(q*q)

cdef double zerofunc_der(double m, double x, double y, double z,double q0, double qinf, double rq):

    cdef double q=qfunc(m,q0,qinf,rq), qder=qfunc_der(m,q0,qinf,rq)

    return 2*(m+(z*z/(q*q*q))*qder)

cdef double newton_raphson(double m0, double x, double y, double z,double q0, double qinf, double rq, double p, double toll, int maxiter):

    cdef int i
    cdef double h, mnew=0, mold=m0

    i=0
    while i<=maxiter:
        h=zerofunc(mold,x,y,z,q0,qinf,rq,p)/zerofunc_der(mold,x,y,z,q0,qinf,rq)
        mnew=mold-h
        if fabs(h) < toll:
            return mnew
        mold=mnew
        i+=1

    return mnew

#######################


####Qtan #TESTATO FUNZIONA FIDATI!
cdef double qfunc_tan(double m,double q0, double qinf, double rq, double eta):

    cdef double tm,t0,C

    tm=tanh((m-rq)/eta)
    t0=tanh(-rq/eta)
    C=(qinf-q0)/(1-t0)

    return qinf+C*(tm-1)

cdef double qfunc_tan_der(double m,double q0, double qinf, double rq, double eta):

    cdef double cm,t0,C,xx

    xx=((m-rq)/eta)
    cm=cosh( xx*xx )
    t0=tanh(-rq/eta)
    C=(qinf-q0)/(1-t0)

    return C/(eta*cm)


cdef double zerofunc_tan(double m, double x, double y, double z,double q0, double qinf, double rq, double p, double eta):

    cdef double q=qfunc_tan(m,q0,qinf,rq,eta)

    return m*m - x*x - y*y/(p*p) - (z*z)/(q*q)

cdef double zerofunc_der_tan(double m, double x, double y, double z,double q0, double qinf, double rq, double eta):

    cdef double q=qfunc_tan(m,q0,qinf,rq,eta), qder=qfunc_tan_der(m,q0,qinf,rq,eta)

    return 2*(m+(z*z/(q*q*q))*qder)

cdef double newton_raphson_tan(double m0, double x, double y, double z,double q0, double qinf, double rq, double p, double eta, double toll, int maxiter):

    cdef int i
    cdef double h, mnew=0, mold=m0

    i=0
    while i<=maxiter:
        h=zerofunc_tan(mold,x,y,z,q0,qinf,rq,p,eta)/zerofunc_der_tan(mold,x,y,z,q0,qinf,rq,eta)
        mnew=mold-h
        if fabs(h) < toll:
            return mnew
        mold=mnew
        i+=1

    return mnew


cdef double newton_raphson_tan2(double m0, double x, double y, double z,double q0, double qinf, double rq, double p, double eta, double toll, int maxiter):

    cdef int i
    cdef double h, mnew=0, mold=m0
    cdef  double cm,tm,t0,C,xx, cyl_rad
    cdef double q, qder, zfunc, zfunc_der, q2, z2

    t0=tanh(-rq/eta)
    C=(qinf-q0)/(1-t0)
    cyl_rad=+x*x+y*y/(p*p)
    z2=z*z

    i=0
    while i<=maxiter:

        xx=((mold-rq)/eta)
        tm=tanh(xx)
        cm=cosh( xx*xx )

        q=qinf+C*(tm-1)
        q2=q*q
        qder=C/(eta*cm)

        zfunc=mold*mold - cyl_rad - z2/(q2)
        zfunc_der=2*(mold+(z2/(q2*q))*qder)


        h=zfunc/zfunc_der
        mnew=mold-h
        if fabs(h) < toll:
            return mnew
        mold=mnew
        i+=1




#############

#test funcs

def newton(double m0, double x, double y, double z, double q0, double qinf, double rq, double p, eta=None, toll=1e-8,maxiter=30):

    cdef double ceta

    if eta is None:

        return newton_raphson(m0,x,y,z,q0,qinf,rq,p,toll,maxiter)

    else:

        ceta=eta

        return newton_raphson_tan(m0,x,y,z,q0,qinf,rq,p,ceta,toll,maxiter)

def nr(double m, double x, double y, double z,double q0, double qinf, double rq, double p, eta=None):

    cdef double ceta=eta

    if eta is None:

        return zerofunc(m,  x, y, z, q0, qinf,  rq, p)

    else:

        ceta=eta

        return zerofunc_tan(m,  x, y, z, q0, qinf,  rq, p, ceta)
######




def calc_m(double[:] x, double[:] y, double[:] z, double q0, double qinf, double rq, double p = 1, eta=None, double toll=1e-4,int maxiter=30):

    cdef unsigned int i, n=len(x)
    cdef double mmt,xx,yy,zz,r, ceta
    cdef double[:] res=np.empty(n,dtype=np.float64)

    if q0>qinf: raise ValueError('Calc_m function: qinf<q not allowed')

    if eta is None:

        for i in range(n):


            xx=x[i]
            yy=y[i]
            zz=z[i]

            r=sqrt(xx*xx+yy*yy+zz*zz)
            res[i]=newton_raphson(r,xx,yy,zz,q0,qinf,rq,p,toll,maxiter)

    else:

        ceta=eta

        for i in range(n):


            xx=x[i]
            yy=y[i]
            zz=z[i]

            r=sqrt(xx*xx+yy*yy+zz*zz)
            res[i]=newton_raphson_tan(r,xx,yy,zz,q0,qinf,rq,p,ceta,toll,maxiter)

    return res

def calc_m2(double[:] x, double[:] y, double[:] z, double q0, double qinf, double rq, double p=1, eta=None, double toll=1e-4,int maxiter=30):

    cdef unsigned int i, n=len(x)
    cdef double mmt,xx,yy,zz,r, ceta
    cdef double[:] res=np.empty(n,dtype=np.float64)

    if q0>qinf: raise ValueError('Calc_m function: qinf<q not allowed')

    if eta is None:

        for i in range(n):


            xx=x[i]
            yy=y[i]
            zz=z[i]

            r=sqrt(xx*xx+yy*yy+zz*zz)
            res[i]=newton_raphson(r,xx,yy,zz,q0,qinf,rq,p,toll,maxiter)

    else:

        ceta=eta

        for i in range(n):


            xx=x[i]
            yy=y[i]
            zz=z[i]

            r=sqrt(xx*xx+yy*yy+zz*zz)
            res[i]=newton_raphson_tan2(r,xx,yy,zz,q0,qinf,rq,p,ceta,toll,maxiter)

    return res