
#!/usr/bin/env python

import os, sys
import copy
import datetime

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.mlab import rms_flat
from matplotlib import cm

from sklearn import linear_model
from sklearn import metrics

import hofTracker_regression as hot

#########################
if __name__=='__main__':

    print_min = 0.00

    iFakeWeight = True

    oset = 0
    censor = False
    rseed = 3
    mnyear = 2013
    mxyear = 2014
    pyear = 2015
    nmin = 2
    ncl = 3
    ninit = 10
    isave = False
    idoPCA = False
    idoLogReg = False
    idoRidge = False
    vbose = 0
    fit_intercept = True
    max_aa = 2
    min_aa = -1
    iLogOdds = False

    theta_deg = 0.0

    iAddPar = True
    nsamp = 20
    nboot = 5
    withReplace = True
    iFitNon = False

    for ia, a in enumerate(sys.argv):
        if a=='-mnyear':
            mnyear = int(sys.argv[ia+1])
        if a=='-mxyear':
            mxyear = int(sys.argv[ia+1])
        if a=='-pyear':
            pyear = int(sys.argv[ia+1])
        if a=='-nmin':
            nmin = int(sys.argv[ia+1])
        if a=='-ncl':
            ncl = int(sys.argv[ia+1])
        if a=='-ninit':
            ninit = int(sys.argv[ia+1])
        if a=='-isave':
            isave = bool(int(sys.argv[ia+1]))
        if a=='-idoPCA':
            idoPCA = bool(int(sys.argv[ia+1]))
        if a=='-idoLogReg':
            idoLogReg = bool(int(sys.argv[ia+1]))
        if a=='-idoRidge':
            idoRidge = bool(int(sys.argv[ia+1]))
        if a=='-fit_intercept':
            fit_intercept = bool(int(sys.argv[ia+1]))
        if a=='-censor':
            censor = bool(int(sys.argv[ia+1]))
        if a=='-vbose':
            vbose = int(sys.argv[ia+1])
        if a=='-rseed':
            rseed = int(sys.argv[ia+1])
        if a=='-theta':
            theta_deg = float(sys.argv[ia+1])
        if a=='-max_aa':
            max_aa = float(sys.argv[ia+1])
        if a=='-min_aa':
            min_aa = float(sys.argv[ia+1])
        if a=='-oset':
            oset = float(sys.argv[ia+1])
        if a=='-print_min':
            print_min = float(sys.argv[ia+1])
        if a=='-iLogOdds':
            iLogOdds = bool(int(sys.argv[ia+1]))
        if a=='-iAddPar':
            iAddPar = bool(int(sys.argv[ia+1]))
        if a=='-iFakeWeight':
            iFakeWeight = bool(int(sys.argv[ia+1]))
        if a=='-withReplace':
            withReplace = bool(int(sys.argv[ia+1]))
        if a=='-iFitNon':
            iFitNon = bool(int(sys.argv[ia+1]))
        if a=='-nsamp':
            nsamp = int(sys.argv[ia+1])
        if a=='-nboot':
            nboot = int(sys.argv[ia+1])

    np.random.seed(rseed)

    ht = hot.hofTracker(vbose=vbose)

    data14 = ht.procFileToData('hofTracker_bd_2014.csv')
    data15 = ht.procFileToData('hofTracker_bd_2015.csv')

    X14, pls14, vts14, aa14 = ht.dataToArray(data14)
    X15, pls15, vts15, aa15 = ht.dataToArray(data15)

    mm14 = np.mean(X14, 1)
    mm15 = np.mean(X15, 1)

    uvts = set(vts15).intersection(set(vts14))
    print uvts
    print len(uvts)
    

    X14, vts14 = ht.filterArray(X14, vts14, uvts)
    X15, vts15 = ht.filterArray(X15, vts15, uvts)

    alphas=[10**i for i in range(-4,4)]
    fitter = linear_model.RidgeCV(alphas=alphas
                                  , fit_intercept=fit_intercept)

    cc14 = np.logical_and(mm14>0, aa14>=min_aa)
    fitter.fit(X14[cc14], aa14[cc14])

    cc15 = np.where(mm15>0)
    pp = fitter.predict(X15[cc15])
    for i, p in enumerate(pp):
        print p, pls15[cc15][i]
    

    xtrn = X14[cc14]
    atrn = aa14[cc14]
    ws = np.floor(1.0/(atrn*(1-atrn)) + 0.5)

# if aa is the truth and mm is the public, then the private (nn) is
# aa = (pp*Np + pn*Nn)/Na
# pn = (Na*aa-pp*Np)/Nn
# mm = pp*Np
# nn = pn*Nn
    ntot = 573.0
    npub = 1.0*len(uvts)
    atrn_non = (ntot*atrn-npub*mm14[cc14])/(ntot-npub)

    print atrn
    print mm14[cc14]
    print atrn_non

    ptrn = pls14[cc14]

    xtst = X15[cc15]
    ptst = pls15[cc15] 

    if iFitNon:
        att = atrn_non[:]
    else:
        att = atrn[:]



    xx, yy, ss, mm, meda, pk = ht.doBootStrap(xtrn, att, ptrn, xtst, ptst
                                              ,nboot = nboot
                                              ,withReplace=withReplace
                                              ,nsamp=nsamp
                                              ,ws=ws
                                              )

    efrac = ss/yy

    if iFitNon:
        for v in [yy, mm, meda]:
            a = (v*(ntot-npub)+mm15[cc15]*npub)/ntot
            v = a[:]

        ss = yy*efrac

    ht.bootStrapPlot(xx, yy, ss, mm, meda, pk)

    print '************************'
    for i in range(len(xx)):
        kk = pk[i].split('_')[0]
        print '%+.3f %+.3f (%+.3f - %+.3f) %s' % (yy[i], meda[i], yy[i]-2*ss[i], yy[i]+2*ss[i], kk)

    d = datetime.datetime.now()
    dstr = d.strftime("%m/%d/%Y %H:%M")
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

#    plt.text(xmax-1, ymax-0.1, '%s' % dstr
#             ,horizontalalignment='right'
#             ,verticalalignment='top'
#             ,fontsize='small'
#             ,fontweight='bold'
#             )

    plt.show()
