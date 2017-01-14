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


def data_fro_comparison_models():
    for i in range(len(xx)):
        a = adict[pk[i]]
        kk = pk[i].split('_')[0]

        cc = np.logical_and(data15['voter'] == 'tango_model', data15['player'] == kk)
        tm = data15[cc]['ivote'][0] / 100.0

        #        tm = 0.0
        tmr['yy'].append(tm)
        tmr['ss'].append(0.01)
        tmr['rr'].append(tm - a)

        cc = np.logical_and(data15['voter'] == 'baseballot', data15['player'] == kk)
        tm = data15[cc]['ivote'][0] / 100.0

        bbr['yy'].append(tm)
        bbr['rr'].append(tm - a)
        bbr['ss'].append(0.01)

    tmr['yy'] = np.array(tmr['yy'])
    tmr['ss'] = np.array(tmr['ss'])
    tmr['rr'] = np.array(tmr['rr'])
    tmr['rms'] = rms_flat(tmr['rr'])

    bbr['yy'] = np.array(bbr['yy'])
    bbr['ss'] = np.array(bbr['ss'])
    bbr['rr'] = np.array(bbr['rr'])
    bbr['rms'] = rms_flat(bbr['rr'])


#########################
if __name__=='__main__':

    print_min = 0.00

    iCheckTangoModel = False
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
    nsamp = 1000
    nboot = 5
    withReplace = True
    iFitNon = True

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
        if a=='-iCheckTangoModel':
            iCheckTangoModel = bool(int(sys.argv[ia+1]))
        if a=='-nsamp':
            nsamp = int(sys.argv[ia+1])
        if a=='-nboot':
            nboot = int(sys.argv[ia+1])

    np.random.seed(rseed)

    ht = hot.hofTracker(vbose=vbose)

    data14 = ht.procFileToData('./data/csv/hofTracker_bd_2016.csv')
    data15 = ht.procFileToData('./data/csv/hofTracker_bd_2017.csv')

    X14, pls14, vts14, aa14 = ht.dataToArray(data14)
    X15, pls15, vts15, aa15 = ht.dataToArray(data15)

    mm14 = np.mean(X14, 1)
    mm15 = np.mean(X15, 1)

    uvts = set(vts15).intersection(set(vts14))
    uvts.discard('StraightArrow')
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
    if iFakeWeight:
        ws = np.floor(1.0/(atrn*(1-atrn)) + 0.5)
    else:
        ws = np.ones(len(atrn))

# if aa is the truth and mm is the public, then the private (nn) is
# aa = (pp*Np + pn*Nn)/Na
# pn = (Na*aa-pp*Np)/Nn
# mm = pp*Np
# nn = pn*Nn
    ntot = 450.0
    npub = 1.0*len(uvts)
    npriv = ntot-npub
    atrn_non = (ntot*atrn-npub*mm14[cc14])/npriv

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


#    checkWeights(ht, xtrn, xtst, att, vts15, pls14[cc14], pls15[cc15], ws=ws)
#    sys.exit()

    ans = ht.doBootStrap(xtrn, att, ptrn, xtst, ptst
                         ,nboot = nboot
                         ,withReplace=withReplace
                         ,nsamp=nsamp
                         ,ws=ws
                         )

    xx, yy, ss, mm, meda, pk, inprob, outprob, amat = ans

    mat = []
    for k in pk:
        mat.append(amat[k])
    mat = np.array(mat)
    xx = np.array(xx)*3

    mean_public = []
    for pl in pk:
        idx = np.where(pls15 == pl)[0][0]
        mean_public.append(mm15[idx])
    mean_public = np.array(mean_public)

    if iFitNon:
        newa = (mat * (ntot - npub) + (mean_public * npub).reshape((-1, 1))) / ntot
        yy = np.mean(newa, 1)
        meda = np.median(newa, 1)
        ss = np.std(newa, 1)
        inprob = np.sum(newa >= 0.75, 1)/(1.0 * newa.shape[1])
        outprob = np.sum(newa < 0.05, 1)/(1.0 * newa.shape[1])

    warnings = ''
    print (' {:5s} | {:s} | '
           '{:s} | {:s} | {:s}'.format('mean%'
                                       ,'95% bootstrap '
                                       ,'priv. - public % '
                                       ,'res. - public % '
                                       ,'player')
           )

    idx = range(len(pk))
    tmp = idx[4]
    idx[4] = idx[3]
    idx[3] = tmp

    for i in idx:
        pl = pk[i]
        ymin = yy[i] - 2 * ss[i]
        ymax = yy[i] + 2 * ss[i]

        # hack hack hack!
        if ymin<0:
            warnings += 'warning; {} has ymin less than 0\n'.format(pl)
            ymin = 0
        diff_respublic = 100 * (yy[i] - mean_public[i])
        diff_privpublic = 100 * (mat.mean(1)[i] - mean_public[i])
        print (' {:4.1f}   ({:4.1f} - {:4.1f}) '
               '          {:+5.1f} '
               '          {:+5.1f} '
               '         {}'.format(100*yy[i],
                                    100*ymin,
                                    100*ymax,
                                    diff_privpublic,
                                    diff_respublic,
                                    pl.split('_')[0]))

    if len(warnings) > 0:
        print '\n\n', warnings


    if aa15 is not None:
        adict = {}
        for i, p in enumerate(pls15):
            adict[p] = aa15[i]
    else:
        adict = None

    tmr = {}
    tmr['yy'] = []
    tmr['ss'] = []
    tmr['rr'] = []

    bbr = {}
    bbr['yy'] = []
    bbr['ss'] = []
    bbr['rr'] = []


    if aa15 is not None:
        adict = {}
        for i, p in enumerate(pls15):
            adict[p] = aa15[i]
    else:
        adict = None

    tmr = {}
    tmr['yy'] = []
    tmr['ss'] = []
    tmr['rr'] = []

    bbr = {}
    bbr['yy'] = []
    bbr['ss'] = []
    bbr['rr'] = []

    ht.bootStrapPlot(xx, yy, ss, mm, meda, pk, act=adict)


    for i, v in enumerate(ht.fitter.coef_):
        print i, vts14[i],  v

    print '************************'
    for i in range(len(xx)):
        kk = pk[i].split('_')[0]
        print '%+.3f %+.3f %+.3f (%+.3f - %+.3f) %s' % (yy[i], meda[i], ss[i], yy[i]-2*ss[i], yy[i]+2*ss[i], kk)

    iPrintActual = True

    if iPrintActual:

        print '************************'
        chi2 = [0.0, 0.0, 0]
        rr = []
        for i in range(len(xx)):
            a = adict[pk[i]]
            kk = pk[i].split('_')[0]


            if iCheckTangoModel:
                pl = pk[i].split('_')[0]
                cc = np.logical_and(data15['voter']=='tango_model', data15['player']==kk)
                tm = data15[cc]['ivote'][0]/100.0

                yy[i] = tm
                meda[i] = tm


            rr.append(yy[i]-a)

            if iCheckTangoModel:
                sd = 1.0
            else:
                sd = ss[i]


            print '%+.3f %+.3f %+.3f (%+.3f - %+.3f) %+.1f %+.1f %+.1f %s' % (yy[i], meda[i], sd, yy[i]-2*sd, yy[i]+2*sd, 100.0*a, 100*(yy[i]-a), 100*(meda[i]-a), kk)

            plt.plot(xx[i]-0.4, a, 'bx')
            
            chi2[0] += ((a-yy[i])/sd)**2
            chi2[1] += ((a-meda[i])/sd)**2
            chi2[2] += 1
#            print ((a-yy[i])/sd), ((a-meda[i])/sd)
    print 'chi2', '%.1f %.1f %d' % (chi2[0], chi2[1], chi2[2])
    rms = rms_flat(rr)
    print 'rms', '%.3f' % rms 
    d = datetime.datetime.now()
    dstr = d.strftime("%m/%d/%Y %H:%M")
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    iDateText = True
    iDateText = False
    col = 'k'
    if iDateText:
        if iPrintActual>=2:
            dstr = 'Final Results'
            col = 'b'

        plt.text(xmax-1, ymax-0.1, '%s\nN=%4d' % (dstr, len(uvts))
                 ,color=col
                 ,horizontalalignment='right'
                 ,verticalalignment='top'
                 ,fontsize='small'
                 ,fontweight='bold'
                 )


#        

    rmi = [rms, tmr['rms'], bbr['rms']]
    cols = ['y', 'r','g']
    labs = ['BD', 'Tango', 'BsBallot']
        
    xmin, xmax = plt.xlim()
    dx = xmax-xmin
    for i, c in enumerate(cols):
        dstr = '%s: rms = %.3f' % (labs[i], rmi[i])
        plt.text(xmax-0.05*dx, ymax-0.1-0.05*i, '%s' % dstr
                 ,color=cols[i]
                 ,horizontalalignment='right'
                 ,verticalalignment='top'
                 ,fontsize='small'
                 ,fontweight='bold'
                 )

    plist = ['Randy Johnson','Pedro Martinez','John Smoltz','Craig Biggio','Mike Piazza','Jeff Bagwell','Tim Raines','Curt Schilling','Roger Clemens','Barry Bonds','Lee Smith','Edgar Martinez','Alan Trammell','Mike Mussina','Jeff Kent','Fred McGriff','Larry Walker','Gary Sheffield','Mark McGwire','Don Mattingly','Sammy Sosa','Nomar Garciaparra','Carlos Delgado']
    ofp = sys.stdout
    for i in range(len(xx)):
        a = adict[pk[i]]
        kk = plist[i]
#pk[i].split('_')[0]
#        print '%.1f %s' % (yy[i]*100.0, kk)
 #       print '%.1f %s' % (tmr['yy'][i]*100.0, kk)
 
#        ofp.write('%.1f,' % (tmr['yy'][i]*100.0))
        ofp.write('%.1f,' % (bbr['yy'][i]*100.0))
    ofp.write('\n')
    plt.show()
