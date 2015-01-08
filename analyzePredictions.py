#!/usr/bin/env python

import os, sys
import csv

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.mlab import rms_flat

##########################
def procData(ifile='predictions.csv'):
    ofp = open(ifile)
    csvr = csv.reader(ofp)
    hd = csvr.next()
    ans = {}
    for row in csvr:
        print zip(hd, row)
        ans[row[0]] = np.array(zip(hd,row)[1:], dtype=np.dtype([('name','S64'),('sc','f8')]))
        
    return ans
##########################
if __name__=='__main__':
    ans = procData()
    pls = None

    vts = ans.keys()
    vts.remove('actual')

    truth = ans['actual']
    votes = {}

    cols = ['k','b','r','y','g','c']

    xxs = {}
    yys = {}
    rmi = {}

    pps = {}

    pl_str = {}
    for k in ans:
        yys[k] = []
        xxs[k] = []
        rmi[k] = []

        if k=='actual':
            continue

        ivt = vts.index(k)

        for i, v in enumerate(ans[k]):
            xx = i*2
            print k, i, ivt, v, v['sc'], truth[i], truth[i]['sc']
            r = v['sc']-truth[i]['sc']
            plt.plot(xx+0.15*ivt, r, 's', color=cols[ivt], markersize=4)
            xxs[k].append(xx)
            yys[k].append(r)
            rmi[k].append(r)


    plt.axhline(0,color='k', linestyle='--')
    xmin, xmax = plt.xlim()
    dx = xmax-xmin
    plt.xlim(xmin-0.1*dx, xmax+0.1*dx)
    xmin, xmax = plt.xlim()
    dx = xmax-xmin

    ymin, ymax = plt.ylim()
    dy = ymax-ymin
    y0 = max([abs(ymin), abs(ymax)])
    plt.ylim(-y0-0.1*dy, y0+0.1*dy)

    ymin, ymax = plt.ylim()
    dy = ymax-ymin

    cx = xmax-0.05*dx
    cy = ymax

    k = 'BD'
    for i, v in enumerate(ans[k]):
        pstr = '%s (%.1f)' % (v['name'], ans['actual'][i]['sc'])
        plt.text(xxs[k][i], ymin+0.02*dy, pstr
                 , fontsize='x-small'
                 , rotation='vertical'
                 , verticalalignment='bottom') 

    for ivt, vt in enumerate(vts):
#        plt.plot(xxs[vt], yys[vt], '-%s' % cols[ivt], drawstyle='steps-mid')
        plt.plot(xxs[vt], yys[vt], '-%s' % cols[ivt], alpha=0.3)
        r = rms_flat(rmi[vt])

        cy -= 0.035*dy
        plt.text(cx, cy, '%s RMS: %.1f%c' % (vt, r, '%')
                 , fontsize='small'
                 ,color=cols[ivt]
                 ,horizontalalignment='right'
                 ,verticalalignment='top'
                 )

    ax = plt.gca()
    ax.set_xticklabels([])
    plt.ylabel('[prediction - actual] (%)')
    plt.savefig('hofTracker_compare_01.png')
    plt.show()
