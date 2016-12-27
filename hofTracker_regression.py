
#!/usr/bin/env python

import os, sys
import copy
import re
import datetime
import csv
import pandas as pd

import numpy as np
import fuzzywuzzy

from matplotlib import pyplot as plt
from matplotlib.mlab import rms_flat
from matplotlib import cm

from sklearn import linear_model
from sklearn import metrics


class hofTracker:

    def __init__(self, fit_intercept=True, vbose=0):
        self.makeNameDict()
        self.vbose = vbose
        self.alphas = [10**i for i in range(-4, 4)] 
        self.fitter = linear_model.RidgeCV(alphas=self.alphas, fit_intercept=True)
        self.default_header_path = './data/headers'
        self.default_data_path = './data/csv'

    def locate_date(self, string_array):
        for s in string_array:
            m = re.search('([0-9]{1,2})/([0-9]{1,2})/([0-9]{1,4})', s)
            if m:
                month, day, year = [int(i) for i in m.groups()]
                if year<2000:
                    year +=  2000
                return datetime.date(year, month, day)
        return None

    def get_timestamp(self, date):
        if date is None:
            return 0
        date_reference = datetime.date(1901,1,1)
        return (date - date_reference).days

    def get_merged_data_by_year(self, year):

        return self.merge_header_csv('{}/hofTracker_header_{:d}.txt'.format(self.default_header_path, year),
                                     '{}/hofTracker_rt_trimmed_{:d}.csv'.format(self.default_data_path, year)
                                     )

    def clean_voter_name(self, s):
        s = re.sub('\(.+\)', '', s)
        s = re.sub('\*','',s)
        s = re.sub('\'', '', s)
        s = re.sub('\.', '', s)
        for k, v in self.nameDict.items():
            s = s.replace(k, v)
        return s.lstrip().strip()

    def get_best_fuzzy_name_match(self, query, choices):
        ans = fuzzywuzzy.process.extract(query, choices)
        return ans

    def convert_dict(self, merged_data):
        converted_data = {}
        voters = sorted(merged_data.keys())
        hd = sorted(merged_data[voters[0]].keys())
        converted_data['hd'] = hd
        converted_data['payload'] = {}
        for voter, voter_choices in merged_data.items():
            converted_data['payload'][voter] = \
                [voter_choices[k] for k in hd]
        return converted_data

    def write_merged_csv(self, merged_data, outfile):
        converted_data = self.convert_dict(merged_data)
        voter_keys = sorted(converted_data['payload'].keys())
        # open in binary to avoid windows carriage return issues
        with open(outfile, 'wb') as fh:
            csvw = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC)
            csvw.writerow(converted_data['hd'])
            for v in voter_keys:
                row = [v] + [str(i) for i in converted_data['payload'][v]]
                csvw.writerow(row)

    def merge_header_csv(self, header_path, csv_path):
        merged_data = {}
        hd = [l.strip() for l in open(header_path).readlines()][0]
        players = [s.replace('\"', '') for s in hd.split(',')]
        lines = [l.strip() for l in open(csv_path).readlines()]
        for l in lines:
            st = l.split(',')
            voter_name = self.clean_voter_name(st[0])
            vote_data = st[1:(len(players)+1)]

            def _parse_vote(s):
                return 0 if s == '' else 1
            vote_data = [_parse_vote(s) for s in vote_data]
            merged_data[voter_name] = dict(zip(players, vote_data))
            merged_data[voter_name]['timestamp'] = self.get_timestamp(self.locate_date(st))
        return merged_data

    def make_tidy_csv(self):
        dfs = []
        for yr in range(2009, 2017+1):
            md = self.get_merged_data_by_year(yr)
            df = pd.DataFrame.from_dict(md, orient="index")
            df = df.assign(year=yr, voter=df.index)
            tmp = pd.melt(df, id_vars=["voter", "year", "timestamp"])
            dfs.append(tmp)
        all_data = pd.concat(dfs)
        all_data = all_data.assign(value=np.array(all_data.value, dtype='i4'))
        all_data = all_data.rename(columns={'variable': 'player'})
        return all_data

    def procFileToData(self, ifile, listOfVoters=[]):
        yr = ifile.replace('.csv','').split('_')[-1]
        yr = int(yr)
        lines = [l.strip() for l in open(ifile).readlines()]

        hd = lines[0]
        ks = hd.split(',')

        data = []
        for l in lines[1:]:
            if len(l)<1:
                continue
            if self.vbose>=2:
                print l

            st = l.split(',')
            nam = st[0]
            nam = nam.strip()
            nam = nam.replace('*','')
            nam = nam.replace('\'', '')
            nam = nam.split('(')[0]
            nam = nam.strip()
            nam = nam.replace('.', '')
            for k, v in self.nameDict.items():
                nam = nam.replace(k, v)

            for i, k in enumerate(ks):
                if i==0:
                    continue

                if 'Known' in k:
                    continue
                if len(k)==0:
                    continue
                if k=='298':
                    continue
                if k=='2560':
                    continue
                if 'baseballthinkfactory' in k:
                    continue

                try:
                    v = float(st[i])
                except ValueError:
                    print 'warning ValueError:', 'st=', st
                    v = 0

                tmp = [yr, nam, k, v]

                data.append(tuple(tmp))
        dt = np.dtype([('yr','i4'),('voter','S64'),('player','S64'),('ivote','f4')])
        data = np.array(data, dtype=dt)
        return data


##############################
    def makeNameDict(self):
        rs = {}
        rs['Barry M Bloom'] =  'Barry Bloom'
        rs['Brayant'] = 'Bryant'
        rs['Kurkijan'] = 'Kurkjian'
        rs['Steven M Goldman'] = 'Steven Goldman'
        rs['Richard F Telander'] = 'Rick Telander'
        rs['Randy J Miller'] = 'Randy Miller'
        rs['Andrew Baggerly'] ='Andrew Baggarly'
        rs['Dan Shuaghnessy'] = 'Dan Shaughnessy'
        rs['Dan Shaugnessy'] = 'Dan Shaughnessy'
        rs['David M Wilheim'] = 'David Wilhelm'
        rs['https://twitter.com/NotMrTibbs/status/546387088707301376'] = 'Art Spander'
        rs['https://twittercom/NotMrTibbs/status/546387088707301376'] = 'Art Spander'
        
        rs['Pres. Jose de Jesus Ortiz'] = 'Jose de Jesus Ortiz'
        rs['Pres Jose de Jesus Ortiz'] = 'Jose de Jesus Ortiz'
        
        rs['http://bit.ly/1CJv758'] = 'C Trent Rosencrans'
        rs['http://bitly/1CJv758'] = 'C Trent Rosencrans'

        rs['Steven Geitschier'] = 'Steven Gietschier'
        rs['Dejan Kovavecic'] = 'Dejan Kovacevic'
        rs['Dave Ammenheuser'] = 'David Ammenheuser'
        rs['Bob Keunster'] = 'Bob Kuenster'
        rs['Edward B. Almada'] = 'Eduardo Almada'
        rs['Edward B Almada'] = 'Eduardo Almada'
        rs['Pete Abraham'] = 'Peter Abraham'
        rs['Richard F. Telander'] = 'Rick Telander'
        rs['Dave Lariviere'] = 'David Lariviere'
        rs['Charles R Scoggins Jr'] = 'Chaz Scoggins'
        rs['William Ballou'] = 'Bill Ballou'
        rs['Joseph E Hoppel'] = 'Joe Hoppel'
        rs['Mike Silverman'] = 'Michael Silverman'
        rs['Chris De Luca'] = 'Chris DeLuca'
        rs['William Center'] = 'Bill Center'
        rs['La Velle E Neal III'] = 'La Velle Neal'
        rs['Jon Haakenson'] = 'Joe Haakenson'
        rs['Terrence Moore'] = 'Terence Moore'
        rs['Mark Topkin'] = 'Marc Topkin'
        rs['Evan P Grant'] = 'Evan Grant'
        rs['Mike Gonzales'] = 'Mark Gonzales'
        self.nameDict = copy.copy(rs)

#########################
    def dataToArray(self, data, vbose=0):
        dact = {}
        adata = {}
        for d in data:
            vt = d['voter']
            pl = d['player']
            iv = d['ivote']
            yr = d['yr']
        
            print 'vt', vt, len(vt), d

            if 'actual50' in vt:
                continue

            k = '%s_%d' % (pl, yr)

            if vt=='actual':
                dact[k] = iv/100.0
                continue

            if not vt in adata:
                adata[vt] = {}
            
            adata[vt][k] = iv

        vts = adata.keys()
        vts.sort()
        try:
            print 'vts', vts
            pls = adata[vts[0]].keys()
        except IndexError:
            print 'IndexError'
            print 'mnyear', mnyear
            print 'mxyear', mxyear
            print 'pyear', pyear
            print 'vts', vts
            sys.exit()
        pls.sort()
    
        if vbose>=1:
            for vt in vts:
                print vt
        
            for pl in pls:
                print pl

        nvt = len(vts)
        npl = len(pls)
        print npl, 'rows', nvt, 'cols'
        X = np.ones((npl, nvt))
        aa = np.ones(npl)

#        print 'dact', dact
        for i in range(npl):
            pl = pls[i]

            if pl in dact:
                aa[i] = dact[pl]
            else:
                aa[i] = None
            for j in range(nvt):
                vt = vts[j]
                iv = adata[vt][pl]
                X[i][j] = iv

                if vbose>=1:
                    print i, pl, j, vt, iv
    
        if np.isnan(np.sum(aa)):
            aa = None
        return X, np.array(pls), np.array(vts), aa

    def filterArray(self, data, vts, com=None):
        tmp = []
        nvts = []

        for i, v in enumerate(np.transpose(data)):
            vt = vts[i]
            print i, vt, vt in com
            if vts[i] in com:
                tmp.append(v)
                nvts.append(vts[i])
        return np.transpose(np.array(tmp)), np.array(nvts)

#########################
    def getActual(self, data):
        ans = {}
        cc = np.where(data['voter']==actual)
        tmp = data[cc]
        pls = np.unique(tmp['player'])
        for pl in pls:
            cc = np.logical_and(tmp['player']==pl, True)
            ans[k] = np.mean(tmp[cc]['ivote'])
        return ans[k]
    
#########################
    def doRidgeFit(self, X, aa
                   ,ws=None
                   ,iFakeWeight=True
                   ,fit_intercept=True
                   ,min_aa=0.00
                   ,max_aa=1.00
                   ,iAddPar=True
                   ,print_min=0.00
                   ,vbose=0):


        tmp = []
        taa = []

        for iv, v in enumerate(X):
            p = aa[iv]        
#        p = aa[cc][iv] + mm[cc][iv]

            if not ws is None:
                n2do = int(ws[iv])
            elif p<=0 or p>=1:
                n2do = 1
            else:
                n2do = int(np.floor(1.0/(p*(1-p))+0.5))

            if self.vbose>=1:
                print 'p n2do', '%.2f' % p, n2do

            
            for i in range(n2do):
                tmp.append(v.tolist())
                dp = np.random.randn()*0.005
                newp = p+dp
                if newp>1:
                    newp = 1
                elif newp<0:
                    newp = 0
                taa.append(newp)

        tmp = np.array(tmp)
        taa = np.array(taa)

        self.fitter.fit(tmp, taa)

#########################
    def doBootStrap(self
                    ,Xtrain
                    ,aatrain
                    ,plstrain
                    ,Xtest
                    ,plstest
                    ,nboot=10
                    ,nsamp=10000
                    ,withReplace=True
                    ,ws=None
                    ):

        aa = copy.deepcopy(aatrain)
        oX = copy.deepcopy(Xtrain)
        pls = copy.deepcopy(plstrain)
        oxtest = copy.deepcopy(Xtest)

        aa = np.transpose(np.array(aa))
        xshape = np.shape(oX)
        print 'xshape', xshape, 'aashape', np.shape(aa)

        nvt = np.shape(oX)[1]

        frms = []
        ans = {}
        
        for i in range(nboot):
            if withReplace:
                if nsamp>nvt:
                    nsamp = nvt
                cc = np.random.random_integers(low=0, high=nvt-1, size=nsamp)
            else:
                cc = np.random.permutation(range(nvt))[0:nsamp]    

            cc.sort()
            if self.vbose>=1:
                print 'cc', cc

            X = np.transpose(oX)
            X = X[cc]
            X = np.transpose(X)

            print 'xshape fit', np.shape(X), 'aashape', np.shape(aa)
            self.doRidgeFit(X, aa
                            ,fit_intercept=True
                            ,min_aa=0.00
                            ,max_aa=1.00
                            ,iAddPar=True
                            ,print_min=0.00
                            ,vbose=0
                            ,ws=ws
                            )

            val = self.fitter.predict(X)
            fit_rms = rms_flat(val-aa)
            frms.append(fit_rms)

            pls = plstest[:]

            Xtest = np.transpose(oxtest)
            Xtest = Xtest[cc]
            Xtest = np.transpose(Xtest)

            print 'xtest shape', np.shape(Xtest)

            val = self.fitter.predict(Xtest)

            for i, v in enumerate(val):
                k = pls[i]
                print i, v, pls[i]
                if not k in ans:
                    ans[k] = []
                ans[k].append(v)

        ks = ans.keys()
        ks.sort()
        frms = np.array(frms)
        ww = 1.0/frms**2
        xx= []
        yy = []
        ss = []
        mm = []
        xs = []
        meda = []

        pos = {}
        pk = []

        tmp = []
        for k in ks:
            ans[k] = np.array(ans[k])
            tt = np.mean(ans[k])
            tmp.append([tt, k])
        tmp.sort(reverse=True)

        ks = []
        for t in tmp:
            ks.append(t[1])

        for i, k in enumerate(ks):

#            print k, ans[k], np.mean(ans[k]), np.std(ans[k])
            print '%+.3f %+.3f %+.3f %s' % (np.mean(ans[k]), np.std(ans[k]), np.sum(ans[k]*ww)/np.sum(ww), k)

            umed = np.median(ans[k])
            umean = np.mean(ans[k])
            ustd  = np.std(ans[k])
            wmean = np.sum(ans[k]*ww)/np.sum(ww)

            pk.append(k)
            xx.append(i)
            yy.append(umean)
            ss.append(ustd)
            mm.append(wmean)
            xs.append(k)
            meda.append(umed)
            pos[k] = (i, umean)
 
        print frms

        xx = np.array(xx)
        yy = np.array(yy)
        ss = np.array(ss)
        mm = np.array(mm)
        meda = np.array(meda)        

        return xx, yy, ss, mm, meda, pk

#####################
    def bootStrapPlot(self, xx, yy, ss, mm, meda, pk, act=None):
        plt.clf()
        plt.plot(xx, yy, 'k.')
        print len(xx), len(yy), len(ss)
        plt.errorbar(xx, yy, ss, ss*0.0, fmt='k.')
        plt.plot(xx, mm, 'r.')
        plt.plot(xx, meda, 'y.')
        ax = plt.gca()
        plt.xticks(xx)
#        ax.set_xticklabels(xs, fontsize='xx-small', rotation=45)
        ax.set_xticklabels([])

        for i, k in enumerate(pk):
            cx = xx[i]
            cy = yy[i]
            print k, cx, cy
            kk = pk[i].split('_')[0]
            if cy>0.3:
                ccy = cy-0.23*cy
                va = 'top'
            else:
                ccy= cy+0.5*cy
                va = 'bottom'
            plt.text(cx+0.25, ccy, kk, fontsize='xx-small', rotation=90, horizontalalignment='center', verticalalignment=va, alpha=0.85)

            if not act is None:                
                plt.plot(xx[i]-0.2, act[pk[i]], 'b-')

        xmin, xmax = plt.xlim()
        dx = xmax-xmin
        plt.xlim(xmin-0.085*dx, xmax+0.085*dx)
        plt.axhline(0.0, color='k', linestyle='--')
        plt.axhline(1.0, color='k', linestyle='--')
        plt.axhline(0.75, color='b', linestyle='-', linewidth=1, alpha=0.2)
        plt.axhline(0.05, color='r', linestyle='-', linewidth=1, alpha=0.2)

        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()

        plt.xlim(xmin-1, xmax+1)
        plt.ylim(-0.05, 1.05)
        
#        plt.show()


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
    nridge = 1
    withReplace = True

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
        if a=='-nsamp':
            nsamp = int(sys.argv[ia+1])
        if a=='-nridge':
            nridge = int(sys.argv[ia+1])

    np.random.seed(rseed)

    

