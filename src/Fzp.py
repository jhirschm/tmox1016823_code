import numpy as np
import typing
from typing import List
import h5py
import sys
import math

def getCentroid(data,pct=.8):
    csum = np.cumsum(data.astype(float))
    s = float(csum[-1])*pct
    csum /= csum[-1]
    inds = np.where((csum>(.5-pct/2.))*(csum<(.5+pct/2.)))
    tmp = np.zeros(data.shape,dtype=float)
    tmp[inds] = data[inds].astype(float)
    num = np.sum(tmp*np.arange(data.shape[0],dtype=float))
    return (num/s,np.uint64(s))


class Fzp:
    def __init__(self,thresh) -> None:
        self.v = []
        self.vsize = int(0)
        self.vc = []
        self.vs = []
        self.initState = True
        self.fzpthresh = thresh
        self.winstart = 0
        self.winstop = 1<<11
        return

    @classmethod
    def update_h5(cls,f,spect,fzpEvents, store_data:bool = False):
        grpfzp = None
        if 'fzp' in f.keys():
            grpfzp = f['fzp']
        else:
            grpfzp = f.create_group('fzp')

        grpfzp.create_dataset('centroids',data=spect.vc,dtype=np.float16)
        grpfzp.create_dataset('sum',data=spect.vs,dtype=np.uint64)
        grpfzp.attrs.create('size',data=spect.vsize,dtype=np.int32)
        grpfzp.create_dataset('events',data=fzpEvents)
        if store_data:
            grpfzp.create_dataset('data',data=spect.v,dtype=int)
        return 
    
    def setthresh(self,x):
        self.fzpthresh = x
        return self

    def setwin(self,low,high):
        self.winstart = int(low)
        self.winstop = int(high)
        return self

    def test(self,fzpwv, mean_val_ind=800):
        mean = np.int16(0)
        if type(fzpwv)==type(None):
            return False
        try:
            mean = np.int16(np.mean(fzpwv[mean_val_ind:])) # this subtracts baseline
        except:
            print('Damnit, Fzp!')
            return False
        else:
            if (np.max(fzpwv)-mean)<self.fzpthresh:
                #print('weak Vls!')
                return False
        return True
    
    def process(self, fzpwv, mean_val_ind=800):
        mean = np.int16(np.mean(fzpwv[mean_val_ind:])) # this subtracts baseline
        if (np.max(fzpwv)-mean)<self.fzpthresh:
            return False
        d = np.copy(fzpwv-mean*np.ones(len(fzpwv))).astype(np.int16)
        c,s = getCentroid(d[self.winstart:self.winstop],pct=0.8)
        if self.initState:
            self.v = [d]
            self.vsize = len(self.v)
            self.vc = [np.float16(c)]
            self.vs = [np.uint64(s)]
            self.initState = False
        else:
            self.v += [d]
            self.vc += [np.float16(c)]
            self.vs += [np.uint64(s)]
        return True
    
    def set_initState(self,state: bool):
        self.initState = state
        return self

    def print_v(self):
        print(self.v[:10])
        return self

    