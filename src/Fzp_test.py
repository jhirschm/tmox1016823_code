import psana
import numpy as np
import sys
import re
import h5py
import os
import socket

from typing import Type,List

from Fzp import *

print("Hello from Fzp_test.py")

runfzp=True
fzps = []
runnums = [22]
expname = 'tmoc00123'

port = {}
chankeys = {}
hsds = {}
hsdstring = {}
ds = psana.DataSource(exp=expname,run=runnums)
detslist = {}
hsdnames = {}

for r in runnums:
    run = next(ds.runs())
    rkey = run.runnum
   
    port.update({rkey:{}})
    hsds.update({rkey:{}})
    chankeys.update({rkey:{}})
    detslist.update({rkey:[s for s in run.detnames]})

    print(port)
    print(chankeys)
    print(hsds)
    print(detslist)