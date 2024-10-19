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
    rkey = run.r
    print(rkey)