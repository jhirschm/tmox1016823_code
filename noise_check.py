import psana
import matplotlib.pyplot as plt
import numpy as np


ds = psana.DataSource(exp='tmoc00123',run=22)

run = next(ds.runs())
ch = 22
evt = next(run.events())
evt = next(run.events())
hsd = next(run.Detector('mrco_hsd'))
y = hsd.raw.peaks(evt)[ch][0][1][0]
w =  hsd.raw.waveforms(evt)[ch][0][1][0]

plt.figure()
plt.plot(w*(1<<3),label="wave")
plt.plot(y.astype(float)/2+float(1<<13),label="peak")
plt.legend()
plt.show()
