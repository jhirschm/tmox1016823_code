import psana
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Set up argument parser to take show_fourier as a command-line argument
parser = argparse.ArgumentParser(description='Experiment.')
parser.add_argument('--exp-name', type=str, help='Name of the experiment.')

# Parse the arguments
args = parser.parse_args()

# Access the experiment name
exp_name = args.exp_name




evt = next(run.events())
evt = next(run.events())

hsd = run.Detector('mrco_hsd')

channels = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
labels = list(range(17))  # Labeled as 0 through 16


runs_list = [6,7,8,9,10,11,13]
mcp_bias = [1450,1500,1550,1600,1650,1700,1800]


channel_max_values = np.zeros((len(runs_list),len(channels)))

for i, run_num in enumerate(runs_list):
    ds = psana.DataSource(exp=exp_name,run=run_num)

    run = next(ds.runs())
    events_list = list(run.events())
    print(events_list)
    # evt = next(run.events())
    # while True:
    #     try:
    #         w_event = hsd.raw.waveforms(evt)   # Extract the waveform data
    #     except:
    #         evt = next(run.events())           # Try the next event if the current one fails
    #     else:
    #         break
    