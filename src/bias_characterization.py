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






channels = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
labels = list(range(17))  # Labeled as 0 through 16

runs_list = [14, 15, 16, 17, 5,6,7,8,9,10,11,13]
mcp_bias = [1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1800]


channel_max_values = np.zeros((len(runs_list),len(channels)))

for i, run_num in enumerate(runs_list):
    ds = psana.DataSource(exp=exp_name, run=run_num)
    run = next(ds.runs())
    hsd = run.Detector('mrco_hsd')

    # Loop through all events in the run
    for num, evt in enumerate(run.events()):
        try:
            # Try to extract the waveform data
            waveforms = hsd.raw.waveforms(evt)
            if waveforms is not None:
                # If waveforms exist, get the maximum value
                for j, chan in enumerate(channels):
                    max_value = waveforms[chan][0].max()
                    if max_value > channel_max_values[i,j]:
                        channel_max_values[i,j] = max_value
                # print(f"Max value of the waveform: {max_value}")
            
        except:
            # Handle any error (like missing data) and continue to next event
            continue
        if num >= 1000:
            break


print(channel_max_values)

# Initialize a new plot
plt.figure(figsize=(10, 6))

# Iterate through each channel and plot its max values across runs
for j, chan in enumerate(channels):
    plt.plot(mcp_bias, channel_max_values[:, j], label=f'Channel {chan}')

# Set plot labels and title
plt.xlabel('MCP Bias (Voltage)', fontsize=12)
plt.ylabel('Max Value', fontsize=12)
plt.title('Max Waveform Values Across Runs for Each Channel', fontsize=14)

# Add a legend to differentiate the channels
plt.legend(loc='best', fontsize=10)

# Show the grid
plt.grid(True)

# Display the plot
plt.show()