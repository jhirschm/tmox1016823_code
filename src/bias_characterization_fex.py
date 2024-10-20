import psana
import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools
from scipy.optimize import curve_fit
from matplotlib.widgets import TextBox

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set up argument parser to take show_fourier as a command-line argument
parser = argparse.ArgumentParser(description='Experiment.')
parser.add_argument('--exp-name', type=str, help='Name of the experiment.')
parser.add_argument('--load-path', type=str, default=None, help='Path to load data (default: None).')
parser.add_argument('--save-plot-path', type=str, default=None, help='Path to load data (default: None).')



# Parse the arguments
args = parser.parse_args()


# Access the experiment name
exp_name = args.exp_name
load_path = args.load_path
save_plot_path = args.save_plot_path







channels = [0, 270]#, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
labels = list(range(17))  # Labeled as 0 through 16

runs_list = [17, 16, 15, 14 ,5,6,7,8,9, 10,11, 12,13]
mcp_bias = np.array([1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800])
# ds = psana.DataSource(exp=exp_name, run=7)
# run = next(ds.runs())
# hsd = run.Detector('mrco_hsd')
# evt = next(run.events())
# evt = next(run.events())

# evt = next(run.events())

# ch=0
# _=[plt.plot(np.arange(hsd.raw.peaks(evt)[ch][0][0][i],hsd.raw.peaks(evt)[ch][0][0][i]+len(hsd.raw.peaks(evt)[ch][0][1][i])),hsd.raw.peaks(evt)[ch][0][1][i]//(1<<3)) for i in range(len(hsd.raw.peaks(evt)[ch][0][1]))]
# plt.show()

# # Create a histogram for the fex values  for max peak heights in each window in the fex for an event 
# binvals = np.arange(0,(1<<15)+1,1<<5)
# hist = np.zeros(len(binvals)-1)
# for i, run_num in enumerate(runs_list):
#     ds = psana.DataSource(exp=exp_name, run=run_num)
#     run = next(ds.runs())
#     hsd = run.Detector('mrco_hsd')

#     # Loop through all events in the run
#     for num, evt in enumerate(run.events()):
        
#         peaks = hsd.raw.peaks(evt)
#         for j, chan in enumerate(channels):
#             for k in range(len(hsd.raw.peaks(evt)[ch][0][1])):
#                 max = hsd.raw.peaks(evt)[ch][0][1][k].max()
#                 # Find the appropriate bin index for the single value
#                 bin_index = np.digitize(max, binvals) - 1  # Subtract 1 to convert to zero-based index

#                 # Ensure the bin index is within the valid range
#                 if 0 <= bin_index < len(hist):
#                     hist[bin_index] += 1  # Increment the count in the appropriate bin
#                 else:
#                     print("Value out of range for histogram bins.")


# Create a 3D array for histograms: (number of runs, number of channels, number of bins)
binvals = np.arange(0,(1<<15)+1,1<<5)
binvals = np.arange(0,(1<<15)+1,1<<7)

num_runs = len(runs_list)
num_channels = len(channels)
histograms = np.zeros((num_runs, num_channels, len(binvals) - 1))

for i, run_num in enumerate(runs_list):
    ds = psana.DataSource(exp=exp_name, run=run_num)
    run = next(ds.runs())
    hsd = run.Detector('mrco_hsd')

    # Loop through all events in the run
    for num, evt in enumerate(run.events()):
        peaks = hsd.raw.peaks(evt)

        for j, chan in enumerate(channels):
            # Ensure the channel exists in the peaks
            if chan in peaks:
                for k in range(len(peaks[chan][0][1])):
                    max_value = peaks[chan][0][1][k].max()  # Get the maximum value for the peak
                    
                    # Find the appropriate bin index for the single value
                    bin_index = np.digitize(max_value, binvals) - 1  # Subtract 1 to convert to zero-based index

                    # Ensure the bin index is within the valid range
                    if 0 <= bin_index < len(histograms[i, j]):
                        histograms[i, j, bin_index] += 1  # Increment the count in the appropriate bin
                    else:
                        print(f"Value out of range for histogram bins for run {run_num}, channel {chan}: {max_value}")
        if num >= 100:
            break
np.save("histograms2.npy", histograms)
np.save("binvals2.npy", binvals)
# At this point, `histograms` contains the 3D histogram counts
# You can access them like this:
for run_idx in range(num_runs):
    for chan_idx in range(num_channels):
        print(f"Run {runs_list[run_idx]}, Channel {channels[chan_idx]} Histogram Counts: {histograms[run_idx, chan_idx]}")


fig, axs = plt.subplots(nrows=len(mcp_bias), ncols=2, figsize=(10, 18), sharex=True)
for (j,chan) in enumerate(channels):
    for i, (bias) in enumerate(zip(mcp_bias)):
        bar = axs[i].bar(binvals[:-1],histograms[i,j,:], width=np.diff(binvals), align='edge', edgecolor='black', alpha=0.7)
        axs[i].set_xticks([])
plt.show()
# # Loop over each channel and create a 2D histogram
# for j, chan in enumerate(channels):
#     plt.subplot(4, 4, j + 1)  # Use a 4x4 grid for 16 channels
    
#     # Extract counts for the current channel across runs
#     counts = histograms[:, j, :]  # Shape: (number of runs, number of bins)
    
#     # Create the 2D histogram
#     plt.hist2d(mcp_bias.repeat(counts.shape[1]), 
#                np.tile(np.arange(counts.shape[1]), counts.shape[0]), 
#                weights=counts.flatten(), 
#                bins=[binvals, len(mcp_bias)], 
#                cmap='viridis', 
#                cmin=0)  # Set cmin to ensure counts below a threshold aren't plotted

#     plt.title(f'Channel {chan} 2D Histogram')
#     plt.xlabel('Bin Values')
#     plt.ylabel('MCP Bias Voltage')
#     plt.colorbar(label='Counts')
#     # Set x ticks to show only the first and last bin values
#     plt.xticks([binvals[0], binvals[-1]], [f'{binvals[0]}', f'{binvals[-1]}'])  # Label first and last bin

# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()
# if load_path is not None:
#     channel_max_values = np.load(load_path)
#     print(f"NumPy array loaded from {load_path}")

# else:   
#     channel_max_values = np.zeros((len(runs_list),len(channels)))

#     for i, run_num in enumerate(runs_list):
#         ds = psana.DataSource(exp=exp_name, run=run_num)
#         run = next(ds.runs())
#         hsd = run.Detector('mrco_hsd')

#         # Loop through all events in the run
#         for num, evt in enumerate(run.events()):
            
#             peaks = hsd.raw.peaks(evt)
            
#             for j, chan in enumerate(channels):
#                 max_value = peaks[chan][0].max()
#                 # max_value_past_threshold
#                 if max_value > channel_max_values[i,j]:
#                     channel_max_values[i,j] = max_value
#                 # print(f"Max value of the waveform: {max_value}")
                
#             if num >= 10000:
#                 break


# print(channel_max_values)
# if load_path is None:   
#     # Define file paths for saving the array and plot
#     numpy_file_path = "channel_max_values.npy"  # Save as .npy
#     plot_file_path = "channel_max_values_plot.png"  # Save as .png

#     # Save the NumPy array
#     np.save(numpy_file_path, channel_max_values)
#     print(f"NumPy array saved to {numpy_file_path}")

# # Define a list of colors and line styles
# colors = plt.cm.get_cmap('tab10', len(channels)).colors  # Use tab10 colormap for distinct colors
# line_styles = ['-', '--']  # Solid and dashed lines

# # Initialize the plot
# plt.figure(figsize=(10, 6))

# # Cycle through line styles and colors
# line_style_cycle = itertools.cycle(line_styles)

# # Plot each channel's max values with different colors and alternating line styles
# for j, chan in enumerate(channels):
#     color = colors[j % len(colors)]  # Cycle through the colors
#     line_style = next(line_style_cycle)  # Alternate between solid and dashed lines
#     plt.plot(mcp_bias, channel_max_values[:, j], label=f'Channel {chan}', color=color, linestyle=line_style)

# # Set plot labels and title
# plt.xlabel('MCP Bias (Voltage)', fontsize=12)
# plt.ylabel('Max Value', fontsize=12)
# plt.title('Max Waveform Values Across Runs for Each Channel', fontsize=14)

# # Add a legend to differentiate the channels
# plt.legend(loc='best', fontsize=10)

# # Show the grid
# plt.grid(True)

# if save_plot_path is not None:
#     # Save the plot
#     plt.savefig(save_plot_path)
#     print(f"Plot saved to {save_plot_path}")

# # Display the plot
# plt.show()


# # Determine best location to run each detector
# # Takes as input the target value and must select operating bias voltage for mcp to achieve this target value
# # then plot this line on the plot
# # assume can interpolate between the values

# # Plotting
# colors = plt.cm.get_cmap('tab10', len(channels)).colors
# line_styles = ['-', '--']

# plt.figure(figsize=(10, 6))
# line_style_cycle = itertools.cycle(line_styles)

# for j, chan in enumerate(channels):
#     color = colors[j % len(colors)]
#     line_style = next(line_style_cycle)
#     plt.plot(mcp_bias, channel_max_values[:, j], label=f'Channel {chan}', color=color, linestyle=line_style)

# # Set plot labels and title
# plt.xlabel('MCP Bias (Voltage)', fontsize=12)
# plt.ylabel('Max Value', fontsize=12)
# plt.title('Max Waveform Values Across Runs for Each Channel', fontsize=14)

# # Add a legend
# plt.legend(loc='best', fontsize=10)
# plt.grid(True)

# # Specify a target value for interpolation
# target_value = 3250  # Example target value
# target_bias = []

# for j in range(len(channels)):
#     # Interpolate the bias voltage corresponding to the target value for each channel
#     bias = np.interp(target_value, channel_max_values[:, j], mcp_bias)
#     target_bias.append(bias)

#     # Plot a black dot at the interpolated bias value and the target value
#     plt.scatter(bias, target_value, color='black', zorder=5)

    
#     # # Add the text of the bias value near the black dot (to the right and slightly above)
#     # plt.text(bias + 10, target_value + 50 - (j * 100), f'Ch {chan}: Bias={bias:.2f}', 
#     #          fontsize=9, color='black', verticalalignment='center')

# print(target_bias)
    
# plt.axhline(y=target_value, color='red', linestyle='--', linewidth=2, label=f'Max Value Target: {target_value}')

# # Adjust the layout to create space for the text on the side
# plt.subplots_adjust(right=0.75)  # Leave space on the right side of the plot for text box

# # Create the text box to the right of the plot
# bias_text = "\n".join([f"Ch {chan}: Bias={bias:.2f}" for chan, bias in zip(channels, target_bias)])

# # Add the text box to the right side using figtext
# plt.figtext(0.78, 0.5, bias_text, fontsize=10, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black'))

# # Show the plot
# plt.savefig("channel_max_values_plot_biasLine_labeled.png")
# print("channel_max_values_plot_biasLine.png")
# plt.show()