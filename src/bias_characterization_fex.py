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
from matplotlib.backends.backend_pdf import PdfPages


# Set up argument parser to take show_fourier as a command-line argument
parser = argparse.ArgumentParser(description='Experiment.')
parser.add_argument('--exp-name', type=str, help='Name of the experiment.')
parser.add_argument('--max-load-path', type=str, default=None, help='Path to load data (default: None).')
parser.add_argument('--hist-load-path', type=str, default=None, help='Path to load data (default: None).')
parser.add_argument('--bins-load-path', type=str, default=None, help='Path to load data (default: None).')

parser.add_argument('--save-plot-path', type=str, default=None, help='Path to load data (default: None).')



# Parse the arguments
args = parser.parse_args()


# Access the experiment name
exp_name = args.exp_name
save_plot_path = args.save_plot_path
max_load_path = args.max_load_path
hist_load_path = args.hist_load_path
bins_load_path = args.bins_load_path

save = False

mcp_bias = np.array([1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800])

if bins_load_path is not None:
    binvals = np.load(bins_load_path)
else:
# Create a 3D array for histograms: (number of runs, number of channels, number of bins)
    binvals = np.arange(0,(1<<15)+1,1<<6)








channels = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
labels = list(range(17))  # Labeled as 0 through 16

runs_list = [17, 16, 15, 14 ,5,6,7,8,9, 10,11, 12,13]



num_runs = len(runs_list)
num_channels = len(channels)
if hist_load_path is not None:
    histograms = np.load(hist_load_path)
else:
    histograms = np.zeros((num_runs, num_channels, len(binvals) - 1))

if max_load_path is not None:
    channel_max_values = np.load(max_load_path)
    print(f"NumPy array loaded from {max_load_path}")
else:
    channel_max_values = np.zeros((len(runs_list),len(channels)))

if max_load_path is None and hist_load_path is None:
    for i, run_num in enumerate(runs_list):
        ds = psana.DataSource(exp=exp_name, run=run_num)
        run = next(ds.runs())
        hsd = run.Detector('mrco_hsd')

        # Loop through all events in the run
        for num, evt in enumerate(run.events()):
            peaks = hsd.raw.peaks(evt)

            for j, chan in enumerate(channels):
                
                                
                if peaks is not None:
                    for k in range(len(peaks[chan][0][1])):
                        if len(peaks[chan][0][1]) > 2:
                            max_value = peaks[chan][0][1][k].max()  # Get the maximum value for the peak
                            
                            # Find the appropriate bin index for the single value
                            bin_index = np.digitize(max_value, binvals) - 1  # Subtract 1 to convert to zero-based index

                            # Ensure the bin index is within the valid range
                            if 0 <= bin_index < len(histograms[i, j]):
                                histograms[i, j, bin_index] += 1  # Increment the count in the appropriate bin
                            else:
                                print(f"Value out of range for histogram bins for run {run_num}, channel {chan}: {max_value}")
                            if max_value > channel_max_values[i,j]:
                                channel_max_values[i,j] = max_value

            if num >= 10000:
                break

if save:
    np.save("histograms6.npy", histograms)
    np.save("binvals6.npy", binvals)
    np.save("channel_max_values_fex6.npy", channel_max_values)


colors = plt.cm.get_cmap('tab10', len(channels)).colors
line_styles = ['-', '--']

plt.figure(figsize=(10, 6))
line_style_cycle = itertools.cycle(line_styles)

for j, chan in enumerate(channels):
    color = colors[j % len(colors)]
    line_style = next(line_style_cycle)
    plt.plot(mcp_bias, channel_max_values[:, j], label=f'Channel {chan}', color=color, linestyle=line_style)

# Set plot labels and title
plt.xlabel('MCP Bias (Voltage)', fontsize=12)
plt.ylabel('Max Value', fontsize=12)
plt.title('Max Waveform Values Across Runs for Each Channel', fontsize=14)

# Add a legend
plt.legend(loc='best', fontsize=10)
plt.grid(True)

# Specify a target value for interpolation
target_value = 26000  # Example target value
target_bias = []

for j in range(len(channels)):
    # Interpolate the bias voltage corresponding to the target value for each channel
    bias = np.interp(target_value, channel_max_values[:, j], mcp_bias)
    target_bias.append(bias)

    # Plot a black dot at the interpolated bias value and the target value
    plt.scatter(bias, target_value, color='black', zorder=5)

    
    # # Add the text of the bias value near the black dot (to the right and slightly above)
    # plt.text(bias + 10, target_value + 50 - (j * 100), f'Ch {chan}: Bias={bias:.2f}', 
    #          fontsize=9, color='black', verticalalignment='center')

# print(target_bias)
    
plt.axhline(y=target_value, color='red', linestyle='--', linewidth=2, label=f'Max Value Target: {target_value}')

# Adjust the layout to create space for the text on the side
plt.subplots_adjust(right=0.75)  # Leave space on the right side of the plot for text box

# Create the text box to the right of the plot
bias_text = "\n".join([f"Ch {chan}: Bias={bias:.2f}" for chan, bias in zip(channels, target_bias)])

# Add the text box to the right side using figtext
plt.figtext(0.78, 0.5, bias_text, fontsize=10, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black'))

# Show the plot
# plt.savefig("channel_max_values_plot_biasLine_labeled_thresh5.png")
print("channel_max_values_plot_biasLine_thresh6.png")
plt.show()

# At this point, `histograms` contains the 3D histogram counts
# You can access them like this:
# for run_idx in range(num_runs):
#     for chan_idx in range(num_channels):
#         print(f"Run {runs_list[run_idx]}, Channel {channels[chan_idx]} Histogram Counts: {histograms[run_idx, chan_idx]}")
# Create a PDF file to save the plots
with PdfPages('channel_plots.pdf') as pdf:
    for i, chan in enumerate(channels):
        fig, axs = plt.subplots(nrows=len(mcp_bias), ncols=1, figsize=(10, 18), sharex=True)

        for j in range(len(mcp_bias)):
            # Plot each histogram with legends
            axs[j].bar(binvals[300:-1], histograms[j, i, 300:], width=np.diff(binvals[300:]), align='edge', edgecolor='black', alpha=0.7)

            # Set y-axis limits
            axs[j].set_ylim(0, 60)
            axs[j].set_ylabel('Counts', fontsize=12)


            # Add title for each subplot with the MCP Bias value
            axs[j].set_title(f'MCP Bias: {mcp_bias[j]}; tmox1016823 Run Number: {runs_list[j]}', fontsize=12)

        # Set the main title for the entire plot based on channel number
        fig.suptitle(f'Channel {chan} - FEX Max Pulse Height per Window', fontsize=16)

        # Set the x-label and y-label for the last subplot
        axs[-1].set_xlabel('FEX Max Pulse Height per Window', fontsize=12)
        
        # Adjust layout for better spacing with your specific settings
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.125, right=0.9, hspace=0.7, wspace=0.6)

        # Save the current figure to the PDF
        pdf.savefig(fig)
        plt.close(fig)  # Close the figure to avoid displaying it


