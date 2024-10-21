import psana
import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools
from matplotlib.backends.backend_pdf import PdfPages

event_num_break = 1000
# Set up argument parser to take command-line arguments
parser = argparse.ArgumentParser(description='Experiment.')
parser.add_argument('--exp-name', type=str, help='Name of the experiment.')
parser.add_argument('--max-load-path', type=str, default=None, help='Path to load max histograms (default: None).')
parser.add_argument('--min-load-path', type=str, default=None, help='Path to load min histograms (default: None).')
parser.add_argument('--hist-load-path', type=str, default=None, help='Path to load histogram data (default: None).')
parser.add_argument('--bins-load-path', type=str, default=None, help='Path to load bin data (default: None).')
parser.add_argument('--save-plot-path', type=str, default=None, help='Path to save the plots.')

args = parser.parse_args()

# Access the experiment name and file paths
exp_name = args.exp_name
save_plot_path = args.save_plot_path
max_load_path = args.max_load_path
min_load_path = args.min_load_path
hist_load_path = args.hist_load_path
bins_load_path = args.bins_load_path

# Initialize MCP bias
mcp_bias = np.array([1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800])

# Load bin values
if bins_load_path is not None:
    binvals = np.load(bins_load_path)
else:
    binvals = np.arange(0, (1 << 15) + 1, 1 << 6)

channels = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
runs_list = [17, 16, 15, 14, 5, 6, 7, 8, 9, 10, 11, 12, 13]
num_runs = len(runs_list)
num_channels = len(channels)

# Initialize histograms for max and min values
if hist_load_path is not None:
    histograms_max = np.load(hist_load_path)
else:
    histograms_max = np.zeros((num_runs, num_channels, len(binvals) - 1))

# Initialize histogram for min values
if min_load_path is not None:
    histograms_min = np.load(min_load_path)
else:
    histograms_min = np.zeros((num_runs, num_channels, len(binvals) - 1))

# Initialize max and min value tracking arrays
channel_max_values = np.zeros((len(runs_list), len(channels)))
channel_min_values = np.full((len(runs_list), len(channels)), np.inf)  # Initialize with a large number for min values

if max_load_path is None and hist_load_path is None and min_load_path is None:
    for i, run_num in enumerate(runs_list):
        ds = psana.DataSource(exp=exp_name, run=run_num)
        run = next(ds.runs())
        hsd = run.Detector('mrco_hsd')

        # Loop through all events in the run
        processed_events = 0
        for num, evt in enumerate(run.events()):
            peaks = hsd.raw.peaks(evt)

            for j, chan in enumerate(channels):
                if peaks is not None:
                    for k in range(len(peaks[chan][0][1])):
                        if len(peaks[chan][0][1]) > 2:
                            values = peaks[chan][0][1][k]
                            max_value = values.max()  # Get the maximum value for the peak
                            min_value = values.min()  # Get the minimum value for the peak

                            # Find the appropriate bin index for the max and min values
                            max_bin_index = np.digitize(max_value, binvals) - 1
                            min_bin_index = np.digitize(min_value, binvals) - 1

                            # Ensure the bin indices are within the valid range
                            if 0 <= max_bin_index < len(histograms_max[i, j]):
                                histograms_max[i, j, max_bin_index] += 1  # Increment the count for max value
                            if 0 <= min_bin_index < len(histograms_min[i, j]):
                                histograms_min[i, j, min_bin_index] += 1  # Increment the count for min value

                            # Track the overall max and min values
                            if max_value > channel_max_values[i, j]:
                                channel_max_values[i, j] = max_value
                            if min_value < channel_min_values[i, j]:
                                channel_min_values[i, j] = min_value
                            processed_events += 1

            if num >= event_num_break:  # Break the loop after processing a certain number of events
                break
        print(f"Run {run_num} completed with {processed_events} events processed after cycling over {num} events.")
# Optionally save the histograms and bin values
if save_plot_path:
    np.save(f"{save_plot_path}/histograms_max.npy", histograms_max)
    np.save(f"{save_plot_path}/histograms_min.npy", histograms_min)
    np.save(f"{save_plot_path}/binvals_minMax.npy", binvals)
    np.save(f"{save_plot_path}/channel_max_values_minMax.npy", channel_max_values)
    np.save(f"{save_plot_path}/channel_min_values_minMax.npy", channel_min_values)

# Now histograms_max and histograms_min contain the separate histograms for max and min values
# Proceed to create the plots for both sets of histograms

colors = plt.cm.get_cmap('tab10', len(channels)).colors
line_styles = ['-', '--']
plt.figure(figsize=(10, 6))
line_style_cycle = itertools.cycle(line_styles)

print("********** Plotting Max Histograms **********")
# Plot the max histograms
with PdfPages(f'{save_plot_path}/channel_plots_max.pdf') as pdf:
    for i, chan in enumerate(channels):
        fig, axs = plt.subplots(nrows=len(mcp_bias), ncols=1, figsize=(10, 18), sharex=True)

        for j in range(len(mcp_bias)):
            # Plot each max histogram with legends
            axs[j].bar(binvals[300:-1], histograms_max[j, i, 300:], width=np.diff(binvals[300:]), align='edge', edgecolor='black', alpha=0.7)
            axs[j].set_ylim(0, 60)
            axs[j].set_ylabel('Counts', fontsize=12)
            axs[j].set_title(f'MCP Bias: {mcp_bias[j]}; Run: {runs_list[j]}', fontsize=12)

        fig.suptitle(f'Channel {chan} - Max FEX Pulse Height per Window', fontsize=16)
        axs[-1].set_xlabel('FEX Max Pulse Height per Window', fontsize=12)
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.125, right=0.9, hspace=0.7, wspace=0.6)
        pdf.savefig(fig)
        plt.close(fig)

print("********** Plotting Min Histograms **********")
# Plot the min histograms
with PdfPages(f'{save_plot_path}/channel_plots_min.pdf') as pdf:
    for i, chan in enumerate(channels):
        fig, axs = plt.subplots(nrows=len(mcp_bias), ncols=1, figsize=(10, 18), sharex=True)

        for j in range(len(mcp_bias)):
            # Plot each min histogram with legends
            axs[j].bar(binvals[0:100], histograms_min[j, i, 0:100], width=np.diff(binvals[0:100+1]), align='edge', edgecolor='red', alpha=0.7)
            axs[j].set_ylim(0, 60)
            axs[j].set_ylabel('Counts', fontsize=12)
            axs[j].set_title(f'MCP Bias: {mcp_bias[j]}; Run: {runs_list[j]}', fontsize=12)

        fig.suptitle(f'Channel {chan} - Min FEX Pulse Height per Window', fontsize=16)
        axs[-1].set_xlabel('FEX Min Pulse Height per Window', fontsize=12)
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.125, right=0.9, hspace=0.7, wspace=0.6)
        pdf.savefig(fig)
        plt.close(fig)

print("********** Plotting Max/Min Histograms **********")
with PdfPages(f'{save_plot_path}/channel_plots_min_max.pdf') as pdf:
    for i, chan in enumerate(channels):
        fig, axs = plt.subplots(nrows=13, ncols=2, figsize=(12, 24), sharex=True)

        for j in range(len(mcp_bias)):
            row = j % 13
            col_min = 0  # First column for min
            col_max = 1  # Second column for max

            # Min histogram plot (red, left column)
            axs[row, col_min].bar(binvals[0:100], histograms_min[j, i, 0:100], width=np.diff(binvals[0:100+1]), align='edge', edgecolor='red', alpha=0.7)
            axs[row, col_min].set_ylim(0, 60)
            axs[row, col_min].set_ylabel('Counts', fontsize=12)
            axs[row, col_min].set_title(f'Min: MCP Bias {mcp_bias[j]} Run {runs_list[j]}', fontsize=10)

            # Max histogram plot (black, right column)
            axs[row, col_max].bar(binvals[300:-1], histograms_max[j, i, 300:], width=np.diff(binvals[300:]), align='edge', edgecolor='black', alpha=0.7)
            axs[row, col_max].set_ylim(0, 60)
            axs[row, col_max].set_title(f'Max: MCP Bias {mcp_bias[j]} Run {runs_list[j]}', fontsize=10)

        fig.suptitle(f'Channel {chan} - Min (Red) vs Max (Black) FEX Pulse Heights', fontsize=16)
        axs[-1, 0].set_xlabel('FEX Min Pulse Height', fontsize=12)
        axs[-1, 1].set_xlabel('FEX Max Pulse Height', fontsize=12)
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.1, right=0.9, hspace=0.6, wspace=0.3)

        pdf.savefig(fig)
        plt.close(fig)
