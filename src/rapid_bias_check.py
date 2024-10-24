import psana
import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime


event_num_break = 10000
# Set up argument parser to take command-line arguments
parser = argparse.ArgumentParser(description='Experiment.')
parser.add_argument('--exp-name', type=str, help='Name of the experiment.')
parser.add_argument('--hist-load-path', type=str, default=None, help='Path to load histogram data (default: None).')
parser.add_argument('--wrapped-hist-load-path', type=str, default=None, help='Path to load wrapped histogram data (default: None).')
parser.add_argument('--bins-load-path', type=str, default=None, help='Path to load bin data (default: None).')
parser.add_argument('--save-folder-path', type=str, default=None, help='Path to save the plots.')

# Add argument for either a single run or a list of runs
parser.add_argument('--run', type=int, nargs='+', help='Single run or a list of runs.')
parser.add_argument('--mcp-bias', type=int, nargs='+', default=None, help='MCP biases for run.')

# Add argument for maximum number of events
parser.add_argument('--max-events', type=int, help='Maximum number of events to process.')

# Add argument for number of channels with a default list of channels
default_channels = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
parser.add_argument('--channels', type=int, nargs='+', default=default_channels, help='List of channels to process (default: [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]).')

# Add argument for channel polarity as a list, defaulting to +1 for all channels
parser.add_argument('--channel-polarity', type=int, nargs='+', default=[1]*len(default_channels), 
                    help='List of channel polarities (1 or -1) corresponding to each channel (default: all 1).')

# Add an argument for the thresholds
parser.add_argument('--pos-threshold', type=float, default=10000, help='Threshold for positive polarity channels (default: 10000).')
parser.add_argument('--neg-threshold', type=float, default=22000, help='Threshold for negative polarity channels (default: 22000).')

args = parser.parse_args()

# Access the experiment name and file paths
exp_name = args.exp_name
save_folder_path = args.save_folder_path
hist_load_path = args.hist_load_path
bins_load_path = args.bins_load_path
max_num_events = args.max_events
channels = args.channels
channel_polarity = args.channel_polarity
runs = args.run
secondary_hist_load_path = args.wrapped_hist_load_path

# Validate MCP bias length
if args.mcp_bias is not None:
    if len(args.mcp_bias) != len(runs):
        raise ValueError(f'The length of MCP biases ({len(args.mcp_bias)}) must match the number of runs ({len(runs)}).')




if bins_load_path is not None:
    binvals = np.load(bins_load_path)
else:
# Create a 3D array for histograms: (number of runs, number of channels, number of bins)
    binvals = np.arange(0,(1<<15)+1,1<<6)

# Find the bin index corresponding to the thresholds
pos_threshold_bin_index = np.digitize([args.pos_threshold], binvals)[0] - 1  # Subtract 1 for zero-based index
neg_threshold_bin_index = np.digitize([args.neg_threshold], binvals)[0] - 1

num_runs = len(runs)
num_channels = len(channels)

if hist_load_path is not None:
    histograms = np.load(hist_load_path)
else:
    histograms = np.zeros((num_runs, num_channels, len(binvals) - 1))

if secondary_hist_load_path is not None:    
    secondary_histograms = np.load(secondary_hist_load_path)
else:
    secondary_histograms = np.zeros((num_runs, num_channels, len(binvals) - 1))

# Create a save folder if it exists or append datetime
if save_folder_path:
    if os.path.exists(save_folder_path):
        # If the directory exists, append the current datetime
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        new_save_folder_path = f"{save_folder_path}_{current_time}"
        os.makedirs(new_save_folder_path)
        save_folder_path = new_save_folder_path
    else:
        # If the directory does not exist, create it
        os.makedirs(save_folder_path)
    print(f"Saving data to {save_folder_path}")
else:
    print("No save folder path provided. Not saving run...")

# Check if 

print(runs)
if hist_load_path is None:
    for i, run_num in enumerate(runs):
        ds = psana.DataSource(exp=exp_name, run=run_num)
        run = next(ds.runs())
        hsd = run.Detector('mrco_hsd')

        # Loop through all events in the run
        for num, evt in enumerate(run.events()):
            peaks = hsd.raw.peaks(evt)

            for j, chan in enumerate(channels):
                polarity = channel_polarity[j]  # Get the polarity for the current channel
                
                if peaks is not None:
                    if len(peaks[chan][0][1]) > 2:
                        for k in range(1,len(peaks[chan][0][1])-1):
                        
                            # Primary histogram logic based on polarity
                            if polarity == 1:
                                # For positive polarity, look for the maximum value
                                signal_value = peaks[chan][0][1][k].max()
                            elif polarity == -1:
                                # For negative polarity, look for the minimum value
                                signal_value = peaks[chan][0][1][k].min()

                            # Find the appropriate bin index for the signal value
                            bin_index = np.digitize(signal_value, binvals) - 1  # Subtract 1 to convert to zero-based index

                            # Ensure the bin index is within the valid range
                            if 0 <= bin_index < len(histograms[i, j]):
                                histograms[i, j, bin_index] += 1  # Increment the count in the appropriate bin
                            else:
                                print(f"Value out of range for histogram bins for run {run_num}, channel {chan}: {signal_value}")
                            
                            # Secondary histogram logic based on thresholds
                            if polarity == 1:
                                # For positive polarity, look for max value below threshold
                                indices = np.where((peaks[chan][0][1][k] < args.pos_threshold) & (peaks[chan][0][1][k] >= 0))
                                filtered_values = peaks[chan][0][1][k][indices]
                                max_value = np.max(filtered_values) if filtered_values.size > 0 else None
                                # print(max_value)
                                if max_value is not None:
                                    secondary_bin_index = np.digitize(max_value, binvals) - 1
                                    if 0 <= secondary_bin_index < len(secondary_histograms[i, j]):
                                        secondary_histograms[i, j, secondary_bin_index] += 1  # Increment in secondary histogram
                            elif polarity == -1:
                                # For negative polarity, look for min value above threshold
                                print("not implemented")

            if num >= max_num_events:
                break

if save_folder_path:
    np.save(f"{save_folder_path}/histograms.npy", histograms)
    np.save(f"{save_folder_path}/wrapping_histograms.npy", secondary_histograms)
    np.save(f"{save_folder_path}/binvals.npy", binvals)
# Determine the number of rows needed for plotting based on the number of channels
num_channels = len(channels)
num_rows = min(16, num_channels)

# Create the PDF file for saving the plots
with PdfPages(f'{save_folder_path}/experiment_plots.pdf') as pdf:
    width_ratio = len(binvals[250:-1]) / len(binvals[0:pos_threshold_bin_index])

    if num_runs == 1:
        # Single run case: All channels on the same page
        # fig, axs = plt.subplots(num_rows, 2, figsize=(10, num_rows * 3))
        fig, axs = plt.subplots(nrows=len(runs), ncols=2, figsize=(10, 24),
                                gridspec_kw={'width_ratios': [1, width_ratio]}, sharex=False)

        if args.mcp_bias is not None:
            fig.suptitle(f'Run {runs[0]} (MCP Bias: {args.mcp_bias[0]})', fontsize=16)
        else:
            fig.suptitle(f'Run {runs[0]}', fontsize=16)
        
        max_value_1 =np.max(secondary_histograms[0, :,:])#np.max(histograms[:,:, 300:])  # Get the maximum value for setting the y-axis limits
        max_value_2 =np.max(histograms[0, :,300:])
        for j, chan in enumerate(channels):
            row = j // 2
            axs[j, 0].bar(binvals[0:pos_threshold_bin_index], secondary_histograms[0, j,0:pos_threshold_bin_index], width=np.diff(binvals[0:pos_threshold_bin_index+1]), edgecolor="red", facecolor="red")
            axs[j, 0].set_title(f'Channel {chan} (Rollover)', fontsize=12)
            axs[j, 0].set_ylim(0, max_value_1 * 1.1)  # Set ylim to max value with a 10% margin

            # Plot primary histograms on the right column
            axs[j, 1].bar(binvals[250:-1], histograms[i, j,250:], width=np.diff(binvals[250:]), edgecolor="black", facecolor="black")
            axs[j, 1].set_title(f'Channel {chan} (Primary)', fontsize=12)
            axs[j, 1].set_ylim(0, max_value_2 * 1.1)  # Set ylim to max value with a 10% margin
        #Set the x-label and y-label for the last subplot
        axs[-1, 0].set_xlabel(f'FEX Max Pulse Height <{pos_threshold_bin_index} per Window', fontsize=12)
        axs[-1, 1].set_xlabel('FEX Max Pulse Height per Window', fontsize=12)
        
        # Adjust layout for better spacing with your specific settings
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.125, right=0.9, hspace=0.7, wspace=0.6)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)  # Save the current figure to the PDF
        plt.close(fig)  # Close the figure to free memory

    else:
        # Multiple runs case: One channel per page, all runs on the same page for that channel
        for j, chan in enumerate(channels):
            # Set up the figure with variable column widths
            fig, axs = plt.subplots(nrows=len(runs), ncols=2, figsize=(10, 24),
                                    gridspec_kw={'width_ratios': [1, width_ratio]}, sharex=False)

            fig.suptitle(f'Channel {chan}', fontsize=16)
            max_value_1 =np.max(secondary_histograms[:, j,:])#np.max(histograms[:,:, 300:])  # Get the maximum value for setting the y-axis limits
            max_value_2 =np.max(histograms[:, j,300:])
            for i, run_num in enumerate(runs):
                # Plot primary histograms for this run on the right column
                axs[i, 0].bar(binvals[0:pos_threshold_bin_index], secondary_histograms[i, j,0:pos_threshold_bin_index], width=np.diff(binvals[0:pos_threshold_bin_index+1]), edgecolor="red", facecolor="red")
                if args.mcp_bias is not None:
                    axs[i, 0].set_title(f'Run {run_num} (Roll Over) MCP Bias: {args.mcp_bias[i]}', fontsize=12)
                else:
                    axs[i, 0].set_title(f'Run {run_num} (Roll Over)', fontsize=12)
                axs[i, 0].set_ylim(0, max_value_1*1.1)  # Set ylim to max value with a 10% margin
                axs[i, 0].set_ylabel('Counts', fontsize=12)
                
                # Plot primary histograms for this run on the right column
                axs[i, 1].bar(binvals[250:-1], histograms[i, j,250:], width=np.diff(binvals[250:]), edgecolor="black", facecolor="black")
                axs[i, 1].set_title(f'Run {run_num} (Primary)', fontsize=12)
                axs[i, 1].set_ylim(0, max_value_2 * 1.1)  # Set ylim to max value with a 10% margin
            # Set the x-label and y-label for the last subplot
            axs[-1, 0].set_xlabel(f'FEX Max Pulse Height <{pos_threshold_bin_index} per Window', fontsize=12)
            axs[-1, 1].set_xlabel('FEX Max Pulse Height per Window', fontsize=12)
        
            # Adjust layout for better spacing with your specific settings
            plt.subplots_adjust(top=0.92, bottom=0.05, left=0.125, right=0.9, hspace=0.7, wspace=0.6)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)  # Save the current figure to the PDF
            plt.close(fig)  # Close the figure to free memory









