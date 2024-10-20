import psana
import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools


# Set up argument parser to take show_fourier as a command-line argument
parser = argparse.ArgumentParser(description='Experiment.')
parser.add_argument('--exp-name', type=str, help='Name of the experiment.')
parser.add_argument('--load-path', type=str, default=None, help='Path to load data (default: None).')


# Parse the arguments
args = parser.parse_args()


# Access the experiment name
exp_name = args.exp_name
load_path = args.load_path







channels = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
labels = list(range(17))  # Labeled as 0 through 16

runs_list = [17, 16, 15, 14 , 5,6,7,8,9,10,11, 12,13]
mcp_bias = [1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800]

if load_path is not None:
    channel_max_values = np.load(load_path)
    print(f"NumPy array loaded from {load_path}")

else:   
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
            # if num >= 10000:
            #     break


print(channel_max_values)
if load_path is None:   
    # Define file paths for saving the array and plot
    numpy_file_path = "channel_max_values.npy"  # Save as .npy
    plot_file_path = "channel_max_values_plot.png"  # Save as .png

    # Save the NumPy array
    np.save(numpy_file_path, channel_max_values)
    print(f"NumPy array saved to {numpy_file_path}")

# Define a list of colors and line styles
colors = plt.cm.get_cmap('tab10', len(channels)).colors  # Use tab10 colormap for distinct colors
line_styles = ['-', '--']  # Solid and dashed lines

# Initialize the plot
plt.figure(figsize=(10, 6))

# Cycle through line styles and colors
line_style_cycle = itertools.cycle(line_styles)

# Plot each channel's max values with different colors and alternating line styles
for j, chan in enumerate(channels):
    color = colors[j % len(colors)]  # Cycle through the colors
    line_style = next(line_style_cycle)  # Alternate between solid and dashed lines
    plt.plot(mcp_bias, channel_max_values[:, j], label=f'Channel {chan}', color=color, linestyle=line_style)

# Set plot labels and title
plt.xlabel('MCP Bias (Voltage)', fontsize=12)
plt.ylabel('Max Value', fontsize=12)
plt.title('Max Waveform Values Across Runs for Each Channel', fontsize=14)

# Add a legend to differentiate the channels
plt.legend(loc='best', fontsize=10)

# Show the grid
plt.grid(True)

if save_plot_path is not None:
    # Save the plot
    plt.savefig(plot_file_path)
    print(f"Plot saved to {plot_file_path}")

# Display the plot
plt.show()