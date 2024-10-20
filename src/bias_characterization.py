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







channels = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
labels = list(range(17))  # Labeled as 0 through 16

runs_list = [17, 16, 15, 14 ,5,6,7,8,9, 10,11, 12,13]
mcp_bias = [1200, 1250, 1300, 1350,1400,1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800]

thresh = 6100

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
                        max_value = waveforms[chan][0][thresh:].max()
                        if max_value > channel_max_values[i,j]:
                            channel_max_values[i,j] = max_value
                    # print(f"Max value of the waveform: {max_value}")
                
            except:
                # Handle any error (like missing data) and continue to next event
                continue
            if num >= 10000:
                break


print(channel_max_values)
if load_path is None:   
    # Define file paths for saving the array and plot
    numpy_file_path = "channel_max_values_thresh.npy"  # Save as .npy
    plot_file_path = "channel_max_values_plot_thresh.png"  # Save as .png

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
    plt.savefig(save_plot_path)
    print(f"Plot saved to {save_plot_path}")

# Display the plot
plt.show()


# Determine best location to run each detector
# Takes as input the target value and must select operating bias voltage for mcp to achieve this target value
# then plot this line on the plot
# assume can interpolate between the values

# Plotting
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
target_value = 3250  # Example target value
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

print(target_bias)
    
plt.axhline(y=target_value, color='red', linestyle='--', linewidth=2, label=f'Max Value Target: {target_value}')

# Adjust the layout to create space for the text on the side
plt.subplots_adjust(right=0.75)  # Leave space on the right side of the plot for text box

# Create the text box to the right of the plot
bias_text = "\n".join([f"Ch {chan}: Bias={bias:.2f}" for chan, bias in zip(channels, target_bias)])

# Add the text box to the right side using figtext
plt.figtext(0.78, 0.5, bias_text, fontsize=10, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black'))

# Show the plot
plt.savefig("channel_max_values_plot_biasLine_labeled_thresh.png")
print("channel_max_values_plot_biasLine_thresh.png")
plt.show()


# Define a fitting function (e.g., a second-degree polynomial)
def poly3(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

# Create subplots (4 rows, 4 columns)
fig, axes = plt.subplots(4, 4, figsize=(16, 12))  # 16 subplots in total
axes = axes.flatten()  # Flatten the 2D array of axes to iterate over them

# Perform curve fitting for each channel for max values between 2500 and 3750
for j, chan in enumerate(channels):
    # Extract data for this channel
    y_data = channel_max_values[:, j]
    x_data = np.array(mcp_bias)

    # Filter values between 2500 and 3750
    valid_indices = (y_data >= 2000) & (y_data <= 4100)
    x_fit = x_data[valid_indices]
    y_fit = y_data[valid_indices]

    print(x_fit)

    # Plot the original data in black on the corresponding subplot
    axes[j].plot(x_data, y_data, color='black', label=f'Channel {chan}', linestyle='-', marker='o')

    if len(x_fit) > 0 and len(y_fit) > 0:
        # Fit a third-degree polynomial to the filtered data
        try:
            popt, _ = curve_fit(poly3, x_fit, y_fit)
            a, b, c, d = popt

            # Generate the fitted curve using the obtained coefficients
            x_curve = np.linspace(min(x_fit), max(x_fit), 100)
            y_curve = poly3(x_curve, a, b, c, d)

            # Plot the fitted curve in red dashed lines
            axes[j].plot(x_curve, y_curve, color='red', linestyle='--', label=f'Fit for Channel {chan}')
        
        except:
            print(f"Curve fitting failed for Channel {chan}")

    # Set the x and y limits for all subplots
    axes[j].set_xlim([1150, 1850])
    axes[j].set_ylim([2000, 4250])

    # Set the title for each subplot
    axes[j].set_title(f'Channel {chan}', fontsize=10)

    # Add labels to the last subplot in the column
    if j % 4 == 0:
        axes[j].set_ylabel('Max Value', fontsize=8)
    if j >= 12:
        axes[j].set_xlabel('MCP Bias (Voltage)', fontsize=8)

# Adjust layout to make sure titles and labels fit properly
plt.tight_layout()

# Save and show the plot
plt.savefig("fitted_channel_max_values_subplots_thresh.png")
print("Plot saved as fitted_channel_max_values_subplots_thresh.png")
plt.show()


# Set up tkinter window
root = tk.Tk()
root.title("Interactive Plot for MCP Bias")

# Create input for maximum value
max_value_label = tk.Label(root, text="Enter Maximum Value:")
max_value_label.pack(pady=5)

max_value_entry = tk.Entry(root)
max_value_entry.pack(pady=5)

# Create Matplotlib figure and subplots
fig, axs = plt.subplots(4, 4, figsize=(15, 15))
axs = axs.ravel()  # Flatten the 2D array of axes
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Store the fitted parameters globally so they aren't recalculated
fit_parameters = []

# # Initial plot setup: plot the data and fits
# def initial_plot_setup():
#     for j, chan in enumerate(channels):
#         y_data = channel_max_values[:, j]
#         x_data = np.array(mcp_bias)

#         # Filter values between 2000 and 4000
#         valid_indices = (y_data >= 2000) & (y_data <= 4000)
#         x_fit = x_data[valid_indices]
#         y_fit = y_data[valid_indices]

#         if len(x_fit) > 0 and len(y_fit) > 0:
#             try:
#                 popt, _ = curve_fit(poly3, x_fit, y_fit)
#                 fit_parameters.append(popt)

#                 # Generate the fitted curve
#                 x_curve = np.linspace(min(x_fit), max(x_fit), 100)
#                 y_curve = poly3(x_curve, *popt)

#                 # Plot the original data and the fitted curve
#                 axs[j].plot(x_data, y_data, 'o', color='black', label=f'Channel {chan}')
#                 axs[j].plot(x_curve, y_curve, color='red', linestyle='--', label='Fit')

#                 # Set plot limits and titles
#                 axs[j].set_xlim(1150, 1850)
#                 axs[j].set_ylim(2000, 4250)
#                 axs[j].set_title(f'Channel {chan} - No Max Value Set')
#                 axs[j].legend()

#             except:
#                 axs[j].set_title(f"Channel {chan} - Fit Failed")

#     canvas.draw()

# # Update only the markers and bias text when the max value is changed
# def update_plots():
#     max_value = float(max_value_entry.get())

#     for j, chan in enumerate(channels):
#         a, b, c, d = fit_parameters[j]

#         # Find the bias voltage for the given max_value
#         fitted_bias_voltage = np.roots([a, b, c, d - max_value])

#         # We only want real roots, select one within the MCP bias range
#         fitted_bias_voltage = fitted_bias_voltage[np.isreal(fitted_bias_voltage)].real
#         fitted_bias_voltage = fitted_bias_voltage[(fitted_bias_voltage >= 1150) & (fitted_bias_voltage <= 1850)]

#         if len(fitted_bias_voltage) > 0:
#             fitted_bias_voltage = fitted_bias_voltage[0]  # Use the first valid bias voltage

#             # Clear previous marker and add the new marker at the max value
#             axs[j].collections.clear()  # Clear previous markers
#             axs[j].plot(fitted_bias_voltage, max_value, 'p', color='blue', markersize=10, label='Selected Point')

#             # Update the title with the bias voltage
#             axs[j].set_title(f'Channel {chan} - Bias {fitted_bias_voltage:.2f}')

#     canvas.draw()

# # Initial plot setup (run only once)
# initial_plot_setup()

# # Create a button to update plots
# update_button = tk.Button(root, text="Update Plots", command=update_plots)
# update_button.pack(pady=5)

# # Run the tkinter main loop
# root.mainloop()