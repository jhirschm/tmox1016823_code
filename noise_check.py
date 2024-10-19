import psana
import matplotlib.pyplot as plt
import numpy as np


ds = psana.DataSource(exp='tmoc00123',run=22)

run = next(ds.runs())
ch = 22
evt = next(run.events())
evt = next(run.events())
hsd = run.Detector('mrco_hsd')

channels = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
labels = list(range(17))  # Labeled as 0 through 16

# Plotting setup
fig, axs = plt.subplots(4, 4, figsize=(12, 12))  # 4x4 grid for 16 channels

for i, ch in enumerate(channels):
    row = i // 4  # Determine row in 4x4 grid
    col = i % 4   # Determine column in 4x4 grid

    y = hsd.raw.peaks(evt)[ch][0][1][0]  # Extract the peak data
    while True:
        print("1")
        try:
            print("here")

            w = hsd.raw.waveforms(evt)[ch][0][1][0]  # Extract the waveform data
        except:
            evt = next(run.events())
            evt = next(run.events())

        else:
            break
    
    # Plot on respective subplot
    axs[row, col].plot(y.astype(float)/2+float(1<<13), label="peak")
    # axs[row, col].plot(w*(1<<3), label="wave")
    
    # Set subtitle indicating both the channel and label
    axs[row, col].set_title(f'Channel {ch}, Label {labels[i]}')
    axs[row, col].legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# y = hsd.raw.peaks(evt)[ch][0][1][0]
# w =  hsd.raw.waveforms(evt)[ch][0][1][0]

# plt.figure()
# # plt.plot(w*(1<<3),label="wave")
# plt.plot(y.astype(float)/2+float(1<<13),label="peak")
# plt.legend()
# plt.show()


# Channels and corresponding labels
