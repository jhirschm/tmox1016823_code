import psana
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Set up argument parser to take show_fourier as a command-line argument
parser = argparse.ArgumentParser(description='Plot time-domain and optional Fourier transform.')
parser.add_argument('--show_fourier', action='store_true', help='Plot Fourier transform (power spectrum).')

# Parse the arguments and assign show_fourier
args = parser.parse_args()
show_fourier = args.show_fourier  # This is the variable that will be True or False


ds = psana.DataSource(exp='tmoc00123',run=22)
ds = psana.DataSource(exp='tmox1016823', run=7)


run = next(ds.runs())
ch = 22
evt = next(run.events())
evt = next(run.events())



hsd = run.Detector('mrco_hsd')

channels = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
labels = list(range(17))  # Labeled as 0 through 16

while True:
    evt = next(run.events())
    # Plotting setup
    # Determine the number of rows based on whether Fourier transforms are shown
    if show_fourier:
        fig, axs = plt.subplots(8, 4, figsize=(12, 24))  # 8x4 grid: 4 for time-domain, 4 for power spectrum
    else:
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))  # 4x4 grid: only time-domain

    if show_fourier:
        for i, ch in enumerate(channels):
            row_time = i // 4  # Determine row in the first 4 rows for time-domain plots
            col = i % 4        # Determine column in 4x4 grid

            evt = next(run.events())
            while True:
                try:
                    w = hsd.raw.waveforms(evt)[ch][0]  # Extract the waveform data
                except:
                    evt = next(run.events())           # Try the next event if the current one fails
                else:
                    break
            
            y = hsd.raw.peaks(evt)[ch][0][1][0]  # Extract the peak data

            # Plot time-domain data on the first 4 rows
            axs[row_time, col].plot(y.astype(float) / 2 + float(1<<13), label="peak")
            axs[row_time, col].plot(w * (1 << 3), label="wave")
            axs[row_time, col].set_title(f'Channel {ch}, Label {labels[i]}')

            # Take Fourier transforms of the peak and waveform data
            peak_fft = np.power(np.fft.fft(y), int(2)).real
            wave_fft = np.power(np.fft.fft(w), int(2)).real

            # Create corresponding frequency-domain plots (Fourier transform)
            row_fft = row_time + 4  # The FFT plots go in the next 4 rows
            freqs_y = np.fft.fftfreq(len(y))  # Frequency axis for FFT
            freqs_w = np.fft.fftfreq(len(w))
            # axs[row_fft, col].plot(freqs_y, np.abs(peak_fft)**2, label="FFT of peak")
            # axs[row_fft, col].plot(freqs_w, np.abs(wave_fft)**2, label="FFT of wave")
            axs[row_fft, col].plot(np.log(peak_fft[-2:2]), label="Log FFT of peak")
            axs[row_fft, col].plot(np.log(wave_fft[-2:2]), label="Log FFT of wave")
            axs[row_fft, col].set_title(f'FFT Channel {ch}, Label {labels[i]}')

        # Create a single legend for both "peak" and "wave" across all subplots
        fig.legend(['peak', 'wave'], loc='upper center', ncol=2)

        # Adjust layout and display the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space at the top for the legend
        plt.show()
    else:
        for i, ch in enumerate(channels):
            row = i // 4  # Determine row in 4x4 grid
            col = i % 4   # Determine column in 4x4 grid

            evt = next(run.events())
            while True:
                try:
                    w = hsd.raw.waveforms(evt)[ch][0]  # Extract the waveform data
                except:
                    evt = next(run.events())

                else:
                    break
            y = hsd.raw.peaks(evt)[ch][0][1][0]  # Extract the peak data
            # Plot on respective subplot
            axs[row, col].plot(y.astype(float)/2+float(1<<13), label="peak")
            axs[row, col].plot(w*(1<<3), label="wave")
            
            # Set subtitle indicating both the channel and label
            axs[row, col].set_title(f'Channel {ch}, Label {labels[i]}')



        # Adjust layout and display the plot
        # Create a single legend for both "peak" and "wave" across all subplots
        fig.legend(['peak', 'wave'], loc='upper center', ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space at the top for the legend
        plt.show()




    # y = hsd.raw.peaks(evt)[ch][0][1][0]
    # w =  hsd.raw.waveforms(evt)[ch][0][1][0]

    # plt.figure()
    # # plt.plot(w*(1<<3),label="wave")
    # plt.plot(y.astype(float)/2+float(1<<13),label="peak")
    # plt.legend()
    # plt.show()


    # Channels and corresponding labels
