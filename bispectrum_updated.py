#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
from scipy.signal import convolve2d
import scipy
from scipy.signal import stft
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
from scipy.signal import butter, filtfilt, welch

np.set_printoptions(linewidth=120)


def bispectrumi(y, nlag=None, nsamp=None, overlap=None,
                flag='biased', nfft=None, wind=None):
    """
    Parameters:
      y       - data vector or time-series
      nlag    - number of lags to compute [must be specified]
      segsamp - samples per segment    [default: row dimension of y]
      overlap - percentage overlap     [default = 0]
      flag    - 'biased' or 'unbiased' [default is 'unbiased']
      nfft    - FFT length to use      [default = 128]
      wind    - window function to apply:
                if wind=0, the Parzen window is applied (default)
                otherwise the hexagonal window with unity values is applied.

    Output:
      Bspec   - estimated bispectrum  it is an nfft x nfft array
                with origin at the center, and axes pointing down and to the right
      waxis   - frequency-domain axis associated with the bispectrum.
              - the i-th row (or column) of Bspec corresponds to f1 (or f2)
                value of waxis(i).
    """

    (ly, nrecs) = y.shape
    if ly == 1:
        y = y.reshape(1, -1)
        ly = nrecs
        nrecs = 1

    if not overlap:
        overlap = 0
    overlap = min(99, max(overlap, 0))
    if nrecs > 1:
        overlap = 0
    if not nsamp:
        nsamp = ly
    if nsamp > ly or nsamp <= 0:
        nsamp = ly
    if not 'flag':
        flag = 'biased'
    if not nfft:
        nfft = 128
    if not wind:
        wind = 0

    nlag = min(nlag, nsamp-1)
    if nfft < 2*nlag+1:
        nfft = 2 ^ nextpow2(nsamp)

    # create the lag window
    Bspec = np.zeros([nfft, nfft])
    if wind == 0:
        indx = np.array([range(1, nlag+1)]).T
        window = make_arr(
            (1, np.sin(np.pi*indx/nlag) / (np.pi*indx/nlag)), axis=0)
    else:
        window = np.ones([nlag+1, 1])
    window = make_arr((window, np.zeros([nlag, 1])), axis=0)

    # cumulants in non-redundant region
    overlap = np.fix(nsamp * overlap / 100)
    nadvance = nsamp - overlap
    nrecord = np.fix((ly*nrecs - overlap) / nadvance)

    c3 = np.zeros([nlag+1, nlag+1])
    ind = np.arange(nsamp)
    y = y.ravel(order='F')

    s = 0
    for k in range(int(nrecord)):
        x = y[ind].ravel(order='F')
        x = x - np.mean(x)
        ind = ind + int(nadvance)

        for j in range(nlag+1):
            z = x[range(nsamp-j)] * x[range(j, nsamp)]
            for i in range(j, nlag+1):
                Sum = np.dot(z[range(nsamp-i)].T, x[range(i, nsamp)])
                if flag == 'biased':
                    Sum = Sum/nsamp
                else:
                    Sum = Sum / (nsamp-i)
                c3[i, j] = c3[i, j] + Sum

    c3 = c3 / nrecord

    # cumulants elsewhere by symmetry
    c3 = c3 + np.tril(c3, -1).T   # complete I quadrant
    c31 = c3[1:nlag+1, 1:nlag+1]
    c32 = np.zeros([nlag, nlag])
    c33 = np.zeros([nlag, nlag])
    c34 = np.zeros([nlag, nlag])
    for i in range(nlag):
        x = c31[i:nlag, i]
        c32[nlag-1-i, 0:nlag-i] = x.T
        c34[0:nlag-i, nlag-1-i] = x
        if i+1 < nlag:
            x = np.flipud(x[1:len(x)])
            c33 = c33 + np.diag(x, i+1) + np.diag(x, -(i+1))

    c33 = c33 + np.diag(c3[0, nlag:0:-1])

    cmat = make_arr(
        (make_arr((c33, c32, np.zeros([nlag, 1])), axis=1),
         make_arr((make_arr((c34, np.zeros([1, nlag])), axis=0), c3), axis=1)),
        axis=0
    )

    # apply lag-domain window
    wcmat = cmat
    if wind != -1:
        indx = np.arange(-1*nlag, nlag+1).T
        window = window.reshape(-1, 1)
        for k in range(-nlag, nlag+1):
            wcmat[:, k+nlag] = (cmat[:, k+nlag].reshape(-1, 1) *
                                window[abs(indx-k)] *
                                window[abs(indx)] *
                                window[abs(k)]).reshape(-1,)

    # compute 2d-fft, and shift and rotate for proper orientation
    Bspec = np.fft.fft2(wcmat, (nfft, nfft))
    Bspec = np.fft.fftshift(Bspec)  # axes d and r; orig at ctr

    if nfft % 2 == 0:
        waxis = np.transpose(np.arange(-1*nfft/2, nfft/2)) / nfft
    else:
        waxis = np.transpose(np.arange(-1*(nfft-1)/2, (nfft-1)/2+1)) / nfft

    # cont = plt.contourf(waxis, waxis, abs(Bspec), 100, cmap=plt.cm.Spectral_r)
    # plt.colorbar(cont)
    # plt.title('Bispectrum estimated via the indirect method')
    # plt.xlabel('f1')
    # plt.ylabel('f2')
    # plt.show()

    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')

    # # create a meshgrid for 3D plot
    # X, Y = np.meshgrid(waxis, waxis)

    # # 3D surface plot
    # surf = ax.plot_surface(X, Y, np.abs(Bspec), cmap=plt.cm.Spectral_r, linewidth=0, antialiased=False)

    # # Add a color bar which maps values to colors
    # fig.colorbar(surf)

    # plt.title('Bispectrum estimated via the indirect method')
    # ax.set_xlabel('f1')
    # ax.set_ylabel('f2')
    # ax.set_zlabel('Magnitude')
    # plt.show()

    return (Bspec, waxis)





## Helper function
def nextpow2(num):
  '''
  Returns the next highest power of 2 from the given value.
  Example
  -------
  >>>nextpow2(1000)
  1024
  >>nextpow2(1024)
  2048

  Taken from: https://github.com/alaiacano/frfft/blob/master/frfft.py
  '''

  npow = 2
  while npow <= num:
      npow = npow * 2
  return npow


def flat_eq(x, y):
  """
  Emulate MATLAB's assignment of the form
  x(:) = y
  """
  z = x.reshape(1, -1)
  z = y
  return z.reshape(x.shape)


def make_arr(arrs, axis=0):
  """
  Create arrays like MATLAB does
  python                                 MATLAB
  make_arr((4, range(1,10)), axis=0) => [4; 1:9]
  """
  a = []
  ctr = 0
  for x in arrs:
    if len(np.shape(x)) == 0:
      a.append(np.array([[x]]))
    elif len(np.shape(x)) == 1:
      a.append(np.array([x]))
    else:
      a.append(x)
    ctr += 1
  return np.concatenate(a, axis)


def shape(o, n):
  """
  Behave like MATLAB's shape
  """
  s = o.shape
  if len(s) < n:
    x = tuple(np.ones(n-len(s)))
    return s + x
  else:
    return s


def here(f=__file__):
  """
  This script's directory
  """
  return os.path.dirname(os.path.realpath(f))


def process_eeg_channel_power(eeg_data, lowcut, highcut, sampling_rate, time):
    """
    Process the specified EEG channel: apply a band-pass filter, slice into 10-second time windows,
    and calculate the band power for each time window.

    :param eeg_data: 2D numpy array of shape (num_channels, num_samples)
    :param channel_index: Index of the EEG channel to process
    :param lowcut: Lower frequency limit of the band-pass filter in Hz
    :param highcut: Upper frequency limit of the band-pass filter in Hz
    :param sampling_rate: Sampling rate of the EEG data in Hz
    :return: A list containing the band power for each 10-second time window
    """
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def calculate_band_power(eeg_data, sampling_rate, lowcut, highcut):
        freqs, psd = welch(eeg_data, sampling_rate, nperseg=sampling_rate)
        band_power = np.trapz(psd[(freqs >= lowcut) & (freqs <= highcut)], dx=np.diff(freqs)[0])
        return band_power

    # Apply the band-pass filter to the specified channel
    channel_data = eeg_data
    b, a = butter_bandpass(lowcut, highcut, sampling_rate)
    filtered_channel_data = filtfilt(b, a, channel_data)

    # Calculate the band power for each 10-second time window
    window_size = time * sampling_rate
    num_windows = filtered_channel_data.shape[0] // window_size
    band_powers = []

    for i in range(num_windows):
        window_data = filtered_channel_data[i * window_size:(i + 1) * window_size]
        band_power = calculate_band_power(window_data, sampling_rate, lowcut, highcut)
        band_powers.append(band_power)

    return band_powers

def test():
    qpc = sio.loadmat(here(__file__) + '/demo/qpc.mat')
    dbic = bispectrumi(qpc['zmat'],  21, 64, 0, 'unbiased', 128, 1)

def test2():
  mat = scipy.io.loadmat("/Users/wenxuan/Documents/Github/phi_calculation/sub_0010-mr_0009-ecr_echo1_EEG_pp.mat")
  print(mat["EEG"]["data"][0][0].shape)
  EEG = mat["EEG"]["data"][0][0]
  eeg_signal = EEG[9]
  eeg_signal = eeg_signal.reshape((1, len(eeg_signal)))
  # important parameters
  bispectrumi(eeg_signal, 1, 1000, 0, 'unbiased', 1000, 1)


def test3():
    mat = scipy.io.loadmat("/Users/wenxuan/Documents/Github/phi_calculation/sub_0010-mr_0009-ecr_echo1_EEG_pp.mat")
    print(mat["EEG"]["data"][0][0].shape)
    EEG = mat["EEG"]["data"][0][0]
    eeg_signal = EEG[9]
    # eeg_signal = eeg_signal.reshape((1, len(eeg_signal)))
    sampling_rate = 250  # Update this to your actual sampling rate
    window_size = sampling_rate * 2  # 1 second windows
    print(window_size)
    num_windows = len(eeg_signal) // window_size
    print(num_windows)

    images = []
    images2 = []
    
    Bspecs = []

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        windowed_signal = eeg_signal[start:end]
        windowed_signal = windowed_signal.reshape((1, len(windowed_signal)))
        Bspec, waxis = bispectrumi(windowed_signal, 1, 1000, 0, 'unbiased', 1000, 1)
        Bspecs.append(abs(Bspec))

    global_vmin = np.min(Bspecs)
    global_vmax = np.max(Bspecs)/2
    print(global_vmin, global_vmax)
    norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)

    for i in range(num_windows):
        Bspec = Bspecs[i]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(waxis, waxis)
        surf = ax.plot_surface(X, Y, np.abs(Bspec), cmap=plt.cm.Spectral_r, norm=norm, linewidth=0, antialiased=False)
        fig.colorbar(surf)
        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        ax.set_zlabel('Magnitude')
        fig_name = f"3d_figure_{i}.png"
        plt.title(f'3D: Bispectrum estimated via the indirect method (second: {i*2})')
        plt.savefig(fig_name)
        images.append(imageio.imread(fig_name))
        plt.close(fig)

  
        fig, ax = plt.subplots()
        cont = ax.contourf(waxis, waxis, abs(Bspec), 100, cmap=plt.cm.Spectral_r, norm=norm)

        # Formatting
        plt.colorbar(cont)
        plt.title(f'Bispectrum estimated via the indirect method (second: {i*2})')
        plt.xlabel('f1')
        plt.ylabel('f2')
        fig_name2 = f"2d_figure_{i}.png"
        plt.savefig(fig_name2)
        images2.append(imageio.imread(fig_name2))
        plt.close(fig)

    # Create gif
    imageio.mimsave('output.gif', images)
    imageio.mimsave('output2.gif', images2)


def animate_bispectrum(y, window_size):
    images = []
    # Find out the max and min across all windows for color range
    vmin = 0
    vmax = 10000
    num_windows = len(y) // window_size
    # Loop through each time window
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        # Select the data for this time window
        y_window = y[start : end]
        y_window = y_window.reshape((1, len(y_window)))
        # Compute the bispectrum
        Bspec, waxis = bispectrumi(y_window, 1, 1000, 0, 'unbiased', 1000, 1)

        # Plot the bispectrum
        fig, ax = plt.subplots()
        cont = ax.contourf(waxis, waxis, abs(Bspec), 100, cmap=plt.cm.Spectral_r, vmin=vmin, vmax=vmax)

        # Formatting
        plt.colorbar(cont)
        plt.title(f'Bispectrum estimated via the indirect method (window {i})')
        plt.xlabel('f1')
        plt.ylabel('f2')
        print(i)

        # Add the image to the list
        images.append(image)

        plt.close(fig)

    # Save the images as a GIF
    imageio.mimsave('bispectrum.gif', images, fps=20)

from scipy.signal import butter, filtfilt

def bandpass_filter(eeg_data, lowcut, highcut, sampling_rate, order=4):
    """
    Apply a Butterworth band-pass filter to the given EEG data.

    :param eeg_data: 2D numpy array of shape (num_channels, num_samples)
    :param lowcut: Lower frequency limit of the band-pass filter in Hz
    :param highcut: Upper frequency limit of the band-pass filter in Hz
    :param sampling_rate: Sampling rate of the EEG data in Hz
    :param order: Order of the Butterworth filter (default is 4)
    :return: 2D numpy array of filtered EEG data with the same shape as input eeg_data
    """
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    num_channels, num_samples = eeg_data.shape
    filtered_eeg_data = np.zeros((num_channels, num_samples))

    b, a = butter_bandpass(lowcut, highcut, sampling_rate, order)

    for channel in range(num_channels):
        filtered_eeg_data[channel] = filtfilt(b, a, eeg_data[channel])

    return filtered_eeg_data

def SFS():
  mat = scipy.io.loadmat("/Users/wenxuan/Documents/Github/phi_calculation/sub_0010-mr_0009-ecr_echo1_EEG_pp.mat")
  print(mat["EEG"]["data"][0][0].shape)
  EEG = mat["EEG"]["data"][0][0]
  # eeg_signal = EEG[9]
  eeg_signal = EEG

  fs = 250

  band_powers_wide = bandpass_filter(eeg_signal, 0.5, 14, fs)
  band_powers_narrow = bandpass_filter(eeg_signal, 8, 14, fs)

  band_powers_wide = band_powers_wide[9]
  band_powers_narrow = band_powers_narrow[9]

  sampling_rate = 250  # Update this to your actual sampling rate
  window_size = sampling_rate * 10  # 10 second windows
  num_windows = len(band_powers_wide) // window_size

  Bspecs_sum = []

  for i in range(num_windows):
      start = i * window_size
      end = start + window_size
      windowed_signal_wide = band_powers_wide[start:end]
      windowed_signal_narrow =  band_powers_narrow[start:end]
      windowed_signal_wide = windowed_signal_wide.reshape((1, len(windowed_signal_wide)))
      windowed_signal_narrow = windowed_signal_narrow.reshape((1, len(windowed_signal_narrow)))

      Bspec1, waxis = bispectrumi(windowed_signal_wide, 1, 1000, 0, 'unbiased', 1000, 1)
      Bspec2, waxis = bispectrumi(windowed_signal_narrow, 1, 1000, 0, 'unbiased', 1000, 1)
      Bspecs_sum.append(np.log10(np.sum(Bspec1)/(np.sum(Bspec2))))

  return num_windows, Bspecs_sum
  
  


def logscaleFFT():
  mat = scipy.io.loadmat("/Users/wenxuan/Documents/Github/phi_calculation/sub_0010-mr_0009-ecr_echo1_EEG_pp.mat")
  print(mat["EEG"]["data"][0][0].shape)
  EEG = mat["EEG"]["data"][0][0]
  eeg_signal = EEG[9]

  time_windows = 20
  lowcut, highcut = 0.5, 4
  fs = 250

  band_powers_delta = process_eeg_channel_power(eeg_signal, 0.5, 4, fs, time_windows)
  band_powers_alpha = process_eeg_channel_power(eeg_signal, 8, 12, fs, time_windows)

  frequencies, times, Zxx = stft(eeg_signal, fs, nperseg=1000)  # Adjust nperseg as needed

  f, t, Sxx = scipy.signal.spectrogram(eeg_signal, fs, nperseg=1000)
  
  # Add a subplot for the band power
  fig, ax1 = plt.subplots(figsize=(10, 6))
  ax2 = ax1.twinx()
  cax = ax1.pcolormesh(t, f, np.log10(Sxx), cmap='jet', vmin = -8, vmax = 4)
  fig.colorbar(cax, ax=ax1, label='Magnitude')
  ax1.set_yscale('symlog')
  ax1.set_ylabel('Frequency [Hz]')
  ax1.set_xlabel('Time [sec]')
  ax1.set_title('Spectrogram with Log Frequency Scale')
  ax1.set_xlim([0, max(t)])  # Set the x-axis to start from 0
  
  # Plot the band power
  time_windows_delta = np.arange(len(band_powers_delta)) * time_windows
  time_windows_alpha = np.arange(len(band_powers_alpha)) * time_windows
  ax2.plot(time_windows_delta, band_powers_delta, 'k-', label='Band Power Delta')
  ax2.plot(time_windows_alpha, band_powers_alpha, 'b-', label='Band Power Alpha')
  ax2.plot()
  ax2.set_ylabel('Band Power', color='k')
  ax2.tick_params('y', colors='k')
  ax2.legend(loc="upper right")
  
  plt.savefig('log_spec.png')
  plt.show()

  # Calculate spectrogram
  spectrogram = np.abs(Zxx)
  frequencies += 1e-10
  spectrogram = np.log10(spectrogram)

  # first plot with linear y-axis
  plt.figure(figsize=(10, 6))
  # plt.ylim([0, 10])
  plt.jet()
  plt.pcolormesh(times, frequencies, spectrogram)
  plt.title('Spectrogram with Linear Frequency Scale')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.colorbar(label='Magnitude')
  plt.savefig('linear_spec.png')
  plt.show()

  # subtract the mean spectrum across time from the spectrogram
  spectrogram -= np.mean(spectrogram, axis=1, keepdims=True)

  # second plot with linear y-axis after subtracting the mean
  plt.figure(figsize=(10, 6))
  plt.ylim([0, 30])
  plt.pcolormesh(times, frequencies, spectrogram)
  plt.jet()
  plt.title('Linear Frequency Scale: Spectrogram with Mean Spectrum Subtracted')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.colorbar(label='Magnitude')
  plt.savefig('linear_spec_subtracted.png')
  plt.show()


if __name__ == '__main__':
  # logscaleFFT()
  # test3()
  x, SFS_values = SFS()
  # Modify the tick values
  plt.plot(SFS_values)
  plt.title('SyncFastSlow')
  plt.ylabel('SFS')
  plt.xlabel('Time [10 sec]')
  plt.savefig('log_SFS_changes.png')
  plt.show()

  

