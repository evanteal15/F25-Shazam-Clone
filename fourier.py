from scipy import signal as scipy_signal
import numpy as np
import matplotlib.pyplot as plt


def hamming_window(N):
    """ Generate the Hamming window for the size of our audio file. """
    nums = np.arange(N)
    return .54 - .46 * np.cos((2 * np.pi * nums) / (N - 1))

def pad_signal(signal, nperseg):
    """ Pad the audio to the correct length. """
    
    # sl = len(signal)
    
    pl = nperseg // 2
    
    return np.concatenate([np.zeros(pl), signal, np.zeros(pl)])

def manual_dft(audio, sr):
    """ 
    Calculate the dft for the given audio file. 
    
    Transforms audio from the time domain to the frequency domain by calculating
    the fourier coefficients for each frequency bins. The value in each bin is the
    magnitude of that frequency.
    """
    
    N = len(audio)
    X = np.zeros(N, dtype = complex)
    
    # Calculate the transform value for each frequency bin
    for k in range(N):
        
        # For discrete units we can take the summation accross all samples instead of the integral
        for n in range(N):
            
            X[k] += audio[n] * np.exp(-2j * np.pi * k * n / N)
            
    # Return the final transformed data
    return X

def manual_fft(audio, sr):
    """ 
    Fast Fourier Transform of the audio file.
    Uses the Cooley-Turkey formula.
    
    
    """
    N = len(audio)
    
    # Base Case
    if N <= 1:
        return np.array(audio)
    
    # After a sufficient number of splits, use dft
    if N <= 32:
        return manual_dft(audio, sr)
    
    # Take advatange of ft symmetry so divide the dft computation into two parts
    # This drastically reduces computation time
    even = manual_fft(audio[0::2], sr)
    odd = manual_fft(audio[1::2], sr)
    
    # factor for the odd terms
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + factor[:N // 2] * odd, even + factor[N // 2:] * odd])
    

# we will use the discrete forier transform formula since in
# digital audio files, audio is stored in discrete packets for 
# each timestep.
def manual_stft(audio, sr):
    """
    Input:
    audio: NP-Array cntain the amplitude over time for this file
    sr (Sampling rate): int for the rate 
    
    Ooutput:
    frequencies:
    """
    nperseg = 1024
    hop_length = nperseg + (nperseg // 2)
    
    noverlap = nperseg - hop_length
    
    N = len(audio)
    # output array
    X = np.zeros(N, dtype=complex)
    
    # pad the audio file
    ap = pad_signal(audio, nperseg)
    
    
    # create the hamming window for this audio file
    windower = hamming_window(nperseg)
    
    # window_norm = np.sqrt(np.sum(windower**2))
    
    # get the total size of our 
    sl = len(ap)
    
    # number of complete windows we can fit
    num_segs = (sl - nperseg) // hop_length + 1
    
    # only keep positive frequencies
    freq_bins = nperseg // 2 + 1
    
    stft_matrix = np.zeros((freq_bins, num_segs), dtype = complex)
    
    # STFT algorith,
    for i in range(num_segs):
        start = i * hop_length
        end = start + nperseg
        
        if end > sl:
            # edge case where our index is longer than our audio file - logic error
            break
        
        # get the windowed section of the audio file
        segment = ap[start:end]
        windowed_segment = segment * windower
        
        # calculate the fft of this segment
        fft_result = manual_fft(windowed_segment, sr)
        
        # normalize the result by the sampling rate
        fft_result /= (sr)
        
        # apply Nyquist function
        stft_matrix[0, i] = fft_result[0]
        stft_matrix[1:-1, i] = fft_result[1:nperseg//2] * 2  # Scale others by 2
        stft_matrix[-1, i] = fft_result[nperseg//2]
        
    # get each frequency bin
    frequencies = np.linspace(0, sr / 2, freq_bins)
        
    # account for the pad in our time calculation
    offset = -nperseg / 2 / sr
        
    # get the timesteps for each measurement
    times = np.arange(num_segs) * hop_length / sr + nperseg / 2 / sr + offset
    
    # return the frequency bins, time stamps, and frequency/maginitude measurements
    return frequencies, times, stft_matrix
    

def compare_with_scipy(audio, fs, nperseg=1024, noverlap=None):
    """
    Compare manual implementation with SciPy's STFT
    """
    if noverlap is None:
        noverlap = nperseg - (nperseg // 4)  # 75% overlap
        
    nperseg = 1024
    hop_length = nperseg + (nperseg // 2)
    
    
    print(f"STFT Parameters:")
    print(f"  nperseg: {nperseg}")
    print(f"  noverlap: {noverlap}")
    print(f"  hop_length: {hop_length}")
    print(f"  fs: {fs}")
    
    # Manual implementation
    freq_manual, time_manual, stft_manual = manual_stft(audio, fs)
    
    # SciPy implementation for comparison
    freq_scipy, time_scipy, stft_scipy = scipy_signal.stft(
        audio, fs=fs, window="hamming", nperseg=nperseg, noverlap=nperseg - hop_length,
        padded=True, return_onesided=True
    )
    
    print(f"\nResults comparison:")
    print(f"Manual - Frequencies shape: {freq_manual.shape}, Times shape: {time_manual.shape}")
    print(f"Manual - STFT shape: {stft_manual.shape}")
    print(f"SciPy  - Frequencies shape: {freq_scipy.shape}, Times shape: {time_scipy.shape}")
    print(f"SciPy  - STFT shape: {stft_scipy.shape}")
    
    # Calculate difference
    if stft_manual.shape == stft_scipy.shape:
        diff = np.abs(stft_manual - stft_scipy)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"Max difference in STFT: {max_diff:.2e}")
        print(f"Mean difference in STFT: {mean_diff:.2e}")
    
    return (freq_manual, time_manual, stft_manual), (freq_scipy, time_scipy, stft_scipy)

def plot_stft_comparison(audio, fs, nperseg=1024):
    """
    Plot STFT spectrograms for comparison
    """
    (freq_manual, time_manual, stft_manual), (freq_scipy, time_scipy, stft_scipy) = \
        compare_with_scipy(audio, fs, nperseg)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Manual implementation spectrogram
    stft_db_manual = 20 * np.log10(np.abs(stft_manual) + 1e-10)
    im1 = axes[0,0].imshow(stft_db_manual, aspect='auto', origin='lower',
                          extent=[time_manual[0], time_manual[-1], 
                                 freq_manual[0], freq_manual[-1]])
    axes[0,0].set_title('Manual STFT Implementation')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axes[0,0], label='Magnitude (dB)')
    
    # SciPy implementation spectrogram
    stft_db_scipy = 20 * np.log10(np.abs(stft_scipy) + 1e-10)
    im2 = axes[0,1].imshow(stft_db_scipy, aspect='auto', origin='lower',
                          extent=[time_scipy[0], time_scipy[-1], 
                                 freq_scipy[0], freq_scipy[-1]])
    axes[0,1].set_title('SciPy STFT Implementation')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=axes[0,1], label='Magnitude (dB)')
    
    # Difference plot
    if stft_manual.shape == stft_scipy.shape:
        diff_db = 20 * np.log10(np.abs(stft_manual - stft_scipy) + 1e-10)
        im3 = axes[1,0].imshow(diff_db, aspect='auto', origin='lower',
                              extent=[time_manual[0], time_manual[-1], 
                                     freq_manual[0], freq_manual[-1]])
        axes[1,0].set_title('Difference (Manual - SciPy)')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Frequency (Hz)')
        plt.colorbar(im3, ax=axes[1,0], label='Difference (dB)')
    
    # Original signal
    t_signal = np.arange(len(audio)) / fs
    axes[1,1].plot(t_signal, audio)
    axes[1,1].set_title('Original Audio Signal')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Amplitude')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
# test signal
# Create test signal
sr = 1000  # Sample rate
duration = 2
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# Multi-frequency signal
signal_test = (np.sin(2 * np.pi * 50 * t) +        # 50 Hz
                0.5 * np.sin(2 * np.pi * 120 * t) +  # 120 Hz
                0.3 * np.sin(2 * np.pi * 200 * t))   # 200 Hz

# Add some time-varying component
signal_test += 0.2 * np.sin(2 * np.pi * (300 + 100 * t) * t)  # Frequency sweep

plot_stft_comparison(signal_test, sr)
