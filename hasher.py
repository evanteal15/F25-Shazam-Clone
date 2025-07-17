import librosa
import numpy as np
from scipy import signal

def create_constellation_map(audio , FS):
    window_len = 0.5
    window_samples = int(FS * window_len)
    
    num_peaks = 20

    pad = window_samples - (audio.size % window_samples)
    
    # pad the audio signal to make it a multiple of window_samples
    audio = np.pad(audio, (0, pad), mode='constant')
    
    # do a fast fourier transform on the audio data to get the frequency spectrum
    # turns raw audio data into a "spectrogram"
    frequencies, times, stft = signal.stft(audio, FS, nperseg=window_samples)
    
    constellation_map = []
    
    # iterate through each time window in the spectrogram
    for idx, window in enumerate(stft.T):
        spectrum = np.abs(window)
        
        # find the peaks in the spectrum
        peaks, props = signal.find_peaks(spectrum, height=0.1, distance=5)

        # sort the peaks by their heights
        sorted_peaks = sorted(peaks, key=lambda x: spectrum[x], reverse=True)

        # keep only the top N peaks
        top_peaks = sorted_peaks[:num_peaks]

        # add the top peaks to the constellation map
        constellation_map.append(top_peaks)

    return constellation_map

def filter_peaks(spectrogram, audioDuration):
    # logarithmic frequency bans since lower frequencies are amplified and will usually dominate
    bands = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 512)]
    
    peaks = []
    binDuration = audioDuration / len(spectrogram)
    for i, bin in enumerate(spectrogram):
        maxFreqs = []
        freqInds = []
        
        binBandMax = []
        # get the maximum frequency in each band
        for band in bands:
            max = (0, 0)
            for j, freq in enumerate(bin[band[0]:band[1]]):
                if freq > max[0]:
                    max = (freq, j + band[0])
            binBandMax.append(max)
        
        maxFreqs = [max[0] for max in binBandMax]
        freqInds = [max[1] for max in binBandMax]
       
       
        avgFreqs = np.mean(maxFreqs)
        
        # add peaks to the list if they are above the average
        for j, freq in enumerate(maxFreqs):
            if freq > avgFreqs:
                # calculate the absolute time of the peak
                peak_time = (i * binDuration) + (freqInds[j] * (binDuration / len(bin)))
                peaks.append((freq, peak_time))

    return peaks

def hash_constellation(peaks, songID):
    fingerprints = {}
    
    # iterate through each anchor point in the constellation map
    for i, peak in enumerate(peaks):
        # iterate through each target point for that anchor point
        for j in range(i+1, len(peaks)):
            target = peaks[j]
            
            address = create_address(peak, target)
            anchorT = peak[1]

            fingerprints[address] = (anchorT, songID)
            
    return fingerprints


def create_address(anchor, target):
    # get relevant information from the anchor and target points
    anchor_freq = anchor[0]
    target_freq = target[0]
    deltaT = target[1] - anchor[1]
    
    return f"{anchor_freq}_{target_freq}_{deltaT}"

def score_hash():
    pass