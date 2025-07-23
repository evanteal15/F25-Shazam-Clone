import librosa
import numpy as np
from scipy import signal

from DBcontrol import retrieve_song, add_hash

def create_constellation_map(audio, FS):
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

def filter_peaks(spectrogram, audioDuration: float):
    # logarithmic frequency bands since lower frequencies are amplified and will usually dominate
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

def create_address(anchor: tuple[int, int], target: tuple[int, int]) -> int:
    # get relevant information from the anchor and target points
    anchor_freq = anchor[0]
    target_freq = target[0]
    deltaT = target[1] - anchor[1]
    
    # 32 bit hash using bit shifting
    #hash = int(anchor_freq) | (int(target_freq) << 10) | (int(deltaT) << 20)
    hash = int(anchor_freq) | (int(target_freq) << 10) | (int(deltaT) << 20)
    # int(anchor_freq)         occupies bits 0-9,    anchor_freq <= 1023
    # int(target_freq) << 10   occupies bits 10-19,  target_freq <= 1023
    # int(deltaT) << 20        occupies bits 20-31,  deltaT <= 4095
    return hash

def hash_constellation(peaks, song_id):
    fingerprints = {}
    
    # iterate through each anchor point in the constellation map
    for i, peak in enumerate(peaks):
        # iterate through each target point for that anchor point
        for j in range(i+1, len(peaks)):
            target = peaks[j]
            
            address = create_address(peak, target)
            anchorT = peak[1]

            fingerprints[address] = (anchorT, song_id)
            
    return fingerprints


def create_hashes(constellation_map, song_id: int):
    song = retrieve_song(song_id)
    duration_s = song["duration_s"]
    audio_path = song["audio_path"]


    Fs, audio = librosa.load(audio_path)
    constellation = create_constellation_map(audio, Fs)
<<<<<<< HEAD
    #peaks = filter_peaks(spectrogram=None, audioDuration=duration_s)
=======
    peaks = filter_peaks(spectrogram=constellation, audioDuration=duration_s)
>>>>>>> e2a75a176256c808dcd5ba6e817e41e1bf9d7b3d
    hashes = hash_constellation(peaks, song_id)
    for address, (anchorT, _) in hashes.items():
        add_hash(address, anchorT, song_id)

def score_hash():
    pass

def score_hashes(hashes: dict):
    num_matches = {}
    for address, (anchorT, _) in hashes.items():
        pass