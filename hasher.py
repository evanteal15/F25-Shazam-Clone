import os

import librosa
import numpy as np
from scipy import signal
from collections import defaultdict
import matplotlib.pyplot as plt


from DBcontrol import connect_to_db, retrieve_song, \
    retrieve_song_ids, retrieve_hashes, add_hashes, \
    create_hash_index, create_tables, add_songs

def convert_to_decibel(magnitude: np.array):
    """
    returns spectrogram's amplitude at each freq/time bin, measured in dB
    """
    return 20*np.log10(magnitude + 1e-6)

def visualize_map(audio_path, show_overlay=True, apply_filter=False, prominence=0, distance=200, height=None, width=None):
    audio, sr = preprocess_audio(audio_path)
    frequencies, times, magnitudes = compute_fft(audio, sr)
    magnitudes = convert_to_decibel(magnitudes)

    # spectrogram
    plt.figure(figsize=(24, 6))
    plt.pcolormesh(times, frequencies, magnitudes, shading='gouraud', cmap='inferno')
    plt.colorbar(label="Magnitude (dB)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("Spectrogram with Constellation Map")

    # overlay constellation peaks
    if show_overlay:
        constellation_map = find_peaks(frequencies, times, magnitudes, prominence=prominence, distance=distance, width=width)
        if apply_filter:
            #constellation_map = filter_peaks(constellation_map)
            constellation_map = filter_peaks(frequencies, times, magnitudes)

        peak_times = [times[t] for t, f in constellation_map]
        peak_freqs = [f for t, f in constellation_map]
        plt.scatter(peak_times, peak_freqs, color='cyan', s=10, marker='x', label="Peaks")

    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_fft(audio, sr):
    #window_len_s = 0.5
    #window_samples = int(window_len_s * sr)
    window_size = 1024

    pad = window_size - (audio.size % window_size)

    
    # pad the audio signal to make it a multiple of window_size
    audio = np.pad(audio, (0, pad), mode='constant')
    #audio_duration_s = len(audio) / sr

    
    ##########################
    # params of signal.stft():
    ##########################
    # window: used to avoid the spread of strong true frequencies to their neighbors
    # https://en.wikipedia.org/wiki/Spectral_leakage
    # https://dsp.stackexchange.com/questions/63405/spectral-leakage-in-laymans-terms
    # signal.windows.hann  # default
    # signal.windows.hamming
    # nperseg: length of each segment
    # noverlap: number of points to overlap between segments
    # nfft: length of the FFT used, if a zero padded FFT is desired
    #https://docs.scipy.org/doc/scipy/tutorial/signal.html#comparison-with-legacy-implementation
    ##########################

    # do a fast fourier transform on the audio data to get the frequency spectrum
    # turns raw audio data into a "spectrogram"
    # fft is used for its O(nlog(n)) time complexity, vs dft's O(n^2) complexity
    # TODO: this could be done manually for understanding, then later on in the project
    #       signal.stft could be introduced for its speed and versatility
    frequencies, times, stft = signal.stft(audio, sr, 
                                           window="hamming",
                                           nperseg=window_size,
                                           #noverlap=window_size // 2,
                                           #nfft=window_size,
                                           #return_onesided=True,
                                           )

    # stft is a 2D complex array of shape (len(frequencies), len(times))
    # complex values encode amplitude and phase offset of each sine wave component
    # We're interested in just the magnitude / strength of the frequency components
    # (phase offset information is useful though for time-stretching/pitch-shifting, or 
    # reconstructing signal via inverse STFT)
    magnitude = np.abs(stft)

    # frequencies is an array of frequency bin centers (in Hz)
    # times is an array of time bin centers (in seconds)
    # magnitude is a 2D real array of shape (len(frequencies), len(times))
    return frequencies, times, magnitude
    
def find_peaks(frequencies, times, stft, **kwargs):
    num_peaks = 10
    constellation_map = []
    
    # iterate through each time window in the spectrogram
    for time_idx, window in enumerate(stft.T):
        spectrum = np.abs(window)
        
        ## find the peaks in the spectrum
        #peaks, props = signal.find_peaks(spectrum, height=0.1, distance=5)
        ## sort the peaks by their heights
        #sorted_peaks = sorted(peaks, key=lambda x: spectrum[x], reverse=True)
        ## keep only the top N peaks
        #top_peaks = sorted_peaks[:num_peaks]
        ## add the top peaks to the constellation map
        #constellation_map.append(top_peaks)

        ## find the peaks in the spectrum
        # height      Minimum power threshold (in dB). Higher = fewer peaks.
        # distance    Minimum frequency bin spacing between peaks. Prevents clustering.
        # prominence  How much a peak stands out from its surroundings. Helps focus on strong peaks.
        # width       Can ensure peaks are broad enough to be meaningful. Optional.
        peaks, props = signal.find_peaks(spectrum, height=1, **kwargs)
        n_peaks = min(num_peaks, len(peaks))
        #if time_idx > 40:
            #print("peaks")
            #print(peaks)
            #print(len(peaks))
            #print("heights")
            #print(props["peak_heights"])
            #print(len(props["peak_heights"]))
            #exit(0)

        # select top n peaks, ranked by prominence
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])

    # assuming window_size = 1024
    #bands = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 512)]


    return constellation_map

def create_constellation_map(audio, sr) -> list[list[int]]:
    frequencies, times, stft = compute_fft(audio, sr)
    constellation_map = find_peaks(frequencies, times, stft)
    return constellation_map


def filter_peaks(frequencies: np.ndarray, times: np.ndarray, magnitude: np.ndarray) -> list[tuple[int, float]]:
    """
    For each time index, keep the highest magnitude peak in each frequency band.

    Returns a list of (time_idx, frequency) pairs.
    """
    # logarithmic frequency bands since lower frequencies are amplified and will usually dominate
    # assuming window_size = 1024
    bands = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 512)]
    peaks = []

    n_times = magnitude.shape[1]
    for time_idx in range(n_times):
        spectrum = magnitude[:, time_idx]
        for band_start, band_end in bands:
            band = spectrum[band_start:band_end]
            if band.size == 0:
                continue
            max_idx_in_band = np.argmax(band)
            max_mag = band[max_idx_in_band]
            if max_mag > 0:
                freq_idx = band_start + max_idx_in_band
                freq = frequencies[freq_idx]
                peaks.append((time_idx, freq))
    return peaks

def create_address(anchor: tuple[int, int], target: tuple[int, int], sr: int) -> int:
    # get relevant information from the anchor and target points
    anchor_freq = anchor[1]
    target_freq = target[1]
    deltaT = target[0] - anchor[0]

    ##############################################
    # Creating a 32 bit hash f1:f2:dt (2002 paper)
    ##############################################

    # MP3 files are downloaded at 48 kHz sampling rate
    # preprocess_audio() resamples to 11 kHz
    # => max frequency is sr/2 = 5512.5
    # (by the Nyquist–Shannon sampling theorem)
    # use a value slightly higher than this (to avoid overflow I think?):
    max_frequency = np.ceil(sr / 2) + 10

    # transform frequencies to fit in 10 bits (0-1023)
    # results in some loss of information (int -> smaller float -> int)
    n_bits = 10
    anchor_freq = (anchor_freq / max_frequency) * (2 ** n_bits)
    target_freq = (target_freq / max_frequency) * (2 ** n_bits)
    
    # bit shifting to obtain 32 bit hash
    # int(anchor_freq)         occupies bits 0-9,    anchor_freq <= 1023
    # int(target_freq) << 10   occupies bits 10-19,  target_freq <= 1023
    # int(deltaT) << 20        occupies bits 20-31,  deltaT <= 4095
    hash = int(anchor_freq) | (int(target_freq) << 10) | (int(deltaT) << 20)
    return hash

def create_hashes(peaks, song_id: int = None, sr: int = None, fanout=10):
    """
    fanout:
        specify the fan-out factor used for determining the target zone

        = number of timesteps forward from anchor to use for target points
    """
    fingerprints = {}
    
    # iterate through each anchor point in the constellation map
    for i, anchor in enumerate(peaks):
        # iterate through each point in target zone for that anchor point
        for j in range(i+1, len(peaks)):
            # select targets from a zone in front of the anchor point
            target = peaks[j]
            time_diff = target[0] - anchor[0]
            # TODO: use freq diff as well
            if time_diff <= 1:
                continue
            if time_diff > fanout:
                # constellation points are sorted by time
                # => no need to check more potential targets
                break
            
            address = create_address(anchor, target, sr)
            anchorT = anchor[0]

            fingerprints[address] = (anchorT, song_id)
            
    return fingerprints

def preprocess_audio(audio_path, sr = 11_025):
    """
    returns `(audio, sr)`
    """
    # resample to 11 kHz (11,025 Hz)
    # TODO: analyze optimal resampling rate
    #       based on the freq distribution (max freq)
    #       of tracks in dataset, eyeballing 
    #       approx max 15 kHz? (=> sr = 30 kHz)
    #
    #       also consider that musically relevent
    #       frequencies will more than likely fall 
    #       within 20-5512 Hz range
    #
    #       Fs == sr == sample_rate
    # uses a low pass filter to filter the higher frequencies to avoid aliasing (Nyquist-Shannon)
    # then takes sequential samples of size 4 and computes average of each sample
    audio, sr = librosa.load(audio_path, sr=sr)

    ## equivalent to:
    #max_freq_cutoff = 5512 # Hz
    ## Calculate the new sampling rate (at least twice the cutoff frequency)
    #new_sr = max_freq_cutoff * 2
    #new_sr += 1  # = 11025 to match librosa documentation / Chigozirim vid
    ## Resample the audio, which implicitly applies a low-pass filter
    #y_filtered = librosa.resample(y=audio, orig_sr=Fs, target_sr=new_sr, res_type='kaiser_best')

    return audio, sr

def compute_source_hashes(song_ids: list[int] = None):
    if song_ids is None:
        song_ids = retrieve_song_ids()
    
    for song_id in song_ids:
        print(f"{song_id:03} ================================================")
        song = retrieve_song(song_id)
        print(f"{song['title']} by {song['artist']}")
        #duration_s = song["duration_s"]
        audio_path = song["audio_path"]

        audio, sr = preprocess_audio(audio_path)
        constellation = create_constellation_map(audio, sr)
        hashes = create_hashes(constellation, song_id, sr)
        add_hashes(hashes)
    
    create_hash_index()


def score_hashes(hashes: dict[int, tuple[int, int]]) -> list[tuple[int, int]]:
    # TODO: refactor and comment
    con, cur = connect_to_db()
    matches_per_song = defaultdict(list)
    for address, (sampleT, _) in hashes.items():
        matching_hashes = retrieve_hashes(address, cur)
        if matching_hashes is not None:
            for _, sourceT, song_id in matching_hashes:
                matches_per_song[song_id].append((address, sampleT, sourceT))
                
    # time coherience score
    scores = {}
    for song_id, matches in matches_per_song.items():
        song_scores_by_offset = defaultdict(int)
        for address, sampleT, sourceT in matches:
            deltaT = sourceT - sampleT
            song_scores_by_offset[deltaT] += 1

        max_score = (0, 0)
        for offset, score in song_scores_by_offset.items():
            if score > max_score[1]:
                max_score = (offset, score)
        scores[song_id] = max_score
    
    scores = list(sorted(scores.items(), key=lambda x: x[1][1], reverse=True)) 
    
    con.close()
    return scores

def init_db(tracks_dir: str = None, n_songs: int = None):
    """
    tracks_dir:
        Path to the mp3 dataset downloaded by `musicdl`. 
        Default is to look for a folder or zip archive matching the pattern "`tracks*`"
    n_songs:
        Take a sample from the top of the tracks dataset. Default is to read all songs

    """
    create_tables()
    add_songs(tracks_dir, n_songs)
    compute_source_hashes()

def recognize_music(sample_wav_path: str) -> list[tuple[int, int]]:
    """
    returns sorted list of `(song_id, score)` tuples
    
    Access top prediction with `scores[0][0]`

    ```
    init_db(tracks_dir = "tracks-2025-07-22", n_songs=5)

    # record from microphone
    sample_wav_path = record_audio(n_seconds = 5)

    song_id = recognize_music(sample_wav_path)[0][0]
    """
    sample, sr = preprocess_audio(sample_wav_path)
    os.remove(sample_wav_path)
    constellation = create_constellation_map(sample, sr)
    hashes = create_hashes(constellation, None, sr)
    scores = score_hashes(hashes)
    return scores