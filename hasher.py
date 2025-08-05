import os

import librosa
import numpy as np
from scipy import signal
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.graph_objects as go


from DBcontrol import connect_to_db, retrieve_song, \
    retrieve_song_ids, retrieve_hashes, add_hashes, \
    create_hash_index, create_tables, add_songs

def convert_to_decibel(magnitude: np.array):
    """
    returns spectrogram's amplitude at each freq/time bin, measured in dB
    """
    return 20*np.log10(magnitude + 1e-6)

def visualize_map_interactive(audio_path):
    audio, sr = preprocess_audio(audio_path)
    frequencies, times, magnitudes = compute_fft(audio, sr)
    magnitudes = convert_to_decibel(magnitudes)
    print(magnitudes.shape)

    constellation_map = find_peaks(frequencies, times, magnitudes)
    peak_times = [times[t] for t, f in constellation_map]
    peak_freqs = [f for t, f in constellation_map]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=magnitudes,
        x=times,
        y=frequencies,
        colorscale='Inferno',
        colorbar=dict(title='Magnitude (dB)'),
        zsmooth='best',
        name="Spectrogram"
    ))

    fig.add_trace(go.Scatter(
        x=peak_times,
        y=peak_freqs,
        mode='markers',
        marker=dict(size=7, color='white', symbol='square-open'),
        name='Constellation Map',
        visible=True
    ))

    # overlay constellation peaks with a toggleable checkbox
    fig.update_layout(
        title='Spectrogram with Constellation Map',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        yaxis=dict(
            range=[frequencies.min(), frequencies.max()],
            autorange=False  # Prevent auto-padding when scatter appears
        ),
        xaxis=dict(
            range=[times.min(), times.max()],
            autorange=False  # Optional: lock X-axis too
        ),
        margin=dict(t=40, b=40, l=60, r=40),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(label="Show Peaks",
                         method="update",
                         args=[{"visible": [True, True]}]),
                    dict(label="Hide Peaks",
                         method="update",
                         args=[{"visible": [True, False]}]),
                ],
                showactive=True,
                x=1.05,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ]
    )

    fig.show()


def compute_fft(audio, sr, n_fft: int = None, hop_length: int = None):
    """
    compute a fast fourier transform on the audio data to get the frequency spectrum

    turns raw audio data into a "spectrogram"

    n_fft (int): FFT size; number of samples used to calculate each FFT

    hop_length (int): number of samples the window slides over each time (step size)
    
    """
    if n_fft is None:
        #window_len_s = 0.5
        #window_samples = int(window_len_s * sr)
        fft_window_size = 1024
    else:
        fft_window_size=n_fft

    if hop_length is None:
        hop_length = fft_window_size + (fft_window_size // 2)

    
    ##########################
    # params of signal.stft():
    ##########################
    # window: used to avoid the spread of strong true frequencies to their neighbors
    # https://en.wikipedia.org/wiki/Spectral_leakage
    # https://dsp.stackexchange.com/questions/63405/spectral-leakage-in-laymans-terms
    # signal.windows.hann  # default
    # signal.windows.hamming
    #https://docs.scipy.org/doc/scipy/tutorial/signal.html#comparison-with-legacy-implementation
    ##########################

    # fft is used for its O(nlog(n)) time complexity, vs dft's O(n^2) complexity
    # TODO: this could be done manually for understanding, then later on in the project
    #       signal.stft could be introduced for speed and versatility
    frequencies, times, stft = signal.stft(audio, fs=sr, 
                                           window="hamming",
                                           nperseg=fft_window_size,
                                           noverlap=fft_window_size - hop_length,
                                           padded=True,
                                           return_onesided=True
                                           #noverlap=window_size // 2,
                                           )


    # stft is a 2D complex array of shape (len(frequencies), len(times))
    # complex values of stft encode amplitude and phase offset of each sine wave component
    # We're interested in just the magnitude / strength of the frequency components
    # (phase offset information is useful though for time-stretching/pitch-shifting, or 
    # reconstructing signal via inverse STFT)
    magnitude = np.abs(stft)

    # frequencies is an array of frequency bin centers (in Hz)
    # times is an array of time bin centers (in seconds)
    # magnitude is a 2D real array of shape (len(frequencies), len(times))
    #           that for each time step gives a spectrum: an array of frequency bins
    return frequencies, times, magnitude



def peaks_are_duplicate(peak1: tuple[int, float] = None, peak2: tuple[int,float] = None):
    if peak1 is None or peak2 is None:
        return False
    delta_time = 10
    delta_freq = 300
    t1, f1 = peak1
    t2, f2 = peak2
    if abs(t1 - t2) <= delta_time and abs(f1 - f2) <= delta_freq:
        return True
    return False

def remove_duplicate_peaks(peaks: list[tuple[int, float]]):
    peaks=peaks.copy()
    peaks.sort(key=lambda x: x[0])
    # for each peak, search for duplicates within the next 10 peaks (ordered by time)
    for i in range(len(peaks)):
        for j in range(len(peaks[i:min(i+15, len(peaks)-1)])):
            j = j+i+1
            if peaks_are_duplicate(peaks[i], peaks[j]):
                peaks[j] = None
    return [peak for peak in peaks if peak is not None]



def find_peaks(frequencies, times, magnitude,
                             window_size=25,
                             candidates_per_band=5):
    """
    find the peaks in the spectrum using a sliding window

    within each window spanning the entire frequency range, find the local maxima within sub tiles of the window, then select `peaks_per_window` peaks across all local maxima

    this helps avoid peaks from being clustered too close together

    use `sub_tile_height=None` to just extract top `peaks_per_window` peaks per window across the audio
    """
    constellation_map = []
        
    # Attempt 3: sliding window across time, extract top peaks from each window after
    #            computing local maxima within frequency bands
    num_freq_bins, num_time_bins = magnitude.shape
    constellation_map = []

    # assuming fft_window_size = 1024
    bands = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 512)]

    # slide a window across time axis
    # height: entire frequency range
    # width:  window_size
    for t_start in range(0, num_time_bins, window_size):
        t_end = min(t_start + window_size, num_time_bins)
        window = magnitude[:, t_start:t_end]

        peak_candidates = []

        # find local maxima within bands of the window
        # height: variable based on current band
        # width:  window_size
        #for f_start in range(0, num_freq_bins, sub_tile_height):
            #f_end = min(f_start + sub_tile_height, num_freq_bins)
        for f_start, f_end in bands:
            sub_tile = window[f_start:f_end, :]

            # Flatten and get indices of top candidates in this frequency band
            flat_indices = np.argpartition(sub_tile.ravel(), -candidates_per_band)[-candidates_per_band:]

            for idx in flat_indices:
                f_local, t_local = np.unravel_index(idx, sub_tile.shape)
                f_idx = f_start + f_local
                t_idx = t_start + t_local
                mag = magnitude[f_idx, t_idx]
                peak_candidates.append((t_idx, f_idx, mag))

        # Keep top peaks per time window (sorted by magnitude)
        proportion_keep = 0.95
        peak_candidates.sort(key=lambda x: x[2], reverse=True)
        for t_idx, f_idx, _ in peak_candidates[0:int(np.floor(proportion_keep*len(peak_candidates)))]:
            freq = frequencies[f_idx]
            peak = (t_idx, freq)
            constellation_map.append(peak)

    return remove_duplicate_peaks(constellation_map)


    # Attempt 1: find lots of peaks using signal.find_peaks(), keep top ten highest from each window
    # iterate through each time window in the spectrogram
    #for time_idx, window in enumerate(magnitude.T):
        #peaks, props = signal.find_peaks(spectrum, height=0.1, distance=5)
        ## sort the peaks by their heights
        #sorted_peaks = sorted(peaks, key=lambda x: spectrum[x], reverse=True)
        ## keep only the top N peaks
        #top_peaks = sorted_peaks[:num_peaks]
        ## add the top peaks to the constellation map
        #constellation_map.append(top_peaks)

    # Attempt 2: find lots of peaks using signal.find_peaks(), then find top
    #            ten per window by "prominence": a metric measuring distance from other peaks
    # height      Minimum power threshold (in dB). Higher = fewer peaks.
    # distance    Minimum frequency bin spacing between peaks. Prevents clustering.
    # prominence  How much a peak stands out from its surroundings. Helps focus on strong peaks.
    # width       Can ensure peaks are broad enough to be meaningful. Optional.
    # iterate through each time window in the spectrogram
    #for time_idx, window in enumerate(magnitude.T):
        #peaks, props = signal.find_peaks(spectrum, height=1, **kwargs)
        #n_peaks = min(num_peaks, len(peaks))
        # select top n peaks, ranked by prominence
        #largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]


def create_constellation_map(audio, sr, hop_length=None) -> list[list[int]]:
    frequencies, times, magnitude = compute_fft(audio, sr, hop_length=hop_length)
    constellation_map = find_peaks(frequencies, times, magnitude)
    return constellation_map


#def filter_peaks(frequencies: np.ndarray, times: np.ndarray, magnitude: np.ndarray) -> list[tuple[int, float]]:
    #"""
    #For each time index, keep the highest magnitude peak in each frequency band.

    #Returns a list of (time_idx, frequency) pairs.
    #"""
    ## logarithmic frequency bands since lower frequencies are amplified and will usually dominate
    ## assuming window_size = 1024
    #bands = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 512)]
    #peaks = []

    #n_times = magnitude.shape[1]
    #for time_idx in range(n_times):
        #spectrum = magnitude[:, time_idx]
        #for band_start, band_end in bands:
            #band = spectrum[band_start:band_end]
            #if band.size == 0:
                #continue
            #max_idx_in_band = np.argmax(band)
            #max_mag = band[max_idx_in_band]
            #if max_mag > 0:
                #freq_idx = band_start + max_idx_in_band
                #freq = frequencies[freq_idx]
                #peaks.append((time_idx, freq))
    #return peaks

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
        constellation = find_peaks(audio, sr)
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
    constellation = find_peaks(sample, sr)
    hashes = create_hashes(constellation, None, sr)
    scores = score_hashes(hashes)
    return scores