import os

import librosa
import numpy as np
from scipy import signal
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.graph_objects as go


from DBcontrol import connect, retrieve_song, \
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
    
    # This line is needed to save the interactive plot as an HTML file when working on WSL
    # Use this command in the terminal to open:  explorer.exe my_spectrogram.html
    # fig.write_html("my_spectrogram.html")


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
                             window_size=10,
                             candidates_per_band=6):

    return find_peaks_windowed(frequencies, times, magnitude, window_size, candidates_per_band)

    # this one has a good chance of working
    #return find_peaks_max_prominence(frequencies, times, magnitude, window_size, candidates_per_band)

    #return find_peaks_max_magnitude(frequencies, times, magnitude, window_size, candidates_per_band)

def find_peaks_max_magnitude(frequencies, times, magnitude, window_size, candidates_per_band):
    # Attempt 1: find lots of peaks using signal.find_peaks(), keep top ten highest from each window
    # iterate through each time window in the spectrogram
    constellation_map = []
    num_peaks = 20
    for time_idx, window in enumerate(magnitude.T):
        #peaks, props = signal.find_peaks(window, height=0.1, distance=5)
        #peaks, props = signal.find_peaks(window, height=0.1, prominence=0, distance=200)
        peaks, props = signal.find_peaks(window, height=0.1, prominence=0, distance=200)
        n_peaks = min(num_peaks, len(peaks))
        largest_peaks = np.argpartition(peaks, -n_peaks)[-n_peaks:]
        print(peaks)
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])
    return constellation_map

def find_peaks_max_prominence(frequencies, times, magnitude, window_size, candidates_per_band):
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
    constellation_map = []
    num_peaks = 20
    for time_idx, window in enumerate(magnitude.T):
        #peaks, props = signal.find_peaks(window, height=0.1, distance=5)
        peaks, props = signal.find_peaks(window, prominence=0, distance=200)
        n_peaks = min(num_peaks, len(peaks))
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])
    return constellation_map

def find_peaks_windowed(frequencies, times, magnitude,
                             window_size=10,
                             candidates_per_band=6):
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
    #bands = [(0, 512)]
    #bands = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 160), (160, 330), (330, 512)]
    bands = [(0, 40), (40, 80), (80, 160), (160, 240), (240, 512)]

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
    # use a value slightly higher than this
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

def create_hashes(peaks, song_id: int = None, sr: int = None, fanout_t=100, fanout_f=3000):
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
            freq_diff = target[1] - anchor[1]

            # determine if peak is within target zone
            if time_diff <= 1:
                continue
            if np.abs(freq_diff) >= fanout_f:
                continue
            if time_diff > fanout_t:
                # constellation points are sorted by time
                # => no need to check more potential targets
                break
            
            address = create_address(anchor, target, sr)
            anchorT = anchor[0]

            fingerprints[address] = (int(anchorT), song_id)
            
    return fingerprints

def preprocess_audio(audio_path, sr = 11_025):
    """
    returns `(audio, sr)`

    11025 Hz
    44100 Hz

    Note: using mp3 prevents reading file blob directly
          using BytesIO stream, we have to save it to a
          temp file first (audio_path = BytesIO(...wavfile...)
          does work though)
    """
    # resample to 11 kHz (11,025 Hz)
    # uses a low pass filter to filter the higher frequencies to avoid aliasing (Nyquist-Shannon)
    # then takes sequential samples of size 4 and keeps the first of each ("decimate")
    audio, sr = librosa.load(audio_path, sr=sr)

    ## equivalent to:
    #max_freq_cutoff = 5512 # Hz
    ## Calculate the new sampling rate (at least twice the cutoff frequency)
    #new_sr = max_freq_cutoff * 2
    #new_sr += 1  # = 11025 to match librosa documentation / Chigozirim vid
    ## Resample the audio, which implicitly applies a low-pass filter
    #y_filtered = librosa.resample(y=audio, orig_sr=Fs, target_sr=new_sr, res_type='kaiser_best')

    return audio, sr

def compute_source_hashes(song_ids: list[int] = None, resample_rate: None|int = 11025):
    """
    `song_ids=None` (default) use all song_ids from database

    `resample_rate=None` to use original sampling rate from file

    Assumes all songs are the same sampling rate
    """
    if song_ids is None:
        song_ids = retrieve_song_ids()
    
    for song_id in song_ids:
        print(f"{song_id:03} ================================================")
        song = retrieve_song(song_id)
        print(f"{song['title']} by {song['artist']}")
        duration_s = song["duration_s"]
        audio_path = song["audio_path"]

        #waveform = song["waveform"]
        #audio_path = "temp_audio.mp3"
        #with open(audio_path, 'wb') as f:
            #f.write(waveform)
        #print(audio_path)

        audio, sr = preprocess_audio(audio_path, sr=resample_rate)
        constellation_map = create_constellation_map(audio, sr)
        hashes = create_hashes(constellation_map, song_id, sr)
        add_hashes(hashes)
    
    create_hash_index()
    return sr

def create_samples(track_info: dict, sr: int = None, n_samples: int = 5, n_seconds: int = 5, seed=1):
    audio, sr = preprocess_audio(track_info["audio_path"], sr)
    samples = []
    window_size = n_seconds * sr
    if len(audio) < window_size:
        raise ValueError(f"{track_info['title']}: could not create samples of length {n_seconds} sec")
    max_start_idx = len(audio) - window_size - 1

    np.random.seed(seed)
    start_indices = np.random.randint(0, max_start_idx, size=n_samples)
    for start_idx in start_indices:
        sample = audio[start_idx:start_idx + window_size]
        samples.append(sample)
    return samples

def add_noise(audio, noise_weight: float = 0.5):

    # BONUS: add noise to a file
    # brownian noise: x(n+1) = x(n) + w(n)
    #                 w(n) = N(0,1)

    #audio, sr = librosa.load(audio_path, sr=None) 
    def peak_normalize(x):
        # Normalize the audio to be within the range [-1, 1]
        return x / np.max(np.abs(x))

    noise = np.random.normal(0, 1, audio.shape[0])
    noise = np.cumsum(noise)
    noise = peak_normalize(noise)

    # 16-bit integer: [-32768, 32767]
    # (2**15 - 1) = 32767
    #scaled = np.int16(peak_normalize(audio_with_noise) * 32767)
    #scipy.io.wavfile.write('audio_with_noise.wav', sr, scaled)
    #ipd.Audio("audio_with_noise.wav")

    audio_with_noise = (audio + noise*noise_weight)
    return peak_normalize(audio_with_noise)


    # YOUR ANSWER HERE


def score_hashes(hashes: dict[int, tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[int, set[int, int]]]:
    """
    returns two values:

    ```
    result[0]: [(top_song_id, top_song_score), (2nd_song_id, 2nd_song_score, ...)]  # sorted
    result[1]: {song_id_1: {(sourceT, sampleT), (sourceT, sampleT), ...}, ...}
    ```
    
    """
    con = connect()
    # buffered=True here solves an issue that occurs when repeatedly calling
    #               retrieve_hashes()
    # https://stackoverflow.com/questions/29772337/python-mysql-connector-unread-result-found-when-using-fetchone
    cur = con.cursor()

    # 2.3: Searching and Scoring
    
    # Each hash from the sample is used to search in the 
    # database for matching hashes

    # For each matching hash found in the database, the
    # corresponding offset times from the beginning of the
    # sample and database files are associated into time pairs.

    # The time pairs are distributed into bins according to the 
    # track ID associated with the matching database hash

    time_pair_bins = defaultdict(set)
    for address, (sampleT, _) in hashes.items():
        matching_hashes = retrieve_hashes(address, cur)
        if matching_hashes is not None:
            #print(matching_hashes)
            #exit(0)
            for _, sourceT, song_id in matching_hashes:
                time_pair_bins[song_id].add((sourceT, sampleT))
            
    # After all sample hashes have been used to search in the
    # database to form matching time pairs, the bins are scanned
    # for matches. 

    #################################################
    # Histogram Method for scanning for matches     #
    # (detecting diagonal line within scatterplot)  #
    #################################################

    # Within each bin, the set of time pairs represents
    # a scatter plot of association between the sample and database
    # sound files.

    # If the files match, matching features should occur at similar
    # relative offsets from the beginning of the file, i.e. a sequence
    # of hashes in one file should also occur in the matching file with
    # the same relative time sequence.

    # The problem of deciding whether a match has been found reduces to
    # detecting a significant cluster of points forming a diagonal line
    # within the scatterplot.

    # Assume that the slope of the diagonal line is 1.0

    # Then corresponding times of matching features between matching
    # files have the relationship:
    # sourceT = sampleT + offset
    # => offset = sourceT - sampleT

    scores = {}
    for song_id, time_pair_bin in time_pair_bins.items():
        # For each (sourceT, sampleT) coordinate in the scatterplot,
        # we calculate:
        # deltaT = sourceT - sampleT
        deltaT_values = [sourceT - sampleT for (sourceT, sampleT) in time_pair_bin]

        # Then we calculate a histogram of these deltaT values
        # and scan for a peak.
        #hist = defaultdict(lambda: 0)
        #for dT in deltaT_values:
            #hist[dT] += 1
        hist, bin_edges = np.histogram(deltaT_values, bins=max(len(np.unique(deltaT_values)), 10))

        # The score of the match is the number of matching points
        # in the histogram peak
        scores[song_id] = hist.max()
        #scores[song_id] = max(hist.values()) if hist else 0

    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    con.close()
    return scores, time_pair_bins

def init_db(tracks_dir: str = None, n_songs: int = 0, specific_songs: list[str] = None):
    """
    tracks_dir:
        Path to the audio dataset downloaded by `musicdl`. 
        Default is to look for a folder or zip archive matching the pattern "`tracks*`"
    n_songs:
        Take a sample from the top of the tracks dataset. Default is to read all songs

    """
    create_tables()
    add_songs(tracks_dir, n_songs, specific_songs)
    compute_source_hashes()

def recognize_music(sample_audio_path: str, sr: None|int = None, remove_sample: bool = True) -> list[tuple[int, int]]:
    """
    returns sorted list of `(song_id, score)` tuples
    
    Access top prediction with `scores[0][0]`

    ```
    init_db(tracks_dir = "tracks-2025-07-22", n_songs=5)

    # record from microphone
    sample_wav_path = record_audio(n_seconds = 5)

    song_id = recognize_music(sample_wav_path)[0][0]
    """
    sample, sr = preprocess_audio(sample_audio_path, sr=sr)
    if remove_sample:
        os.remove(sample_audio_path)
    constellation_map = create_constellation_map(sample, sr)
    hashes = create_hashes(constellation_map, None, sr)
    scores, time_pair_bins = score_hashes(hashes)
    return scores, time_pair_bins

def visualize_scoring(sample_wav_path: str) -> None:
    scores, time_pair_bins = recognize_music(sample_wav_path, remove_sample=False)
    song_ids = [score[0] for score in scores[:min(len(scores), 5)]]
    if 20 not in song_ids:
        song_ids.append(20)
    for song_id in song_ids:
        time_pair_bin = time_pair_bins[song_id]
        song = retrieve_song(song_id)
        if song is None:
            continue
        deltaT_values = [sourceT - sampleT for (sourceT, sampleT) in time_pair_bin]
        hist, bin_edges = np.histogram(deltaT_values, bins=max(len(np.unique(deltaT_values)), 10))
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        fig.suptitle(f"{song['title']} by {song['artist']}", fontsize=16)

        # scatterplot
        sourceT_vals = [pair[0] for pair in time_pair_bin]
        sampleT_vals = [pair[1] for pair in time_pair_bin]
        axes[0].scatter(sourceT_vals, sampleT_vals, alpha=0.7)
        axes[0].set_xlabel('Source Time')
        axes[0].set_ylabel('Sample Time')
        axes[0].set_title('Scatterplot of matching hash locations')

        # histogram
        axes[1].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', align='edge')
        axes[1].set_xlabel('Offset t')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Histogram of differences of time offsets')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    track = '/home/evanteal15/F25-Shazam-Clone/notebooks/sample.wav'
    visualize_map_interactive(track)
