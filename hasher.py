import os

import librosa
import numpy as np
from scipy import signal
from collections import defaultdict


from DBcontrol import connect_to_db, retrieve_song, \
    retrieve_song_ids, retrieve_hashes, add_hashes, \
    create_hash_index, create_tables, add_songs

# TODO: visualize constellation map overlaid on spectrogram to interpret parameters of 
#       scipy.signal.stft, scipy.signal.find_peaks, and result of using filter_peaks()
# TODO: break this up so that we can do a parameter sweep
#       and visualize the results (with a utils script)

def create_spectrogram(audio, sr):
    window_len_s = 0.5
    window_samples = int(window_len_s * sr)

    pad = window_samples - (audio.size % window_samples)
    
    # pad the audio signal to make it a multiple of window_samples
    audio = np.pad(audio, (0, pad), mode='constant')
    
    # do a fast fourier transform on the audio data to get the frequency spectrum
    # turns raw audio data into a "spectrogram"

    # window:
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html  # default
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hamming.html
    # nperseg: length of each segment
    # noverlap: number of points to overlap between segments
    # nfft: length of the FFT used, if a zero padded FFT is desired
    #https://docs.scipy.org/doc/scipy/tutorial/signal.html#comparison-with-legacy-implementation
    frequencies, times, stft = signal.stft(audio, sr, 
                                           window="hamming",
                                           nperseg=window_samples,
                                           #noverlap=window_samples // 2,
                                           #nfft=window_samples,
                                           #return_onesided=True,
                                           )
    return frequencies, stft
    


def create_constellation_map(audio, sr) -> list[list[int]]:
    frequencies, stft = create_spectrogram(audio, sr)

    num_peaks = 20
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
        peaks, props = signal.find_peaks(spectrum, prominence=0, distance=200)
        n_peaks = min(num_peaks, len(peaks))
        # select top n peaks, ranked by prominence
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])


    #duration_s as parameter to create_constellation_map(), or pass song_id and retrieve from db
    # TODO:
    #peaks = filter_peaks(constellation_map=constellation, audioDuration=duration_s)
    return constellation_map

# TODO: incorperate binned filtering via filter_peaks() (avoid overrepresenting lower frequencies)
def filter_peaks(constellation_map: list[list[int]], audioDuration: float) -> list[list[int]]:
    # logarithmic frequency bands since lower frequencies are amplified and will usually dominate
    bands = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 512)]
    
    peaks = []
    binDuration = audioDuration / len(constellation_map)
    for i, bin in enumerate(constellation_map):
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
                peaks.append((peak_time, freq))

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
    # use a value slightly higher than this:
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