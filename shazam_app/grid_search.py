import numpy as np
import pandas as pd
from itertools import product
import librosa

from cm_visualizations import visualize_map_interactive

# updated add_noise that uses signal-to-noise ratio
#from cm_helper import create_samples, add_noise
from cm_helper import create_samples

from DBcontrol import init_db, connect, retrieve_song

from const_map import create_constellation_map
from hasher import create_hashes
from search import score_hashes

from parameters import set_parameters, read_parameters
from pathlib import Path

# score_hashes_no_index
from collections import defaultdict
from DBcontrol import retrieve_hashes

# measuring search time
import time

# GridViewer
import pickle
import json

# logging errors in GridViewer
import traceback
import datetime

#
# 1 / 3: list samples to use to evaluate recognition performance
#

microphone_sample_list = [
    ("Plastic Beach (feat. Mick Jones and Paul Simonon)", "Gorillaz", "audio_samples/plastic_beach_microphone_recording.wav"),
    #("DOGTOOTH", "Tyler, The Creator", "tracks/audio/TylerTheCreator_DOGTOOTH_QdkbuXIJHfk.flac")
    # ...
]


microphone_sample_list = [
    {"title": v[0], "artist": v[1], "audio_path": v[2]} 
    for v in microphone_sample_list
    ]


def add_noise(audio, snr_dB):
    """
    add noise to librosa audio with a desired 
    signal-to-noise ratio (snr), measured in decibels.
    ```
    snr_dB = 10 log_10(P_signal / P_noise)

    P_signal / P_noise = (A_signal / A_noise)**2

    => snr_dB = 20 log_10(A_signal / A_noise)

    => A_signal / A_noise = 10**(snr_dB / 20)

    => A_noise = A_signal / 10**(SNR_dB / 20)
    ```
    
    We use the root mean square (RMS) of the amplitude for the 
    signal and noise. To achieve the desired SNR for our final 
    audio, we calculate a weight to multiply the brownian noise by:

    ```
    A_noise_initial * noise_weight = A_noise

    => noise_weight = A_noise / A_noise_initial

    => noise_weight = rms_signal / (10**(snr_dB / 20)) / (rms_noise)

    (avoid division by zero by adding 1e-10 to denominator)
    ```
    """

    # brownian noise: x(n+1) = x(n) + w(n)
    #                 w(n) = N(0,1)

    def peak_normalize(x):
        # Normalize the audio to be within the range [-1, 1]
        return x / np.max(np.abs(x))

    noise = np.random.normal(0, 1, audio.shape[0])
    noise = np.cumsum(noise)
    noise = peak_normalize(noise)

    rms_signal = np.sqrt(np.mean(audio**2))
    rms_noise = np.sqrt(np.mean(noise**2))

    noise_weight = rms_signal / (10**(snr_dB / 20)) / (rms_noise + 1e-10)



    audio_with_noise = (audio + noise*noise_weight)
    return peak_normalize(audio_with_noise)

def augment_samples(sr, snr):
    con = connect()
    cur = con.cursor()
    samples = []
    # TODO for members: Write this SQL query
    for sample in microphone_sample_list:
        cur.execute(
            "SELECT id AS song_id "
            "FROM songs "
            "WHERE title LIKE ? "
            "AND artist = ?",
            (sample["title"], sample["artist"])
        )

        res = cur.fetchone()
        if res: 
            song_id = res[0]
        else:
            raise ValueError(f'\'{sample["title"]}\' by \'{sample["artist"]}\' not found.\nSpelling of name/artist might be different from the spelling in tracks/audio/tracks.csv?')
        
        sample_slices = create_samples(sample["audio_path"], sr, n_samples = 20)
        sample_slices_noisy = [add_noise(audio, snr) for audio in sample_slices]
        samples.extend([
            {"song_id": song_id,
             "microphone": s[0],
             #"mic_noisy": s[1]
             } for s in zip(sample_slices, sample_slices_noisy)
        ])
    return samples



#
# 2 / 3: Compute some metrics using output of search.py
#

def std_of_deltaT(song_id, time_pair_bins, sr=11025):
    """
    heuristic that measures the distribution of deltaT values

    results in standard deviation between 0 ms and 1000 ms

    lower is better - suggests that the time offsets are behaving consistently
    """
    time_pair_bin = time_pair_bins[song_id]

    # 1) take the difference in time between sourceT and sampleT
    #    values represent the offset suggested by the presence of the 
    # respective hash match
    # ex: offset = 40
    # data might look like [-4, 5, 8, 40, 40, 40, ..., 40, 40, 40, 56, 68, 80]
    deltaT_vals = sorted([sourceT-sampleT for (sourceT, sampleT) in time_pair_bin])

    #    Potential idea: then, take the difference between adjacent deltaT values
    #    removes influence of offset / outlier offset values
    #    values are close to zero for consistent time offsets
    # ex: [9, 3, 0, 0, 0, ..., 0, 0, 16, 12, 12]
    #second_order_differences = [deltaT_vals[i] - deltaT_vals[i-1] 
                                #for i in range(1, len(deltaT_vals))]
    # not doing this step makes the metric more interpretable
                                

    # 3) compute the standard deviation of these differences, measured in seconds
    #    return a metric between 0 and 1000
    #    (stddev between 0 and 1000 milliseconds, 0 and 1 seconds)
    #
    # T is in units of STFT bins, 
    # multiply by 1000*(hop_length / sr) to get in units of milliseconds
    #
    # Unit conversion:
    # bins * ((n_samples_to_jump/bin) / (samples/second)) = bins * (seconds / bin) = seconds
    # seconds * (1000 ms / second) = milliseconds
    # std(x * v) = x * std(v)
    hop_length = 1024 // 2  # sidenote: not 1024 + (1024 // 2) as in cm_helper.py

    return min(np.std(deltaT_vals) * (hop_length/sr), 1000)


def count_hash_matches(song_id, time_pair_bins, n_sample_hashes):
    """
    naive heuristic that counts the number of hash matches between sample and source
    
    note: there can be repeated matches for a single hash value present in sample

    higher is better - suggests that many hashes show up in both sample and source
    """
    n_matches = len(time_pair_bins[song_id]) 
    return n_matches


def compute_performance_metrics(song_id, time_pair_bins, n_sample_hashes, sr=11025):
    """
    given the output from search.py:score_hashes(), computes metrics that
    evaluate how close a given song_id matches the sample audio

    returns a dictionary containing metrics

    ## Metrics:
     
    - `std_of_deltaT`: less is better, (0-1000 ms)
    - `n_hash_matches`: more is better
    - `prop_hash_matches`: `min(cout_hash_matches / n_sample_hashes, 1)
    
    """
    metrics = {
        "std_of_deltaT": std_of_deltaT(song_id, time_pair_bins, sr),
        "n_hash_matches": count_hash_matches(song_id, time_pair_bins, n_sample_hashes),
        "n_sample_hashes": n_sample_hashes
    }
    metrics["prop_hash_matches"] = min(metrics["n_hash_matches"] / metrics["n_sample_hashes"], 1)
    return metrics


def old_simulate_microphone_from_source(y, sr):

    # ensure 1D
    y = np.asarray(y)
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # 1) bandwidth limitation (simulate cheap mic low-pass) via resampling
    target_sr = min(8000, sr)
    if target_sr < sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        y = librosa.resample(y, orig_sr=target_sr, target_sr=sr)

    # 2) simple dynamic range compressor (frame RMS -> dB -> gain reduction)
    frame_len = max(1, int(0.02 * sr))  # 20 ms window
    window = np.ones(frame_len) / frame_len
    power = np.convolve(y * y, window, mode="same")
    rms = np.sqrt(power + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)

    thresh_db = -25.0   # threshold (dB)
    ratio = 4.0         # compression ratio
    over_db = np.maximum(db - thresh_db, 0.0)
    reduction_db = over_db * (1.0 - 1.0 / ratio)

    # smooth gain changes to simulate attack/release
    #smooth_len = max(1, int(0.01 * sr))  # 10 ms smoothing
    #smooth_k = np.ones(smooth_len) / smooth_len
    #reduction_db = np.convolve(reduction_db, smooth_k, mode="same")

    #gain = 10.0 ** (-reduction_db / 20.0)
    #gain = 10.0 ** (-reduction_db / 10.0)
    gain = 10.0 ** (-reduction_db / 10.0)
    y_compressed = y * gain

    # makeup gain to restore perceived loudness
    makeup_db = 6.0
    y_compressed *= 10.0 ** (makeup_db / 20.0)

    # 3) soft clipping to simulate mic preamp saturation
    #y_compressed = np.tanh(y_compressed)

    # 4) reduce effective bit depth (quantization noise)
    #quant_bits = 16
    #max_val = 2 ** (quant_bits - 1) - 1
    #y_compressed = np.round(y_compressed * max_val) / float(max_val)

    # 5) add low-level microphone noise/hiss
    #noise_level = 1e-3 * max(1.0, np.std(y_compressed))
    #y_compressed = y_compressed + np.random.normal(0.0, noise_level, size=y_compressed.shape)

    # clamp to [-1, 1]
    #y_compressed = np.clip(y_compressed, -1.0, 1.0)

    return y_compressed

def simulate_microphone_from_source(y, sr, drc_threshold=-40, drc_ratio=4.0, drc_attack=10, drc_release=50):
    """
    does two things:

    1) Harmonic-Percussive Source Separation (HPSS) using librosa

    2) Dynamic Range Compression using pydub
    """
    ##################################
    # HPSS
    # maintain percussion voices
    # after filtering and dynamic
    # range compression
    # 
    # issue: this may take too long to
    # be worth it, but also drc takes
    # a similar amount of time
    ##################################
    #D = librosa.stft(y)
    #H, P = librosa.decompose.hpss(D)
    #y_harmonic = librosa.istft(H)
    ## should always pad with 256 zeros  (128 [...] 128)
    #y_harmonic = np.pad(y_harmonic,
                          #int(max(len(y) - len(y_harmonic), 0)/2))
    #y_percussive = librosa.istft(P)
    #y_percussive = np.pad(y_percussive,
                          #int(max(len(y) - len(y_percussive), 0)/2))


    ##################################
    # convert to 16 bit integer
    # (librosa to pydub)
    ##################################

    max_amplitude = np.iinfo(np.int16()).max
    y_int16 = (y * max_amplitude).astype(np.int16)

    from pydub import AudioSegment
    from pydub.effects import compress_dynamic_range, low_pass_filter
    y_audioseg = AudioSegment(
        data = y_int16.tobytes(),
        frame_rate = sr,
        sample_width = y_int16.dtype.itemsize,
        channels=1)
    
    ##################################
    # low pass filter: 6000 Hz cutoff
    ##################################
    # https://github.com/jiaaro/pydub/blob/master/pydub/effects.py#L222
    y_audioseg = low_pass_filter(y_audioseg, 6000)

    ##################################
    # dynamic range compression
    ##################################
    # https://github.com/jiaaro/pydub/blob/master/pydub/effects.py#L116
    # https://en.wikipedia.org/wiki/Dynamic_range_compression
    y_audioseg = compress_dynamic_range(
        y_audioseg,
        threshold=drc_threshold,
        ratio = drc_ratio,
        attack = drc_attack,
        release=drc_release
    )

    ##################################
    # convert back to 32 bit float
    # (pydub to librosa)
    ##################################
    y_compressed = (np.array(y_audioseg.get_array_of_samples()) / max_amplitude).astype(np.float32)
    #return np.clip(y_compressed + (y_percussive * 0.3), -1, 1)
    #return y_compressed + (y_percussive * 0.3)
    return y_compressed


def sample_from_source_audios(n_songs: int = None, tracks_dir = "./tracks", sr=11025, snr=None):
    """
    This function attempts to simulate the act of recording many microphone samples
    by doing some dynamic range compression and filtering of the source audio files

    spectrogram looks similar to actual microphone recordings, primary trait 
    of microphone recordings compared to the source audio is having a lower 
    amplitude in the waveform on average (lower dynamic range)

    assumes that song_ids correspond to order that tracks appear in csv

    (first row => song_id = 1, ...)
    """
    samples = []
    tracks_dir = Path(tracks_dir)
    df = pd.read_csv(tracks_dir/"tracks.csv")
    for idx, track in df.iterrows():
        song_id = idx + 1
        if n_songs is not None:
            if song_id > n_songs:
                break
        sample_slices = create_samples(tracks_dir/track["audio_path"], sr, n_samples=5, n_seconds=5, seed=1)
        sample_slices = [simulate_microphone_from_source(s, sr) for s in sample_slices]
        sample_slices_with_noise = [add_noise(audio, snr) for audio in sample_slices]
        samples.extend([
            {"song_id": song_id,
             "microphone": s[0],
             "microphone_with_noise": s[1]
             } for s in zip(sample_slices, sample_slices_with_noise)
        ])
    return samples

def recompute_hashes():
    with connect() as con:
        con.execute("DELETE FROM hashes")
        con.commit()
    from DBcontrol import compute_source_hashes
    compute_source_hashes()

class GridViewer():
    """
    save summary statistics for each combination of parameters used in grid search

    use for visualization of the four main audio fingerprint system parameters:
    - *Reliability*: can the model actually recognize tracks?
    - *Robustness*: how resistant is the model to noise?
    - *Fingerprint size*: how much disk space is used?
    - *Search speed*: how long does it take to search for a match?
    """
    def __init__(self, n_seconds_of_source_audio: float):
        self.param_stats = defaultdict(list)  # {snr: [ (params_1, stats_1), (params_2, stats_2), ... ], ...}
        self.errors = []  # dict keys: {"time_of_error", "all_params", "stacktrace"}
        self.n_seconds_of_source_audio = n_seconds_of_source_audio
        self.filename = "gridviewer.pkl"
        # k = frozenset(d.items()) # idea: recursively use this to index based on paramset
        # (immutable keys)

    def summary_statistics(self, results, database_size):
        """
        summarize the results table to be stored with 
        its corresponding parameters
        """
        results_df = pd.DataFrame(results)
        return {
            "proportion_correct": results_df["correct"].mean(),
            "avg_search_time": results_df["search_time_with_index"].mean(),
            "total_fingerprint_size_MB": database_size[0] + database_size[1],
            "misc_info": {
                "avg_search_time_without_index": results_df["search_time_without_index"].mean(),
                "fingerprint_size_MB": (('hashes', database_size[0]), ('hash_index', database_size[1])),
            }
        }

    def append(self, all_parameters, results, snr, database_size: tuple[float,float]):
        self.param_stats[snr].append(
            (
                all_parameters, 
                self.summary_statistics(results, database_size)
            )
        )

    def log_exception(self, all_parameters, stacktrace, snr):
        """
        obtain stacktrace string with `traceback.format_exc()` inside `except` clause.
        """
        time_of_error = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.errors.append({
            "time_of_error": time_of_error,
            "all_params": all_parameters,
            "stacktrace": stacktrace,
            "snr": snr
            })

    
    @classmethod
    def from_pickle(cls, filename="gridviewer.pkl"):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def to_pickle(self, filename = None):
        with open(
            filename if filename is not None else self.filename, 
            'wb') as f:
            pickle.dump(self, f)

    def to_json(self, json_filename="gridviewer.json"):
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump({
                "param_stats": self.param_stats,
                "errors": self.errors,
                "n_seconds_of_source_audio": self.n_seconds_of_source_audio
            })



def score_hashes_no_index(hashes: dict[int, tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[int, set[int, int]]]:
    """
    same as `search.py:score_hashes()`, except does not use the `hashes` table index when retrieving matching hashes

    used for computing search time metrics to show effectiveness of database index
    """
    con = connect()
    cur = con.cursor()
    time_pair_bins = defaultdict(set)
    for address, (sampleT, _) in hashes.items():
        # DBcontrol.py:retrieve_hashes(), without using index
        cur.execute("SELECT hash_val, time_stamp, song_id FROM hashes NOT INDEXED WHERE hash_val = ?", (address,))
        matching_hashes = cur.fetchall()
        if matching_hashes is not None:
            for _, sourceT, song_id in matching_hashes:
                time_pair_bins[song_id].add((sourceT, sampleT))
    scores = {}
    for song_id, time_pair_bin in time_pair_bins.items():
        deltaT_values = [sourceT - sampleT for (sourceT, sampleT) in time_pair_bin]
        hist, bin_edges = np.histogram(deltaT_values, bins=max(len(np.unique(deltaT_values)), 10))
        scores[song_id] = hist.max()
    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    con.close()
    return scores, time_pair_bins


def perform_recognition_test(n_songs=None, samples=None):
    """
    returns a tuple:
    `(n_correct / n_samples, performance results for each sample)`

    first value is the proportion of correct recognitions

    samples specified based on slicing of samples in `microphone_sample_list`
    using `grid_search.py:augment_samples()`
    """
    #init_db()
    #init_db(n_songs=n_songs)
    #if samples is None:
        #snr = ???
        #samples = augment_samples(sr=sr, snr=snr)

    sr = 11025
    results = []
    for sample in samples:
        result = {}
        ground_truth_song_id = sample["song_id"]

        # peaks -> hashes
        constellation_map = create_constellation_map(sample["microphone_with_noise"], sr=sr)
        hashes = create_hashes(constellation_map, None, sr)

        # hashes -> metrics
        start_time = time.time()
        scores, time_pair_bins = score_hashes(hashes)
        search_time_with_index = time.time() - start_time

        start_time = time.time()
        score_hashes_no_index(hashes)
        search_time_without_index = time.time() - start_time

        n_sample_hashes = len(hashes)
        n_potential_matches = min(len(scores), 5)
        metrics_per_potential_match = {}
        for potential_song_id, potential_score in scores[:n_potential_matches]:
            metrics_per_potential_match[potential_song_id] = compute_performance_metrics(
                potential_song_id, time_pair_bins, n_sample_hashes, sr
            )
            metrics_per_potential_match[potential_song_id]["histogram_max_height"] = potential_score
        

        # store metrics for each sample
        result = {
            "ground_truth": ground_truth_song_id,
            "prediction": scores[0][0],
            "correct": ground_truth_song_id == scores[0][0],
            "metrics": metrics_per_potential_match,
            "search_time_with_index": search_time_with_index,
            "search_time_without_index": search_time_without_index,
            "microphone_audio": sample["microphone_with_noise"]
        }

        results.append(result)

    return sum(r["correct"] for r in results) / len(samples), results

def compute_fingerprint_size():
    """
    returns `(all_hashes_size_MB, hash_index_size_MB)`

    https://sqlite.org/dbstat.html
    """
    with connect() as con:
        cur = con.cursor()
        cur.execute(
            "SELECT name, SUM(pgsize) / (1024.0*1024.0) as size_in_mb "
            "FROM dbstat "
            "GROUP BY name "
            "HAVING name IN ('hashes', 'idx_hash_val') "
            "ORDER BY "
                "CASE name "
                "WHEN 'hashes' THEN 1 "
                "ELSE 2 "
                "END"
        )
        res = cur.fetchall()
        size_hashes = res[0][1]
        if len(res) == 1 or res[1][0] != "idx_hash_val":
            # idx_hash_val doesn't exist
            size_hash_index = 0
        else:
            size_hash_index = res[1][1]

        return size_hashes, size_hash_index

    

def compute_total_length_of_source_audio():
    """
    number of seconds of source audio in database

    can be used to normalize fingerprint size to Megabytes per second of audio
    """
    with connect() as con:
        cur = con.cursor()
        cur.execute(
            "SELECT SUM(duration_s) as total_duration_s "
            "FROM songs"
        )
        total_duration_s = cur.fetchone()[0]
    
    return total_duration_s


def run_grid_search(n_songs=None):
    max_proportion_correct = 0
    best_params = 0
    best_results = {}
    proportion_correct = 0
    results = {}

    # {0.1: (best_results, best_params), 0.2: (best_results, best_params), ...}
    best_model_per_snr = {}

    grid_cmws = [4,5,10]
    grid_cpb = [5,7]
    grid_bands = [[(0,20), (20, 40), (40,80), (80,160), (160, 320), (320,512)]]

    #grid_noise_weights = np.arange(0.1, 1, 0.1)
    grid_signal_to_noise_ratios = [-3, 0, 3, 6]

    param_grid = list(product(grid_cmws, grid_cpb, grid_bands))

    grid_viewer = GridViewer(
        n_seconds_of_source_audio = compute_total_length_of_source_audio()
        )

    ##################################
    # grid search:
    # try all combinations
    # compute metrics for each
    # run on Great Lakes HPC overnight
    ##################################

    for i, snr in enumerate(grid_signal_to_noise_ratios):
        print(f"snr={snr}: {i+1} / {len(grid_signal_to_noise_ratios)}")

        samples = sample_from_source_audios(n_songs=n_songs, snr=snr)
        init_db(n_songs=n_songs)


        for j, (cm_window_size, candidates_per_band, bands) in enumerate(param_grid):
            print(f"\tparamset {j+1} / {len(param_grid)}")
            parameters = {
                "cm_window_size": cm_window_size,
                "candidates_per_band": candidates_per_band,
                "bands": bands
                # ...
            }

            # unspecified parameters are set to their defaults
            #
            # **dict notation:
            # converts {"key1": value1, "key2": value2}
            #          -> key1=value1, key2=value2
            set_parameters(**parameters)
            recompute_hashes()
            database_size = compute_fingerprint_size()
            # return length of time for searching
            all_parameters = read_parameters("all_parameters")
            try:
                proportion_correct, results = perform_recognition_test(n_songs, samples)
                if proportion_correct > max_proportion_correct:
                    max_proportion_correct = proportion_correct
                    best_params = all_parameters
                    best_results = results
                grid_viewer.append(all_parameters, results, snr, database_size)
            except:
                # there can be invalid combinations of parameters
                #
                # example: first band is 5 freq bins tall, cm_window_size is 5, and
                # we attempt to extract candidates_per_band=40 peaks from
                # the bottom left spectrogram window
                #
                # if we run into an error, log exception traceback
                print("\t exception occured")
                grid_viewer.log_exception(
                    all_parameters=all_parameters, 
                    stacktrace=traceback.format_exc(),
                    snr=snr
                    )


        best_model_per_snr[snr] = (best_results, best_params)

    return best_model_per_snr, grid_viewer
    

def main():
    #max_results, max_params = run_grid_search(n_songs=4)
    best_model_per_noise_weight, grid_viewer = run_grid_search(n_songs=4)
    max_results, max_params = best_model_per_noise_weight[6]
    print("=============")
    print("max parameters:")
    print("=============")

    print(max_params)

    print("=============")
    print("results:")
    print("=============")

    #print(max_results)
    max_results_df = pd.DataFrame(max_results)
    max_results_df.to_pickle("max_results.pkl")
    grid_viewer.to_json()
    grid_viewer.to_pickle()
    print("saved best results to:         max_results.pkl")
    print("saved best parameters to:      parameters.json")
    print(f"saved grid viewer to:         {grid_viewer.filename}")
    print("                           and gridviewer.json")

    #plastic_beach = retrieve_song(1)["audio_path"]
    #visualize_map_interactive(plastic_beach)

if __name__ == "__main__":
    main()
    #pb_source = "tracks/audio/Gorillaz_PlasticBeachfeatMickJonesandPaulSimonon_pWIx2mz0cDg.flac"
    #from cm_helper import preprocess_audio
    #y, sr = preprocess_audio(pb_source)
    #y = np.clip(y, -1, 1)
    #y_compressed = simulate_microphone_from_source(y,sr)
    #import soundfile as sf
    #sf.write("pb_compressed.wav", y_compressed, sr)

    






