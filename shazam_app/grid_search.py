import numpy as np
import pandas as pd
from itertools import product
import librosa

from cm_visualizations import visualize_map_interactive
from cm_helper import create_samples, add_noise
from DBcontrol import init_db, connect, retrieve_song

from const_map import create_constellation_map
from hasher import create_hashes
from search import score_hashes

from parameters import set_parameters, read_parameters
from pathlib import Path

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

def augment_samples(sr, noise_weight):
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
        sample_slices_noisy = [add_noise(audio, noise_weight) for audio in sample_slices]
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

def simulate_microphone_from_source(y, sr):
    """
    does two things:

    1) Harmonic-Percussive Source Separation (HPSS) using librosa

    2) Dynamic Range Compression using pydub
    """
    D = librosa.stft(y)
    H, P = librosa.decompose.hpss(D)

    y_harmonic = librosa.istft(H)
    y_harmonic = np.pad(y_harmonic,
                          int(max(len(y) - len(y_harmonic), 0)/2))

    y_percussive = librosa.istft(P)
    y_percussive = np.pad(y_percussive,
                          int(max(len(y) - len(y_percussive), 0)/2))


    from pydub import AudioSegment
    from pydub.effects import compress_dynamic_range, low_pass_filter
    max_amplitude = np.iinfo(np.int16()).max
    y_int16 = (y * max_amplitude).astype(np.int16)
    y_audioseg = AudioSegment(
        data = y_int16.tobytes(),
        frame_rate = sr,
        sample_width = y_int16.dtype.itemsize,
        channels=1)
    
    y_audioseg = low_pass_filter(y_audioseg, 6000)

    threshold = -40  # default -20
    ratio = 4.0
    attack = 10
    release = 50
    # https://github.com/jiaaro/pydub/blob/master/pydub/effects.py#L116
    # https://en.wikipedia.org/wiki/Dynamic_range_compression
    y_audioseg = compress_dynamic_range(
        y_audioseg,
        threshold=threshold,
        ratio = ratio,
        attack = attack,
        release=release
    )
    # cutoff in Hz
    # https://github.com/jiaaro/pydub/blob/master/pydub/effects.py#L222

    y_compressed = (np.array(y_audioseg.get_array_of_samples()) / max_amplitude).astype(np.float32)
    #return np.clip(y_compressed + (y_percussive * 0.3), -1, 1)
    return y_compressed + (y_percussive * 0.3)




def sample_from_source_audios(tracks_dir = "./tracks", sr=11025):
    """
    assumes that song_ids correspond to order that tracks appear in csv

    (first row => song_id = 1, ...)
    """
    samples = []
    noise_weight = 0.4
    tracks_dir = Path(tracks_dir)
    df = pd.read_csv(tracks_dir/"tracks.csv")
    for idx, track in df.iterrows():
        song_id = idx + 1
        sample_slices = create_samples(tracks_dir/track["audio_path"], sr, n_samples=5, n_seconds=5, seed=1)
        sample_slices_noisy = [add_noise(audio, noise_weight) for audio in sample_slices]
        samples.extend([
            {"song_id": song_id,
             "microphone": s[0],
             "microphone_with_noise": s[1]
             } for s in zip(sample_slices, sample_slices_noisy)
        ])
    return samples


def perform_recognition_test(n_songs=None):
    """
    returns a tuple:
    `(n_correct / n_samples, performance results for each sample)`

    first value is the proportion of correct recognitions

    samples specified based on slicing of samples in `microphone_sample_list`
    using `grid_search.py:augment_samples()`
    """
    #init_db()
    sr = 11025
    init_db(n_songs=n_songs)
    results = []
    samples = augment_samples(sr=sr, noise_weight=0.3)
    for sample in samples:
        result = {}
        ground_truth_song_id = sample["song_id"]

        # peaks -> hashes
        constellation_map = create_constellation_map(sample["microphone"], sr=sr)
        hashes = create_hashes(constellation_map, None, sr)

        # hashes -> metrics
        scores, time_pair_bins = score_hashes(hashes)
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
            "microphone_audio": sample["microphone"]
        }

        results.append(result)

    return sum(r["correct"] for r in results) / len(samples), results

def run_grid_search(n_songs=None):
    max_proportion_correct = 0
    max_params = 0
    max_results = {}
    proportion_correct = 0
    results = {}

    grid_cmws = [4,5,10]
    grid_cpb = [5,7]
    grid_bands = [[(0,20), (20, 40), (40,80), (80,160), (160, 320), (320,512)]]

    for (
        cm_window_size, candidates_per_band, bands
        ) in list(product(grid_cmws, grid_cpb, grid_bands)):
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
        proportion_correct, results = perform_recognition_test(n_songs)
        if proportion_correct > max_proportion_correct:
            max_proportion_correct = proportion_correct
            max_params = read_parameters("all_parameters")
            max_results = results

    return max_results, max_params
    

def main():
    max_results, max_params = run_grid_search(n_songs=2)
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
    print("saved results to pkl file")

    plastic_beach = retrieve_song(1)["audio_path"]
    visualize_map_interactive(plastic_beach)

if __name__ == "__main__":
    #main()
    pb_source = "tracks/audio/Gorillaz_PlasticBeachfeatMickJonesandPaulSimonon_pWIx2mz0cDg.flac"
    from cm_helper import preprocess_audio
    y, sr = preprocess_audio(pb_source)
    y = np.clip(y, -1, 1)
    y_compressed = simulate_microphone_from_source(y,sr)
    import soundfile as sf
    sf.write("pb_compressed.wav", y_compressed, sr)
    






