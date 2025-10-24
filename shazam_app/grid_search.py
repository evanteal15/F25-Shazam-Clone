import numpy as np

from cm_helper import create_samples, add_noise
from DBcontrol import init_db, connect

from const_map import create_constellation_map
from hasher import create_hashes
from search import score_hashes

from parameters import set_parameters

microphone_sample_list = [
    ("Plastic Beach", "Gorillaz", "audio_samples/plastic_beach_microphone_recording.wav")
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
            "WHERE title ILIKE %?% "  # could use LIKE, could use =
            "AND artist ILIKE %?% ",
            (sample["title"], sample["artist"])
        )

        song_id = cur.fetchall()[0]["song_id"]
        sample_slices = create_samples(sample["audio_path"], sr, n_samples = 20)
        sample_slices_noisy = [add_noise(audio, noise_weight) for audio in sample_slices]
        samples.extend([
            {"song_id": song_id,
             "microphone": s[0],
             #"mic_noisy": s[1]
             } for s in zip(sample_slices, sample_slices_noisy)
        ])
    return samples



def perform_recognition_test():
    """
    """
    #init_db()
    sr = 11025
    init_db(n_songs=2)
    results = []
    samples = augment_samples(sr=sr, noise_weight=0.3)
    for sample in samples:
        result = {}
        ground_truth_song_id = sample["song_id"]

        # peaks -> hashes -> scores
        constellation_map = create_constellation_map(sample["microphone"], sr=sr)
        hashes = create_hashes(constellation_map, None, sr)
        scores, time_pair_bins = score_hashes(hashes)
        result = {
            "ground_truth": ground_truth_song_id,
            "prediction": scores[0][0],
            "correct": ground_truth_song_id == scores[0][0]
        }

        results.append(result)


def std_of_deltaT(song_id, time_pair_bins, sr=11025):
    """
    heuristic that measures the distribution of deltaT values

    results in standard deviation between 0 ms and 1000 ms

    lower is better - suggests that the time offsets are behaving consistently
    """
    time_pair_bin = time_pair_bins[song_id]
    deltaT_vals = sorted([sourceT-sampleT for (sourceT, sampleT) in time_pair_bin])

    # 1) take the difference in time between sourceT and sampleT
    #    values represent the offset suggested by the presence of a hash match
    # ex: offset = 40
    # data might look like [-4, 5, 8, 40, 40, 40, ..., 40, 40, 40, 56, 68, 80]
    first_order_differences = [deltaT_vals[i] - deltaT_vals[i-1] 
                                for i in range(1, len(deltaT_vals))]

    # 2) then, take the difference between adjacent differences
    #    removes influence of offset
    #    values are close to zero for consistent time offsets
    # ex: [9, 3, 0, 0, 0, ..., 0, 0, 16, 12, 12]
    second_order_differences = [first_order_differences[i] - first_order_differences[i-1]
                                for i in range(1, len(first_order_differences))]

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

    return np.max(np.std(second_order_differences) * 1000 * (hop_length/sr), 1000)

def count_hash_matches(song_id, time_pair_bins, n_sample_hashes):
    """
    naive heuristic that counts the number of hash matches between sample and source
    
    divides by the number of sample hashes to give a number between 0 and 1

    multiply by `n_sample_hashes` to get number of matches

    note: there can be repeated matches for a single hash value present in sample
    - cap metric at one if n_matches > n_sample_hashes

    higher is better - suggests that many hashes show up in both sample and source
    """
    n_matches = len(time_pair_bins[song_id]) 
    return np.max(n_matches / n_sample_hashes, 1)


def compute_performance_metrics():
    pass


def run_grid_search():
    pass


if __name__ == "__main__":
    run_grid_search()