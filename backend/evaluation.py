import pandas as pd

from .hasher import score_hashes

def evaluate_model(samples):
    """
    `samples: ["song_id": 1, "info": "...", "hashes": {...}]`
    """
    results = []
    for sample in samples:
        sample_clean = sample["clean"]
        sample_noise = sample["noise"]
        song_id = sample_clean["song_id"]
        scores_clean, _ = score_hashes(sample_clean["hashes"])
        scores_noise, _ = score_hashes(sample_noise["hashes"])
        results.append({
            "pred_clean": scores_clean[0][0],
            "pred_noise": scores_noise[0][0],
            "ground_truth": song_id,
            "youtube_url": sample_clean["youtube_url"]
            #"scores": scores
        })
    results_df = pd.DataFrame(results)
    results_df["correct_clean"] = results_df["pred_clean"] == results_df["ground_truth"]
    results_df["correct_noise"] = results_df["pred_noise"] == results_df["ground_truth"]
    # Using the definition of false positive and false negative 
    # as described in the paper
    # "A Highly Robust Audio Fingerprinting System" by Haitsma and Kalker:
    #
    # A false negative occurs when the fingerprints of 
    # perceptually similar audio clips are too different 
    # to lead to a positive match.
    results_df["false_negative"] = results_df["correct_clean"] & (~results_df["correct_noise"])

    #by_song_id_df = results_df.groupby("ground_truth")
    by_song_id_df = results_df.groupby("youtube_url")

    group_stats = by_song_id_df.agg({
        "correct_clean": "mean",
        "correct_noise": "mean"
    })
    group_stats["false_positive_rate_clean"] = 1 - group_stats["correct_clean"]
    group_stats["false_positive_rate_noise"] = 1 - group_stats["correct_noise"]


    n_true_positives_clean = results_df["correct_clean"].sum()
    n_true_positives_noise = results_df["correct_noise"].sum()
    n_false_negatives = results_df["correct_noise"].sum()
    print("================================")
    print(f"Number of True Positives (clean): {n_true_positives_clean} / {len(results)}")
    print(f"Number of True Positives (noise): {n_true_positives_noise} / {len(results)}")
    print(f"Number of False Negatives / Number of True Positives (clean)\n"
          f"   (correct on clean audio, but incorrect on noisy audio):\n"
          f"   {n_false_negatives} / {n_true_positives_clean}")
    print("================================")
    print(group_stats)
