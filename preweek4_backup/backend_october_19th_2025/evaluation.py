import pandas as pd

from .hasher_og import score_hashes

def evaluate_model(samples):
    """
    `samples: ["song_id": 1, "info": "...", "hashes": {...}]`
    """
    results = []
    for sample in samples:
        sample_clean = sample["source"]
        sample_mic = sample["mic"]
        song_id = sample_clean["song_id"]
        scores_clean, _ = score_hashes(sample_clean["hashes"])
        scores_mic, _ = score_hashes(sample_mic["hashes"])
        results.append({
            "pred_source": scores_clean[0][0],
            "pred_mic": scores_mic[0][0],
            "ground_truth": song_id,
            "youtube_url": sample_clean["youtube_url"]
            #"scores": scores
        })
    results_df = pd.DataFrame(results)
    results_df["correct_source"] = results_df["pred_source"] == results_df["ground_truth"]
    results_df["correct_noise"] = results_df["pred_mic"] == results_df["ground_truth"]
    # Using the definition of false positive and false negative 
    # as described in the paper
    # "A Highly Robust Audio Fingerprinting System" by Haitsma and Kalker:
    #
    # A false negative occurs when the fingerprints of 
    # perceptually similar audio clips are too different 
    # to lead to a positive match.

    # this might not be working correctly
    results_df["false_negative"] = results_df["correct_source"] & (~results_df["correct_mic"])

    #by_song_id_df = results_df.groupby("ground_truth")
    by_song_id_df = results_df.groupby("youtube_url")

    group_stats = by_song_id_df.agg({
        "correct_source": "mean",
        "correct_mic": "mean"
    })
    group_stats["false_positive_rate_source"] = 1 - group_stats["correct_source"]
    group_stats["false_positive_rate_mic"] = 1 - group_stats["correct_mic"]


    n_true_positives_source = results_df["correct_source"].sum()
    n_true_positives_mic = results_df["correct_mic"].sum()
    n_false_negatives = results_df["correct_mic"].sum()
    print("================================")
    print(f"Number of True Positives (source): {n_true_positives_source} / {len(results)}")
    print(f"Number of True Positives (mic): {n_true_positives_mic} / {len(results)}")
    print(f"Number of False Negatives / Number of True Positives (source)\n"
          f"   (correct on source audio, but incorrect on microphone audio):\n"
          f"   {n_false_negatives} / {n_true_positives_source}")
    print("================================")
    print(group_stats)
