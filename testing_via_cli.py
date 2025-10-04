#!/usr/bin/env python

import os

from musicdl import MusicDownloader
from musicdl.dataloader import extract_zip, create_zip, load

from backend.hasher_og import compute_source_hashes, add_noise, create_samples, create_constellation_map, create_hashes

from backend.DBcontrol import create_tables, add_songs, retrieve_song_id

from backend.evaluation import evaluate_model

from backend.recorder import record_audio

def create_manual_test_set():
    pass

def download_week3_data():
    mdl = MusicDownloader(audio_directory="./week3tracks", audio_format="flac", use_ytdlp_cli=True)
    mdl.download([
        "https://open.spotify.com/playlist/4b1BfkHbghTEN0fUN0KGtf?si=68cc7a74eb9740e7"
    ])
    create_zip(zip_file="./week3tracks.zip", audio_directory="./week3tracks")

def download_week3_data_tiny():
    mdl = MusicDownloader(audio_directory="./week3tracks_tiny", audio_format="mp3", use_ytdlp_cli=True)
    mdl.download([
        "https://open.spotify.com/playlist/4b1BfkHbghTEN0fUN0KGtf?si=8b5de18bf2764b2e"
    ])
    create_zip(zip_file="./week3tracks_tiny.zip", audio_directory="./week3tracks_tiny")

def main():
    # 1) create dataset of WAV files, loadable via zip file loader function (no dependencies)
    # default file format is flac
    #mdl = MusicDownloader()
    #mdl.download([
        #"https://www.youtube.com/watch?v=5K6fO9Bdlgg",
        #"https://www.youtube.com/watch?v=CI2Oa1Ds6Z8",
        #"https://www.youtube.com/watch?v=iVzXMytwCCo",
        #"https://www.youtube.com/watch?v=THV2TDARpS0",
        #"https://www.youtube.com/watch?v=MXtQ5I0asKg",
        #"https://open.spotify.com/album/0ESBFn4IKNcvgD53QJPlpD?si=PNFoDMvHTi6IaT6ia-s1Jw"

    #])

    # for dumping and loading zip archives:
    # create_zip(zip_file="./tracks.zip", audio_directory="./tracks")
    # if not os.path.exists("./tracks") and os.path.exists("./tracks.zip"):
        # extract_zip(zip_file="./tracks.zip", audio_directory="./tracks")

    tracks_info = load(audio_directory="./tracks")

    # 2) function to create fingerprints of all audio files
    #create_tables()
    #add_songs(audio_directory = "./tracks")

    ##for sr in (None, 11025):
    sr = 11025
    #sr = compute_source_hashes(resample_rate=sr)

    test_set = []

    for track in tracks_info:
        n_seconds = 5
        seed = 1
        song_id = retrieve_song_id(track["youtube_url"])
        samples = create_samples(track, sr=sr, n_samples=5, n_seconds=n_seconds, seed=seed)
        for sample in samples:
            test_sample = {"clean": None, "noise": None}
            # 3) function to create labeled test set of 5s samples from dataset songs
            constellation_map = create_constellation_map(sample, sr)
            hashes = create_hashes(constellation_map, song_id, sr)
            test_sample["clean"] = {"title": track["title"], "youtube_url": track["youtube_url"], "song_id": song_id, "info": f"noiseweight0_sr{sr}_nseconds{n_seconds}_seed{seed}", "hashes": hashes}

            # 4) function to create labeled test set of same 5s samples with added noise
            noise_weight = 0.5
            sample = add_noise(sample)
            constellation_map = create_constellation_map(sample, sr)
            hashes = create_hashes(constellation_map, song_id, sr)
            test_sample["noise"] = {"title": track["title"], "youtube_url": track["youtube_url"], "song_id": song_id, "info": f"noiseweight{noise_weight}_sr{sr}_nseconds{n_seconds}_seed{seed}", "hashes": hashes}
            test_set.append(test_sample)

    # 5) manual dataset of labeled microphone samples (play via phone into macbook mic)
    # test_set_manual = []

    # 6) an evaluator to display false negative rate with/without noise augmentation
    evaluate_model(test_set)

    #================================
    #Number of True Positives (clean): 34 / 40
    #Number of True Positives (noise): 33 / 40
    #Number of False Negatives / Number of True Positives (clean)
    #(correct on clean audio, but incorrect on noisy audio):
    #33 / 34  <- this is incorrect for some reason
    #================================
                                                #correct_clean  correct_noise  false_positive_rate_clean  false_positive_rate_noise
    #youtube_url                                                                                                                    
    #https://www.youtube.com/watch?v=5K6fO9Bdlgg            1.0            1.0                        0.0                        0.0
    #https://www.youtube.com/watch?v=CI2Oa1Ds6Z8            0.8            0.4                        0.2                        0.6
    #https://www.youtube.com/watch?v=H0gtQyD0BKc            0.4            0.8                        0.6                        0.2
    #https://www.youtube.com/watch?v=MXtQ5I0asKg            1.0            0.8                        0.0                        0.2
    #https://www.youtube.com/watch?v=PZr9g8KezlQ            0.8            1.0                        0.2                        0.0
    #https://www.youtube.com/watch?v=THV2TDARpS0            0.8            0.6                        0.2                        0.4
    #https://www.youtube.com/watch?v=YWMVwL-Jpr4            1.0            1.0                        0.0                        0.0
    #https://www.youtube.com/watch?v=iVzXMytwCCo            1.0            1.0                        0.0                        0.0

if __name__ == "__main__":
    #download_week3_data()
    #download_week3_data_tiny()
    main()
