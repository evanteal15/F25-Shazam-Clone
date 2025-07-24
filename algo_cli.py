import os
import sys

import pyaudio
import wave
from hasher import create_constellation_map, \
                   compute_source_hashes, create_hashes, score_hashes, preprocess_audio
from DBcontrol import init_db, retrieve_song

def display_result(song_id: int):
    song = retrieve_song(song_id)
    hline = "="*43
    if song is not None:
        s = "\n".join([
            hline,
            "",
            "You are listening to:".center(len(hline)),
            song["title"].center(len(hline)),
            "by".center(len(hline)),
            song["artist"].center(len(hline)),
            "",
            song["youtube_url"],
            "",
            hline,
            "",
            "ALBUM ART:".center(len(hline)),
            song["artwork_url"],
            "",
        ])
        print(s)
    else:
        raise ValueError("song_id not found")

def display_scores(scores):
    for song_id, score in scores[:5]:
        song_name = retrieve_song(song_id)["title"]
        print(f"{song_name}: Score of {score[1]}")

def record_audio(n_seconds: int = 5) -> str:
    """
    records audio using computer microphone
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000

    outfile = "sample.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    for i in range(0, int(RATE / CHUNK * n_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(outfile, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return outfile

def recognize_music() -> list[tuple[int,int]]:
    print("🎤 Listening for music", end="\r")
    sample_path = record_audio()

    print("🔎 Searching for a match", end="\r")
    sample, sr = preprocess_audio(sample_path)
    os.remove(sample_path)
    constellation = create_constellation_map(sample, sr)
    hashes = create_hashes(constellation, None, sr)
    scores = score_hashes(hashes)
    display_result(scores[0][0])
    #display_scores(scores)

def main():
    init_db(tracks_dir = "tracks-2025-07-22", n_songs = 5)
    compute_source_hashes()

    input("🔊 Tap (Enter) to recognize_music()\n")
    recognize_music()

if __name__ == "__main__":
    main()