import pyaudio
import wave
from hasher import init_db, recognize_music, visualize_map_interactive
from DBcontrol import retrieve_song
import argparse

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
    for _ in range(0, int(RATE / CHUNK * n_seconds)):
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

def shazamesque():
        input("🔊 Tap (Enter) to Shazamesque\n")
        print("🎤 Listening for music", end="\r")
        sample_wav_path = record_audio()
        print("🔎 Searching for a match", end="\r")
        scores = recognize_music(sample_wav_path)
        song_id = max(scores, key=scores.get)
        display_result(song_id)
        display_scores(scores)

def main():
    parser = argparse.ArgumentParser(description="Music recognition CLI")
    parser.add_argument('--init', action='store_true', help='Initialize the database and exit')
    parser.add_argument('--recognize', action='store_true', help='Run the Shazamesque Algorithm on repeat')
    parser.add_argument('--visualize', action='store_true', help='Visualize constellation map overlaid on spectrogram')
    args = parser.parse_args()

    if args.init:
        init_db(tracks_dir="tracks-2025-07-22")
        print("Database initialized with songs and hashes.")
        return
    elif args.visualize:
        visualize_map_interactive("sacrifice.mp3")
        return
    elif args.recognize:
        print("Ctrl+C to exit")
        try:
            while True:
                shazamesque()
        except KeyboardInterrupt:
            return
    else:
        parser.print_help()
        return

if __name__ == "__main__":
    main()