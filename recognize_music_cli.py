import pyaudio
import wave
import librosa
import sqlite3
from hasher import create_constellation_map, create_hashes, score_hashes

def recognize_music():
    #db_dict = {}
    #with sqlite3.connect('shazam.db') as conn:
        #cur = conn.cursor()

        #cur.execute("SELECT hash_val, time_val, song_id FROM hashes")# WHERE song_id IN (?, ?)", 

        #for hash, source_time, song_index in cur.fetchall():
            #if hash not in db_dict:
                #db_dict[hash] = []
            #db_dict[hash].append((source_time, song_index))
        #print(len(db_dict))


    Fs, song = librosa.load("output.wav", sr=None)
    constellation = create_constellation_map(song, Fs)
    hashes = create_hashes(constellation, None)

    #scores = score_hashes_against_database(hashes, db_dict)[:5]
    #scores = score_hashes(hashes, db_dict)[:3]
    scores = score_hashes(hashes)[:3]
    return scores


def record_audio(outfile: str ="output.wav", n_seconds: int = 5):
    """
    records audio using computer microphone
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Start recording...")

    frames = []
    for i in range(0, int(RATE / CHUNK * n_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("recording stopped")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(outfile, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

#def main():
    #songs = "./tracks-2025-07-22/tracks.csv"
    #df = pd.read_csv(songs)[["track_id", "track_name", "artist_name", "audio_path"]].head(3)
    #print(df.head())

    #create_db(df)
    #_ = input("Press enter to begin recording")
    #record_audio("output.wav")
    #scores = recognize_music()
    #for song_id, score in scores:
        #track_name = df.loc[df["track_id"] == song_id, "track_name"].iloc[0]
        #print(f"{track_name}: Score of {score[1]} at {score[0]}")







#if __name__ == "__main__":
    #main()