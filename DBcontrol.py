import sqlite3
import librosa
import os
import glob
import pandas as pd

library = "sql/library.db"

def add_song(track_data: dict, tracks_dir: str) -> None:
    # connect to local db
    with sqlite3.connect(library) as con:
        cur = con.cursor()

        audio_path = os.path.join(tracks_dir, track_data["audio_path"])
        duration_s = librosa.get_duration(audio_path)

        cur.execute("""
            INSERT INTO songs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                track_data["track_name"],
                track_data["artist_name"],
                track_data["album_name"],
                track_data["image_url"],
                f"https://open.spotify.com/track/{track_data['track_id']}",
                f"https://www.youtube.com/watch?v={track_data['video_id']}",
                track_data["release_date"],
                duration_s,
                audio_path
            )
        )
        con.commit()

def add_songs(tracks_dir: str = None) -> None:
    if tracks_dir is None:
        tracks_dirs = glob.glob("tracks-*")
        if tracks_dirs:
            tracks_dir = os.path.abspath(tracks_dirs[0])
        else:
            raise FileNotFoundError(f"Could not find tracks folder in {os.getcwd()}")
        
    df = pd.read_csv(os.path.join(tracks_dir, "tracks.csv"))
    for row in df.to_dict(orient="records"):
        add_song(row)
    
def retrieve_song(song_id) -> dict|None:
    con = sqlite3.connect(library)
    df = pd.read_sql_query("SELECT * FROM songs WHERE id = ?", con, params=(song_id,))
    if df.empty:
        con.close()
        return None
    row = df.iloc[0].to_dict()
    con.close()
    return row
     
def remove_song(song_id) -> None:
    with sqlite3.connect(library) as con:
        cur = con.cursor()
        cur.execute("DELETE FROM songs WHERE id = ?", (song_id,))
        con.commit()
    
def add_hash(hash_val: int, time_stamp: float, song_id: int) -> None:
    # todo: do some testing to see if it's possible to have hash_val
    # collisions (different spotify track but same youtube audio)
    # if so, handle collisions by appending to a list
    with sqlite3.connect(library) as con:
        cur = con.cursor()
        cur.execute("INSERT INTO hashes VALUES (? ? ?)", (hash_val, time_stamp, song_id))
        con.commit()

def retrieve_hash(hash_val: int) -> tuple[int, float, int]|None:
    pass