import os

import sqlite3
import librosa
import glob
import pandas as pd
import zipfile

library = "sql/library.db"

def connect_to_db() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    con = sqlite3.connect(library)
    cur = con.cursor()
    return con, cur

def create_hash_index():
    with sqlite3.connect(library) as con:
        cur = con.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_hash_val ON hashes(hash_val)")
        con.commit()

def add_song(track_data: dict) -> None:
    with sqlite3.connect(library) as con:
        cur = con.cursor()

        audio_path = track_data["audio_path"]
        duration_s = librosa.get_duration(path=audio_path)

        cur.execute("""
            INSERT INTO songs 
                    (title, artist, album, artwork_url, spotify_url, youtube_url, release_date, duration_s, audio_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def find_tracks_dir(tracks_dir: str = None):
    if tracks_dir is not None:
        return tracks_dir
    tracks_dirs = glob.glob("tracks*")
    if tracks_dirs:
        tracks_dir = None
        non_zip_dirs = [d for d in tracks_dirs if not d.endswith(".zip")]
        if non_zip_dirs:
            tracks_dir = os.path.abspath(non_zip_dirs[0])
        else:
            zip_path = os.path.abspath(tracks_dirs[0])
            extract_dir = os.path.splitext(zip_path)[0]
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            tracks_dir = extract_dir
    else:
        raise FileNotFoundError(f"Could not find folder or zip archive matching the pattern 'tracks*' in {os.getcwd()}")


def add_songs(tracks_dir: str = None, n_songs: int = None, specific_songs: list[str] = None) -> None:
    if n_songs == 0:
        return
    
    tracks_dir = find_tracks_dir(tracks_dir)
        
    df = pd.read_csv(os.path.join(tracks_dir, "tracks.csv"))
    df["audio_path"] = df["audio_path"].apply(lambda x: os.path.join(tracks_dir, x))

    if n_songs is not None:
        df = df.head(n_songs)

    for row in df.to_dict(orient="records"):
        if specific_songs is not None: 
            if row["track_name"] in specific_songs:
                add_song(row)
        else:
            add_song(row)

def retrieve_song(song_id) -> dict|None:
    with sqlite3.connect(library) as con:
        df = pd.read_sql_query("SELECT * FROM songsdemo WHERE id = ?", con, params=(song_id,))
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        return row
    
def retrieve_song_ids() -> list[int]:
    with sqlite3.connect(library) as con:
        cur = con.cursor()
        cur.execute("SELECT id FROM songs ORDER BY id ASC")
        ids = [row[0] for row in cur.fetchall()]
    return ids
      
def add_hash(hash_val: int, time_stamp: float, song_id: int, cur: sqlite3.Cursor) -> None:
    cur.execute("""INSERT INTO hashes 
                    (hash_val, time_stamp, song_id) 
                    VALUES (?, ?, ?)""", 
                    (hash_val, time_stamp, song_id))
    
def add_hashes(hashes: dict[int, tuple[int, int]]):
    with sqlite3.connect(library) as con:
        cur = con.cursor()
        for address, (anchorT, song_id) in hashes.items():
            add_hash(address, anchorT, song_id, cur)
        con.commit()

def retrieve_hashes(hash_val: int, cur: sqlite3.Cursor) -> tuple[int, int, int]|None:
    result = cur.execute("SELECT hash_val, time_stamp, song_id FROM hashes WHERE hash_val = ?", (hash_val,)).fetchall()
    return result if result else None

def create_tables():
    if os.path.exists(library):
        os.remove(library)
    with sqlite3.connect(library) as con:
        cur = con.cursor()
        with open("sql/schema.sql", "r") as f:
            schema_sql = f.read()
        cur.executescript(schema_sql)
        con.commit()