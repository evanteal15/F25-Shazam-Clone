import os
from xmlrpc import server

import mysql.connector
import librosa
import glob
import pandas as pd
import zipfile
import pyodbc

from dotenv import load_dotenv

load_dotenv(dotenv_path='/home/evanteal15/F25-Shazam-Clone/env/.env')

db_name = "shazamesque"

def connect(): #-> tuple[sqlite3.Connection, sqlite3.Cursor]:
    #con = sqlite3.connect(library)
    
    server = 'tcp:shazesq.database.windows.net,1433'
    database = 'shazamesque'
    username = os.getenv('USER_NAME')
    password = os.getenv('THE_PASSWORD')
    driver = '{ODBC Driver 18 for SQL Server}'

    connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    # Driver={ODBC Driver 18 for SQL Server};Server=tcp:shazesq.database.windows.net,1433;Database=shazamesque;Uid=CloudSAce2ffd30;Pwd={your_password_here};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;
    
    # con = mysql.connector.connect(
    #     host="localhost",
    #     user="root",
    #     database=db_name,
    #     password="password"
    # )
    con = pyodbc.connect(connection_string)

    return con

def create_hash_index():
    with connect() as con:
        cur = con.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_hash_val ON hashes(hash_val) USING HASH;")
        con.commit()

def add_song(track_data: dict) -> str:
    """
    track_data = {
        "youtube_url": https://youtube.com/watch?v=video_id
        "title": name of track, or youtube video title
        "artist": name of artist, or youtube video author
        "artwork_url": url to spotify album art, or youtube thumbnail
        "audio_path": local path to mp3 file
    }

    returns corresponding song id via SELECT last_insert_rowid();
    """

    #https://img.youtube.com/vi/_r-nPqWGG6c/0.jpg


    with connect() as con:
        cur = con.cursor()

        youtube_url = track_data["youtube_url"]
        audio_path = track_data["audio_path"]
        with open(audio_path, 'rb') as f:
            waveform = f.read()
        duration_s = librosa.get_duration(path=audio_path)


        cur.execute("""
            INSERT INTO songs 
                    (youtube_url, title, artist, artwork_url, waveform, duration_s)
                    VALUES (?, ?, ?, ?, ?, ?)
            """, (
                youtube_url,
                track_data["title"],
                track_data["artist"],
                track_data["artwork_url"],
                waveform,
                duration_s,
            )
        )
        con.commit()
        cur.execute("SELECT last_insert_rowid()")
        song_id = cur.fetchone()[0]
        return song_id



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
            if row["title"] in specific_songs:
                add_song(row)
        else:
            add_song(row)

def retrieve_song(song_id) -> dict|None:
    with connect as con:
        df = pd.read_sql_query("SELECT * FROM songsdemo WHERE id = ?", con, params=(song_id,))
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        return row
    
def retrieve_song_ids() -> list[int]:
    with connect as con:
        cur = con.cursor()
        cur.execute("SELECT id FROM songs ORDER BY id ASC")
        ids = [row[0] for row in cur.fetchall()]
    return ids
      
def add_hash(hash_val: int, time_stamp: float, song_id: int, cur: "mysqlcursor") -> None:
    cur.execute("""INSERT INTO hashes 
                    (hash_val, time_stamp, song_id) 
                    VALUES (?, ?, ?)""", 
                    (hash_val, time_stamp, song_id))
    
def add_hashes(hashes: dict[int, tuple[int, int]]):
    with connect() as con:
        cur = con.cursor()
        for address, (anchorT, song_id) in hashes.items():
            add_hash(address, anchorT, song_id, cur)
        con.commit()

def retrieve_hashes(hash_val: int, cursor) -> tuple[int, int, int]|None:
    result = cursor.execute("SELECT hash_val, time_stamp, song_id FROM hashes WHERE hash_val = ?", (hash_val,)).fetchall()
    return result if result else None

def create_tables():
    con = mysql.connector.connect(
        host="localhost",
        user="root",
    )
    cur = con.cursor()
    cur.execute(f"CREATE DATABASE {db_name}")
    cur = con.cursor()
    with open("sql/schema.sql", "r") as f:
        schema_sql = f.read()
    cur.executescript(schema_sql)
    con.commit()
    con.close()

