from DBcontrol import connect
import sqlite3
import pandas as pd
from cm_helper import preprocess_audio
from hasher import create_hashes
from const_map import create_constellation_map
import librosa

# TODO: Implement this function to check if a song with the given YouTube URL already exists in the database
def check_if_song_exists(youtube_url: str) -> bool:
    with connect() as con:
        cur = con.cursor()
        
        # TODO: Write a SQL query to count how many songs have the given YouTube URL
        query = "SELECT COUNT(*) FROM songs WHERE youtube_url = ?"
        cur.execute(query, (youtube_url,))
        count = cur.fetchone()[0]
        return count > 0
    
def add_song(track_info: dict, resample_rate: None|int = 11025) -> int:
    with connect() as con:
        cur = con.cursor()

        # get the duration of the audio file
        audio_path = track_info["audio_path"]
        duration_s = librosa.get_duration(path=audio_path)
        
        # TODO: Insert the song metadata into the songs table
        cur.execute("""
            INSERT INTO songs 
                    (youtube_url, title, artist, artwork_url, audio_path, duration_s)
                    VALUES (?, ?, ?, ?, ?, ?)
            """, (
                track_info["youtube_url"],
                track_info["title"],
                track_info["artist"],
                track_info["artwork_url"],
                audio_path,
                duration_s,
            )
        )
        con.commit()

        # select the last inserted song id
        cur.execute("SELECT last_insert_rowid()")
        song_id = cur.fetchone()[0]
        
        audio, sr = preprocess_audio(track_info["audio_path"], sr=resample_rate)
        constellation_map = create_constellation_map(audio, sr)
        hashes = create_hashes(constellation_map, song_id, sr)
        
        # TODO: Insert the hashes into the fingerprints table
        for hash in hashes:
            cur.execute("""
                INSERT INTO hashes (song_id, hash_val, offset)
                VALUES (?, ?, ?)
            """, (hash[0], hash[1], hash[2]))
        con.commit()
        
        return song_id
        