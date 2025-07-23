CREATE TABLE songs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    artist TEXT,
    album TEXT,
    artwork_url TEXT,
    spotify_url TEXT UNIQUE,
    youtube_url TEXT,
    release_date TEXT,
    duration_s FLOAT NOT NULL,
    audio_path TEXT NOT NULL,
)

CREATE TABLE hashes(
    hash_val INTEGER PRIMARY KEY,
    time_stamp INTEGER NOT NULL,
    song_id INTEGER NOT NULL,
    FOREIGN KEY (song_id) REFERENCES songs(id)
)