CREATE TABLE songs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    youtube_url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    artwork_url TEXT NOT NULL,
    waveform MEDIUMBLOB NOT NULL,
    duration_s FLOAT NOT NULL
);

CREATE TABLE hashes(
    hash_val INTEGER NOT NULL,
    time_stamp INTEGER NOT NULL,
    song_id INTEGER NOT NULL,
    FOREIGN KEY (song_id) REFERENCES songs(id)
);