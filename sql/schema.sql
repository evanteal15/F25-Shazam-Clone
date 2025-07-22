CREATE TABLE songs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    artist TEXT,
    album TEXT,
    youtube_link TEXT,
    duration INTEGER
)

CREATE TABLE hashes(
    hash_val INTEGER PRIMARY KEY,
    time_stamp INTEGER NOT NULL,
    song_id INTEGER NOT NULL,
    FOREIGN KEY (song_id) REFERENCES songs(id)
)