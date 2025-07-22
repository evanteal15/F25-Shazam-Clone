import sqlite3

def add_song(song_data):
    # connect to local db
    con = sqlite3.connect("sql/library.db")
    cur = con.cursor()
    
    cur.execute("""
        INSERT INTO songs VALUES
            ({song_data[title]}, {song_data[album]})
        """)
    
    con.commit()
    
def retreive_song(song_id):
    pass
     
def edit_song():
    pass

def remove_song():
    pass
    
def add_hash():
    pass