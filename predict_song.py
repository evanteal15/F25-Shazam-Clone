from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from hasher import recognize_music, compute_source_hashes, init_db
from DBcontrol import retrieve_song
import scipy.io.wavfile as wavfile
import tempfile
import os
from pydub import AudioSegment
import io
import subprocess
import yt_dlp
import sqlite3
from pytube import YouTube
import ffmpeg

app = Flask(__name__)
CORS(app)

library = "sql/library.db"
idx = 0

@app.route('/predict', methods=['POST'])
def predict():
    # get audio from the request
    if 'audio' not in request.files:
        print("No audio found")
        return jsonify({'error': 'No audio found'})
    
    audio_file = request.files['audio']
    
    # Create a temporary file for the webm conversion
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_webm:
        audio_file.save(temp_webm.name)
        temp_webm_path = temp_webm.name
    
    # Convert to WAV using FFmpeg
    temp_wav_path = temp_webm_path.replace('.webm', '.wav')
    
    subprocess.run([
        'ffmpeg', '-i', temp_webm_path,
        '-ar', '44100',  # Sample rate
        '-ac', '1',      # Mono
        '-y',            # Overwrite
        temp_wav_path
    ], check=True, capture_output=True)
    
    
    # Use scipy to parse the WAV file properly
    sample_rate, audio_array = wavfile.read(temp_wav_path)
    
    # Convert to float32 and normalize
    audio_array = audio_array.astype(np.float32) / (2**15)
    
    print(f"Audio loaded: {len(audio_array)} samples at {sample_rate}Hz")
    
     
    if audio_file is None:
        return jsonify({'error': 'Could not read the audio file'}), 400
    
    # TODO: Get the prediction
    # audio_array, sample_rate = librosa.load(temp_wav_path, sr=44100)
    
    # print(f"Audio loaded: {len(audio_array)} samples at {sample_rate}Hz")
    
    scores, _ = recognize_music(temp_wav_path)

    # top5 = scores[:5]

    print(scores)
    print(scores[0])
    print(scores[0][0])
    
    urls = []
    for id in scores:
        urls.append(retrieve_song(id[0])["youtube_url"])

    return jsonify({
        'best': scores[0][0],
        'confidence': float(scores[0][1]),
        'urls': urls[0]
    })
    
@app.route('/add', methods=['POST'])
def add_song():
    song_data = request.json
    if not song_data:
        return jsonify({'error': 'Could not read the audio file'}), 400
    
    # TODO: song_data contains the youtube url for a song
    # get a wav file corresponding to this song and try to retrieve the artist and song name
    
    youtube_url = song_data.get('youtube_url')
    
    con = sqlite3.connect(library)
    cur = con.cursor()

    if cur.execute("SELECT COUNT(*) FROM songsdemo WHERE youtube_url = ?", (youtube_url,)).fetchone()[0] != 0:
        print("Song already exists in database")
        return jsonify({'error': 'Song already exists in database'}), 400
    
    global idx
    ydl_opts = {
        'format': 'bestaudio/best',
        'cookiefile': 'cookies.txt',
        'outtmpl': f'output_{idx}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }
    idx += 1

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        
        ydl.download([youtube_url])
        stream = ffmpeg.input(f'output_{idx-1}.m4a')
        stream = ffmpeg.output(stream, f'output_{idx-1}.wav')

    # print(f"Attempting to fetch: {youtube_url}")
    
    # yt = YouTube(youtube_url)
    
    # print(f"Attempting to fetch: {yt}")
    # audio_stream = yt.streams.filter(only_audio=True).first()
    # downloaded_file = audio_stream.download(output_path="audio")
    
    # # Step 2: Convert to WAV
    # # pydub can open MP4, WebM (and will invoke ffmpeg)
    # file_root, ext = os.path.splitext(downloaded_file)
    # wav_file = file_root + ".wav"
    # audio = AudioSegment.from_file(downloaded_file)
    # audio.export(wav_file, format="wav")
    
    con = sqlite3.connect(library)
    cur = con.cursor()

    print("Inserting song into database")

    cur.execute("""
        INSERT INTO songsdemo (youtube_url, audio_path)
        VALUES (?, ?)
    """, (youtube_url, f'output_{idx-1}.wav'))
    con.commit()
    
    
    # TODO: add the wav file hashes to the database
    # retrieve the id of the most recently added song
    cur.execute("SELECT last_insert_rowid()")
    song_id = cur.fetchone()[0]
    print(f"Added song with ID: {song_id}")
    
    compute_source_hashes([song_id])
    
    
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     ydl_opts = {
    #         'format': 'bestaudio/best',
    #         'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
    #         'postprocessors': [{
    #             'key': 'FFmpegExtractAudio',
    #             'preferredcodec': 'wav',
    #         }],
    #     }
        
    #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    #         ydl.download([youtube_url])
            
    #         # Find the wav file
    #         wav_file = os.path.join(temp_dir, 'audio.wav')
            
    #         # add the wav file to database
    return jsonify({'status': 'Song added successfully'}), 200
            

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5003)