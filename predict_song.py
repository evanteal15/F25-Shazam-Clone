from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import tensorflow as tf
from hasher import recognize_music
import scipy.io.wavfile as wavfile
import tempfile
import os
import librosa
from pydub import AudioSegment
import io
import subprocess

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
# @cross_origin()
def predict():
    print(request.files)
    if 'audio' not in request.files:
        print("No audio found")
        return jsonify({'error': 'No audio found'})
    
    audio_file = request.files['audio']
    
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
    
    # with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_file:
    #     # Save the uploaded file to the temporary file
    #     audio_file.save(temp_file.name)
    #     temp_file_path = temp_file.name
    
    # audio_file.seek(0)
    # audio_bytes = audio_file.read()
    
    # Use scipy to parse the WAV file properly
    sample_rate, audio_array = wavfile.read(temp_wav_path)
    
    # Convert to float32 and normalize
    audio_array = audio_array.astype(np.float32) / (2**15)
    
    print(f"Audio loaded: {len(audio_array)} samples at {sample_rate}Hz")
    
    # audio_bytes = audio_file.read()
    # # print(audio_bytes)
    # print(len(audio_bytes))
    # print("(((((((((())))))))))")
    # audio_array = np.frombuffer(audio_bytes, np.int16)
    # audio = tf.convert_to_tensor(audio_array, dtype=tf.float32)
    
    # sample_rate = 44100  # You'll need to know or detect the sample rateau
    # print(len(audio_array))
    # print("(((((((((())))))))))")
    # wavfile.write('output.wav', sample_rate, audio_array)
     
    if audio_file is None:
        return jsonify({'error': 'Could not read the audio file'}), 400
    
    # TODO: Get the prediction
    # audio_array, sample_rate = librosa.load(temp_wav_path, sr=44100)
    
    # print(f"Audio loaded: {len(audio_array)} samples at {sample_rate}Hz")
    
    scores, _ = recognize_music(temp_wav_path)
    
    print(scores[0])
    
    
    # Read the file data
    # audio_file.seek(0)
    # audio_data = audio_file.read()
    
    # # Load M4A with pydub
    # audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="m4a")
    
    # # Convert to numpy array
    # samples = audio_segment.get_array_of_samples()
    # audio_array = np.array(samples, dtype=np.float32)
    
    # # Normalize to [-1, 1] range (like librosa)
    # if audio_segment.sample_width == 2:  # 16-bit
    #     audio_array = audio_array / (2**15)
    # elif audio_segment.sample_width == 4:  # 32-bit
    #     audio_array = audio_array / (2**31)
    
    # # Convert stereo to mono if needed
    # if audio_segment.channels == 2:
    #     audio_array = audio_array.reshape((-1, 2)).mean(axis=1)
    
    # sample_rate = audio_segment.frame_rate
    
    # print(f"Audio processed: shape={audio_array.shape}, sample_rate={sample_rate}")
    
    return jsonify({
        'best': "heheha"
        # 'confidence': float(probs[0]),
        # 'class_id': int(top5[0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)