from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.file:
        print("No audio found")
        return jsonify({'error': 'No audio found'})
    
    audio_file = request.file['audio']
    
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)