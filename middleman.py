from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import ast
from PIL import Image
import io
from hasher import recognize_music

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    """
    TODO: Implement a fruit classification endpoint that:
    1. Accepts an image file
    2. Preprocesses the image
    3. Makes a prediction using the model
    4. Returns the predicted fruit with confidence score
    """
    
    # TODO: Check if image is provided in the request
    print(request.files)
    # Return error if no audio is found
    if 'song' not in request.files:
        print('No audio found')
        return jsonify({'error': 'No audio found'}), 400
    
    # TODO: Read and decode the audio
    audio_file = request.files['song']
    audio_bytes = audio_file.read()
    audio_array = np.frombuffer(audio_bytes, np.int16)
    
    
    # TODO: convert our audio into a wav function
    # audio_bytes = audio_file.read()
    # audio_array = np.frombuffer(audio_bytes, np.int16)
    # audio = tf.convert_to_tensor(audio_array, dtype=tf.float32)

    if audio_file is None:
        return jsonify({'error': 'Could not read the audio file'}), 400
    
    # TODO: Get the prediction
    scores, _ = recognize_music(audio_file)
    
    ans = scores[0]
    
    # TODO: Load the model
    # Hint: Use try-except for error handling
    # try:
    #     model = tf.keras.models.load_model('fruitclassifier.keras')
    # except Exception as e:
    #     print('cant open model')
    #     return jsonify({'error': 'cant open model'}), 500
    
    # # TODO: Make prediction
    # # Hint: Use model.predict() and handle exceptions
    # try:
    #     predictions = model.predict(np.expand_dims(file, 0))
    # except Exception as e:
    #     print('prediction failed')
    #     return jsonify({'error': 'Prediction failed'}), 500
    
    # # TODO: Get top 5 predictions
    # # Hint: Use np.argsort()
    # top5 = np.argsort(predictions[0])[-5:][::-1]
    # probs = predictions[0][top5]

    # print('top5:', top5)
    
    # TODO: Load fruits dictionary from 'Backend/directories.txt'
    # Hint: Use ast.literal_eval()
    # with open('directories.txt', 'r') as f:
    #     fruits = ast.literal_eval(f.read())
    
    # TODO: Return prediction
    # Format: {
    #   'fruit': fruit_name,
    #   'confidence': confidence_score,
    #   'class_id': class_id
    # }

    return jsonify({
        'best': ans
        # 'confidence': float(probs[0]),
        # 'class_id': int(top5[0])
    })

    # for pred_class, confidence in zip(top5, probs):
    #     pred_class = int(pred_class)
    #     if pred_class in fruits:
    #         whole_fruit = fruits[pred_class]
    #         fruit = whole_fruit.split()[0]
    #         return jsonify({
    #             'fruit': fruit,
    #             'confidence': float(confidence),
    #             'class_id': pred_class
    #         })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
