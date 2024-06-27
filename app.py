from flask import Flask, jsonify, request
import os
from image_recognition import reconized_image
from flask_cors import CORS,cross_origin

app = Flask(__name__)

CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/upload/', methods=['POST'])
@cross_origin(origin='*')
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = file.filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            data = reconized_image(img_path)
            os.remove(img_path)
            return jsonify({'data': data, 'filename': filename}), 201
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/')
def home():
    return "Welcome to the Flask API!"



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
