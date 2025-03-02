from flask import Flask, render_template, request
from jinja2 import Template
from deepface import DeepFace
import os
import pickle
from keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained deep learning model
model = load_model("deepface_model.h5")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        left_images = request.files.getlist('left_images')
        right_image = request.files['right_image']
        
        left_image_paths = []
        for img in left_images:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            img.save(img_path)
            left_image_paths.append(img_path)
        
        right_image_path = os.path.join(app.config['UPLOAD_FOLDER'], right_image.filename)
        right_image.save(right_image_path)
        
        results = compare_images(left_image_paths, right_image_path)
        
        # Save results to a pickle file
        with open('deepface_results.pkl', 'wb') as file:
            pickle.dump(results, file)
        
        return render_template('index.html', results=results, left_images=left_image_paths, right_image=right_image_path)
    
    return render_template('index.html', results=None, left_images=[], right_image=None)

def compare_images(left_images, right_image):
    results = []
    for img_path in left_images:
        try:
            result = DeepFace.verify(img_path, right_image, model_name='VGG-Face')
            similarity = result["distance"]
            percentage_similarity = (1 - similarity) * 100
            results.append({
                'image': img_path,
                'verified': result['verified'],
                'similarity': f"{percentage_similarity:.2f}%"
            })
        except Exception as e:
            results.append({'image': img_path, 'error': str(e)})
    return results

@app.route('/load_results', methods=['GET'])
def load_results():
    try:
        with open('deepface_results.pkl', 'rb') as file:
            results = pickle.load(file)
    except FileNotFoundError:
        results = None
    return render_template('index.html', results=results, left_images=[], right_image=None)

if __name__ == '__main__':
    app.run(debug=True)
