from flask import Flask, render_template, request, jsonify
from  yolo import initialize_context, generate_response
from werkzeug.utils import secure_filename


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
'''
@app.route('/upload', methods=['POST'])

def upload():
    if request.method == 'POST':
        file = request.files['image']
        image_path = "uploads/" + file.filename
        file.save(image_path)
        context = perform_object_detection(image_path)  # Get the context dynamically
        print(context)
        return render_template('response.html', context=context, image_path=image_path)
'''
import os

upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(upload_folder, filename)
            file.save(image_path)
            context = initialize_context(image_path)  # Get the context dynamically
            print(context)
            return render_template('chatbot.html', context=context, image_path=image_path)
        return "No file uploaded", 400

@app.route('/query', methods=['POST'])
def query():
    if request.method == 'POST':
        query = request.form['query']
        print("abc")
        response = generate_response(request.form['context'], query)
        return jsonify({'response': response})

app.run(debug=True)