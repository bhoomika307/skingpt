from flask import Flask, render_template, request, jsonify, send_from_directory
from yolo import initialize_context
from memory import store_diagnosis, search_similar, get_memory_context
from orchestrator import orchestrate_initial, orchestrate_chat, orchestrate_followup
from werkzeug.utils import secure_filename
import os
import json

app = Flask(__name__)

upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory('uploads', filename)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(upload_folder, filename)
            file.save(image_path)

            # Detect disease
            context, disease, confidence = initialize_context(image_path)

            # Search memory for similar past cases
            similar_cases = search_similar(image_path)
            memory_context = get_memory_context(similar_cases)

            # Store this diagnosis in memory
            store_diagnosis(image_path, disease, confidence)

            # Combine detection + memory into full context
            full_context = f"{context}\n{memory_context}"

            # Run the agent loop — it will THINK → ACT → OBSERVE → repeat
            result = orchestrate_initial(
                disease, confidence, image_path, memory_context, similar_cases
            )

            # Log the agent's reasoning trace
            for i, step in enumerate(result.get('transcript', []), 1):
                print(f"  Step {i}: {step['action']} → {step['observation'][:80]}")

            return render_template('chatbot.html',
                                   context=full_context,
                                   disease=disease,
                                   confidence=confidence,
                                   similar_cases=similar_cases,
                                   image_path=image_path,
                                   memory_context=memory_context,
                                   initial_action=result['action'],
                                   initial_content=result['content'],
                                   transcript=json.dumps(result.get('transcript', [])))
        return "No file uploaded", 400


@app.route('/query', methods=['POST'])
def query():
    if request.method == 'POST':
        user_query = request.form['query']
        full_context = request.form['context']

        # Case metadata for the agent
        disease = request.form.get('disease', '')
        confidence = float(request.form.get('confidence', 0))
        image_path = request.form.get('image_path', '')
        memory_context = request.form.get('memory_context', '')
        followup_question = request.form.get('followup_question', '')

        if followup_question:
            # Resume agent loop with user's follow-up answer
            result = orchestrate_followup(
                disease, confidence, image_path, memory_context,
                user_query, followup_question
            )
        else:
            # New chat question — run agent loop
            result = orchestrate_chat(
                full_context, user_query,
                disease=disease, confidence=confidence,
                image_path=image_path, memory_context=memory_context
            )

        # Log trace
        for i, step in enumerate(result.get('transcript', []), 1):
            print(f"  Step {i}: {step['action']} → {step['observation'][:80]}")

        return jsonify({
            'response': result['content'],
            'action': result['action'],
            'thought': result.get('thought', ''),
            'steps': len(result.get('transcript', [])),
        })


app.run(debug=True)
