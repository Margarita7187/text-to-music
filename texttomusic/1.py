from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
import scipy
from transformers import pipeline

app = Flask(__name__)
os.makedirs('static/audio', exist_ok=True)

print("🎵 Загрузка MusicGen... (первый раз 3-5 минут, потом быстро)")
# Загружаем модель локально — она сама скачается в кеш
synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")
print("✅ Модель готова!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    return send_from_directory('static/audio', filename)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('text', 'эпическая оркестровая музыка')
    
    print(f"🎵 Генерируем: {prompt}")
    music = synthesiser(prompt, forward_params={"do_sample": True})
    
    filename = f"music_{uuid.uuid4().hex}.wav"
    filepath = os.path.join('static/audio', filename)
    scipy.io.wavfile.write(filepath, rate=music["sampling_rate"], data=music["audio"])
    
    return jsonify({'success': True, 'audio_url': f'/static/audio/{filename}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)