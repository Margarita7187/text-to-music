import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

# Импорт библиотек для модели
try:
    from transformers import pipeline
    import scipy.io.wavfile as wavfile
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Библиотека transformers не установлена. Установите: pip install transformers")

app = Flask(__name__)

# Глобальная переменная для модели
music_generator = None

def load_model():
    """Загрузка модели MusicGen (первый раз скачивается)"""
    global music_generator
    if not TRANSFORMERS_AVAILABLE:
        return False
    
    try:
        print("=" * 50)
        print("🎵 Загрузка модели MusicGen...")
        print("Это первый запуск. Модель скачивается ~1.5GB")
        print("Ожидайте 10-15 минут...")
        print("=" * 50)
        
        # Загрузка модели для генерации музыки
        music_generator = pipeline(
            "text-to-audio", 
            model="facebook/musicgen-small",
            device="cpu"  # Используем CPU (для компьютера без видеокарты)
        )
        
        print("=" * 50)
        print("✅ Модель успешно загружена!")
        print("🎶 Теперь можно генерировать музыку")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return False

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/status')
def status():
    """Проверка статуса модели"""
    if music_generator is None:
        return jsonify({'loaded': False, 'message': 'Модель не загружена'})
    return jsonify({'loaded': True, 'message': 'Модель готова к работе'})

@app.route('/generate', methods=['POST'])
def generate_music():
    """Генерация музыки из текста"""
    global music_generator
    
    # Проверка загрузки модели
    if music_generator is None:
        # Пытаемся загрузить модель
        if not load_model():
            return jsonify({
                'success': False,
                'error': 'Модель не загружена. Проверьте установку библиотек и интернет-соединение.'
            }), 500
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Введите текст для генерации музыки'}), 400
    
    # Ограничение длины текста для экономии памяти
    if len(text) > 300:
        text = text[:300]
    
    try:
        print(f"🎵 Начинаю генерацию музыки для текста: {text[:50]}...")
        
        # Генерация музыки
        result = music_generator(
            text,
            forward_params={
                "do_sample": True,  # Включаем случайность для разнообразия
                "temperature": 0.8,  # Температура для креативности
                "guidance_scale": 3,  # Насколько строго следуем тексту
                "max_new_tokens": 256  # Длительность аудио
            }
        )
        
        # Получаем аудио массив и частоту дискретизации
        audio_array = result["audio"]
        sampling_rate = result["sampling_rate"]
        
        # Конвертируем в int16 если нужно
        if audio_array.dtype != np.int16:
            # Нормализуем и конвертируем в int16
            audio_array = audio_array / np.max(np.abs(audio_array))
            audio_array = (audio_array * 32767).astype(np.int16)
        
        # Сохраняем в WAV формат
        wav_io = io.BytesIO()
        wavfile.write(wav_io, sampling_rate, audio_array)
        wav_io.seek(0)
        
        # Конвертируем в base64 для отправки в браузер
        audio_b64 = base64.b64encode(wav_io.read()).decode('utf-8')
        
        print("✅ Музыка успешно сгенерирована!")
        
        return jsonify({
            'success': True,
            'audio': audio_b64,
            'duration': len(audio_array) / sampling_rate,
            'message': f'🎵 Музыка создана! Длительность: {len(audio_array) / sampling_rate:.1f} секунд'
        })
        
    except Exception as e:
        print(f"❌ Ошибка генерации: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Ошибка генерации: {str(e)}'
        }), 500

@app.route('/download', methods=['POST'])
def download_audio():
    """Скачивание сгенерированного аудио"""
    data = request.get_json()
    audio_b64 = data.get('audio', '')
    
    if not audio_b64:
        return jsonify({'error': 'Нет аудио для скачивания'}), 400
    
    try:
        audio_bytes = base64.b64decode(audio_b64)
        return send_file(
            io.BytesIO(audio_bytes),
            mimetype='audio/wav',
            as_attachment=True,
            download_name='generated_music.wav'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Пытаемся загрузить модель при старте
    print("🚀 Запуск приложения Text-to-Music...")
    load_model()
    
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)