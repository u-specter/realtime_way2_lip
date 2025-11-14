#!/usr/bin/env python3
"""
Простая интеграция Wav2Lip с Zoom через OBS Studio
Использует Flask для показа видео, которое OBS захватывает и транслирует в Zoom
"""

import cv2
import numpy as np
import pyaudio
import time
from flask import Flask, Response
from inference import Wav2LipInference
import threading
import queue

app = Flask(__name__)

class Wav2LipZoom:
    def __init__(self, image_path):
        """
        Инициализация Wav2Lip для Zoom

        Args:
            image_path: путь к изображению аватара
        """
        self.image_path = image_path

        # Загружаем изображение
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # Размер для Zoom (1280x720)
        self.output_width = 1280
        self.output_height = 720

        # Изменяем размер изображения
        self.original_image = cv2.resize(self.original_image, (self.output_width, self.output_height))

        # Инициализируем Wav2Lip
        print("Загрузка модели Wav2Lip...")
        self.wav2lip = Wav2LipInference(device='cpu')

        # PyAudio настройки
        self.CHUNK = 8000  # 0.5 секунды при 16000 Hz
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        self.p = pyaudio.PyAudio()
        self.stream = None

        # Очередь для кадров
        self.frame_queue = queue.Queue(maxsize=2)
        self.current_frame = self.original_image.copy()
        self.running = False

        # Для статистики
        self.frame_count = 0
        self.start_time = time.time()

        print("Wav2Lip готов!")

    def start_audio_stream(self):
        """Запуск аудио потока"""
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        print("✅ Аудио поток запущен")

    def stop_audio_stream(self):
        """Остановка аудио потока"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("Аудио поток остановлен")

    def process_audio_loop(self):
        """
        Фоновый цикл обработки аудио
        Работает в отдельном потоке
        """
        print("🎤 Начинаю захват аудио...")

        while self.running:
            try:
                # Захватываем аудио
                audio_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

                # Проверяем тишину
                audio_rms = np.sqrt(np.mean(audio_array**2))
                silence_threshold = 150

                if audio_rms < silence_threshold:
                    # Тишина - используем оригинальное изображение
                    frame = self.original_image.copy()
                else:
                    # Есть звук - генерируем синхронизацию губ
                    frame = self.wav2lip.process_with_audio(
                        self.original_image,
                        audio_array,
                        self.RATE
                    )

                # Обновляем текущий кадр
                self.current_frame = frame

                # Статистика
                self.frame_count += 1
                if self.frame_count % 50 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    print(f"📊 Кадров: {self.frame_count}, FPS: {fps:.1f}, RMS: {audio_rms:.0f}")

            except Exception as e:
                print(f"❌ Ошибка обработки: {e}")
                time.sleep(0.1)

    def generate_frames(self):
        """
        Генератор кадров для Flask streaming
        """
        while self.running:
            try:
                # Берем текущий кадр
                frame = self.current_frame.copy()

                # Добавляем текст с информацией
                cv2.putText(
                    frame,
                    f"WAV2LIP ZOOM AVATAR - FPS: {self.frame_count/(time.time()-self.start_time+0.001):.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Кодируем в JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()

                # Отправляем в формате multipart
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Контроль FPS (~30 FPS)
                time.sleep(0.033)

            except Exception as e:
                print(f"❌ Ошибка генерации кадра: {e}")
                time.sleep(0.1)

    def start(self):
        """Запуск обработки"""
        self.running = True
        self.start_audio_stream()

        # Запускаем фоновый поток для обработки аудио
        self.audio_thread = threading.Thread(target=self.process_audio_loop, daemon=True)
        self.audio_thread.start()

        print("✅ Wav2Lip для Zoom запущен!")

    def stop(self):
        """Остановка обработки"""
        self.running = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=2)
        self.stop_audio_stream()
        print("⏹️  Wav2Lip остановлен")


# Глобальный экземпляр
wav2lip_zoom = None


@app.route('/')
def index():
    """Главная страница"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wav2Lip Zoom Avatar</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                background: #000;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                font-family: Arial, sans-serif;
            }
            .container {
                text-align: center;
            }
            img {
                max-width: 100%;
                max-height: 90vh;
                border: 2px solid #0f0;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0, 255, 0, 0.5);
            }
            h1 {
                color: #0f0;
                margin-bottom: 20px;
            }
            .instructions {
                color: #fff;
                max-width: 800px;
                margin: 20px auto;
                text-align: left;
                background: rgba(0, 255, 0, 0.1);
                padding: 20px;
                border-radius: 10px;
            }
            .instructions h2 {
                color: #0f0;
            }
            .instructions ol {
                line-height: 1.8;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Wav2Lip Zoom Avatar</h1>
            <img src="/video_feed" alt="Zoom Avatar">

            <div class="instructions">
                <h2>📋 Инструкции для Zoom:</h2>
                <ol>
                    <li><b>Откройте OBS Studio</b> (если не установлен: <a href="https://obsproject.com" style="color: #0f0;">obsproject.com</a>)</li>
                    <li><b>В OBS:</b> Sources → + → Window Capture</li>
                    <li><b>Выберите:</b> это окно браузера</li>
                    <li><b>Нажмите:</b> Start Virtual Camera в OBS</li>
                    <li><b>Откройте Zoom</b> → Settings → Video → Camera</li>
                    <li><b>Выберите:</b> OBS Virtual Camera</li>
                    <li><b>Говорите в микрофон</b> - аватар будет двигать губами! 🎤</li>
                </ol>
                <p><b>Совет:</b> Откройте этот браузер в полноэкранном режиме (F11)</p>
            </div>
        </div>
    </body>
    </html>
    """


@app.route('/video_feed')
def video_feed():
    """Видео поток для OBS/Zoom"""
    global wav2lip_zoom
    if wav2lip_zoom is None or not wav2lip_zoom.running:
        return "Wav2Lip не запущен", 503

    return Response(
        wav2lip_zoom.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def main():
    import sys
    global wav2lip_zoom

    if len(sys.argv) < 2:
        print("Использование: python3 zoom_simple.py <путь_к_изображению_аватара>")
        print("\nПример:")
        print("  python3 zoom_simple.py assets/uploaded_images/avatar.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print("=" * 70)
    print("  WAV2LIP ZOOM INTEGRATION (через OBS Studio)")
    print("=" * 70)
    print()

    try:
        # Создаем экземпляр Wav2Lip
        wav2lip_zoom = Wav2LipZoom(image_path=image_path)

        # Запускаем обработку
        wav2lip_zoom.start()

        print()
        print("=" * 70)
        print("  🌐 Откройте в браузере: http://127.0.0.1:5000")
        print("=" * 70)
        print()
        print("Следуйте инструкциям на странице для настройки OBS и Zoom")
        print()
        print("Нажмите Ctrl+C для остановки")
        print()

        # Запускаем Flask
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except KeyboardInterrupt:
        print("\n\n⏹️  Остановка...")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if wav2lip_zoom:
            wav2lip_zoom.stop()


if __name__ == "__main__":
    main()
