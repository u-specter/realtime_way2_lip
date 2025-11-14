#!/usr/bin/env python3
"""
Virtual Camera for Zoom Integration
Создает виртуальную камеру с аватаром Wav2Lip для Zoom
"""

import cv2
import numpy as np
import pyaudio
import time
from inference import Wav2LipInference
import pyvirtualcam
from pyvirtualcam import PixelFormat

class VirtualCameraWav2Lip:
    def __init__(self, image_path, device='cpu'):
        """
        Инициализация виртуальной камеры с Wav2Lip

        Args:
            image_path: путь к изображению аватара
            device: 'cpu' или 'cuda'
        """
        self.image_path = image_path
        self.device = device

        # Загружаем изображение
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # Размер для Zoom (рекомендуется 1280x720 или 640x480)
        self.output_width = 1280
        self.output_height = 720

        # Изменяем размер изображения если нужно
        self.original_image = cv2.resize(self.original_image, (self.output_width, self.output_height))

        # Инициализируем Wav2Lip
        print("Загрузка модели Wav2Lip...")
        self.wav2lip = Wav2LipInference(device=device)

        # PyAudio настройки
        self.CHUNK = 8000  # 0.5 секунды при 16000 Hz
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 0.5

        self.p = pyaudio.PyAudio()
        self.stream = None

        print("Виртуальная камера готова!")

    def start_audio_stream(self):
        """Запуск аудио потока"""
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        print("Аудио поток запущен")

    def stop_audio_stream(self):
        """Остановка аудио потока"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("Аудио поток остановлен")

    def process_frame(self):
        """
        Обработка одного кадра с синхронизацией губ

        Returns:
            numpy.ndarray: обработанный кадр в формате RGB
        """
        try:
            # Захватываем аудио
            audio_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

            # Проверяем тишину
            audio_rms = np.sqrt(np.mean(audio_array**2))
            silence_threshold = 150

            if audio_rms < silence_threshold:
                # Тишина - возвращаем оригинальное изображение
                frame = self.original_image.copy()
            else:
                # Есть звук - генерируем синхронизацию губ
                frame = self.wav2lip.process_with_audio(
                    self.original_image,
                    audio_array,
                    self.RATE
                )

            # Конвертируем BGR в RGB для pyvirtualcam
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb

        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            # В случае ошибки возвращаем оригинал
            return cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

    def run(self):
        """
        Запуск виртуальной камеры
        Zoom сможет выбрать эту камеру как источник видео
        """
        print(f"\nЗапуск виртуальной камеры {self.output_width}x{self.output_height}...")
        print("Zoom увидит камеру как 'OBS Virtual Camera' или 'pyvirtualcam'")
        print("\nИнструкции:")
        print("1. Откройте Zoom")
        print("2. Settings → Video → Camera")
        print("3. Выберите 'OBS Virtual Camera' или 'pyvirtualcam'")
        print("4. Говорите в микрофон - аватар будет двигать губами!")
        print("\nНажмите Ctrl+C для остановки\n")

        try:
            # Запускаем аудио поток
            self.start_audio_stream()

            # Создаем виртуальную камеру
            with pyvirtualcam.Camera(
                width=self.output_width,
                height=self.output_height,
                fps=30,
                fmt=PixelFormat.RGB
            ) as cam:
                print(f"✅ Виртуальная камера создана: {cam.device}")
                print(f"   Разрешение: {self.output_width}x{self.output_height}")
                print(f"   FPS: 30")
                print("\n🎤 Говорите в микрофон...")

                frame_count = 0
                start_time = time.time()

                while True:
                    # Обрабатываем кадр с синхронизацией губ
                    frame = self.process_frame()

                    # Отправляем в виртуальную камеру
                    cam.send(frame)

                    # Небольшая задержка для 30 FPS
                    cam.sleep_until_next_frame()

                    # Статистика каждые 100 кадров
                    frame_count += 1
                    if frame_count % 100 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"📊 Кадров обработано: {frame_count}, FPS: {fps:.1f}")

        except KeyboardInterrupt:
            print("\n\n⏹️  Остановка виртуальной камеры...")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
        finally:
            self.stop_audio_stream()
            print("✅ Виртуальная камера остановлена")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Использование: python3 virtual_camera.py <путь_к_изображению_аватара>")
        print("\nПример:")
        print("  python3 virtual_camera.py assets/uploaded_images/avatar.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print("=" * 60)
    print("  WAV2LIP VIRTUAL CAMERA FOR ZOOM")
    print("=" * 60)

    try:
        # Создаем виртуальную камеру
        vcam = VirtualCameraWav2Lip(
            image_path=image_path,
            device='cpu'  # используем CPU, т.к. у вас macOS
        )

        # Запускаем
        vcam.run()

    except Exception as e:
        print(f"\n❌ Ошибка запуска: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
