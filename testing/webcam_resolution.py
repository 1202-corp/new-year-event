import cv2
import numpy as np
import platform
import sys
import os

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.config import Config

def get_camera_backend():
    """Получить подходящий backend для камеры в зависимости от платформы"""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Linux":
        return cv2.CAP_V4L2
    else:
        return 0  # По умолчанию

def setup_camera_exposure(camera):
    """
    Настройка экспозиции камеры с правильным значением AUTO_EXPOSURE в зависимости от backend.
    """
    system = platform.system()
    
    # Пробуем разные значения AUTO_EXPOSURE в зависимости от платформы
    auto_exposure_values = []
    if system == "Windows":
        # Для DirectShow: 1.0 или 0.25 = ручной режим
        auto_exposure_values = [1.0, 0.25, 0.0]
    else:
        # Для Linux/V4L2: 0.25 = ручной режим (не 0.0!)
        auto_exposure_values = [0.25, 0.0, 1.0]
    
    for auto_exp_val in auto_exposure_values:
        try:
            result = camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exp_val)
            if result:
                print(f"Установлен AUTO_EXPOSURE = {auto_exp_val}")
                return True
        except Exception:
            continue
    
    print("Предупреждение: Не удалось установить AUTO_EXPOSURE")
    return False

# Популярные разрешения для выбора
RESOLUTIONS = [
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 1024),
    (1920, 1080),
]

# Инициализируем камеру с настройками из .env
backend = get_camera_backend()
cap = cv2.VideoCapture(Config.SNOWBALL_CAMERA_INDEX, backend)

if not cap.isOpened():
    raise SystemExit(f"Не удалось открыть камеру {Config.SNOWBALL_CAMERA_INDEX}")

# Устанавливаем MJPEG формат для лучшей производительности
try:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    print("Установлен формат MJPEG")
except Exception as e:
    print(f"Не удалось установить MJPEG: {e}")

# Устанавливаем разрешение из .env
cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.SNOWBALL_CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.SNOWBALL_CAMERA_HEIGHT)

# Настраиваем экспозицию (должно быть сделано до установки значения экспозиции)
setup_camera_exposure(cap)

# Применяем настройки камеры из .env
cap.set(cv2.CAP_PROP_EXPOSURE, Config.SNOWBALL_CAMERA_EXPOSURE)
cap.set(cv2.CAP_PROP_BRIGHTNESS, Config.SNOWBALL_CAMERA_BRIGHTNESS)
cap.set(cv2.CAP_PROP_CONTRAST, Config.SNOWBALL_CAMERA_CONTRAST)
cap.set(cv2.CAP_PROP_SATURATION, Config.SNOWBALL_CAMERA_SATURATION)
cap.set(cv2.CAP_PROP_SHARPNESS, Config.SNOWBALL_CAMERA_SHARPNESS)
cap.set(cv2.CAP_PROP_GAIN, Config.SNOWBALL_CAMERA_GAIN)
cap.set(cv2.CAP_PROP_FOCUS, Config.SNOWBALL_CAMERA_FOCUS)

# Отключаем автофокус при установке ручного фокуса
if Config.SNOWBALL_CAMERA_FOCUS >= 0:
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except Exception:
        pass

print("Применены настройки камеры из .env")

# Устанавливаем MJPEG формат для лучшей производительности
try:
    # MJPEG fourcc code
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    print("Установлен формат MJPEG")
except Exception as e:
    print(f"Не удалось установить MJPEG: {e}")

current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Текущее разрешение камеры: {current_width}x{current_height}")

# Находим текущее разрешение в списке
current_resolution_index = 0
for i, (w, h) in enumerate(RESOLUTIONS):
    if w == current_width and h == current_height:
        current_resolution_index = i
        break
else:
    # Если текущее разрешение не найдено, добавляем его в начало
    RESOLUTIONS.insert(0, (current_width, current_height))
    current_resolution_index = 0

# Создаем окно для выбора разрешения
RESOLUTION_WINDOW = "Resolution Selector"
cv2.namedWindow(RESOLUTION_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(RESOLUTION_WINDOW, 400, 200)

# Функция обратного вызова для ползунка разрешения
def on_resolution_change(val):
    if 0 <= val < len(RESOLUTIONS):
        width, height = RESOLUTIONS[val]
        # Проверяем текущее разрешение перед изменением
        current_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        current_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Если разрешение уже такое же, ничего не делаем
        if current_w == width and current_h == height:
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Очищаем буфер камеры - читаем несколько кадров и выбрасываем их
        # Это необходимо, чтобы избавиться от старых кадров с предыдущим разрешением
        for _ in range(5):
            ret, _ = cap.read()
            if not ret:
                break
        
        # Восстанавливаем настройки камеры после изменения разрешения
        setup_camera_exposure(cap)
        cap.set(cv2.CAP_PROP_EXPOSURE, Config.SNOWBALL_CAMERA_EXPOSURE)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, Config.SNOWBALL_CAMERA_BRIGHTNESS)
        cap.set(cv2.CAP_PROP_CONTRAST, Config.SNOWBALL_CAMERA_CONTRAST)
        cap.set(cv2.CAP_PROP_SATURATION, Config.SNOWBALL_CAMERA_SATURATION)
        cap.set(cv2.CAP_PROP_SHARPNESS, Config.SNOWBALL_CAMERA_SHARPNESS)
        cap.set(cv2.CAP_PROP_GAIN, Config.SNOWBALL_CAMERA_GAIN)
        cap.set(cv2.CAP_PROP_FOCUS, Config.SNOWBALL_CAMERA_FOCUS)
        if Config.SNOWBALL_CAMERA_FOCUS >= 0:
            try:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except Exception:
                pass
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Установлено разрешение: {actual_width}x{actual_height} (запрошено: {width}x{height})")

# Создаем ползунок для выбора разрешения
cv2.createTrackbar("Resolution", RESOLUTION_WINDOW, current_resolution_index, len(RESOLUTIONS) - 1, on_resolution_change)

# Создаем окно для видео
VIDEO_WINDOW = "Camera Feed"
cv2.namedWindow(VIDEO_WINDOW, cv2.WINDOW_NORMAL)

print("Используйте ползунок в окне 'Resolution Selector' для изменения разрешения")
print("Нажмите 'Q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Показываем видео в основном окне
    cv2.imshow(VIDEO_WINDOW, frame)
    
    # Обновляем окно выбора разрешения (показываем текущее разрешение)
    resolution_pos = cv2.getTrackbarPos("Resolution", RESOLUTION_WINDOW)
    if 0 <= resolution_pos < len(RESOLUTIONS):
        width, height = RESOLUTIONS[resolution_pos]
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Создаем текст для отображения
        info_text = f"Selected: {width}x{height}"
        actual_text = f"Actual: {actual_width}x{actual_height}"
        # Создаем простое изображение для окна выбора разрешения
        info_img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(info_img, info_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_img, actual_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_img, "Use slider to change", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(RESOLUTION_WINDOW, info_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()