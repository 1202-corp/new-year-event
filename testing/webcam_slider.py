import cv2
import numpy as np
import platform
import sys
import os

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Проверяем наличие флага --env
USE_ENV = '--env' in sys.argv

try:
    from game.config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    if USE_ENV:
        print("Warning: Could not import Config, using default values")

"""
Пример программы для настройки параметров USB‑камеры с помощью ползунков OpenCV.

Программа автоматически находит доступную камеру и создаёт окно
с ползунками для основных параметров: яркость, контраст, насыщенность,
резкость, гамма, баланс белого, компенсация задней подсветки, усиление,
фокус, масштаб и экспозиция. Значения по умолчанию взяты из скриншота.

После изменения ползунка вызывается функция обратного вызова, которая
устанавливает соответствующее свойство камеры через `cap.set()`. Если
какая‑то настройка не поддерживается камерой, она будет проигнорирована.

Нажатие клавиши S выводит текущие значения ползунков в терминал, клавиша
P открывает штатный диалог Windows (CAP_PROP_SETTINGS), а клавиша Q закрывает
приложение.

ПРИМЕЧАНИЕ: Настройка экспозиции и других параметров зависит от драйвера
и API. На Windows Media Foundation многие камеры игнорируют вызовы
`CAP_PROP_EXPOSURE`. В таких случаях попробуйте запустить `cv2.VideoCapture`
с флагом `cv2.CAP_DSHOW` (DirectShow) и предварительно отключить автоматическую
экспозицию с помощью `cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)` или 0.25.
Также учтите, что на Windows значения экспозиции задаются отрицательными
индексами (от 0 до -13), где 0 соответствует ~1 с, а −13 ≈ 122 µs【506170052913158†L16-L39】.
Если камера по‑прежнему не реагирует на изменения, воспользуйтесь
официальным приложением Logitech или вызовите диалог настроек через
`CAP_PROP_SETTINGS`【845147060340318†L846-L873】.
"""


def get_camera_backend():
    """Получить подходящий backend для камеры в зависимости от платформы"""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Linux":
        return cv2.CAP_V4L2
    else:
        return 0  # По умолчанию


def find_available_camera(max_tries=10, backend=None):
    """
    Найти первую доступную камеру.
    
    Args:
        max_tries: Максимальное количество индексов для проверки
        backend: Backend для использования (None = автоматический выбор)
    
    Returns:
        tuple: (camera_index, backend) или (None, None) если камера не найдена
    """
    if backend is None:
        backend = get_camera_backend()
    
    print(f"Поиск доступной камеры (backend: {backend})...")
    
    for i in range(max_tries):
        try:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                # Попробуем прочитать кадр, чтобы убедиться, что камера работает
                ret, _ = cap.read()
                if ret:
                    cap.release()
                    print(f"Найдена камера с индексом {i}")
                    return i, backend
                cap.release()
        except Exception:
            continue
    
    # Если с указанным backend не получилось, попробуем без него
    if backend != 0:
        print(f"Попытка с backend=0 (по умолчанию)...")
        for i in range(max_tries):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        cap.release()
                        print(f"Найдена камера с индексом {i} (backend по умолчанию)")
                        return i, 0
                    cap.release()
            except Exception:
                continue
    
    return None, None


# Попытка использовать индекс из аргументов командной строки
# Игнорируем флаг --env при парсинге индекса камеры
CAMERA_ID = None
BACKEND = None

args_without_env = [arg for arg in sys.argv[1:] if arg != '--env']

if len(args_without_env) > 0:
    try:
        CAMERA_ID = int(args_without_env[0])
        BACKEND = get_camera_backend()  # Используем подходящий backend для платформы
        print(f"Используется камера с индексом {CAMERA_ID} (из аргументов)")
    except ValueError:
        print(f"Неверный индекс камеры: {args_without_env[0]}, будет выполнен поиск...")

# Если индекс не указан, ищем доступную камеру
if CAMERA_ID is None:
    CAMERA_ID, BACKEND = find_available_camera()
    if CAMERA_ID is None:
        raise SystemExit("Не удалось найти доступную камеру. Убедитесь, что камера подключена.")

# Если BACKEND все еще не определен, используем подходящий для платформы
if BACKEND is None:
    BACKEND = get_camera_backend()

# Создаём объект захвата
cap = cv2.VideoCapture(CAMERA_ID, BACKEND)
if not cap.isOpened():
    raise SystemExit(f"Не удалось открыть камеру {CAMERA_ID}")

# Получаем текущее разрешение камеры
current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Текущее разрешение камеры: {current_width}x{current_height}")

# Популярные разрешения для выбора
RESOLUTIONS = [
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 1024),
    (1920, 1080),
]

# Находим текущее разрешение в списке или добавляем его
current_resolution_index = 0
if CONFIG_AVAILABLE and USE_ENV:
    config_width = int(Config.SNOWBALL_CAMERA_WIDTH)
    config_height = int(Config.SNOWBALL_CAMERA_HEIGHT)
    for i, (w, h) in enumerate(RESOLUTIONS):
        if w == config_width and h == config_height:
            current_resolution_index = i
            break
    else:
        # Если разрешение из конфига не найдено, добавляем его в начало
        RESOLUTIONS.insert(0, (config_width, config_height))
        current_resolution_index = 0
else:
    # Ищем текущее разрешение в списке
    for i, (w, h) in enumerate(RESOLUTIONS):
        if w == current_width and h == current_height:
            current_resolution_index = i
            break
    else:
        # Если текущее разрешение не найдено, добавляем его в начало
        RESOLUTIONS.insert(0, (current_width, current_height))
        current_resolution_index = 0

# Желательно выключить автоэкспозицию и автофокус, чтобы значение экспозиции и
# фокус корректно устанавливались. Значения флагов могут отличаться для разных
# бекендов (1/0 vs 0.25/0.75), поэтому пробуем разные значения.
system = platform.system()
if system == "Windows":
    # Для DirectShow: 1.0 или 0.25 = ручной режим
    auto_exposure_values = [1.0, 0.25]
else:
    # Для Linux/V4L2: 0.25 = ручной режим (не 0.0!)
    auto_exposure_values = [0.25, 0.0, 1.0]

for auto_exp_val in auto_exposure_values:
    try:
        result = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exp_val)
        if result:
            print(f"Установлен AUTO_EXPOSURE = {auto_exp_val}")
            break
    except Exception:
        continue

try:
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 — выключить автофокус
except Exception:
    pass

# Настройки окна
WINDOW_NAME = "Logi HD1080p Settings"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Создаем отдельное окно для выбора разрешения
RESOLUTION_WINDOW = "Resolution Selector"
cv2.namedWindow(RESOLUTION_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(RESOLUTION_WINDOW, 400, 200)

# Определяем диапазон экспозиции в зависимости от платформы
system = platform.system()
if system == "Windows":
    # Windows: логарифмические индексы от -13 до 0
    exposure_min = -13
    exposure_max = 0
    # Используем значение из .env, если доступно и флаг --env установлен, иначе -7
    if CONFIG_AVAILABLE and USE_ENV:
        exposure_default = Config.SNOWBALL_CAMERA_EXPOSURE
        # Убедимся, что значение в правильном диапазоне
        exposure_default = max(exposure_min, min(exposure_max, exposure_default))
    else:
        exposure_default = -7
    exposure_is_log = True  # Логарифмическая шкала
else:
    # Linux/V4L2: используем микросекунды, но маппим ползунок 0-13 на диапазон
    # Ползунок 0-13 будет маппироваться на диапазон 1-100000 мкс (логарифмически)
    # Это даст плавное изменение от очень короткой до длинной экспозиции
    exposure_min_us = 1      # 1 микросекунда (очень короткая экспозиция)
    exposure_max_us = 100000  # 100 миллисекунд (длинная экспозиция)
    # Используем значение из .env, если доступно и флаг --env установлен, иначе 1000 мкс
    if CONFIG_AVAILABLE and USE_ENV:
        exposure_default_us = Config.SNOWBALL_CAMERA_EXPOSURE
        # Убедимся, что значение в правильном диапазоне
        exposure_default_us = max(exposure_min_us, min(exposure_max_us, exposure_default_us))
    else:
        exposure_default_us = 1000  # 1 миллисекунда по умолчанию
    exposure_is_log = False  # Используем логарифмическое преобразование для микросекунд

# Диапазоны и начальные значения для ползунков
# Используем значения из .env, если доступны и флаг --env установлен
settings = {
    "Brightness": {
        "id": cv2.CAP_PROP_BRIGHTNESS, 
        "min": 0, 
        "max": 255, 
        "value": int(Config.SNOWBALL_CAMERA_BRIGHTNESS) if (CONFIG_AVAILABLE and USE_ENV) else 161
    },
    "Contrast": {
        "id": cv2.CAP_PROP_CONTRAST, 
        "min": 0, 
        "max": 255, 
        "value": int(Config.SNOWBALL_CAMERA_CONTRAST) if (CONFIG_AVAILABLE and USE_ENV) else 128
    },
    "Saturation": {
        "id": cv2.CAP_PROP_SATURATION, 
        "min": 0, 
        "max": 255, 
        "value": int(Config.SNOWBALL_CAMERA_SATURATION) if (CONFIG_AVAILABLE and USE_ENV) else 128
    },
    "Sharpness": {
        "id": cv2.CAP_PROP_SHARPNESS, 
        "min": 0, 
        "max": 255, 
        "value": int(Config.SNOWBALL_CAMERA_SHARPNESS) if (CONFIG_AVAILABLE and USE_ENV) else 231
    },
    "Gain": {
        "id": cv2.CAP_PROP_GAIN, 
        "min": 0, 
        "max": 255, 
        "value": int(Config.SNOWBALL_CAMERA_GAIN) if (CONFIG_AVAILABLE and USE_ENV) else 102
    },
    "Focus": {
        "id": cv2.CAP_PROP_FOCUS, 
        "min": 0, 
        "max": 255, 
        "value": int(Config.SNOWBALL_CAMERA_FOCUS) if (CONFIG_AVAILABLE and USE_ENV) else 0
    },
    "Zoom": {
        "id": cv2.CAP_PROP_ZOOM, 
        "min": 0, 
        "max": 100, 
        "value": 0  # По умолчанию без зума
    },
    # Экспозиция: разные диапазоны для разных платформ
    "Exposure": {
        "id": cv2.CAP_PROP_EXPOSURE,  
        "min": exposure_min if system == "Windows" else exposure_min_us,
        "max": exposure_max if system == "Windows" else exposure_max_us,
        "value": exposure_default if system == "Windows" else exposure_default_us,
        "is_log": exposure_is_log,
        "is_windows": system == "Windows"
    },
}

# Функция обратного вызова для ползунков
def on_trackbar(val, name):
    conf = settings[name]
    prop_id = conf["id"]
    # Преобразуем значение ползунка в реальное значение
    if name == "Exposure":
        if conf.get("is_windows", False):
            # Windows: логарифмическая шкала (val от 0 до 13 -> от -13 до 0)
            real_val = conf["min"] + val
        else:
            # Linux: ползунок 0-13 маппируется на микросекунды логарифмически
            # Это дает плавное изменение от короткой к длинной экспозиции
            trackbar_max = 13
            if trackbar_max > 0:
                # Логарифмическая интерполяция для плавного изменения
                ratio = val / trackbar_max  # 0.0 до 1.0
                min_us = conf["min"]
                max_us = conf["max"]
                # Логарифмическое преобразование: log(min) до log(max)
                import math
                log_min = math.log(max(min_us, 1))  # Избегаем log(0)
                log_max = math.log(max(max_us, 1))
                log_val = log_min + ratio * (log_max - log_min)
                real_val = math.exp(log_val)
            else:
                real_val = conf["min"]
    else:
        real_val = val
    # Если это фокус, отключаем автофокус перед установкой
    if name == "Focus":
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except Exception:
            pass
    
    # Для экспозиции нужно убедиться, что автоэкспозиция отключена
    if name == "Exposure":
        system = platform.system()
        if system == "Windows":
            # Windows: 1.0 или 0.25 = ручной режим
            auto_exposure_values = [1.0, 0.25]
        else:
            # Linux/V4L2: 0.25 = ручной режим (не 0.0!)
            auto_exposure_values = [0.25, 0.0, 1.0]
        for auto_exp_val in auto_exposure_values:
            try:
                result = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exp_val)
                if result:
                    break
            except Exception:
                continue
    
    # Устанавливаем значение
    try:
        cap.set(prop_id, float(real_val))
    except Exception:
        # Некоторые свойства могут не поддерживаться
        pass

# Создаём ползунки
for name, conf in settings.items():
    if name == "Exposure":
        if conf.get("is_windows", False):
            # Windows: ползунок 0-13 для диапазона -13 до 0
            max_val = conf["max"] - conf["min"]  # 0 - (-13) = 13
            init = conf["value"] - conf["min"]   # -7 - (-13) = 6
        else:
            # Linux: ползунок 0-13 маппируется на микросекунды
            max_val = 13
            # Вычисляем начальную позицию ползунка из значения в микросекундах
            import math
            min_us = conf["min"]
            max_us = conf["max"]
            value_us = conf["value"]
            log_min = math.log(max(min_us, 1))
            log_max = math.log(max(max_us, 1))
            log_val = math.log(max(value_us, 1))
            if log_max > log_min:
                ratio = (log_val - log_min) / (log_max - log_min)
                init = int(ratio * max_val)
            else:
                init = max_val // 2
    else:
        max_val = conf["max"]
        init = conf["value"]
    
    cv2.createTrackbar(name, WINDOW_NAME, int(init), int(max_val), lambda v, n=name: on_trackbar(v, n))

# Функция обратного вызова для ползунка разрешения в отдельном окне
def on_resolution_change(val):
    if 0 <= val < len(RESOLUTIONS):
        width, height = RESOLUTIONS[val]
        # Проверяем текущее разрешение перед изменением
        current_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        current_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Если разрешение уже такое же, ничего не делаем
        if current_w == width and current_h == height:
            return
        
        # Устанавливаем ширину и высоту
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Очищаем буфер камеры - читаем несколько кадров и выбрасываем их
            # Это необходимо, чтобы избавиться от старых кадров с предыдущим разрешением
            for _ in range(5):
                ret, _ = cap.read()
                if not ret:
                    break
            
            # После изменения разрешения нужно заново установить автоэкспозицию и экспозицию
            # чтобы они не сбросились
            system = platform.system()
            if system == "Windows":
                auto_exposure_values = [1.0, 0.25]
            else:
                auto_exposure_values = [0.25, 0.0, 1.0]
            for auto_exp_val in auto_exposure_values:
                try:
                    result = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exp_val)
                    if result:
                        break
                except Exception:
                    continue
            # Восстанавливаем текущее значение экспозиции
            try:
                exposure_pos = cv2.getTrackbarPos("Exposure", WINDOW_NAME)
                exposure_conf = settings["Exposure"]
                if exposure_conf.get("is_windows", False):
                    exposure_val = exposure_conf["min"] + exposure_pos
                else:
                    import math
                    trackbar_max = 13
                    min_us = exposure_conf["min"]
                    max_us = exposure_conf["max"]
                    ratio = exposure_pos / trackbar_max if trackbar_max > 0 else 0
                    log_min = math.log(max(min_us, 1))
                    log_max = math.log(max(max_us, 1))
                    log_val = log_min + ratio * (log_max - log_min)
                    exposure_val = math.exp(log_val)
                cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_val))
            except Exception:
                pass
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Установлено разрешение: {actual_width}x{actual_height} (запрошено: {width}x{height})")
        except Exception as e:
            print(f"Ошибка установки разрешения: {e}")

# Создаем ползунок для выбора разрешения в отдельном окне
cv2.createTrackbar("Resolution", RESOLUTION_WINDOW, current_resolution_index, len(RESOLUTIONS) - 1, on_resolution_change)

print("Нажмите 'S' для вывода текущих значений, 'P' для диалога свойств камеры, 'Q' для выхода.")
print("Используйте ползунок в окне 'Resolution Selector' для изменения разрешения")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать кадр с камеры")
        break
    cv2.imshow(WINDOW_NAME, frame)
    
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
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        # Печатаем текущие значения
        print("Текущие параметры камеры:")
        for name, conf in settings.items():
            pos = cv2.getTrackbarPos(name, WINDOW_NAME)
            if name == "Exposure":
                if conf.get("is_windows", False):
                    # Windows: логарифмическая шкала
                    val = conf["min"] + pos
                    print(f"  {name}: {val} (логарифмический индекс)")
                else:
                    # Linux: вычисляем реальное значение в микросекундах
                    import math
                    trackbar_max = 13
                    min_us = conf["min"]
                    max_us = conf["max"]
                    ratio = pos / trackbar_max if trackbar_max > 0 else 0
                    log_min = math.log(max(min_us, 1))
                    log_max = math.log(max(max_us, 1))
                    log_val = log_min + ratio * (log_max - log_min)
                    val_us = math.exp(log_val)
                    print(f"  {name}: {val_us:.1f} мкс ({val_us/1000:.2f} мс) [ползунок: {pos}/13]")
            else:
                val = pos
                print(f"  {name}: {val}")
        
        # Выводим разрешение отдельно
        resolution_pos = cv2.getTrackbarPos("Resolution", RESOLUTION_WINDOW)
        if 0 <= resolution_pos < len(RESOLUTIONS):
            width, height = RESOLUTIONS[resolution_pos]
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Resolution: {width}x{height} (фактическое: {actual_width}x{actual_height}) [ползунок: {resolution_pos}/{len(RESOLUTIONS)-1}]")
    elif key == ord('p') or key == ord('P'):
        # Открываем диалог свойств камеры, если поддерживается
        try:
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
        except Exception:
            print("CAP_PROP_SETTINGS не поддерживается данным бекендом")

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()