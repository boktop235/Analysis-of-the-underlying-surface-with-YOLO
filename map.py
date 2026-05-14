from ultralytics import YOLO
import cv2
import time
import os
import threading
import numpy as np

SAFETY_CATEGORIES = {
    'SAFE': {
        'color': (0, 255, 0),
        'thickness': 3,
        'fill_color': (0, 255, 0),
        'fill_alpha': 0.4,
        'description': 'Безопасно - можно садиться'
    },
    'CAUTION': {
        'color': (0, 255, 255),
        'thickness': 2,
        'fill_color': (0, 255, 255),
        'fill_alpha': 0.3,
        'description': 'Осторожно - садиться нежелательно'
    },
    'DANGER': {
        'color': (0, 0, 255),
        'thickness': 2,
        'fill_color': (0, 0, 255),
        'fill_alpha': 0.25,
        'description': 'Опасно - избегать посадки'
    }
}

SAFETY_CLASSIFICATION = {
    'building': 'DANGER',
    'ar-marker': 'CAUTION',
    'bald-tree': 'DANGER',
    'bicycle': 'DANGER',
    'car': 'DANGER',
    'dirt': 'SAFE',
    'dog': 'DANGER',
    'door': 'DANGER',
    'fence': 'DANGER',
    'fence-pole': 'DANGER',
    'grass': 'SAFE',
    'gravel': 'CAUTION',
    'human': 'DANGER',
    'log': 'DANGER',
    'metal': 'DANGER',
    'misc': 'CAUTION',
    'mobile-home': 'DANGER',
    'other': 'CAUTION',
    'pickup-truck': 'DANGER',
    'pole': 'DANGER',
    'rocks': 'DANGER',
    'sand': 'SAFE',
    'tree': 'DANGER',
    'wood': 'DANGER',
    'small-vehicle': 'DANGER',
    'large-vehicle': 'DANGER',
    'buildings': 'DANGER',
    'road': 'CAUTION',
    'vegetation': 'SAFE',
    'waterbody': 'DANGER',
}

models_config = {
    'model_2': {
        'path': 'runs/landcover_yolo_model2/weights/best.pt',
        'display_name': 'Building',
        'classes': ['building']
    },
    'model_4': {
        'path': 'runs/landcover_yolo_model4/weights/best.pt',
        'display_name': 'Objects',
        'classes': [
            'ar-marker', 'bald-tree', 'bicycle', 'car', 'dirt',
            'dog', 'door', 'fence', 'fence-pole', 'grass', 'gravel',
            'human', 'log', 'metal', 'misc', 'mobile-home', 'other',
            'pickup-truck', 'pole', 'rocks', 'sand', 'tree', 'wood'
        ]
    },
    'model_14': {
        'path': 'runs/landcover_yolo_model14/weights/best.pt',
        'display_name': 'Vehicles',
        'classes': ['small-vehicle', 'large-vehicle', 'human']
    },
    'model_15': {
        'path': 'runs/landcover_yolo_model15/weights/best.pt',
        'display_name': 'Landcover',
        'classes': ['buildings', 'road', 'vegetation', 'waterbody']
    }
}

models = {}
for key, config in models_config.items():
    if os.path.exists(config['path']):
        try:
            model = YOLO(config['path'])
            models[key] = {
                'model': model,
                'display_name': config['display_name'],
                'classes': config['classes']
            }
            print(f"{config['display_name']}: {len(config['classes'])} классов")
        except Exception as e:
            print(f"Ошибка загрузки {key}: {e}")
            continue
    else:
        print(f"Модель не найдена: {config['path']}")

if not models:
    print("Нет доступных моделей")
    exit()

print(f"\nЗагружено моделей: {len(models)}")

category_counts = {'SAFE': 0, 'CAUTION': 0, 'DANGER': 0}
for class_name, category in SAFETY_CLASSIFICATION.items():
    category_counts[category] += 1

video_path = 'Test5.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Не могу открыть видео")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360

# СОХРАНЕНИЕ ДВУХ ФАЙЛОВ
output_path_normal = 'Test_normal_mode_output.mp4'
output_path_emergency = 'Test_emergency_mode_output.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Видео для штатного режима
out_normal = cv2.VideoWriter(output_path_normal, fourcc, fps, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
# Видео для аварийного режима
out_emergency = cv2.VideoWriter(output_path_emergency, fourcc, fps, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

is_emergency = False
show_detections = True
pause_frame = None
pause_detections = []
current_frame_normal = None
current_frame_emergency = None


def get_safety_category(class_name):
    class_name_lower = class_name.lower()

    if class_name_lower in SAFETY_CLASSIFICATION:
        return SAFETY_CLASSIFICATION[class_name_lower]

    for key, category in SAFETY_CLASSIFICATION.items():
        if key in class_name_lower or class_name_lower in key:
            return category

    return 'DANGER'


def detect_model(model_key, model_data, frame):
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

    conf_threshold = 0.5
    max_det = 40

    if model_key == 'model_4':
        conf_threshold = 0.3
        max_det = 60

    results = model_data['model'](
        small_frame,
        imgsz=320,
        conf=conf_threshold,
        verbose=False,
        max_det=max_det
    )

    detections = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                scale_x = DISPLAY_WIDTH / PROCESS_WIDTH
                scale_y = DISPLAY_HEIGHT / PROCESS_HEIGHT
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

                conf = float(box.conf[0])
                cls_id = int(box.cls[0]) if hasattr(box, 'cls') and box.cls is not None else 0

                class_name = ""
                if cls_id < len(model_data['classes']):
                    class_name = model_data['classes'][cls_id]
                else:
                    class_name = f"class_{cls_id}"

                safety_category = get_safety_category(class_name)
                safety_info = SAFETY_CATEGORIES[safety_category]

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': class_name,
                    'safety_category': safety_category,
                    'color': safety_info['color'],
                    'thickness': safety_info['thickness'],
                    'fill_color': safety_info['fill_color'],
                    'fill_alpha': safety_info['fill_alpha'],
                    'model_name': model_key,
                    'display_name': model_data['display_name']
                })

    return detections


def detect_all_models(frame):
    detections = []
    threads = []
    results = {}

    for key, model_data in models.items():
        thread = threading.Thread(
            target=lambda k=key, m=model_data: results.update({k: detect_model(k, m, frame)})
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join(timeout=0.8)

    for key, model_dets in results.items():
        detections.extend(model_dets)

    return detections


def draw_normal_mode(frame, detections):
    result = frame.copy()

    if not show_detections:
        return result

    danger_detections = [d for d in detections if d['safety_category'] == 'DANGER']
    caution_detections = [d for d in detections if d['safety_category'] == 'CAUTION']
    safe_detections = [d for d in detections if d['safety_category'] == 'SAFE']

    for category_detections in [safe_detections, caution_detections, danger_detections]:
        for det in category_detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            thickness = det['thickness']
            fill_color = det['fill_color']
            fill_alpha = det['fill_alpha']

            overlay = result.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
            cv2.addWeighted(overlay, fill_alpha, result, 1 - fill_alpha, 0, result)

            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            corner_size = 8
            cv2.line(result, (x1, y1), (x1 + corner_size, y1), color, 2)
            cv2.line(result, (x1, y1), (x1, y1 + corner_size), color, 2)
            cv2.line(result, (x2, y1), (x2 - corner_size, y1), color, 2)
            cv2.line(result, (x2, y1), (x2, y1 + corner_size), color, 2)
            cv2.line(result, (x1, y2), (x1 + corner_size, y2), color, 2)
            cv2.line(result, (x1, y2), (x1, y2 - corner_size), color, 2)
            cv2.line(result, (x2, y2), (x2 - corner_size, y2), color, 2)
            cv2.line(result, (x2, y2), (x2, y2 - corner_size), color, 2)

    return result


def draw_emergency_mode_clean(frame, detections):
    result = frame.copy()

    if not show_detections:
        return result

    safe_detections = [d for d in detections if d['safety_category'] == 'SAFE']
    caution_detections = [d for d in detections if d['safety_category'] == 'CAUTION']
    danger_detections = [d for d in detections if d['safety_category'] == 'DANGER']

    if len(safe_detections) > 0:
        priority_detections = safe_detections
    elif len(caution_detections) > 0:
        priority_detections = caution_detections
    elif len(danger_detections) > 0:
        priority_detections = danger_detections
    else:
        priority_detections = []

    for det in priority_detections:
        x1, y1, x2, y2 = det['bbox']
        color = det['color']
        thickness = det['thickness']
        fill_color = det['fill_color']
        fill_alpha = det['fill_alpha']

        enhanced_alpha = fill_alpha + 0.15

        overlay = result.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
        cv2.addWeighted(overlay, enhanced_alpha, result, 1 - enhanced_alpha, 0, result)

        enhanced_thickness = thickness + 1
        cv2.rectangle(result, (x1, y1), (x2, y2), color, enhanced_thickness)

        corner_size = 12
        corner_thickness = 3

        cv2.line(result, (x1, y1), (x1 + corner_size, y1), color, corner_thickness)
        cv2.line(result, (x1, y1), (x1, y1 + corner_size), color, corner_thickness)
        cv2.line(result, (x2, y1), (x2 - corner_size, y1), color, corner_thickness)
        cv2.line(result, (x2, y1), (x2, y1 + corner_size), color, corner_thickness)
        cv2.line(result, (x1, y2), (x1 + corner_size, y2), color, corner_thickness)
        cv2.line(result, (x1, y2), (x1, y2 - corner_size), color, corner_thickness)
        cv2.line(result, (x2, y2), (x2 - corner_size, y2), color, corner_thickness)
        cv2.line(result, (x2, y2), (x2, y2 - corner_size), color, corner_thickness)

        inner_offset = 5
        if det['safety_category'] == 'SAFE':
            cv2.line(result, (x1 + inner_offset, y1 + inner_offset),
                     (x2 - inner_offset, y2 - inner_offset), (0, 200, 0), 1)
            cv2.line(result, (x2 - inner_offset, y1 + inner_offset),
                     (x1 + inner_offset, y2 - inner_offset), (0, 200, 0), 1)
        elif det['safety_category'] == 'CAUTION':
            center_y = (y1 + y2) // 2
            cv2.line(result, (x1 + inner_offset, center_y),
                     (x2 - inner_offset, center_y), (0, 200, 200), 1)
        elif det['safety_category'] == 'DANGER':
            quarter_x1 = x1 + (x2 - x1) // 4
            quarter_x2 = x2 - (x2 - x1) // 4
            cv2.line(result, (quarter_x1, y1 + inner_offset),
                     (quarter_x1, y2 - inner_offset), (0, 0, 200), 1)
            cv2.line(result, (quarter_x2, y1 + inner_offset),
                     (quarter_x2, y2 - inner_offset), (0, 0, 200), 1)

    return result


cv2.namedWindow('Emergency Landing System', cv2.WINDOW_NORMAL)
frame_count = 0
last_detections = []
start_time = time.time()

print(f"\n Запуск системы посадки...")
print("Управление:")
print("  SPACE - Аварийный режим/Пауза (показывает приоритетные зоны)")
print("  1     - Вкл/Выкл подсветку объектов")
print("  ESC   - Выход")
print("\nВ аварийном режиме:")
print("  ТОЛЬКО зеленые зоны (если есть)")
print("  ТОЛЬКО желтые зоны (если нет зеленых)")
print("  ТОЛЬКО красные зоны (если нет других)")
print(f"\nСохранение двух видео:")
print(f"  Штатный режим: {output_path_normal}")
print(f"  Аварийный режим: {output_path_emergency}")

while True:
    if not is_emergency:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 3 == 0:
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            last_detections = detect_all_models(display_frame)
        else:
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    else:
        display_frame = cv2.resize(pause_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        last_detections = pause_detections

    # Генерация кадров для обоих режимов
    normal_frame = draw_normal_mode(display_frame, last_detections)
    emergency_frame = draw_emergency_mode_clean(display_frame, last_detections)

    # Сохранение в оба видеофайла
    out_normal.write(normal_frame)
    out_emergency.write(emergency_frame)

    # Отображение текущего режима
    if is_emergency:
        cv2.imshow('Emergency Landing System', emergency_frame)
    else:
        cv2.imshow('Emergency Landing System', normal_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == 32:
        is_emergency = not is_emergency
        if is_emergency:
            pause_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            print(f"\n АВАРИЙНЫЙ РЕЖИМ на кадре {frame_count}")
            pause_detections = detect_all_models(pause_frame)

            safe_count = len([d for d in pause_detections if d['safety_category'] == 'SAFE'])
            caution_count = len([d for d in pause_detections if d['safety_category'] == 'CAUTION'])
            danger_count = len([d for d in pause_detections if d['safety_category'] == 'DANGER'])

            print(f" Обнаружено зон:")
            print(f"  Безопасные зоны: {safe_count}")
            print(f"  Зоны с осторожностью: {caution_count}")
            print(f"  Опасные зоны: {danger_count}")

            if safe_count > 0:
                safe_classes = list(set([d['class_name'] for d in pause_detections if d['safety_category'] == 'SAFE']))
                print(f" Показываются: ТОЛЬКО зеленые зоны")
                print(f" Безопасные поверхности: {', '.join(safe_classes[:3])}")
            elif caution_count > 0:
                caution_classes = list(
                    set([d['class_name'] for d in pause_detections if d['safety_category'] == 'CAUTION']))
                print(f" Показываются: ТОЛЬКО желтые зоны")
                print(f" Поверхности с осторожностью: {', '.join(caution_classes[:3])}")
            elif danger_count > 0:
                danger_classes = list(
                    set([d['class_name'] for d in pause_detections if d['safety_category'] == 'DANGER']))
                print(f" Показываются: ТОЛЬКО красные зоны")
                print(f" Опасные поверхности: {', '.join(danger_classes[:3])}")
            else:
                print(f" Зоны не обнаружены!")
        else:
            print(f"\n Возврат в обычный режим")
    elif key == 49:
        show_detections = not show_detections
        if show_detections:
            print(f"\n Подсветка объектов ВКЛЮЧЕНА")
        else:
            print(f"\n️ Подсветка объектов ВЫКЛЮЧЕНА")

cap.release()
out_normal.release()
out_emergency.release()
cv2.destroyAllWindows()

print(f"\nСистема посадки завершила работу!")
print(f"Сохранены видеофайлы:")
print(f"  1. Штатный режим: {output_path_normal}")
print(f"  2. Аварийный режим: {output_path_emergency}")
print(f"Обработано кадров: {frame_count}")