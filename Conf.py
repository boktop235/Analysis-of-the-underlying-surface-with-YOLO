# check_four_models_classes.py
from ultralytics import YOLO
import os
import yaml

models_config = {
    'model_2': {
        'path': 'runs/landcover_yolo_model2/weights/best.pt',
        'name': 'Model 2'
    },
    'model_4': {
        'path': 'runs/landcover_yolo_model4/weights/best.pt',
        'name': 'Model 4'
    },
    'model_14': {
        'path': 'runs/landcover_yolo_model14/weights/best.pt',
        'name': 'Model 14'
    },
    'model_15': {
        'path': 'runs/landcover_yolo_model15/weights/best.pt',
        'name': 'Model 15'
    }
}

print("=" * 70)
print("ПРОВЕРКА КЛАССОВ В 4 МОДЕЛЯХ")
print("=" * 70)


def get_model_classes(model_path):
    """Пытается получить классы модели разными способами"""
    classes = []

    try:
        # Способ 1: Загружаем модель и смотрим атрибуты
        model = YOLO(model_path)

        # Пробуем разные пути к атрибутам names
        if hasattr(model, 'names') and model.names:
            classes = list(model.names.values())
        elif hasattr(model.model, 'names') and model.model.names:
            classes = list(model.model.names.values())
        elif hasattr(model.model, 'model') and hasattr(model.model.model, 'names'):
            classes = list(model.model.model.names.values())

        return classes, "из модели"

    except Exception as e:
        # Способ 2: Ищем data.yaml файл
        model_dir = os.path.dirname(os.path.dirname(model_path))  # Поднимаемся на уровень выше weights/

        # Ищем yaml файлы
        yaml_files = []
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith(('.yaml', '.yml')):
                    yaml_files.append(os.path.join(root, file))

        # Проверяем каждый yaml файл
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                    if 'names' in data:
                        if isinstance(data['names'], dict):
                            classes = list(data['names'].values())
                        elif isinstance(data['names'], list):
                            classes = data['names']

                        if classes:
                            return classes, f"из {os.path.basename(yaml_file)}"

                    # Проверяем другие возможные ключи
                    for key in ['nc', 'num_classes', 'classes']:
                        if key in data:
                            if key == 'nc' or key == 'num_classes':
                                print(f"    Найдено количество классов: {data[key]}")
                            elif key == 'classes' and isinstance(data[key], list):
                                classes = data[key]
                                return classes, f"из {os.path.basename(yaml_file)}"

            except Exception as yaml_error:
                continue

        return classes, "не найдены"


# Проверяем каждую модель
for key, config in models_config.items():
    print(f"\n{'=' * 50}")
    print(f"📊 {config['name']} ({key})")
    print(f"{'=' * 50}")

    if os.path.exists(config['path']):
        print(f"  Путь: {config['path']}")

        # Проверяем размер файла модели
        try:
            file_size = os.path.getsize(config['path']) / (1024 * 1024)  # в МБ
            print(f"  Размер: {file_size:.1f} MB")
        except:
            pass

        # Получаем информацию о классах
        classes, source = get_model_classes(config['path'])

        if classes:
            print(f"  ✅ Количество классов: {len(classes)} ({source})")
            print("  Классы:")

            # Отображаем классы
            max_classes_to_show = 10  # Максимум показываем 10 классов
            for i, class_name in enumerate(classes[:max_classes_to_show]):
                print(f"    {i}: {class_name}")

            if len(classes) > max_classes_to_show:
                print(f"    ... и еще {len(classes) - max_classes_to_show} классов")

            # Показываем статистику по именам классов
            print(f"\n  📈 Статистика классов:")

            # Проверяем, есть ли числовые имена классов
            numeric_count = 0
            string_count = 0
            unique_classes = set()

            for cls in classes:
                unique_classes.add(str(cls))
                if isinstance(cls, (int, float)) or (isinstance(cls, str) and cls.isdigit()):
                    numeric_count += 1
                else:
                    string_count += 1

            print(f"    Уникальных классов: {len(unique_classes)}")
            print(f"    Числовых классов: {numeric_count}")
            print(f"    Текстовых классов: {string_count}")

            # Если все классы числовые, предлагаем переименовать
            if numeric_count == len(classes) and len(classes) > 1:
                print(f"\n  💡 Подсказка: Все классы числовые. Можете переименовать их:")
                print("    Например: {0: 'building', 1: 'road', 2: 'tree', 3: 'water'}")

        else:
            print(f"  ⚠️  Классы {source}")

            # Попробуем проанализировать структуру папки
            model_dir = os.path.dirname(os.path.dirname(config['path']))
            print(f"\n  📁 Структура папки {model_dir}:")

            if os.path.exists(model_dir):
                try:
                    # Показываем важные файлы
                    for root, dirs, files in os.walk(model_dir):
                        level = root.replace(model_dir, '').count(os.sep)
                        if level <= 2:  # Показываем только до 2 уровней вложенности
                            indent = '  ' * level
                            print(f"{indent}{os.path.basename(root)}/")

                            # Показываем только важные файлы
                            important_files = [f for f in files if
                                               f.endswith(('.yaml', '.yml', '.txt', '.json', '.pt', '.pth'))]
                            for file in important_files[:5]:  # Первые 5 файлов
                                print(f"{indent}  {file}")

                            if len(important_files) > 5:
                                print(f"{indent}  ... и еще {len(important_files) - 5} файлов")
                except:
                    print("    Не удалось прочитать структуру папки")

    else:
        print(f"  ❌ Модель не найдена: {config['path']}")

        # Показываем что есть в runs/
        print(f"\n  🔍 Поиск доступных моделей в runs/:")
        runs_dir = 'runs'
        if os.path.exists(runs_dir):
            for item in os.listdir(runs_dir):
                item_path = os.path.join(runs_dir, item)
                if os.path.isdir(item_path) and 'landcover_yolo_model' in item:
                    print(f"    📁 {item}")

                    # Проверяем есть ли weights/best.pt
                    weights_path = os.path.join(item_path, 'weights', 'best.pt')
                    if os.path.exists(weights_path):
                        print(f"      ✅ best.pt найден")
                    else:
                        print(f"      ❌ best.pt отсутствует")

print(f"\n{'=' * 70}")
print("ИНСТРУКЦИЯ ПО НАСТРОЙКЕ КЛАССОВ")
print("=" * 70)
print("\nЕсли классы не определены или только числовые:")
print("1. Найдите файл data.yaml рядом с моделью")
print("2. В нем должен быть раздел 'names:' с классами")
print("3. Пример правильного data.yaml:")
print("""
names:
  0: building
  1: road  
  2: tree
  3: water
  4: vehicle
  5: person

nc: 6  # количество классов
""")

print("\n💡 СОВЕТЫ:")
print("- Если классы числовые (0, 1, 2, 3), вы можете:")
print("  1. Создать mapping в коде: {0: 'building', 1: 'road', ...}")
print("  2. Отредактировать data.yaml файл модели")
print("  3. Использовать как есть (будут показываться числа)")

print("\n📍 ВАШИ МОДЕЛИ:")
print(f"1. Model 2: {models_config['model_2']['path']}")
print(f"2. Model 4: {models_config['model_4']['path']}")
print(f"3. Model 14: {models_config['model_14']['path']}")
print(f"4. Model 15: {models_config['model_15']['path']}")
print("=" * 70)

# Дополнительно: создаем пример mapping файла
print("\n📝 ПРИМЕР mapping.py для переименования классов:")
print("""
# class_mapping.py
CLASS_MAPPING = {
    'model_2': {
        0: 'class_0',
    },
    'model_4': {
        0: 'class_0',  
    },
    'model_14': {
        0: 'class_0',
        1: 'class_1',
        2: 'class_2',
        3: 'class_3',
    },
    'model_15': {
        0: 'class_0',
        1: 'class_1', 
        2: 'class_2',
        3: 'class_3',
    }
}
""")