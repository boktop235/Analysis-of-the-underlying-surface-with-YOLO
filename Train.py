from ultralytics import YOLO
import torch

model = YOLO('yolov8n-obb.pt')

# ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ МАКСИМАЛЬНОЙ СКОРОСТИ:
results = model.train(
    data="DOTAv1.yaml",
    epochs=30,
    imgsz=400,  # Увеличиваем до 640 (быстрее конвергенция)
    batch=16,  # Увеличиваем batch для ускорения (если память позволяет)
    device=0,
    workers=0,

    amp=False,
    close_mosaic=0,  # Не отключать мозаику
    cache=False,

    lr0=0.02,  # Увеличиваем learning rate для ускорения
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=2,
    warmup_momentum=0.8,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    augment=True,
    degrees=5.0,  # Уменьшаем поворот
    translate=0.05,  # Уменьшаем смещение
    scale=0.2,  # Уменьшаем масштаб
    shear=1.0,  # Уменьшаем скос
    perspective=0.0002,  # Уменьшаем перспективу
    flipud=0.0,
    fliplr=0.3,  # Реже отражения
    mosaic=0.5,  # Уменьшаем мозаику
    mixup=0.0,
    copy_paste=0.0,
    val=True,
    save=True,
    save_period=5,
    plots=True,
    verbose=True,
    project='runs/train',
    name='fast_training',
    exist_ok=True,
    pretrained=True,
    nbs=64,  # Нормализация batch size
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    max_det=300,
    fraction=1.0,
    single_cls=False,
)