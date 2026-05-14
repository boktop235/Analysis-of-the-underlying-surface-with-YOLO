import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import threading
import os

SAFETY_CATEGORIES = {
    'SAFE': {'color': (0, 255, 0), 'thickness': 3},
    'CAUTION': {'color': (0, 255, 255), 'thickness': 2},
    'DANGER': {'color': (0, 0, 255), 'thickness': 2}
}

CLASS_SAFETY = {
    'grass': 'SAFE',
    'dirt': 'SAFE',
    'sand': 'SAFE',
    'vegetation': 'SAFE',
    'gravel': 'CAUTION',
    'road': 'CAUTION',
    'building': 'DANGER',
    'car': 'DANGER',
    'tree': 'DANGER',
    'human': 'DANGER',
    'bicycle': 'DANGER',
    'dog': 'DANGER',
    'fence': 'DANGER',
    'pole': 'DANGER',
    'waterbody': 'DANGER',
    'rocks': 'DANGER',
    'log': 'DANGER',
    'metal': 'DANGER',
    'mobile-home': 'DANGER',
    'pickup-truck': 'DANGER',
    'small-vehicle': 'DANGER',
    'large-vehicle': 'DANGER',
    'buildings': 'DANGER',
    'bald-tree': 'DANGER',
    'ar-marker': 'CAUTION',
}

MODEL_PATHS = {
    'model2': '/workspace/models/runs/landcover_yolo_model2/weights/best.pt',
    'model4': '/workspace/models/runs/landcover_yolo_model4/weights/best.pt',
    'model14': '/workspace/models/runs/landcover_yolo_model14/weights/best.pt',
    'model15': '/workspace/models/runs/landcover_yolo_model15/weights/best.pt'
}

MODEL_CLASSES = {
    'model2': ['building'],
    'model4': ['ar-marker', 'bald-tree', 'bicycle', 'car', 'dirt', 'dog', 'door', 'fence',
               'fence-pole', 'grass', 'gravel', 'human', 'log', 'metal', 'misc', 'mobile-home',
               'other', 'pickup-truck', 'pole', 'rocks', 'sand', 'tree', 'wood'],
    'model14': ['small-vehicle', 'large-vehicle', 'human'],
    'model15': ['buildings', 'road', 'vegetation', 'waterbody']
}


class DroneDetector(Node):
    def __init__(self):
        super().__init__('drone_detector')

        self.subscription = self.create_subscription(Image, '/camera', self.callback, 10)
        self.br = CvBridge()

        self.display_width = 1280
        self.display_height = 720
        self.process_width = 640
        self.process_height = 360

        self.frame_count = 0
        self.last_detections = []

        self.get_logger().info('Загрузка моделей...')
        self.models = {}
        for name, path in MODEL_PATHS.items():
            if os.path.exists(path):
                self.models[name] = {
                    'model': YOLO(path),
                    'classes': MODEL_CLASSES.get(name, [])
                }
                self.get_logger().info(f"✓ {name} - {len(self.models[name]['classes'])} классов")
            else:
                self.get_logger().error(f"✗ {name} не найдена")

        cv2.namedWindow('Drone Detection', cv2.WINDOW_NORMAL)
        self.get_logger().info('Готово! ESC - выход')

    def get_safety_category(self, class_name):
        return CLASS_SAFETY.get(class_name.lower(), 'DANGER')

    def detect_single_model(self, model_data, frame):
        small_frame = cv2.resize(frame, (self.process_width, self.process_height))
        model = model_data['model']
        classes_list = model_data['classes']

        results = model(small_frame, imgsz=320, conf=0.3, verbose=False, max_det=60)

        detections = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    scale_x = self.display_width / self.process_width
                    scale_y = self.display_height / self.process_height
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0]) if box.cls is not None else 0

                    class_name = classes_list[cls_id] if cls_id < len(classes_list) else f"class_{cls_id}"
                    safety_cat = self.get_safety_category(class_name)

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class_name': class_name,
                        'safety_category': safety_cat
                    })
        return detections

    def detect_all_models(self, frame):
        detections = []
        results = {}
        threads = []

        def detect_thread(name, model_data):
            results[name] = self.detect_single_model(model_data, frame)

        for name, model_data in self.models.items():
            t = threading.Thread(target=detect_thread, args=(name, model_data))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=0.8)

        for dets in results.values():
            detections.extend(dets)

        return detections

    def draw_detections(self, frame, detections):
        result = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            category = det['safety_category']
            color = SAFETY_CATEGORIES[category]['color']
            thickness = SAFETY_CATEGORIES[category]['thickness']

            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            label = f"{det['class_name']} ({category})"
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        safe_count = len([d for d in detections if d['safety_category'] == 'SAFE'])
        caution_count = len([d for d in detections if d['safety_category'] == 'CAUTION'])
        danger_count = len([d for d in detections if d['safety_category'] == 'DANGER'])

        cv2.putText(result, f"SAFE: {safe_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"CAUTION: {caution_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result, f"DANGER: {danger_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return result

    def callback(self, msg):
        try:
            frame = self.br.imgmsg_to_cv2(msg, 'bgr8')
            display_frame = cv2.resize(frame, (self.display_width, self.display_height))

            self.frame_count += 1
            if self.frame_count % 3 == 0:
                self.last_detections = self.detect_all_models(display_frame)

            output = self.draw_detections(display_frame, self.last_detections)
            cv2.imshow('Drone Detection', output)

            if cv2.waitKey(1) & 0xFF == 27:
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Ошибка: {e}")


rclpy.init(args=None)
node = DroneDetector()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
