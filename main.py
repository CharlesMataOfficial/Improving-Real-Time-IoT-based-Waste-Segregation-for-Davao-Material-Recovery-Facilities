from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import hashlib
import imutils
import cv2

class WasteDetection:

    def __init__(self, capture, confidence_threshold, max_age, offset, line):
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.names
        self.confidence_threshold = confidence_threshold
        self.line = line
        self.offset = offset
        self.bio_cls = [0]
        self.recyclable_cls = [1]
        self.special_cls = [2]
        self.tracker = DeepSort(max_age=max_age)
        self.track_detections = {}

    def load_model(self):
        model = YOLO("best55.pt")
        model.fuse()
        return model

    def predict(self, img):
        results = self.model(img,conf=self.confidence_threshold, iou=0.3, device=0,  stream=True)
        return results

    def plot_boxes(self, results, img):
        detections = []

        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                detections.append(([int(x1), int(y1),(int(x2)-int(x1)), (int(y2)-int(y1))], conf, int(cls)))
        return detections, img

    def id_to_color(self, id_str):
        hash_object = hashlib.md5(id_str.encode())
        hash_hex = hash_object.hexdigest()
        r = int(hash_hex[:2], 16) % 256
        g = int(hash_hex[2:4], 16) % 256
        b = int(hash_hex[4:6], 16) % 256
        return r, g, b

    def line_detect(self, track_id, cls):
        if cls in self.bio_cls:
            print(cls, track_id, "detected.. moving BIO servo!")
        elif cls in self.recyclable_cls:
            print(cls, track_id, "detected.. moving RECYCLABLE servo!")
        elif cls in self.special_cls:
            print(cls, track_id, "detected.. moving SPECIAL servo!")
        else:
            print("Detected class is not defined in parameters")

    def track_detect(self, detections, img):
        tracks = self.tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            cls = self.CLASS_NAMES_DICT[track.det_class]
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            x3 = int(x1 + x2) // 2  # x coordinate for center point
            y3 = int(y1 + y2) // 2  # y coordinate for center point
            cv2.circle(img, (x3, y3), 3, (0, 255, 0), 2)
            if (self.line + self.offset) > x3 > (self.line - self.offset):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                self.line_detect(track_id, track.det_class)
                self.track_detections[track_id] = cls

            cv2.putText(img, f'ID: {track_id} {cls} {track.det_conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

        return img

    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        cap.set(3, 640)
        assert cap.isOpened()

        # tracker = DeepSort()

        while True:
            ret, img = cap.read()

            if not ret:
                break

            # img = imutils.resize(img, width=640)  # Resize frame for faster processing
            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img)
            detect_frame = self.track_detect(detections, frames)
            cv2.line(img, (self.line, 1), (self.line, 480), (0, 0, 255), 2)
            print(self.track_detections.items())

            cv2.imshow('Image', detect_frame)
            if cv2.waitKey(5) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Example usage:
detector = WasteDetection(capture=1, confidence_threshold=0.5, max_age=5, offset=20, line=320)
detector()