import cv2
import numpy as np
from pathlib import Path

from yolov8 import YOLOv8

class Yolov8_pure:

    def __init__(self, camera_ip: str):

        self.camera_ip = camera_ip

        self.win_w, self.win_h = (1280, 720)

        self.names = ['obj1', 'obj2']

        # -----------------------------------------------------------------
        # load model

        self.yolo_weights = Path('weights/yolov8n.onnx')
        self.detector = YOLOv8(str(self.yolo_weights), conf_thres=0.5, iou_thres=0.75)

        # load model
        # -----------------------------------------------------------------
        
        rng = np.random.default_rng(3)

        self.colors = rng.uniform(0, 255, size=(len(self.names), 3))

    def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.4):
        mask_img = image.copy()
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = self.colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Draw fill rectangle in mask image
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            label = self.names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            offset = 12

            box_ymax = max(y1, offset)
            box_ymin = max(box_ymax - th, 2)

            cv2.rectangle(det_img, (x1, box_ymin),
                        (x1 + tw, box_ymax), color, -1)
            cv2.rectangle(mask_img, (x1, box_ymin),
                        (x1 + tw, box_ymax), color, -1)
            
            cv2.putText(det_img, caption, (x1, box_ymax),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(mask_img, caption, (x1, box_ymax),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)
    
    def detect_objects_realtime(self):
        cap = cv2.VideoCapture(self.camera_ip)

        while True:
            
            success, frame = cap.read()

            if not success:
                break

            boxes, scores, class_ids = self.detector(frame)

            if len(class_ids) > 0:
                
                frame = self.draw_detections(frame, boxes, scores, class_ids)

            frame_resized = cv2.resize(frame, (self.win_w, self.win_h))

            cv2.imshow('Yolov8', frame_resized)
            
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()
if __name__ == "__main__":

    camera_ip = "0"

    app: Yolov8_pure = Yolov8_pure(camera_ip)

    app.detect_objects_realtime()
