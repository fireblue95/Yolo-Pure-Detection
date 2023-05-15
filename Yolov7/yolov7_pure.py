import cv2
import time
import torch
from pathlib import Path
from typing import Tuple
from warnings import filterwarnings

from models.experimental import attempt_load

filterwarnings('ignore')

import numpy as np

class Yolov7_pure:
    def __init__(self, yolov7_conf: float = 0.5):
        
        # -----------------------------------------------------------------
        # load model

        self.yolov7_conf = yolov7_conf

        self.yolo_weights = Path('./yolov7.pt')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = attempt_load(self.yolo_weights, map_location=self.device)  # load FP32 model
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # Get the class names


        # load model
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # Set configure

        self.win_title = 'Yolov7'

        # Set configure
        # -----------------------------------------------------------------


    def detector(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Yolov7 detect the frame and send back the result.

        The boxes is (640, 640). scales is 0 ~ 640
            type is [xmin, ymin, width, height]
            than will return boxes type:
                (0 ~ ori_w, 0 ~ ori_h)
                [xmin, ymin, xmax, ymax]
        """

        ori_h, ori_w, _ = frame.shape

        # Convert the ROI image to a PyTorch tensor and move it to the GPU
        blob = torch.from_numpy(cv2.resize(frame, (640, 640))).to(self.device)
        blob = blob.permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        # Detect objects in the ROI using the YOLOv7 model
        detections = self.model(blob)[0][0]

        # Keep only detections with a confidence score > 0.5
        detections = detections[detections[:, 4] > self.yolov7_conf]

        boxes, scores, class_ids = detections[..., :4], detections[..., 4], detections[..., 5:]

        if boxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        class_ids = torch.argmax(class_ids, dim=-1)

        xmin, ymin, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

        o_xmin = xmin / 640 * ori_w
        o_ymin = ymin / 640 * ori_h
        o_xmax = (xmin + w) / 640 * ori_w
        o_ymax = (ymin + h) / 640 * ori_h

        out_boxes = torch.stack([o_xmin, o_ymin, o_xmax, o_ymax], axis=-1)

        return out_boxes.cpu().numpy(), scores.cpu().numpy(), class_ids.cpu().numpy()

    def draw_detections(self, image, class_names, boxes, scores, class_ids, mask_alpha=0.3):
        rng = np.random.default_rng(3)

        colors = rng.uniform(0, 255, size=(len(class_names), 3))

        mask_img = image.copy()
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Draw fill rectangle in mask image
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            label = class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.rectangle(mask_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

    @torch.no_grad()
    def detect_objects_realtime(self):
        # Start the video capture
        #cap = cv2.VideoCapture(0)
        
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_drew = frame.copy()
            boxes, scores, class_ids = self.detector(frame)

            frame_drew = self.draw_detections(frame_drew, self.names, boxes, scores, class_ids)

            # Display the ROI image with bounding boxes
            cv2.imshow(self.win_title, frame_drew)
            
            # Wait for a key press to exit
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()

    

if __name__ == "__main__":
    app: Yolov7_pure = Yolov7_pure()
    app.detect_objects_realtime()
