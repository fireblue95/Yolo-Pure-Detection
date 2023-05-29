"""
Use this convert pt or onnx model to TRT model.
URL: https://github.com/fireblue95/YOLOv8-TensorRT
"""

import cv2
import argparse
import numpy as np
from warnings import filterwarnings

from models.utils import blob, det_postprocess, letterbox

filterwarnings('ignore')

class Yolov8_TRT_pure:

    def __init__(self, opts):
        self.opts = opts

        self.names = ['obj1', 'obj2', 'obj3']

        # Yolov8 TRT

        if self.opts.method == 'cudart':
            from models.cudart_api import TRTEngine
        elif self.opts.method == 'pycuda':
            from models.pycuda_api import TRTEngine
        else:
            raise NotImplementedError

        self.Engine = TRTEngine(self.opts.weights)

        self.H, self.W = self.Engine.inp_info[0].shape[-2:]

        # Yolov8 TRT

        self.win_title = 'Yolov8 TRT'

        self.windows_w, self.windows_h = (1280, 720)

        self.colors = np.random.default_rng(3).uniform(0, 255, size=(len(self.names), 3))

    def detector(self, drawed_img):
        blob_frame, ratio, dwdh = letterbox(drawed_img, (self.W, self.H))

        tensor = blob(blob_frame, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)

        # inference
        data = self.Engine(tensor)

        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio

        return bboxes.astype(np.int32), scores, labels
    
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

    def run(self):
        cv2.namedWindow(self.win_title, cv2.WINDOW_NORMAL)
        # cv2.namedWindow(self.win_title, cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty(self.win_title, cv2.WINDOW_NORMAL, cv2.WINDOW_FULLSCREEN)

        cap = cv2.VideoCapture(self.opts.source)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0

        self.video_writer = cv2.VideoWriter('out.mp4', fourcc, fps, (self.windows_w, self.windows_h))

        while True:
            success, frame = cap.read()

            if not success:
                break

            drawed_img = frame.copy()
            
            boxes, scores, class_ids = self.detector(drawed_img)

            if len(class_ids) > 0:
                drawed_img = self.draw_detections(drawed_img, boxes, scores, class_ids)

            self.video_writer.write(cv2.resize(drawed_img, (self.windows_w, self.windows_h)))
             
            cv2.imshow(self.win_title, cv2.resize(drawed_img, (self.windows_w, self.windows_h)))
            
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.engine', help='Engine file')
    parser.add_argument('--source', type=str, default='0', help='video or image source file')
    parser.add_argument('--method', type=str, default='cudart', help='CUDART pipeline')
    opts = parser.parse_args()

    app = Yolov8_TRT_pure(opts)
    app.run()
