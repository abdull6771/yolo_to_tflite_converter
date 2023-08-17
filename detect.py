import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import utils as utils
from config import cfg
from yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import colorsys
import random

import os
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from yolov4 import filter_boxes
'''
Requirements: 
1) Install the tflite_runtime package from here:
https://www.tensorflow.org/lite/guide/python
2) Camera to take inputs
3) [Optional] libedgetpu.so.1.0 installed from here if you want to use the edgetpu:
https://github.com/google-coral/edgetpu/tree/master/libedgetpu/direct
Prepraration:
1) Download label:
$ wget https://raw.githubusercontent.com/google-coral/edgetpu/master/test_data/coco_labels.txt
2) Download models:
$ wget https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_ssd_v2_coco_quant_postprocess.tflite
$ wget https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
Run:
1) With out edgetpu:
$ python3 tflite_cv.py --model mobilenet_ssd_v2_coco_quant_postprocess.tflite --labels coco_labels.txt
2) With edgetpu:
$ python3 tflite_cv.py --model mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels coco_labels.txt --edgetpu True
'''


def load_label(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}
        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to tflite model.', required=True)
    parser.add_argument('--labels', help='Path to label file.', required=True)
    parser.add_argument(
        '--threshold', help='Minimum confidence threshold.', default=0.5)
    parser.add_argument('--source', help='Video source.', default=0)
    parser.add_argument('--edgetpu', help='With EdgeTpu', default=False)
    flags.DEFINE_integer('size', 416, 'resize images to')
    flags.DEFINE_string('weights', './checkpoints/yolov4-tiny',
                    'path to weights file')
    return parser.parse_args()


def main():

    args = get_cmd()

    if args.edgetpu:
        interpreter = Interpreter(args.model, experimental_delegates=[
                                  load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = Interpreter(args.model)

    interpreter.allocate_tensors()
    saved_model_loaded = tf.saved_model.load("./checkpoints/yolov4-tiny", tags=[tag_constants.SERVING])
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width = input_details[0]['shape'][2]
    height = input_details[0]['shape'][1]

    labels = load_label(args.labels)

        # Capturing the video.
    cap = cv2.VideoCapture(args.source)
    image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_counter = 0
    start = time.time()
    while(True):
        frame_counter += 1
        # Acquire frame and resize to expected shape [1xHxWx3]
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        input_data = np.asarray(input_data).astype(np.float32)
        # set frame as input tensors
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # perform inference
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        #print(pred)
        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([416, 416]))
        #print(pred_conf)
        batch_data = tf.constant(input_data)
        #print(batch_data)
        infer = saved_model_loaded.signatures['serving_default']
        pred_bbox = infer(batch_data)
        #print(pred_bbox)
        #break
        for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
        #print(boxes[0])
        #break
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.25
        )
        #print(boxes.numpy())
        #break
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        print(class_names)
        #break
        allowed_classes = list(class_names.values())
        print(allowed_classes)
        #break
        # Get output tensor
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        print(boxes)
        break
        #classes = interpreter.get_tensor(output_details[1]['index'])[0]
        #scores = interpreter.get_tensor(output_details[2]['index'])[0]
        #print(scores)
        image = utils.draw_bbox(frame, pred_bbox, allowed_classes = allowed_classes)
        #print(image.shape)
        hsv_tuples = [(1.0 * x / 2, 1., 1.) for x in range(2)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        #print(hsv_tuples)
        random.seed(0)
        random.shuffle(colors)
        random.seed(None)
        out_boxes, out_scores, out_classes, num_boxes = pred_bbox
        print(pred_bbox)
        #break
        image_h, image_w, _ = frame.shape
        #print(frame)
        break
        for i in range(num_boxes[0]):
          if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > 2: continue
          coor = out_boxes[0][i]
          coor[0] = int(coor[0] * image_h)
          coor[2] = int(coor[2] * image_h)
          coor[1] = int(coor[1] * image_w)
          coor[3] = int(coor[3] * image_w)
          fontScale = 0.5
          score = out_scores[0][i]
          class_ind = int(out_classes[0][i])
          class_name = classes[class_ind]
          #print(class_name)
          if allowed_classes in allowed_classes:
            continue
          else:
              bbox_color = colors[class_ind]
              bbox_thick = int(0.6 * (image_h + image_w) / 600)
              c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
              #c1 = tuple([int(coor[1]),int(coor[0])])
              #c2 = tuple([int(coor[3]),int(coor[2])])
              #cv2.rectangle(image, c1, c2, 255,0,0,)
              print(c1, c2, bbox_color, bbox_thick)
              cv2.rectangle(image, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), bbox_color, bbox_thick)
              if True:
                bbox_mess = '%s: %.2f' % (classes[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                #cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled
                cv2.rectangle(image, (int(c1[0]), int(c1[1])), (int(c3[0]), int(c3[1])), (255, 0, 0), -1)

                #cv2.putText(image, bbox_mess, (c1, c2), cv2.FONT_HERSHEY_SIMPLEX,
                  #         fontScale, (0, 255, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
                cv2.putText(image, bbox_mess, (int(c1[0]), int(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        cv2.imshow('Object detector', image)
        if cv2.waitKey(1) == ord('q'):
            break
        #break
        #image = Image.fromarray(image.astype(np.uint8))
       
    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()