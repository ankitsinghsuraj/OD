import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

text = 'GPU Nvidia GTX 860M 2GB - Intel i7 4710HQ '
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture("test.MP4")
#cap.set(cv2.CAP_PROP_FPS, 30)
video_record = cv2.VideoWriter('test_TF_NN_OD.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (720,480))


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# ## Object detection imports

from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation 

#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'

text = text + MODEL_NAME
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'pet_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Detection

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    while True :
      startTime = time.time()
      ret,image_np = cap.read()
      
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
                                      [detection_boxes, 
                                      detection_scores, 
                                      detection_classes, 
                                      num_detections],
                                      feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh = .5,
          line_thickness=8)
      endTime = time.time()
      #Calculando el total de tiempo
      workTime =  endTime - startTime
             
      # Estructurando resultado
      text2 = "Prediction time: " + str(round(workTime*1000,2)) + " ms"

      image_np = cv2.resize(image_np, (720,480))
      cv2.putText(image_np,text,(20,20), font, 0.5,(255,255,255),2,cv2.LINE_AA)
      cv2.putText(image_np,text2,(20,40), font, 0.5,(255,255,255),2,cv2.LINE_AA)

      cv2.imshow('object detection', image_np)
      video_record.write(image_np)
      if cv2.waitKey(1) & 0x77 == ord('q'):
          cv2.destroyAllWindows()
          video_record.release()
          cap.release()
          break 

cv2.destroyAllWindows()
video_record.release()
cap.release()