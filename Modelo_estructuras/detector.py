# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import visualize_detection as vd
import matplotlib.pyplot as plt
import cv2
import os



# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='exported_model/model_v2.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.1)
detector = vision.ObjectDetector.create_from_options(options)



pathdirectory = 'test/'
files = [f for f in os.listdir(pathdirectory) if os.path.isfile(os.path.join(pathdirectory, f)) and f.endswith('.jpg')]

for f_name in files:
    image = mp.Image.create_from_file(pathdirectory + f_name)
    detection_result = detector.detect(image)
    image_copy = np.copy(image.numpy_view())
    annotated_image = vd.visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        bbox = detection.bounding_box
        x1 = bbox.origin_x
        y1 = bbox.origin_y
        x2 = x1 + bbox.width
        y2 = y1 + bbox.height
        print(result_text)
        print('x1:', x1)
        print('y1:', y1)
        print('x2:', x2)
        print('y2:', y2)
        
    plt.imshow(rgb_annotated_image)
    plt.show()
    

