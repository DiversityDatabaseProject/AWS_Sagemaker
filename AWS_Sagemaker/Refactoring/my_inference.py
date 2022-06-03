import cv2
import sample_utils #sample_utils is used to convert to tensor of a certain dimension for input to prediction.
import object_detection.utils
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import boto3


def inference(image_filename,image_content,predictor,image_key):
    '''
    Compute inference from model on image provided.
    '''
    my_image_tensor = sample_utils.image_file_to_tensor(image_filename)
    result = predictor.predict(my_image_tensor)     # Make prediction on new image
    my_detections1 = np.array(result['predictions'][0]['detection_boxes'])
    my_scores1=np.array(result['predictions'][0]['detection_scores'])
    my_classes1=([int(x) for x in result['predictions'][0]['detection_classes']]) 
    my_classes1=np.array(my_classes1)
    category_label={'name':'face',id:1}   # This creates the category label required as input to viz tool 
    label_id_offset = 1
    
    image_np_with_detections_resize = cv2.resize(image_content, (image_content.shape[1],image_content.shape[0]), interpolation = cv2.INTER_AREA) # Resize image as had been squared for inference
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections_resize,
            my_detections1,
            my_classes1-label_id_offset,
            my_scores1,
            category_label,
            use_normalized_coordinates=True,
            min_score_thresh=.6,
            agnostic_mode=False)
 
    image_key2=image_key[:-4]
    filename=image_key2+"_withlabel.jpg"
    cv2.imwrite(filename, image_np_with_detections_resize) 
    
    # Save to s3
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(filename, 'diversitybucket-v9', "images/"+image_key2+"_withlabel.jpg")
    
    return image_np_with_detections_resize, my_scores1