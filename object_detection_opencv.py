# Specify model imports
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import numpy as np
import os
import tensorflow as tf

# Disable GPU if necessary
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Create object detector
class TFObjectDetector():
  
  # Constructor
  def __init__(self, path_to_object_detection = './models/research/object_detection/configs/tf2',\
    path_to_model_checkpoint = './checkpoint', path_to_labels = './labels.pbtxt',\
      model_name = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'):
    self.model_name = model_name
    self.pipeline_config_path = path_to_object_detection
    self.pipeline_config = os.path.join(f'{self.pipeline_config_path}/{self.model_name}.config')
    self.full_config = config_util.get_configs_from_pipeline_file(self.pipeline_config)
    self.path_to_model_checkpoint = path_to_model_checkpoint
    self.path_to_labels = path_to_labels
    self.setup_model()

    
  # Set up model for usage
  def setup_model(self):
    self.build_model()
    self.restore_checkpoint()
    self.detection_function = self.get_model_detection_function()
    self.prepare_labels()

    
  # Build detection model
  def build_model(self):
    model_config = self.full_config['model']
    assert model_config is not None
    self.model = model_builder.build(model_config=model_config, is_training=False)
    return self.model

  
  # Restore checkpoint into model
  def restore_checkpoint(self):
    assert self.model is not None
    self.checkpoint = tf.train.Checkpoint(model=self.model)
    self.checkpoint.restore(os.path.join(self.path_to_model_checkpoint, 'ckpt-0')).expect_partial()

    
  # Get a tf.function for detection
  def get_model_detection_function(self):
    assert self.model is not None
    
    @tf.function
    def detection_function(image):
      image, shapes = self.model.preprocess(image)
      prediction_dict = self.model.predict(image, shapes)
      detections = self.model.postprocess(prediction_dict, shapes)
      return detections, prediction_dict, tf.reshape(shapes, [-1])
    
    return detection_function


  # Prepare labels
  # Source: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb
  def prepare_labels(self):
    label_map = label_map_util.load_labelmap(self.path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)
    self.label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
    
  # Get keypoint tuples
  # Source: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb
  def get_keypoint_tuples(self, eval_config):
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
      tuple_list.append((edge.start, edge.end))
    return tuple_list

  
  # Prepare image
  def prepare_image(self, image):
    return tf.convert_to_tensor(
      np.expand_dims(image, 0), dtype=tf.float32
    )

  
  # Perform detection
  def detect(self, image, label_offset = 1):
    # Ensure that we have a detection function
    assert self.detection_function is not None
    
    # Prepare image and perform prediction
    image = image.copy()
    image_tensor = self.prepare_image(image)
    detections, predictions_dict, shapes = self.detection_function(image_tensor)

    # Use keypoints if provided
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in detections:
      keypoints = detections['detection_keypoints'][0].numpy()
      keypoint_scores = detections['detection_keypoint_scores'][0].numpy()
    
    # Perform visualization on output image/frame 
    viz_utils.visualize_boxes_and_labels_on_image_array(
      image,
      detections['detection_boxes'][0].numpy(),
      (detections['detection_classes'][0].numpy() + label_offset).astype(int),
      detections['detection_scores'][0].numpy(),
      self.category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=25,
      min_score_thresh=.40,
      agnostic_mode=False,
      keypoints=keypoints,
      keypoint_scores=keypoint_scores,
      keypoint_edges=self.get_keypoint_tuples(self.full_config['eval_config']))
    
    # Return the image
    return image

  
  # Predict image from folder
  def detect_image(self, path, output_path):

    # Load image
    image = cv2.imread(path)

    # Perform object detection and add to output file
    output_file = self.detect(image)
    
    # Write output file to system
    cv2.imwrite(output_path, output_file)
    
    
  # Predict video from folder
  def detect_video(self, path, output_path):
    
    # Set output video writer with codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (1920, 1080))
    
    # Read the video
    vidcap = cv2.VideoCapture(path)
    frame_read, image = vidcap.read()
    count = 0
    
    # Iterate over frames and pass each for prediction
    while frame_read:
        
      # Perform object detection and add to output file
      output_file = self.detect(image)
      
      # Write frame with predictions to video
      out.write(output_file)
      
      # Read next frame
      frame_read, image = vidcap.read()
      count += 1
        
    # Release video file when we're ready
    out.release()

  
if __name__ == '__main__':
  detector = TFObjectDetector('../../tf-models/research/object_detection/configs/tf2', './checkpoint', './labels.pbtxt', 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8')
  detector.detect_image('./1.jpg', './1o.jpg')
  detector.detect_video('./1v.mp4', './v1o.mp4')

  
