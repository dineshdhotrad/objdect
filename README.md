# Object Detection Using TensorFlow API (TensorFlow V2) on Windows 10

## Brief Summary
*Last updated: 10/31/2020 with TensorFlow v2.3.1*

This repository is a tutorial for how to use TensorFlow's Object Detection API to train an object detection classifier for multiple objects on Windows 10.

Original Version with TF_V1 can be found here :- [EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

## Introduction
Fire and Smoke detection using learning based approach.

The purpose of this repo is to explain how to train your own convolutional neural network for detection classifier for smoke and fire, starting from scratch.

Team Members:
 - Dinesh Dhotrad 
 - Shivaraj Chattannavar 

This github project provides the learning based approach of dectecting fire and smoke in images. The non-annotated data is collected from various sources such as kaagle etc. 

Firstly, the training data is annotated and provided to the Faster-RCNN-Inception-V2-COCO model from Tensorflow. The progress of the training job can be viewed by TensorBoard.

After the training of annotated data, the model is trained now which is ready for detecting fire and smoke with the output of bounding boxes.



## Steps

### 1: Install Anaconda, CUDA, and cuDNN

### 2: Set up TensorFlow Directory and Anaconda Virtual Environment
#### 2a: Download or clone TensorFlow Object Detection API repository from GitHub
[TensorFlow Model](https://github.com/tensorflow/models) *

*The Current Version is updated for V2...!
 
#### 2b: Download the ssd_mobilenet_v2_320x320_coco17_tpu-8 model from TensorFlow's model zoo V2
[Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

#### 2c: Download this tutorial's repository from GitHub
After download exract files in Object Detection Folder under models/reserch/object detection

#### 2d: Download the dataset of FIRE and SMOKE
[DATASET](https://drive.google.com/file/d/1I2ZvYfB9Lo4pOMExRsP2YgUtYEoyToOt/view?usp=sharing)
After download exract files in Object Detection Folder under models/reserch/object detection/images


### 3: Create new TensorFlow Environment and install these Dependencies
conda install -c anaconda protobuf
pip install pillow
pip install lxml
pip install Cython
pip install contextlib2
pip install jupyter
pip install matplotlib
pip install pandas

### 4: Configure PYTHONPATH environment variable
set PYTHONPATH=<YOUR PATH FOR TF MODEL>\models;<YOUR PATH FOR TF MODEL>\models\research;<YOUR PATH FOR TF MODEL>\models\research\slim;
  
### 5: Compile Protobufs and run setup.py
```
cd <YOUR PATH FOR TF MODEL>\models\research
```

```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```

```
python setup.py build
python setup.py install
```

### 6: Gather and Label Pictures

Now that the TensorFlow Object Detection API is all set up and ready to go, we need to provide the images it will use to train a new detection classifier.

Since we have already annotated the data-set we can skip this process
    
### 7:Generate Training Data
```
python xml_to_csv.py
```

For example, say you are training a classifier to detect basketballs, shirts, and shoes. You will replace the following code in generate_tfrecord.py:
```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'nine':
        return 1
    elif row_label == 'ten':
        return 2
    elif row_label == 'jack':
        return 3
    elif row_label == 'queen':
        return 4
    elif row_label == 'king':
        return 5
    elif row_label == 'ace':
        return 6
    else:
        None
```
With this:
```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'fire':
        return 1
    elif row_label == 'smoke':
        return 2
    else:
        None
```
Then, generate the TFRecord files by issuing these commands from the \object_detection folder:
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=gen1\train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=gen1\test.record
```

### 7: Create Label Map and Configure Training
Use a text editor to create a new file and save it as labelmap.pbtxt in the <YOUR PATH FOR TF MODEL>\models\research\object_detection\training folder.

```
item {
  id: 1
  name: 'nine'
}

item {
  id: 2
  name: 'ten'
}

item {
  id: 3
  name: 'jack'
}
```

The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file. For the basketball, shirt, and shoe detector example mentioned in Step 4, the labelmap.pbtxt file will look like:

```
item {
  id: 1
  name: 'fire'
}

item {
  id: 2
  name: 'smoke'
}
```

### 8:onfigure training
Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training!

Navigate to C:\tensorflow1\models\research\object_detection\configs\tf2 and copy the ssd_mobilenet_v2_320x320_coco17_tpu-8.config file into the \object_detection\training directory. Then, open the file with a text editor. There are several changes to make to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.

Make the following changes to the faster_rcnn_inception_v2_pets.config file. Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model! Also, the paths must be in double quotation marks ( " ), not single quotation marks ( ' ).

- Line 11. Change num_classes to the number of different objects you want the classifier to detect. For the above basketball, shirt, and shoe detector, it would be num_classes : 2 .
- Line 145. Change fine_tune_checkpoint to:
  - fine_tune_checkpoint : "<YOUR PATH FOR TF MODEL>/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

- Lines 179 and 181. In the train_input_reader section, change input_path and label_map_path to:
  - input_path : "<YOUR PATH FOR TF MODEL>/models/research/object_detection/gen1/train.record"
  - label_map_path: "<YOUR PATH FOR TF MODEL>/models/research/object_detection/training/labelmap.pbtxt"
 
- Lines 191 and 195. In the eval_input_reader section, change input_path and label_map_path to:
  - input_path : "<YOUR PATH FOR TF MODEL>/models/research/object_detection/gen1/test.record"
  - label_map_path: "<YOUR PATH FOR TF MODEL>/models/research/object_detection/training/labelmap.pbtxt"

Save the file after the changes have been made. That’s it! The training job is all configured and ready to go!

### 9: Run the Training

```
python model_main_tf2.py --pipeline_config_path=training/ssd_mobilenet_v2_320x320_coco17_tpu-8.config --model_dir=saved_model/ssd_mobilenet_v2_320x320_coco17_tpu-8/ --checkpoint_dir=training/ --alsologtostderr
```

##### If you get pycocotools error:

S1: Install Visual Studio Build Tools > V14

S2: pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

### To View the log of training:
open anaconda tf environment and run
```
tensorboard --logdir=training/train
```


### 10: Finally after satisfied result freeze the graph for predictiction

export the last checkpoint to inference_graph folder
```
python exporter_main_v2.py --trained_checkpoint_dir=training --pipeline_config_path=training/ssd_mobilenet_v2_320x320_coco17_tpu-8.config --output_directory inference_graph
```

### 11: Test the trained model and evaluate on OWN images
run
```
jupyter notebook objdetect.ipynb
```
### Results for our sample images
![alt-text-1](https://github.com/dineshdhotrad/tfv2_ObjectDetection/blob/main/Results/1.jpeg) ![alt-text-2](https://github.com/dineshdhotrad/tfv2_ObjectDetection/blob/main/Results/2.jpeg)
