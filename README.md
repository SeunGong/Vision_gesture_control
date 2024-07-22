This project is recognization hand gesture for interacting with mobile robot.

  Object Detection Model : yolov8n.pt (for Hands detect)
  Pose Model : yolov8m-pose.pt (for pose detect)
  Python version : 3.9.13 64-bit, 3.9.19(conda)

## Folder list
1. Argumentation
2. ConfusionMatrix
3. Dataset
4. Predict
5. Train
***
### Argumentation
- dataset_argument_test.py: Test file for one image augmentation
- dataset_argumentation.py: Data augmentation multi image and label, this file purpose is augmentation image and lable of one image dataset
  1. set folder
  2. set image_dir
  3. set label_dir

### ConfusionMatrix
- cm_draw.py: confusion matrix draw
- cm_hands.py: draw CM for only hands model
- cm_Integrated.py: draw CM for integreted model that combine hand and pose after post-processing
  
### Dataset
- dataset_camera_test.py: Intel Realsense D455f camera operation test
- dataset_collect.py: Image data collection according to hand type
- dataset_combine.py: Merge data within folders saved by hand type and save them in one folder
- dataset_divide.py: Divide the train, validation, and test data sets by a specified ratio using scikit-learn.
- dataset_remove_duplicate.py: Remove duplicate files by comparing lable and image

### Predict
- predict_f.py: Fuction list
- predict_image : Run model for one image for test
- predict_notebook: notebook enviorment execpt d455f camera
- predict_pc : pc enviorment model inference test with d455f camera
- **predict.py** : Main file for inference model on Jetson Orin Nano device

### Train
- data.yaml: setting file for train
- train.py: trainning model with hyperparameter
