# Classifying Food Items by Image using CNN

Course: **Machine Learning (ML)**
Language: **Serbian** <br>

**Faculty of Mathematics, University of Belgrade** <br>
**May 2023 - Today**

# Testing

Run the `init.py` script to download the food image dataset locally and create the `datasets` directory used for training, validation and testing the ML models as well as the `artifacts` directory used for saving the ML models as well as the graphs of loss and accuracy as well as the confusion matrix.

# Project Structure

All the code is located in the `src` directory. The `src` directory contains the following subdirectories:
- `models` - contains the ML models
- `utils` - contains the utility functions

The project is split in this way because all the ML models differ only by the architecture and all call the same utility functions for loading and preprocessing the data, splitting the dataset into train, validation and test sets, building, training, evaluating and saving the ML models as well as displaying the graphs of loss and accuracy as well as the confusion matrix.

Inside the `utils` folder there are the following files:
- `preprocessing_utils.py` - this file contains the helper class PreprocessingUtils which is used for loading and preprocessing the images and labels and splitting the dataset into train, validation and test sets
- `model_wrapper.py` - this file contains the helper class ModelWrapper which is used for building, training, evaluating and saving the ML models as well as displaying the graphs of loss and accuracy as well as the confusion matrix

Inside the `models` folder there are the following files:
- `alex_net.ipynb` - this file contains the implementation of the AlexNet architecture
- `caffe_net.ipynb` - this file contains the implementation of the CaffeNet architecture
- `google_net.ipynb` - this file contains the implementation of the GoogleNet architecture
- `inception_V3.ipynb` - this file contains the implementation of the InceptionV3 architecture

There is also the subdirectory `dreprecated` inside the `models` directory which contains the previous versions of the ML models.