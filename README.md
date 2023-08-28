# Classifying Food Items by Image using CNN

Course: **Machine Learning (ML)**
Language: **Serbian** <br>

Participants:
1. **Pavle Cvejović** <br>
2. **Viktor Novaković** <br>

**Faculty of Mathematics, University of Belgrade** <br>
**May 2023 - August 2023**

# Project description

## Overview
In this project, we used deep learning architectures from the [research paper](http://cs231n.stanford.edu/reports/2017/pdfs/607.pdf) written in the project specification, alongside a custom-built model to sort out different food images through classification. The aim was to compare the efficiencies of these models and discern the potential of tailored solutions in machine learning tasks.

## Dataset
We used the [Freiburg Groceries Dataset](http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset) which covers an assortment of food items across various categories. In contrast to existing groceries datasets, dataset includes a large variety of perspectives, lighting conditions, and degrees of clutter. The dataset was divided into training, validation, and testing segments to ensure comprehensive model evaluation.

Visualization of the dataset can be found in the `src/data/vizualization.ipynb` notebook.

## Models

For this project, we used the following models:

1. **AlexNet**
2. **GoogLeNet**
3. **CaffeNet**
4. **Inception v3**

Detailed descriptions and specifics about each model are provided in the accompanying presentation.

## Training procedure
During the training phase, the models underwent rigorous training using the dedicated training subset of our dataset. To maintain a level playing field, a consistent set of hyperparameters was adopted across the board. 

### Parameters explored:
- **Epochs**
- **Batch Size**
- **Regularizations**: Dropout, L2
- **Optimizers**: Adam, SGD
- **Learning Rate**
- **Number of Layers**
- **Layer Size**
- etc...

## Evaluation & Findings
The models were tested against the validation and test subsets of the dataset. The metrics of evaluation were:
- **Loss**
- **Accuracy**
- **Classification Report**
- **Confusion Matrix**

Detailed results and visual representations of these metrics can be found in the accompanying presentation.

# Manual testing

First, install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

Then, run the `init.py` script to download the food image dataset locally and create the `datasets` directory used for training, validation and testing the ML models as well as the `artifacts` directory used for saving the ML models as well as the graphs of loss and accuracy as well as the confusion matrix.

# Project structure

All the code is located in the `src` directory. The `src` directory contains the following subdirectories:
- `models` - contains the ML models
- `utils` - contains the utility functions
- `data` - contains the visualization of the dataset

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