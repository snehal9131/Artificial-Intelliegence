# Artificial-Intelliegence
The python program we present is a Convolutional Neuronal Network on a flower dataset that is categorized into 5 different categories: Daisy, Sunflower, Rose, Tulip and Dandelion

# Dataset
The dataset can be found here: https://www.kaggle.com/alxmamaev/flowers-recognition

# Repository Structure
The repository contains only the files for the python program. Model, Dataset are not included.

You can find 2 Subprojects:
-flower CNN (CNN on the dataset using the Image Data Generator) containing a full CNN run.

The second Subproject is the main Project:
-myclassifier for the CNN model
-utils for the preparation of the dataset
-predict for the CNNs prediction on the different images

# Run the project
To run the project you need the following libraries:
- tensorflow
- numpy
- sklearn
- mathplotlib
- opencv
- pickle
- os

You also need to choose the right path to connect your dataset, which can be found and downloaded here: https://www.kaggle.com/alxmamaev/flowers-recognition

Batchsize and Epochs can be adjusted in line 45 of the myclassifier.py 
