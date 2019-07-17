# Real Time Modular Object Detection

This project is aimed to detect any specific type of object in real time by passing a dataset of the object. There are three folders that show the progress of the different algorithms in this project. 

- Has support to use [Zynq UltraScale+](https://www.xilinx.com/products/boards-and-kits/zcu104.html) as a processor for optical flow

**Use the _obj-detection-frcnn_ for best results.**

## Dataset
All of the algorithms are tested on the [Town Centre data set](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets). 
Results for each algorithm are shown below:

#### Faster R-CNN:
#### U-Net:
#### Optical Flow with a Simple Classifier:

## Prerequisites
- Python 3
- Keras
- Tensorflow
- OpenCV
- Imutils
- Numpy

## Train models with Town Centre Data Set (Only works for U-Net and Faster R-CNN)
**Note: Make sure to use a computer with a powerful GPU or Google Cloud/Amazon instance with GPU for best efficiency**
- Download TownCenterXVID.avi from the Dataset linked above and rename the video to test.avi
- Copy and paste the text from TownCentre-groundtruth.top into a file called data.csv
- Create an empty folder called **data** in the corresponding algorithm folder(i.e. obj-detection-frcnn) **where there is a script called create_data.py.**
- Place the test.avi file and data.csv file into the data folder
- Run the create_data.py script with the following command
  ```bash
  python3 create_data.py
  ```
- If using Faster R-CNN, once create_data.py is finished, rename the create train.csv to train.txt 
- Run the following command to train the model/weights:
  - If using Faster R-CNN:
    ```bash
    python3 train_frcnn.py -o simple -p data/train.txt
    ```
    **Note: for optional training arguments, read inside train_frcnn.py to find other arguments**
  - If using U-Net:
    ```bash
    python3 train.py
    ```
## Train models with Custom Data Set
