# Program description
This is a Python program that trains a CNN (convolutional neural network) model to classify handwritten single digits. The CNN model will be trained on the MNIST dataset. The trained classifer can recognize handwritten single digit (0-9) on a scanned image.

# Requirements
- Language: Python 3.8.10
- Python Libraries:
  - torch==1.10.2
  - numpy==1.22.1
  - opencv-python==4.5.5.64
  - python-mnist==0.7

# Training
1. Type the following command in the shell propmt: ```python training.py --filename filename --batch batch --epoch epoch```. Below are the description of the optional command line arguments:
    - filename: filename of the trained CNN model (default: 'cnn')
    - batch: training batch size (default: 16)
    - epoch: training epoches (default: 100)
2. If the training script runs successfully, a CNN model will be trained on the MNIST dataset. You should see similar output in the terminal:
    ```console
    --------------Model Summary--------------
    CNN(
    (conv1): Sequential(
        (0): Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1))
        (1): ReLU()
        (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (conv2): Sequential(
        (0): Conv2d(32, 16, kernel_size=(5, 5), stride=(1, 1))
        (1): ReLU()
        (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (out): Linear(in_features=144, out_features=10, bias=True)
    )

    Epoch (1/100): Testing Accuracy (current, best)=(0.875, 0.875)
    Epoch (2/100): Testing Accuracy (current, best)=(0.917, 0.917)
    ...
    ``` 
3. After the training has completed, the model with the highest testing accuracy in the MNIST problem is saved in the `/model/` folder.

# Running the digit classifier
1. Type the following command in the shell propmt: ```python digit_classifier.py image --model model```. Below are the description of the command line arguments:
    - filename (required): digit image to be classified (in .jpg format)
    - model (optional): CNN model to be used (default: 'cnn')
2. If the script runs successfully, you should see two numbers in the terminal output that are separated by a comma. 
    - 1<sup>st</sup> number: the digit classified by the CNN model
    - 2<sup>nd</sup> number: classification confidence level (higher means greater confidence)
- Example: 'test_image.jpg' is a scanned image of a handwritten digit '2'. To use the included CNN model to classify the handwritten digit, we can type the following command in the shell propmpt: ```python digit_classifier.py test_image.jpg --model cnn```. The terminal output would be as follows:
    ```console
    2,0.9852445721626282
    ```
    The 1<sup>st</sup> number, 2, indicates that the CNN classifies the handwritten digit to be the digit '2'. The 2<sup>nd</sup> number, 0.9852445721626282, indicates that the CNN is about 98.5% confident that the digit is a digit '2'

# Disclaimer
Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license
