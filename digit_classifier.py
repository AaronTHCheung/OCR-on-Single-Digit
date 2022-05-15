import cv2 as cv
from src.Models import CNN
import numpy as np
import argparse
import torch
import sys

if __name__ == "__main__":
    # Reading arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Digit image to be classified (in .jpg format)')
    parser.add_argument('--model', help='CNN model to be used')
    args = parser.parse_args()

    # Path
    image_filepath = args.image
    model_path = 'model/'
    model_filename = args.model if args.model is not None else 'cnn'
    model_filepath = model_path + model_filename + '.pth'


    ## Image denoising
    image = cv.imread(image_filepath)
    denoised_image = cv.fastNlMeansDenoisingColored(image)


    ## Image binarization
    gray_image = cv.cvtColor(denoised_image, cv.COLOR_BGR2GRAY)
    (thresh, bw_image) = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)

    # check if it has a black background, invert image if it does not
    n_zeros = np.count_nonzero(bw_image==0)
    n_ones = bw_image.size - n_zeros
    if (n_zeros < n_ones):  # image has white background
        bw_image = 255-bw_image


    ## Remove empty spaces
    coords = cv.findNonZero(bw_image)
    x, y, w, h = cv.boundingRect(coords)
    cropped_bw_image = bw_image[y:y+h, x:x+w]


    ## Resize the image to 28x28 (the input dimension of the trained CNN)
    # calculate the padding
    digit_dim = (20,20)  # without padding
    padding = int((28-20)/2)

    # add pads to the images
    nonpadded_image = cv.resize(cropped_bw_image, digit_dim)
    final_image = cv.copyMakeBorder(nonpadded_image, top=padding, bottom=padding, left=padding, right=padding, borderType=cv.BORDER_CONSTANT, value = 0)


    ## Use the trained CNN to classify the digit
    # load the model
    model = CNN()
    best_model_save = torch.load(model_filepath)
    model.load_state_dict(best_model_save['state_dict'])

    # transform the image's dimension as CNN's input
    input = torch.tensor(final_image).view(1,1,28,28)
    input = input/255  # CNN takes [0,1] values as input

    # output the classication result to standard output
    pred = model(input)[0]
    digit_hat, digit_hat_probability = torch.argmax(pred).item(), torch.max(pred).item()
    sys.stdout.write(str(digit_hat)+','+str(digit_hat_probability))