# CS6475 - Fall 2021
import cv2
import numpy as np


""" Assignment 0 - Introduction

This file has a number of basic image handling functions that you need
to write python3 code for in order to complete the assignment. We will
be using these operations throughout the course, and this assignment helps
to familiarize yourself with the cv2 and numpy libraries. Please write
the appropriate code, following the instructions in the docstring for each
function. Make sure that you know which commands and libraries you may or may
not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, prints, or writes
    over the image that you are being passed in. Any code line that you may
    have in your code to complete these actions must be commented out when
    you turn in your code. These actions may cause the autograder to crash,
    which will count as one of your limited attempts.

    2. DO NOT import any other libraries aside from the libraries that we
    provide. You may not import anything else, you should be able to complete
    the assignment with the given libraries (and in most cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the course environment.
    You are responsible for ensuring that your code executes properly in the
    course environment and in the Gradescope autograder. Any changes you make
    outside the areas annotated for student code must not impact your performance
    on the autograder system.
    Thank you.
"""


def returnYourName():
    """ When it is called, this function should return your official name as
    shown on your Gradescope Account for full credit.

    To find your name, login to Gradescope, and click ACCOUNT > Edit Account in the
    bottom left corner. Your official Gradescope name is shown down the page in the
    "Course-Specific Information" section, for course CS6475. It is NOT necessarily
    the same as the Full Name at the top.

    If you are still not sure what name to use, then the first time you submit this
    file to A0_Introduction Resources in Gradescope, the autograder will print an
    error message with your name.

    Parameters
    ----------
    input : none

    Returns
    -------
    output : string
        a string formatted as follows: your official name as shown on your
        Gradescope Account
    """
    # WRITE YOUR CODE HERE.
    return 'SeungHui Huh'

    # End of code
    raise NotImplementedError


def imageDimensions(image):
    """ This function takes your input image and returns its array shape.
    You may use a numpy command to find the shape.

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of three dimensions (HxWxD) and type np.uint8
        Your code should work for color (three-channel RGB) images.
    Returns
    ----------
    tuple:  tuple of numpy integers of type np.int
        the tuple returns the shape of the image ordered as
        (rows, columns, channels)
    """
    # WRITE YOUR CODE HERE.
    return (image.shape)

    # End of code
    raise NotImplementedError


def convolutionManual(image, filter):
    """ This function takes your input color (BGR) image and a square, symmetrical
    filter with odd dimensions and convolves them. The returned image must be the same
    size and type as the input image. We may input different sizes of filters,
    do not hard-code a value.

    Your code must use loops (it will be slower) and move the filter past each pixel
    of your image to determine the new pixel values. We assign this exercise to help
    you understand what filtering does and exactly how it is applied to an image.
    Almost every assignment will involve filtering, this is essential understanding.

    **************** FORBIDDEN COMMANDS ******************
    NO CV2 LIBRARY COMMANDS MAY BE USED IN THIS FUNCTION
    NO NUMPY CONVOLVE OR STRIDES COMMANDS
    In general, no use of commands that do your homework for you.

    The use of forbidden commands may result in a zero score
    for the assignment and an honor code violation review
    ******************************************************

    Follow these steps:
    (1) Copy the image into a new array, so that you do not alter the image.
        Change the type to float64, for calculations.
    (2) From the shape of the filter, determine how many rows of padding you need.
    (3) Pad the copied image with mirrored padding. You must use the correct
        amount of padding of the correct type. For example: if a 7x7 filter is
        used you will require three rows/columns of padding around the image.
        A mirrored padding row will look like:

            image row [abcdefgh] ====> padded image row [cba|abcdefgh|hgf]

        Note1: If you use np.pad for this, each channel must be filtered separately.

    (5) Convolve, passing the filter (kernel) through all of the padded image.
        Save new pixel values in the result array. Don't overwrite the padded image!
    (6) Use numpy to round your values.
    (7) Convert to the required output type.

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWxD) and type np.uint8
        Your original image should be a three-channel RGB image.
    filter : numpy.ndarray
        A 2D numpy array of variable dimensions, with type np.float.
        The filter (also called kernel) will be square with odd dimensions,
        e.g. [3,3], [11,11], etc. Your code should be able to handle
        any filter that fits this description.
    Returns
    ----------
    image:  numpy.ndarray
        a convolved numpy array with the same dimensions as the input image
        and with type np.uint8.
    """
    # WRITE YOUR CODE HERE.
    copied_img = np.copy(image)
    copied_img = copied_img.astype(np.float64)
    row, col, chann = copied_img.shape
    kernel = int((filter.shape[0]-1)/2)
    filter_size = filter.shape[0]
    padded_img = np.pad(copied_img, pad_width=((kernel,kernel),(kernel,kernel),(0,0)),mode='symmetric')
    new_img = np.zeros_like(copied_img)
    

    for z in range(chann):
        for x in range(row):
            for y in range(col):
                new_img[x,y,z] = np.around(np.sum(padded_img[x:x+filter_size,y:y+filter_size,z]*filter))

    new_img = new_img.astype(np.uint8) 
    return new_img
    




    # End of code
    raise NotImplementedError
    


def convolutionCV2(image, filter):
    """ This function performs convolution on your image using a square,
    symmetrical odd-dimension 2D filter. You may use cv2 commands to complete this
    function. See the opencv docs for the version of cv2 used in the class env.
    Opencv has good tutorials for image processing.

    *** You may use any cv2 or numpy commands for this function ***

    Follow these steps:
    (1) same as convolutionManual.
    (2) Pad the copied image in the same mirrored style as convolutionManual:

            image row [abcdefgh] ====> padded image row [cba|abcdefgh|hgf]

        With cv2 commands you may not need to code the padding,
        but you must specify the correct Border Type.
        Note: Numpy and cv2 use different names for this type of padding. Be careful.
    (3) Complete the convolution
    (4) Finish is same as convolutionManual.

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWxD) and type np.uint8
        Your original image should be a three-channel RGB image.
    filter : numpy.ndarray
        A 2D numpy array of variable dimensions, with type np.float.
        The filter (also called kernel) will be square with odd dimensions,
        e.g. [3,3], [11,11], etc. Your code should be able to handle
        any filter that fits this description.
    Returns
    ----------
    image:  numpy.ndarray
        a convolved numpy array with the same dimensions as the input image
        and with type np.uint8.
    """
    # WRITE YOUR CODE HERE.
    copied_img = np.copy(image)
    #copied_img2 = copied_img.astype(np.float64)
    filter_size = filter.shape[0]
    new_img = cv2.filter2D(copied_img,-1,kernel = filter,borderType=cv2.BORDER_REFLECT)
    new_img = new_img.astype(np.uint8)
    return new_img

    # End of code
    raise NotImplementedError


# ----------------------------------------------------------
if __name__ == "__main__":
    """ YOU MAY USE THIS AREA FOR CODE THAT ALLOWS YOU TO TEST YOUR FUNCTIONS.
    This section will not be graded, you can change or delete the code below.
    When you are ready to test your code on the autograder, comment out any code
    below, along with print statements you may have in your functions. 
    Any imported libraries or extensive print statements here or in the functions 
    may cause crashes.  e.g. Don't have a print statements that print out 
    thousands of pixel values, or counts them individually.
    
    Uncommented code here is not run by the autograder, but can crash it, 
    costing you a try!
    """
    # WRITE YOUR CODE HERE

    # create filter that meets function requirements
    #filter = np.random.randint(0, 100, size=(3, 3))
    # filter = np.ones((3,3))
    # #read in your image, change image format to match
    #image = cv2.imread("image.png")
    #-- OR --
    #Create a small random toy image for testing
    #image = np.random.randint(0, 255, (5, 4, 3), dtype=(np.uint8))
    #save the original image in .png for the report
    # cv2.imwrite("image.png", image)

    #Code to run the functions and generate results
    #Uncomment whatever you want to test.

    # print("returnYourName:")
    # print(returnYourName(), type(returnYourName()))

    # print("\nimageDimensions:")
    # dims = imageDimensions(image)
    # print(dims, type(dims), type(dims[0]), len(dims))

    # print('\nconvolutionManual:')
    # convolve = convolutionManual(image, filter)
    # print(np.shape(convolve), convolve.dtype)
    # cv2.imwrite("convolveManual.png", convolve)     # save image

    # print('\nconvolutionCV2:')
    # cv2convolve = convolutionCV2(image, filter)
    # print(np.shape(cv2convolve), cv2convolve.dtype)
    # cv2.imwrite("convolveCV2.png", cv2convolve)     # save image

    #End of code
    #DON'T FORGET TO COMMENT OUT YOUR CODE IN THIS SECTION!
    pass