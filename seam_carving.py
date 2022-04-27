# CS6475 - Spring 2021

import numpy as np
import scipy as sp
import cv2
import scipy.signal                     # option for a 2D convolution library
from matplotlib import pyplot as plt    # for optional plots

import copy


""" Project 1: Seam Carving

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

References
----------
See the following papers, available in Canvas under Files:

(1) "Seam Carving for Content-Aware Image Resizing"
    Avidan and Shamir, 2007
    
(2) "Improved Seam Carving for Video Retargeting"
    Rubinstein, Shamir and Avidan, 2008
    
FORBIDDEN:
    1. OpenCV functions SeamFinder, GraphCut, and CostFunction are
    forbidden, along with any similar functions that may be in the class environment.
    2. Numeric Metrics functions of error or similarity; e.g. SSIM, RMSE. 
    These must be coded from their mathematical equations. Write your own versions. 

GENERAL RULES:
    1. ALL CODE USED IN THIS ASSIGNMENT to generate images, red-seam images,
    differential images, and comparison metrics must be included in this file.
    2. YOU MAY NOT USE any library function that essentially completes
    seam carving or metric calculations for you. If you have questions on this,
    ask on Ed. **Usage may lead to zero scores for the project and review
    for an honor code violation.**
    3. DO NOT CHANGE the format of this file. You may NOT change existing function
    signatures, including the given named parameters with defaults.
    4. YOU MAY ADD FUNCTIONS to this file, however it is your responsibility
    to ensure that the autograder accepts your submission.
    5. DO NOT IMPORT any additional libraries other than the ones imported above.
    You should be able to complete the assignment with the given libraries.
    6. DO NOT INCLUDE code that prints, saves, shows, displays, or writes the
    images, or your results. If you have code in the functions that 
    does any of these operations, comment it out before autograder runs.
    7. YOU ARE RESPONSIBLE for ensuring that your code executes properly.
    This file has only been tested in the course environment. Any changes you make
    outside the areas annotated for student code must not impact the autograder
    system or your performance.
    
FUNCTIONS:
    returnYourName
    IMAGE GENERATION:
        beach_back_removal
        dolphin_back_insert with redSeams=True and False
        dolphin_back_double_insert
        bench_back_removal with redSeams=True and False
        bench_for_removal with redSeams=True and False
        car_back_insert
        car_for_insert
    COMPARISON METRICS:
        difference_image
        numerical_comparison
"""


def returnYourName():
    """ This function returns your name as shown on your Gradescope Account.
    """
    # WRITE YOUR CODE HERE.
    return 'SeungHui Huh'
    raise NotImplementedError




# -------------------------------------------------------------------
""" IMAGE GENERATION
    *** ALL IMAGES SUPPLIED OR RETURNED ARE EXPECTED TO BE UINT8 ***
    Parameters and Returns are as follows for all of the removal/insert 
    functions:

    Parameters
    ----------
    image : numpy.ndarray (dtype=uint8)
        Three-channel image of shape (r,c,ch). 
    seams : int
        Integer value of number of vertical seams to be inserted or removed.
        NEVER HARDCODE THE NUMBER OF SEAMS, we check other values in the autograder.
    redSeams : boolean
        Boolean variable; True = produce a red seams image, False = no red seams
        
    Returns
    -------
    numpy.ndarray (dtype=uint8)
        An image of shape (r, c_new, ch) where c_new = new number of columns.
        Make sure you deal with any needed normalization or clipping, so that 
        your image array is complete on return.
"""
def beach_back_removal(image, seams=300, redSeams=False):
    """ Use the backward method of seam carving from the 2007 paper to remove
   the required number of vertical seams in the provided image. Do NOT hard-code the
    number of seams to be removed.
    """
    # WRITE YOUR CODE HERE.
    forwardE_TF = False      
    image = image.astype(np.float64)
    carved_images = img_rem_b(image, seams,redSeams, forwardE_TF).astype(np.uint8)
    return carved_images
    raise NotImplementedError


def dolphin_back_insert(image, seams=100, redSeams=False):
    """ Similar to Fig 8c and 8d from 2007 paper. Use the backward method of seam carving to 
    insert vertical seams in the image. Do NOT hard-code the number of seams to be inserted.
    
    This function is called twice:  dolphin_back_insert with redSeams = True
                                    dolphin_back_insert without redSeams = False
    """
    # WRITE YOUR CODE HERE
    forwardE_TF = False      
    image = image.astype(np.float64)
    carved_images = img_add_rep(image,seams, redSeams,forwardE_TF).astype(np.uint8)
    return carved_images
    raise NotImplementedError

def dolphin_back_double_insert(image, seams=100, redSeams=False):
    """ Similar to Fig 8f from 2007 paper. Use the backward method of seam carving to 
    insert vertical seams by performing two insertions, each of size seams, in the image.  
    i.e. insert seams, then insert seams again.  
    Do NOT hard-code the number of seams to be inserted.
    """
    # WRITE YOUR CODE HERE.
    forwardE_TF = False      
    image = image.astype(np.float64)
    output = img_add_rep2(image, seams, redSeams,forwardE_TF).astype(np.uint8)
    return output
    raise NotImplementedError


def bench_back_removal(image, seams=225, redSeams=False):
    """ Similar to Fig 8 from 2008 paper. Use the backward method of seam carving to 
    remove vertical seams in the image. Do NOT hard-code the number of seams to be removed.
    
    This function is called twice:  bench_back_removal, redSeams = True
                                    bench_back_removal, redSeams = False
    """
    # WRITE YOUR CODE HERE.
    forwardE_TF = False      
    image = image.astype(np.float64)
    output = img_rem_b(image, seams,redSeams,forwardE_TF).astype(np.uint8)
    return output
    raise NotImplementedError


def bench_for_removal(image, seams=225, redSeams=False):
    """ Similar to Fig 8 from 2008 paper. Use the forward method of seam carving to 
    remove vertical seams in the image. Do NOT hard-code the number of seams to be removed.
    
    This function is called twice:  bench_for_removal, redSeams = True
                                    bench_for_removal, redSeams = False
  """
    # WRITE YOUR CODE HERE.
    forwardE_TF = True    
    image = image.astype(np.float64)
    if redSeams:  
        output = img_rem_red(image, seams,redSeams,forwardE_TF).astype(np.uint8)
    else:
        output = img_rem(image,seams, redSeams,forwardE_TF).astype(np.uint8)
    return output
    raise NotImplementedError


def car_back_insert(image, seams=170, redSeams=False):
    """ Fig 9 from 2008 paper. Use the backward method of seam carving to insert
    vertical seams in the image. Do NOT hard-code the number of seams to be inserted.
    """
    # WRITE YOUR CODE HERE.
    forwardE_TF = False      
    image = image.astype(np.float64)
    carved_images = img_add_rep(image,seams, redSeams,forwardE_TF).astype(np.uint8)
    return carved_images
    raise NotImplementedError


def car_for_insert(image, seams=170, redSeams=False):
    """ Similar to Fig 9 from 2008 paper. Use the forward method of seam carving to 
    insert vertical seams in the image. Do NOT hard-code the number of seams to be inserted.
    """
    # WRITE YOUR CODE HERE.
    forwardE_TF = True   
    image = image.astype(np.float64) 
    carved_images = img_add_rep(image,seams, redSeams,forwardE_TF).astype(np.uint8)
    return carved_images  
    raise NotImplementedError
    

# __________________________________________________________________
""" COMPARISON METRICS 
    There are two functions here, one for visual comparison support and one 
    for a quantitative metric. 
"""

def difference_image(result_image, comparison_image):
    """ Take two images and produce a difference image that best visually
    indicates how and where the two images differ in pixel values.
    
    Parameters
    ----------
    result_image, comparison_image : numpy.ndarray (dtype=uint8)
        two BGR images of the same shape (r,c,ch) 
    
    Returns
    -------
    numpy.ndarray (dtype=uint8)
        A BGR image of shape (r, c, ch) representing the differences between two
        images. 
        
    NOTES: MANY ERRORS IN PRODUCING DIFFERENCE IMAGES RELATE TO THESE ISSUES
        1) Do your calculations in floats, so that data is not lost.
        2) Before converting back to uint8, complete any necessary scaling,
           rounding, or clipping. 
    """
    # WRITE YOUR CODE HERE.
    result_image = result_image.astype(np.float64)
    comparison_image = comparison_image.astype(np.float64)
    new_img = result_image.copy()
    h,w,_ = result_image.shape
    h1,w1,_ = comparison_image.shape
    print(h,w,h1,w1)
    for i in range(h):
        for j in range(w):
            diff = abs(result_image[i][j] - comparison_image[i][j])
            diff = np.sum(diff)
            if diff > 191.25 and diff<=382.5:
                new_img[i][j] = [255,0,0]
            if diff>382.5 and diff<= 573.75:
                new_img[i][j] = [0,255,0]
            if diff>573.75:
                new_img[i][j] = [0,0,255]
    new_img = new_img.astype(np.uint8)
    return new_img
    raise NotImplementedError


def numerical_comparison(result_image, comparison_image):
    """ Take two images and produce one or two single-value metrics that
    numerically best indicate(s) how different or similar two images are.
    Only one metric is required, you may submit two, but no more.
    
    If your metric produces a result indicating a total number of pixels, or values,
    formulate it as a ratio of the total pixels in the image. This supports use
    of your results to evaluate code performance on different images.

    ******************************************************************
    NOTE: You may not use functions that perform the whole function for you.
    Research methods, find an algorithm (equation) and implement it. You may
    use numpy array MATHEMATICAL functions such as abs, sqrt, min, max, dot, .T 
    and others that perform a single operation for you.
    
    FORBIDDEN: Library functions of error or similarity; e.g. SSIM, RMSE, etc.
    Use of these functions may result in zero for the assignment and review 
    for an honor code violation.
    ******************************************************************

    Parameters
    ----------
    result_image, comparison_image : numpy.ndarray (dtype=uint8)
        two BGR images of the same shape (r,c,ch) 

    Returns
    -------
        value(s) : float as type np.float64
        One or two single float value metric comparisons
        Return a tuple of single float values if you are using two metrics.

    NOTES:  (1) you may return only one or two values; choose the best one(s)
                for determining image similarity.
            (2) If you do your calculations in float32, change to float64 for
                submission. Some numpy comparison functions do not recognize
                float32.   

    """
    # WRITE YOUR CODE HERE.
    result_image, comparison_image = result_image.astype(np.float64),comparison_image.astype(np.float64)
    k = cv2.getGaussianKernel(3,1)
    c1, c2 = (2.55)**2, (7.65)**2
    windows = np.outer(k,k.transpose())
    x_avg = cv2.filter2D(result_image,-1,windows)[5:-5,5:-5]
    xsq = x_avg**2
    y_avg = cv2.filter2D(comparison_image,-1,windows)[5:-5,5:-5]
    ysq = y_avg**2
    xy = x_avg*y_avg
    sig1 = cv2.filter2D(result_image**2, -1, windows)[5:-5, 5:-5] - xsq
    sig2 = cv2.filter2D(comparison_image**2, -1, windows)[5:-5, 5:-5] - ysq
    sig12 = cv2.filter2D(result_image * comparison_image, -1, windows)[5:-5, 5:-5] - xy

    output = (((2 * xy + c1) * (2 * sig12 + c2)) / ((xsq + ysq + c1) *
                                                            (sig1 + sig2 + c2))).mean()
    return output
    # count = 0
    # h,w,_ = result_image.shape
    # total_pixel = h*w
    # for i in range(h):
    #     for j in range(w):
    #             if (result_image[i][j] == comparison_image[i][j]).all():
    #                 count+=1
    # identical_percentage = 100*count/total_pixel
    # mean_squared_error = (result_image.astype(np.float64)-comparison_image.astype(np.float64))**2
    # mean_squared_error = np.sum(mean_squared_error)
    # mean_squared_error = mean_squared_error/total_pixel
    # return identical_percentage, mean_squared_error
    raise NotImplementedError

def backenergy_generate(image):
    weights = np.array([1,0,-1])
    y_gradient=sp.ndimage.filters.convolve1d(image, weights, axis=0, mode='wrap')
    x_gradient=sp.ndimage.filters.convolve1d(image, weights, axis=1, mode='wrap')
    x = np.sum(x_gradient**2, axis=2)
    y = np.sum(y_gradient**2, axis=2)
    back_energy = np.sqrt(x + y)
    return back_energy

def forwardenergy_generate(image):
    h = image.shape[0]
    w = image.shape[1] 
    # Citation: Adapted from 
    # https://edstem.org/us/courses/7818/discussion/650343   
    # Converting images to real gray
    image = np.average(image, axis=2)
    energy_map = np.zeros((h, w))
    map = np.copy(energy_map)
    for i in range(1, h):
        for j in range(w):
            if j == 0:
                left = w-1
            else: 
                left = j-1
            if j == w-1:
                right = 0
            else:
                right = j+1
            up = i-1
            up_diff = np.abs(image[i,right] - image[i,left])
            left_diff = np.abs(image[i,left] - image[up,j]) + up_diff
            right_diff = np.abs(image[i,right] - image[up,j]) + up_diff
            current_diff = np.array([up_diff, left_diff, right_diff])
            up_total = map[up,j]
            left_total = map[up,left]
            right_total= map[up,right]
            total_diff = np.array([up_total, left_total, right_total]) + current_diff
            min_idx = np.argmin(total_diff)
            map[i,j] = total_diff[min_idx]
            energy_map[i,j] = current_diff[min_idx]
    return energy_map

def find_min_seam(image, forwardE_TF):
    height, width, length = image.shape
    if forwardE_TF == True:
        e_generate = forwardenergy_generate
    else:
        e_generate = backenergy_generate
    e_mat = e_generate(image)
    mat = np.zeros_like(e_mat, dtype= np.int)

    li = []
    for i in range(height-1):
        for j in range(width):
            if j != 0:
                ind = np.argmin(e_mat[i,j-1:j+2])
                mat[i+1,j] = (j-1)+ind
                e_min = e_mat[i,(j-1)+ind]
            else:
                ind = np.argmin(e_mat[i,:2])
                mat[i+1,0] = ind
                e_min = e_mat[i,ind]
            e_mat[i+1, j] += e_min
    rslt = np.ones((height, width), dtype=np.bool)
    jinit = np.argmin(e_mat[-1])
    mat_temp = mat.copy()
    for k in range(height,0,-1):
        rslt[k-1, jinit] = False
        li.append(jinit)
        jinit = mat[k-1, jinit]
    return np.array(li[::-1]), rslt

def img_add(image, ind, redSeams):
    height, width, length = image.shape
    rslt = np.zeros((height, width + 1, 3))
    for i in range(height):
        j = ind[i]
        for k in range(3):
            if j == 0:
                p = sum(image[i,:2,k])/2
                rslt[i,0,k] = image[i,0,k]
                rslt[i,j+1,k] = p
            else:
                p = sum(image[i,j-1:j+1,k])/2
                rslt[i,:j,k] = image[i,:j,k]
                rslt[i,j,k] = p 
            rslt[i,j+1:,k] = image[i,j:,k]
        if redSeams:
            rslt[i,j,:] = np.array([0,0,255])  
    return rslt

def img_rem(image, num_seams, redSeams, forwardE_TF):
    for i in range(num_seams):
        ind, rslt = find_min_seam(image,forwardE_TF)
        height, width, length = image.shape
        path= np.stack([rslt] * 3, axis=2)
        if redSeams == False:
            image = image[path].reshape((height,width-1,3))
        else:
            for i in range(image.shape[0]):                
                image[i,ind[i],:] = np.array([0,0,255])
    return image

def img_rem_red(image, num_seams, redSeams, forwardE_TF):
    finans = []
    temp = image.copy()
    for i in range(num_seams):
        ind, rslt = find_min_seam(temp,forwardE_TF)
        height, width, _ = temp.shape
        path= np.stack([rslt] * 3, axis=2)
        if redSeams == False:
            temp = temp[path].reshape((height,width-1,3))
        else:
            finans.append(ind)
            temp = temp[path].reshape((height,width-1,3))
    for j in range(len(finans)):
        inds = finans[j]
        for k in range(len(inds)):
            image[k,inds[k],:] = np.array([0,0,255])

    return image
def img_rem_b(image, num_seams, redSeams, forwardE_TF):
    for i in range(num_seams):
        ind, rslt = find_min_seam(image, forwardE_TF)
        height, width, length = image.shape
        path= np.stack([rslt] * 3, axis=2)
        if redSeams == False:
            image = image[path].reshape((height,width-1,3))
        else:
            for i in range(image.shape[0]):                
                image[i,ind[i],:] = np.array([0,0,255])
    return image

def img_add_rep(image, num_rep,redSeams,forwardE_TF):
    ind_li = []
    temp = image.copy()
    for i in range(num_rep):
        ind, rslt = find_min_seam(temp, forwardE_TF)
        ind_li.append(ind)
        height, width, length = temp.shape
        temp = temp[np.stack([rslt] * 3, axis=2)]
        temp = temp.reshape((height,width-1,3))
    ind_li = ind_li[::-1]
    for j in range(num_rep):
        ind_seam = ind_li[-1]
        image = img_add(image, ind_seam, redSeams)
        ind_li = ind_li[:-1]
        for rest in ind_li:
            ind_temp = np.where(rest > ind_seam-1)
            rest[ind_temp] += 2         
    return image

def img_add2(image, ind, redSeams):
    height, width, length = image.shape
    rslt = np.zeros((height, width + 2, 3))
    for i in range(height):
        j = ind[i]
        for k in range(3):
            if j == 0:
                p = sum(image[i,:2,k])/2
                rslt[i,0,k] = image[i,0,k]
                rslt[i,j+1,k] = p
                rslt[i,j+2,k] = p
            else:
                p = sum(image[i,j-1:j+1,k])/2
                rslt[i,:j,k] = image[i,:j,k]
                rslt[i,j,k] = p 
                rslt[i,j+1,k] = p 
            rslt[i,j+2:,k] = image[i,j:,k]
        if redSeams:
            rslt[i,j,:] = np.array([0,0,255]) 
    return rslt

def img_add_rep2(image, num_rep, redSeams,forwardE_TF):
    ind_li = []
    temp = image.copy()
    for i in range(num_rep):
        ind, rslt = find_min_seam(temp,forwardE_TF)
        ind_li.append(ind)
        height, width, length = temp.shape
        temp = temp[np.stack([rslt] * 3, axis=2)]
        temp = temp.reshape((height,width-1,3))
    ind_li = ind_li[::-1]
    for j in range(num_rep):
        ind_seam = ind_li[-1]
        image = img_add2(image, ind_seam, redSeams)
        ind_li = ind_li[:-1]
        for rest in ind_li:
            ind_temp = np.where(rest > ind_seam-1)
            rest[ind_temp] += 3         
    return image


if __name__ == "__main__":
    """ You may use this area for code that allows you to test your functions.
    This section will not be graded, and is optional. 
    
    Comment out this section when you submit to the autograder to avoid the chance 
    of wasting time and attempts.
    """
    # WRITE YOUR CODE HERE
    
    pass

# a = cv2.imread('beach_back_removal.png')
# b = cv2.imread('comp_beach_back_rem.png')
# print(numerical_comparison(a,b))