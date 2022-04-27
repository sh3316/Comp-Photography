
import numpy as np
import scipy as sp
import cv2
import scipy.signal                     # option for a 2D convolution library
from matplotlib import pyplot as plt    # optional if you find a use for it

''' Project 2 - Object Removal

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

Reference:
----------
"Object Removal by Exemplar_Based Inpainting" by Creminisi, Perez and Toyama, 2003
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/criminisi_cvpr2003.pdf

FORBIDDEN:
    **Usage of these may lead to zero scores for the project and review
    for an honor code violation.**
    1. YOU MAY NOT USE any library function that essentially completes
    object removal or region inpainting for you. This includes:
        OpenCV function cv2.inpaint,
        any similar functions that may be in the class environment or other libraries.
    2. The use of other algorithms than the one presented in the paper.
    
    If you have questions on this, ask on Ed.
    
    
GENERAL RULES:
    1. ALL CODE USED IN THIS ASSIGNMENT to generate images must be included in this file.
    2. DO NOT CHANGE the format of this file. You may NOT change existing function
    signatures, including the given named parameters with defaults.
    3. YOU MAY ADD FUNCTIONS to this file, however it is your responsibility
    to ensure that the autograder accepts your submission.
    4. DO NOT IMPORT any additional libraries other than the ones imported above.
    You should be able to complete the assignment with the given libraries.
    5. DO NOT INCLUDE code that prints, saves, shows, displays, or writes the
    images, or your results. If you have code in the functions that
    does any of these operations, comment it out before autograder runs.
    6. YOU ARE RESPONSIBLE for ensuring that your code executes properly.
    This file has only been tested in the course environment. Any changes you make
    outside the areas annotated for student code must not impact the autograder
    system or your performance.
    
FUNCTIONS:
    returnYourName
    objectRemoval
'''


def returnYourName():
    """ This function returns your name as shown on your Gradescope Account.
    """
    # WRITE YOUR CODE HERE.
    return 'SeungHui Huh'
    raise NotImplementedError


def calc_patch(point, workImage, halfPatchWidth):
    centre_x, centre_y = point
    h, w = workImage.shape[:2]
    min_x = max(centre_x - halfPatchWidth, 0)
    max_x = min(centre_x + halfPatchWidth, w-1)
    min_y = max(centre_y - halfPatchWidth, 0)
    max_y = min(centre_y+ halfPatchWidth, h-1)
    upper_left = (min_x, min_y)
    lower_right = (max_x, max_y)
    return upper_left, lower_right

def objectRemoval(image, mask, setnum=0, window=(9,9)):
    if setnum == 1:
        window = (15,15)
    input_image = np.copy(image)
    mask = np.copy(mask)
    if mask.shape==3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    updated_mask = np.copy(mask)
    work_image = np.copy(input_image)
    result = np.ndarray(shape=input_image.shape, dtype=input_image.dtype)
    half_patch_width = window[0]//2


    thresh, confidence = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    thresh, confidence = cv2.threshold(confidence, 2, 1, cv2.THRESH_BINARY_INV)
    confidence = np.float32(confidence)

    source_region = np.uint8(np.copy(confidence))
    original_source_region = np.copy(source_region)



    thresh, target_region = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
    thresh, target_region = cv2.threshold(target_region, 2, 1, cv2.THRESH_BINARY)
    target_region = np.uint8(target_region)
    data = np.ndarray(shape=input_image.shape[:2], dtype=np.float32)

    laplacian_kernel = np.float32([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    normal_kernel_x = np.float32([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    normal_kernel_y = cv2.transpose(normal_kernel_x)

    srcGray = cv2.cvtColor(work_image, cv2.COLOR_RGB2GRAY)
    y_gradient = cv2.Scharr(srcGray, cv2.CV_32F, 0, 1)
    y_gradient = cv2.convertScaleAbs(y_gradient)
    y_gradient = np.float32(y_gradient)
    x_gradient = cv2.Scharr(srcGray, cv2.CV_32F, 1, 0)
    x_gradient = cv2.convertScaleAbs(x_gradient)
    x_gradient = np.float32(x_gradient)


    h, w = source_region.shape
    for y in range(h):
        for x in range(w):
            if source_region[y, x] == 0:
                x_gradient[y, x] = 0
                y_gradient[y, x] = 0

    x_gradient /= 255
    y_gradient /= 255
    iterate = True
    while iterate:
        boundry_mat = cv2.filter2D(target_region, cv2.CV_32F, laplacian_kernel)
        boundry_mat = boundry_mat.astype(np.uint8)
        source_gradient_x = cv2.filter2D(source_region, cv2.CV_32F, normal_kernel_x)
        source_gradient_y = cv2.filter2D(source_region, cv2.CV_32F, normal_kernel_y)

        fill_front = []
        normals = []
        h, w = boundry_mat.shape[:2]
        for y in range(h):
            for x in range(w):
                if boundry_mat[y, x] > 0:
                    fill_front.append((x, y))
                    dx = source_gradient_x[y, x]
                    dy = source_gradient_y[y, x]
                    normal_x, normal_y  = dy, -dx
                    temp_F = np.sqrt(pow(normal_x, 2) + pow(normal_y, 2))
                    if not temp_F == 0: 
                        normal_x /= temp_F
                        normal_y /= temp_F
                    normals.append((normal_x, normal_y))

        for p in fill_front:
            x_point, y_point = p
            (ax, ay), (bx, by) = calc_patch(p, work_image, half_patch_width)
            total = 0
            for y in range(ay, by+1):
                for x in range(ax, bx+1):
                    if target_region[y, x] == 0:
                        total += confidence[y, x]
            confidence[y_point, x_point] = total / ((bx-ax+1)*(by-ay+1))

        for i in range(len(fill_front)):
            x, y = fill_front[i]
            current_normal_x, current_normal_y = normals[i]
            data[y, x] = np.abs(x_gradient[y, x] * current_normal_x + y_gradient[y, x] * current_normal_y) + 0.001

        target_index = 0
        priority, max_priority = 0, 0
        for i in range(len(fill_front)):
            x, y = fill_front[i]
            priority = confidence[y, x] * data[y, x]

            if max_priority < priority:
                max_priority = priority
                target_index = i

        min_error , best_patch_variance = 10**10 , 10**10
        current_point = fill_front[target_index]
        (ax, ay), (bx, by) = calc_patch(current_point, work_image, half_patch_width)
        pH = by - ay + 1
        pW = bx - ax + 1
        h, w = work_image.shape[:2]
        work_image = work_image.tolist()

        patchHeight = patchWidth = 0
        if pH != patchHeight or pW != patchWidth:
            patchHeight = pH
            patchWidth = pW
            area = pH * pW
            sum_kernel = np.ones((pH, pW), dtype=np.uint8)

            convolved_mat = cv2.filter2D(original_source_region, cv2.CV_8U, sum_kernel, anchor=(0, 0))
            source_patch_UL_list = []
            
            

            for y in range(h - pH):
                for x in range(w - pW):
                    if convolved_mat[y, x] == area:
                        source_patch_UL_list.append((y, x))
        stack = 0
        S_target = []
        T_target = []
        for y in range(pH):
            for x in range(pW):
                if source_region[ay+y, ax+x] == 1:
                    stack += 1
                    S_target.append((y, x))
                else:
                    T_target.append((y, x))
        for (y, x) in source_patch_UL_list:
            patch_error = 0
            r_avg, g_avg, b_avg = 0,0,0
            for (i, j) in S_target:
                source_pixel = work_image[y+i][x+j]
                target_pixel = work_image[ay+i][ax+j]
                for channel in range(3):
                    difference = float(source_pixel[channel]) - float(target_pixel[channel])
                    patch_error += pow(difference, 2)
                r_avg += source_pixel[0]
                g_avg += source_pixel[1]
                b_avg += source_pixel[2]
            stack = float(stack)
            patch_error = patch_error/stack
            b_avg = b_avg/stack
            g_avg = g_avg/stack
            r_avg = r_avg/stack
            if min_error >= patch_error:
                patch_variance = 0
                for (i, j) in T_target:
                    source_pixel = work_image[y+i][x+j]
                    difference = source_pixel[0] - r_avg
                    patch_variance += pow(difference, 2)
                    difference = source_pixel[1] - g_avg
                    patch_variance += pow(difference, 2)
                    difference = source_pixel[2] - b_avg
                    patch_variance += pow(difference, 2)

                if patch_error < min_error or patch_variance < best_patch_variance:
                    best_patch_variance = patch_variance
                    min_error = patch_error
                    best_match_upper_left = (x, y)
        target_point = fill_front[target_index]
        tx, ty = target_point
        work_image = np.array(work_image)
        (ax, ay), (bx, by) = calc_patch(target_point, work_image, half_patch_width)
        bul_x, bul_y = best_match_upper_left
        pH = by-ay+1
        pW = bx-ax+1

        for (i, j) in T_target:
            work_image[ay+i, ax+j] = work_image[bul_y+i, bul_x+j]
            x_gradient[ay+i, ax+j] = x_gradient[bul_y+i, bul_x+j]
            y_gradient[ay+i, ax+j] = y_gradient[bul_y+i, bul_x+j]
            confidence[ay+i, ax+j] = confidence[ty, tx]
            source_region[ay+i, ax+j] = 1
            target_region[ay+i, ax+j] = 0
            updated_mask[ay+i, ax+j] = 0
        
        height, width = source_region.shape
        iterate_counter = 0
        for y in range(height):
            if iterate_counter>0:
                break
            for x in range(width):
                if source_region[y, x] == 0:
                    iterate_counter+=1
                    break
                else:
                    iterate = False
        if iterate_counter > 0:
            iterate = True



    result = np.copy(work_image)
    result = result.astype(np.uint8)
    return result






