#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt


# From problem 4
def template_match_helper(template, image, threshold=0.999):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    ########## Code starts here ##########
    h, w = template.shape[0:2] # bounding box height and width is the size of the template
    result = cv2.matchTemplate(image, template, method=cv2.TM_CCORR_NORMED)
    toplefts = np.where(result >= threshold) # only consider "good enough" matches
    
    matches = []
    for [y, x] in zip(*toplefts): 
         matches.append(tuple([y, x, h, w]))
    return matches
    ########## Code ends here ##########

def template_match(template, image,
                   num_upscales=2, num_downscales=3,
                   detection_threshold=0.93):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        num_upscales: How many times to 2x-upscale image with Gaussian blur before template matching over it.
        num_downscales: How many times to 0.5x-downscale image with Gaussian blur before template matching over it.
        detection_threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    ########## Code starts here ##########
    # No scaling
    matches = template_match_helper(template, image, threshold=detection_threshold)

    # Upscales
    upscaled = image
    scale = 1
    for i in range(num_upscales):
        upscaled = cv2.pyrUp(upscaled)
        scale *= 2
        upscaled_matches = template_match_helper(template, upscaled, threshold=detection_threshold)
        for upscaled_match in upscaled_matches:
            matches.append(tuple(val//scale for val in upscaled_match)) # Divide to downscale bounding box

    # Downscales
    downscaled = image
    scale = 1
    for i in range(num_upscales):
        downscaled = cv2.pyrDown(upscaled)
        scale *= 2
        downscaled_matches = template_match_helper(template, downscaled, threshold=detection_threshold)
        for downscaled_match in downscaled_matches:
            matches.append(tuple(val*scale for val in downscaled_match)) # Multiply to upscale bounding box

    return matches
    ########## Code ends here ##########


def create_and_save_detection_image(image, matches, filename="image_detections.png"):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.

    Returns
        None, this function will save the detection image in the current folder.
    """
    det_img = image.copy()
    for (y, x, bbox_h, bbox_w) in matches:
        cv2.rectangle(det_img, (x, y), (x + bbox_w, y + bbox_h), (255, 0, 0), 2)

    cv2.imwrite(filename, det_img)


def main():
    template = cv2.imread('messi_face.jpg')
    image = cv2.imread('messipyr.jpg')

    matches = template_match(template, image)
    create_and_save_detection_image(image, matches)

    template = cv2.imread('stop_signs/stop_template.jpg').astype(np.float32)
    for i in range(1, 6):
        image = cv2.imread('stop_signs/stop%d.jpg' % i).astype(np.float32)
        matches = template_match(template, image, detection_threshold=0.87)
        create_and_save_detection_image(image, matches, 'stop_signs/stop%d_detection.png' % i)


if __name__ == '__main__':
    main()
