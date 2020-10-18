#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt


def half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    m, n, c = image.shape
    downscaled = np.zeros((m//2, n//2, c))
    for y in range(m//2):
        for x in range(n//2):
            downscaled[y, x, :] = image[y*2, x*2, :]
    return downscaled
    ########## Code ends here ##########


def blur_half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    blur = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0.7)
    return half_downscale(blur)
    ########## Code ends here ##########


def two_upscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        upscaled_image: A 2x-upscaled version of image.
    """
    ########## Code starts here ##########
    return np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    ########## Code ends here ##########


def bilinterp_upscale(image, scale):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
        scale: How much larger to make the image

    Returns
        upscaled_image: A scale-times upscaled version of image.
    """
    m, n, c = image.shape

    f = (1./scale) * np.convolve(np.ones((scale, )), np.ones((scale, )))
    f = np.expand_dims(f, axis=0) # Making it (1, (2*scale)-1)-shaped
    filt = f.T * f

    ########## Code starts here ##########
    I_scaled = np.zeros((m*scale, n*scale, c))
    for y in range(m):
        for x in range(n):
            I_scaled[y*scale, x*scale, :] = image[y, x, :]
    upscaled = cv2.filter2D(I_scaled, -1, filt)
    return upscaled
    ########## Code ends here ##########


def main():
    # OpenCV actually uses a BGR color channel layout,
    # Matplotlib uses an RGB color channel layout, so we're flipping the 
    # channels here so that plotting matches what we expect for colors.
    test_card = cv2.imread('test_card.png')[..., ::-1].astype(float)
    favicon = cv2.imread('favicon-16x16.png')[..., ::-1].astype(float)
    test_card /= test_card.max()
    favicon /= favicon.max()

    # Note that if you call matplotlib's imshow function to visualize images,
    # be sure to pass in interpolation='none' so that the image you see
    # matches exactly what's in the data array you pass in.
    
    ########## Code starts here ##########
    # Simple downscale
    simple_downscale = test_card
    for i in range(3):
        simple_downscale = half_downscale(simple_downscale)
    plt.imshow(simple_downscale, interpolation='none')
    plt.savefig('out_simple_downscale.png')

    # Blurred downscale
    blur_downscale = test_card
    for i in range(3):
        blur_downscale = blur_half_downscale(blur_downscale)
    plt.imshow(blur_downscale, interpolation='none')
    plt.savefig('out_blur_downscale.png')

    # Simple upscale
    simple_upscale = favicon
    for i in range(3):
        simple_upscale = two_upscale(simple_upscale)
    plt.imshow(simple_upscale, interpolation='none')
    plt.savefig('out_simple_upscale.png')

    # Bilinear interpolation upscale
    bilin_upscale = bilinterp_upscale(favicon, 8)
    plt.imshow(bilin_upscale, interpolation='none')
    plt.savefig('out_bilin_upscale.png')
    ########## Code ends here ##########


if __name__ == '__main__':
    main()
