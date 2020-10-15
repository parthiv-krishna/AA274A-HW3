#!/usr/bin/env python

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
import numpy as np
from PlotFunctions import *


############################################################
# functions
############################################################

def ExtractLines(RangeData, params):
    '''
    This function implements a split-and-merge line extraction algorithm.

    Inputs:
        RangeData: (x_r, y_r, theta, rho)
            x_r: robot's x position (m).
            y_r: robot's y position (m).
            theta: (1D) np array of angle 'theta' from data (rads).
            rho: (1D) np array of distance 'rho' from data (m).
        params: dictionary of parameters for line extraction.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        segend: np array (N_lines, 4) of line segment endpoints. Each row represents [x1, y1, x2, y2].
        pointIdx: (N_lines,2) segment's first and last point index.
    '''

    # Extract useful variables from RangeData
    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]

    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=np.int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    startIdx = 0
    for endIdx in LineBreak:
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, rho, startIdx, endIdx, params)
        N_lines = r_seg.size

        ### Merge Lines ###
        if (N_lines > 1):
            alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    pointIdx = pointIdx.astype(int) # Fix issue with the entries being floats and not being usable as indices
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    ### Filter Lines ###
    # Find and remove line segments that are too short
    goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
                          (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    pointIdx = pointIdx[goodSegIdx, :]
    alpha = alpha[goodSegIdx]
    r = r[goodSegIdx]
    segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r

    return alpha, r, segend, pointIdx


def SplitLinesRecursive(theta, rho, startIdx, endIdx, params):
    '''
    This function executes a recursive line-slitting algorithm, which
    recursively sub-divides line segments until no further splitting is
    required.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        startIdx: starting index of segment to be split.
        endIdx: ending index of segment to be split.
        params: dictionary of parameters.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        idx: (N_lines,2) segment's first and last point index.

    HINT: Call FitLine() to fit individual line segments.
    HINT: Call FindSplit() to find an index to split at.
    '''
    ########## Code starts here ##########
    alpha, r = FitLine(theta[startIdx:endIdx], rho[startIdx:endIdx])
    if endIdx - startIdx < params["MIN_POINTS_PER_SEGMENT"]:
        return np.array([alpha]), np.array([r]), np.array([startIdx, endIdx])
    
    splitIdx = FindSplit(theta[startIdx:endIdx], rho[startIdx:endIdx], alpha, r, params)
    if splitIdx == -1:
        return np.array([alpha]), np.array([r]), np.array([startIdx, endIdx])

    alpha1, r1, idx1 = SplitLinesRecursive(theta, rho, startIdx, startIdx + splitIdx, params)
    alpha2, r2, idx2 = SplitLinesRecursive(theta, rho, startIdx + splitIdx, endIdx, params)

    alpha = np.hstack((alpha1, alpha2))
    r = np.hstack((r1, r2))
    idx = np.vstack((idx1, idx2))
    ########## Code ends here ##########
    return alpha, r, idx

def FindSplit(theta, rho, alpha, r, params):
    '''
    This function takes in a line segment and outputs the best index at which to
    split the segment, or -1 if no split should be made.

    The best point to split at is the one whose distance from the line is
    the farthest, as long as this maximum distance exceeds
    LINE_POINT_DIST_THRESHOLD and also does not divide the line into segments
    smaller than MIN_POINTS_PER_SEGMENT. Otherwise, no split should be made.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: 'alpha' of input line segment (1 number).
        r: 'r' of input line segment (1 number).
        params: dictionary of parameters.
    Output:
        splitIdx: idx at which to split line (return -1 if it cannot be split).
    '''
    ########## Code starts here ##########
    # absolute value of distance
    distances = np.abs(rho*np.cos(theta-alpha) - r)

    # By setting the edges to zero, we avoid putting splitIdx too close 
    # to the edge and creating too short of a segment
    distances[:params["MIN_POINTS_PER_SEGMENT"]] = 0
    distances[len(distances) - params["MIN_POINTS_PER_SEGMENT"] + 1:] = 0

    splitIdx = np.argmax(distances)
    if distances[splitIdx] < params["LINE_POINT_DIST_THRESHOLD"]:
        splitIdx = -1
    ########## Code ends here ##########
    return splitIdx

def FitLine(theta, rho):
    '''
    This function outputs a least squares best fit line to a segment of range
    data, expressed in polar form (alpha, r).

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
    Outputs:
        alpha: 'alpha' of best fit for range data (1 number) (rads).
        r: 'r' of best fit for range data (1 number) (m).
    '''
    ########## Code starts here ##########
    n = len(theta)

    # rho_squared = rho * rho # rho_squared[i] = rho[i]^2
    # sin = np.sin(theta) # sin[i] = sin(theta[i])
    # cos = np.cos(theta) # cos[i] = cos(theta[i])
    # sin2 = np.sin(2*theta) # sin2[i] = sin(2*theta[i])
    # cos2 = np.cos(2*theta) # cos2[i] = cos(2*theta[i])
    # rho_rho = np.outer(rho, rho) # rho_rho[i,j] = rho[i]*rho[j]
    # # cos_sin = np.outer(np.sin(rho), np.cos(rho)) # cos_sin[i,k] = cos(theta[i])*sin(theta[j])
    # theta_columns = np.outer(theta, np.ones(n)) # n columns, each of which is theta
    # # adding the columns to rows gives all combinations of theta[i] + theta[j]
    # cos_theta_plus_theta = np.cos(theta_columns + np.transpose(theta_columns))

    # numerator = np.dot(rho_squared, sin2) - 2.0/n*np.dot(rho, cos)*np.dot(rho, sin)
    # denominator = np.dot(rho_squared, cos2) - 1.0/n*np.sum(rho_rho*cos_theta_plus_theta)
    
    numerator = sum([(rho[i]**2)*np.sin(2*theta[i]) for i in range(n)]) 
    numerator -= 2.0/n*sum([rho[i]*rho[j]*np.cos(theta[i])*np.sin(theta[j]) for i in range(n) for j in range(n)])
    denominator = sum([(rho[i]**2)*np.cos(2*theta[i]) for i in range(n)]) 
    denominator -= 1.0/n*sum([rho[i]*rho[j]*np.cos(theta[i] + theta[j]) for i in range(n) for j in range(n)])
    alpha = 0.5 * np.arctan2(numerator, denominator) + np.pi/2

    r = 1.0/n*sum([rho[i]*np.cos(theta[i] - alpha) for i in range(n)])
    ########## Code ends here ##########
    return alpha, r

def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):
    '''
    This function merges neighboring segments that are colinear and outputs a
    new set of line segments.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        pointIdx: (N_lines,2) segment's first and last point indices.
        params: dictionary of parameters.
    Outputs:
        alphaOut: output 'alpha' of merged lines (rads).
        rOut: output 'r' of merged lines (m).
        pointIdxOut: output start and end indices of merged line segments.

    HINT: loop through line segments and try to fit a line to data points from
          two adjacent segments. If this line cannot be split, then accept the
          merge. If it can be split, do not merge.
    '''
    ########## Code starts here ##########
    alpha_curr = alpha[0]
    r_curr = r[0]

    startIdx = pointIdx[0, 0]
    prevEndIdx = pointIdx[0, 1]

    alphaOut = np.array([])
    rOut = np.array([])
    pointIdxOut = np.empty((0,2))

    for i in range(1, len(r)): # Skip index 0 since we've already accounted for it above
        endIdx = pointIdx[i, 1]
        alpha_test, r_test = FitLine(theta[startIdx:endIdx], rho[startIdx:endIdx])
        splitIdx = FindSplit(theta[startIdx:endIdx], rho[startIdx:endIdx], alpha_test, r_test, params)
 
        if splitIdx == -1: # Not splittable, so we merge
            alpha_curr = alpha_test
            r_curr = r_test
        else: # Splittable, so we don't merge
            alphaOut = np.append(alphaOut, alpha_curr) # Add current segment to the 
            rOut = np.append(rOut, r_curr)
            pointIdxOut = np.vstack((pointIdxOut, [startIdx, prevEndIdx]))
            alpha_curr = alpha[i]
            r_curr = r[i]
            startIdx = pointIdx[i, 0]

        prevEndIdx = endIdx
        
    # Add last segment    
    alphaOut = np.append(alphaOut, alpha_curr)
    rOut = np.append(rOut, r_curr)
    pointIdxOut = np.vstack((pointIdxOut, [startIdx, endIdx]))

    ########## Code ends here ##########
    return alphaOut, rOut, pointIdxOut


#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = 0.05  # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = 0.02  # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 4  # minimum number of points per line segment
    MAX_P2P_DIST = 1.0  # max distance between two adjent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    filename = 'rangeData_5_5_180.csv'
    # filename = 'rangeData_4_9_360.csv'
    # filename = 'rangeData_7_2_90.csv'

    # Import Range Data
    RangeData = ImportRangeData(filename)

    params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
              'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
              'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
              'MAX_P2P_DIST': MAX_P2P_DIST}

    alpha, r, segend, pointIdx = ExtractLines(RangeData, params)

    ax = PlotScene()
    ax = PlotData(RangeData, ax)
    ax = PlotRays(RangeData, ax)
    ax = PlotLines(segend, ax)

    plt.show(ax)

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
