# Based on tutorial:
# https://docs.opencv.org/3.4.1/dc/dc3/tutorial_py_matcher.html
# TODO
# - investigate performance improvement using FLANN based matcher
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import numpy as np
import os

from skimage.measure import compare_ssim as ssim

#plt.ioff()

# collect images to test with, then integrate into TF to use dataset at each epoch
# --> need to check how long it takes..

RESULTS_DIR = "/Users/jack/Dropbox/University/Courses/PROJ/models/cgan_256_mn40s_bs_4_lw1_1_lw2_100_lw3_0_adam_001_9_bn_skip/results/images/"

def get_images():
    steps = glob.glob(RESULTS_DIR + "/*")
    orb_scores = []
    sift_scores = []
    ssim_scores = []
    for idx, step_dir in enumerate(steps):
        orb_score = 0
        sift_score = 0
        ssim_score = 0
        for t_idx, o_idx in [("1_0","1_2"),("2_3","2_5"),("3_6","3_8"),("4_9","4_11")]:
            t_path = os.path.join(step_dir,f"{t_idx}.png")
            o_path = os.path.join(step_dir,f"{o_idx}.png")

            im1 = cv.imread(t_path, 0)
            im2 = cv.imread(o_path, 0)
            ssim_score += ssim(im1, im2, data_range=im2.max() - im2.min())
            orb_score += orb_match(im1, im2)
            sift_score += sift_match(im1, im2)
        orb_scores.append(orb_score)
        sift_scores.append(sift_score)
        ssim_scores.append(ssim_score)
        print(f"{idx}: {orb_score} | {sift_score} | {ssim_score}")

    #plt.plot(orb_scores)
    #plt.plot(sift_scores)
    plt.plot(ssim_scores)
    plt.show()



def sift_match(im1, im2, draw=False):
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # create BFMatcher object
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good = []
    for match in matches:
        try:
            m, n = match
        except ValueError:
            # No match
            continue

        if m.distance < 0.75*n.distance:
            good.append([m])

    if draw:
        # cv.drawMatchesKnn expects list of lists as matches.
        im3 = cv.drawMatchesKnn(im1,kp1,im2,kp2,good,None,flags=2)
        plt.imshow(im3)
        plt.show()

    return sum(m[0].distance for m in good)

def orb_match(im1, im2, draw=False):
    orb = cv.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    if draw:
        im3 = cv.drawMatches(im1,kp1,im2,kp2,matches[:10],None,flags=2)
        plt.imshow(im3)
        plt.show()

    # Take top X matches by their distance and sum to get a total
    return sum(m.distance for m in matches)

if __name__ == "__main__":
    get_images()
