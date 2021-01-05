import cv2
import numpy as np

def get_matches(img_left_gray, img_right_gray, num_keypoints=1000, threshold=0.8):
    '''Function to get matched keypoints from two images using ORB

    Args:
        img_left_gray (numpy array): of shape (H, W, C) with opencv representation of left image (i.e C: B,G,R)
        img_right_gray (numpy array): of shape (H, W, C) with opencv representation of right image (i.e C: B,G,R)
        num_keypoints (int): number of points to be matched (default=100)
        threshold (float): can be used to filter strong matches only. Lower the value stronger the requirements and hence fewer matches.
    Returns:
        match_points_a (numpy array): of shape (n, 2) representing x,y pixel coordinates of left image keypoints
        match_points_b (numpy array): of shape (n, 2) representing x,y pixel coordianted of matched keypoints in right image
    '''
    orb = cv2.ORB_create(nfeatures=num_keypoints)
    kp_l, desc_l = orb.detectAndCompute(img_left_gray, None)
    kp_r, desc_r = orb.detectAndCompute(img_right_gray, None)
    
    dis_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_list = dis_matcher.knnMatch(desc_l, desc_r, k=2) # get the two nearest matches for each keypoint in left image

    # for each keypoint feature in left image, compare the distance of the two matched keypoints in the right image
    # retain only if distance is less than a threshold 
    good_matches_list = []
    for match_1, match_2 in matches_list:
        if match_1.distance < threshold * match_2.distance:
            good_matches_list.append(match_1)
    
    #filter good matching keypoints 
    good_kp_l = []
    good_kp_r = []

    for match in good_matches_list:
        good_kp_l.append(kp_l[match.queryIdx].pt) # keypoint on the left image
        good_kp_r.append(kp_r[match.trainIdx].pt) # matching keypoint on the right image
    return np.array(good_kp_l), np.array(good_kp_r)