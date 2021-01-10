import cv2
import numpy as np

def get_matches(img_left_gray, img_right_gray, num_keypoints=1000, threshold=0.8):
    '''Function to get matched keypoints from two images using ORB

    Args:
        img_left_gray (numpy array): of shape (H, W, C) with opencv representation of left image (i.e C: B,G,R)
        img_right_gray (numpy array): of shape (H, W, C) with opencv representation of right image (i.e C: B,G,R)
        num_keypoints (int): number of points to be matched (default=100)
        threshold (float): can be used to filter strong matches only. Lower the value, stronger the requirements and hence fewer matches.
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


def calculate_homography(points_img_l, points_img_r):
    '''Function to calculate the homography matrix from point corresspondences using Direct Linear Transformation
        Homography H = [h1 h2 h3; 
                        h4 h5 h6;
                        h7 h8 h9]
        u, v ---> point in the left image
        x, y ---> matched point in the right image then,
        with n point correspondences the DLT equation is:
            A.h = 0
        where A = [-x1 -y1 -1 0 0 0 u1*x1 u1*y1 u1;
                   0 0 0 -x1 -y1 -1 v1*x1 v1*y1 v1;
                   ...............................;
                   ...............................;
                   -xn -yn -1 0 0 0 un*xn un*yn un;
                   0 0 0 -xn -yn -1 vn*xn vn*yn vn]
        This equation is then solved using SVD
        (At least 4 point correspondences are required to determine 8 unkwown parameters of homography matrix)
    Args:
        points_img_l (numpy array): of shape (n, 2) representing pixel coordinate points (u, v) in the left image
        points_img_r (numpy array): of shape (n, 2) representing pixel coordinates (x, y) in the right image
    
    Returns:
        h_mat: A (3, 3) numpy array of estimated homography
    '''
    # concatenate the two numpy points array to get 4 columns (u, v, x, y)
    points_lr = np.concatenate((points_img_l, points_img_r), axis=1)
    A = []
    # fill the A matrix by looping through each row of points_ab containing u, v, x, y
    # each row in the points_ab would fill two rows in the A matrix
    for u, v, x, y in points_lr:
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = np.array(A)
    _, _, v_t = np.linalg.svd(A)

    # soltion is the last column of v which means the last row of its transpose v_t
    h_mat = v_t[-1, :].reshape(3,3)
    return h_mat


def compute_outliers(h_mat, points_img_l, points_img_r, threshold=3):
    '''Function to compute the error in the Homography matrix using the matching points in
        image A and image B
    
    Args:
        h_mat (numpy array): of shape (3, 3) representing the homography that transforms points in right image to the left image
        points_img_l (numpy array): of shape (n, 2) representing pixel coordinate points (u, v) in the left image
        points_img_r (numpy array): of shape (n, 2) representing pixel coordinates (x, y) in the right image
        theshold (int): a number that represents the allowable euclidean distance (in pixels) between the transformed pixel coordinate from
            the right image to the supposed to be pixel coordinate in the left image, to be conisdered outliers
    
    Returns:
        error: a scalar float representing the error in the Homography matrix
    '''
    num_points = points_img_l.shape[0]
    outliers_count = 0

    # add a 3rd column of ones to the point numpy representation to make use of matrix multiplication
    ones_col = np.ones((num_points,1))
    points_img_l = np.concatenate((points_img_l, ones_col), axis=1)
    points_img_r = np.concatenate((points_img_r, ones_col), axis=1)
    points_img_r_hat = np.matmul(h_mat, points_img_r.T).T
    points_img_r_hat = points_img_r_hat / (points_img_r_hat[:,2]).reshape(-1,1)
    # let x, y be coordinate representation of points in the left image
    # let x_hat, y_hat be the coordinate representation of transformed points of the right image with resect to left image
    x = points_img_l[:, 0]
    y = points_img_l[:, 1]
    x_hat = points_img_r_hat[:, 0]
    y_hat = points_img_r_hat[:, 1]
    euclid_dis = np.sqrt(np.power((x_hat - x), 2) + np.power((y_hat - y), 2)).reshape(-1)
    for dis in euclid_dis:
        if dis > threshold:
            outliers_count += 1
    return outliers_count
    