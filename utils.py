import cv2
import numpy as np
import os
import re
import exceptions

MINIMUM_MATCH_POINTS = 20

def get_matches(img_a_gray, img_b_gray, num_keypoints=1000, threshold=0.8):
    '''Function to get matched keypoints from two images using ORB

    Args:
        img_a_gray (numpy array): of shape (H, W) representing grayscale image A
        img_b_gray (numpy array): of shape (H, W) representing grayscale image B
        num_keypoints (int): number of points to be matched (default=100)
        threshold (float): can be used to filter strong matches only. Lower the value, stronger the requirements and hence fewer matches.
    Returns:
        match_points_a (numpy array): of shape (n, 2) representing x,y pixel coordinates of image A keypoints
        match_points_b (numpy array): of shape (n, 2) representing x,y pixel coordianted of matched keypoints in image B
    '''
    orb = cv2.ORB_create(nfeatures=num_keypoints)
    kp_a, desc_a = orb.detectAndCompute(img_a_gray, None)
    kp_b, desc_b = orb.detectAndCompute(img_b_gray, None)
    
    dis_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_list = dis_matcher.knnMatch(desc_a, desc_b, k=2) # get the two nearest matches for each keypoint in image A

    # for each keypoint feature in image A, compare the distance of the two matched keypoints in image B
    # retain only if distance is less than a threshold 
    good_matches_list = []
    for match_1, match_2 in matches_list:
        if match_1.distance < threshold * match_2.distance:
            good_matches_list.append(match_1)
    
    #filter good matching keypoints 
    good_kp_a = []
    good_kp_b = []

    for match in good_matches_list:
        good_kp_a.append(kp_a[match.queryIdx].pt) # keypoint in image A
        good_kp_b.append(kp_b[match.trainIdx].pt) # matching keypoint in image B
    
    if len(good_kp_a) < MINIMUM_MATCH_POINTS:
        raise exceptions.NotEnoughMatchPointsError(len(good_kp_a), MINIMUM_MATCH_POINTS)
    
    return np.array(good_kp_a), np.array(good_kp_b)


def calculate_homography(points_img_a, points_img_b):
    '''Function to calculate the homography matrix from point corresspondences using Direct Linear Transformation
        The resultant homography transforms points in image B into points in image A
        Homography H = [h1 h2 h3; 
                        h4 h5 h6;
                        h7 h8 h9]
        u, v ---> point in image A
        x, y ---> matched point in image B then,
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
        points_img_a (numpy array): of shape (n, 2) representing pixel coordinate points (u, v) in image A
        points_img_b (numpy array): of shape (n, 2) representing pixel coordinates (x, y) in image B
    
    Returns:
        h_mat: A (3, 3) numpy array of estimated homography
    '''
    # concatenate the two numpy points array to get 4 columns (u, v, x, y)
    points_a_and_b = np.concatenate((points_img_a, points_img_b), axis=1)
    A = []
    # fill the A matrix by looping through each row of points_a_and_b containing u, v, x, y
    # each row in the points_ab would fill two rows in the A matrix
    for u, v, x, y in points_a_and_b:
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = np.array(A)
    _, _, v_t = np.linalg.svd(A)

    # soltion is the last column of v which means the last row of its transpose v_t
    h_mat = v_t[-1, :].reshape(3,3)
    return h_mat


def compute_outliers(h_mat, points_img_a, points_img_b, threshold=3):
    '''Function to compute the error in the Homography matrix using the matching points in
        image A and image B
    
    Args:
        h_mat (numpy array): of shape (3, 3) representing the homography that transforms points in image B to points in image A
        points_img_a (numpy array): of shape (n, 2) representing pixel coordinate points (u, v) in image A
        points_img_b (numpy array): of shape (n, 2) representing pixel coordinates (x, y) in image B
        theshold (int): a number that represents the allowable euclidean distance (in pixels) between the transformed pixel coordinate from
            the image B to the matched pixel coordinate in image A, to be conisdered outliers
    
    Returns:
        error: a scalar float representing the error in the Homography matrix
    '''
    num_points = points_img_a.shape[0]
    outliers_count = 0

    # add a 3rd column of ones to the point numpy representation to make use of matrix multiplication
    ones_col = np.ones((num_points, 1))
    points_img_a = np.concatenate((points_img_a, ones_col), axis=1)
    points_img_b = np.concatenate((points_img_b, ones_col), axis=1)
    points_img_b_hat = np.matmul(h_mat, points_img_b.T).T
    points_img_b_hat = points_img_b_hat / (points_img_b_hat[:,2]).reshape(-1,1)
    # let x, y be coordinate representation of points in image A
    # let x_hat, y_hat be the coordinate representation of transformed points of image B with respect to image A
    x = points_img_a[:, 0]
    y = points_img_a[:, 1]
    x_hat = points_img_b_hat[:, 0]
    y_hat = points_img_b_hat[:, 1]
    euclid_dis = np.sqrt(np.power((x_hat - x), 2) + np.power((y_hat - y), 2)).reshape(-1)
    for dis in euclid_dis:
        if dis > threshold:
            outliers_count += 1
    return outliers_count


def compute_homography_ransac(matches_a, matches_b):
    """Function to estimate the best homography matrix using RANSAC on potentially matching
    points.
    
    Args:
        matches_a (numpy array): of shape (n, 2) representing the coordinates
            of possibly matching points in image A
        matches_b (numpy array): of shape (n, 2) representing the coordinates
            of possibly matching points in image B

    Returns:
        best_h_mat: A numpy array of shape (3, 3) representing the best homography
            matrix that transforms points in image B to points in image A
    """
    num_all_matches =  matches_a.shape[0]
    # RANSAC parameters
    SAMPLE_SIZE = 5 #number of point correspondances for estimation of Homgraphy
    SUCCESS_PROB = 0.995 #required probabilty of finding H with all samples being inliners 
    min_iterations = int(np.log(1.0 - SUCCESS_PROB)/np.log(1 - 0.5**SAMPLE_SIZE))
    
    # Let the initial error be large i.e consider all matched points as outliers
    lowest_outliers_count = num_all_matches
    best_h_mat = None
    best_i = 0 # just to know in which iteration the best h_mat was found

    for i in range(min_iterations):
        rand_ind = np.random.permutation(range(num_all_matches))[:SAMPLE_SIZE]
        h_mat = calculate_homography(matches_a[rand_ind], matches_b[rand_ind])
        outliers_count = compute_outliers(h_mat, matches_a, matches_b)
        if outliers_count < lowest_outliers_count:
            best_h_mat = h_mat
            lowest_outliers_count = outliers_count
            best_i = i
    return best_h_mat

def get_crop_points(h_mat, img_a, img_b, stich_direc):
    """Function to find the pixel corners to crop the stiched image such that the black space 
        in the stiched image is removed.
        The black space could be because either image B is not of the same dimensions as image A
        or image B is skewed after homographic transformation.
        Example: 
                  (Horizontal stiching)
                ____________                     _________________
                |           |                    |                |
                |           |__________          |                |
                |           |         /          |       A        |
                |     A     |   B    /           |________________|
                |           |       /                |          | 
                |           |______/                 |    B     |
                |___________|                        |          |
                                                     |__________|  <-imagine slant bottom edge
        
        This function returns the corner points to obtain the maximum area inside A and B combined and making
        sure the edges are straight (i.e horizontal and veritcal). 

    Args:
        h_mat (numpy array): of shape (3, 3) representing the homography from image B to image A
        img_a (numpy array): of shape (h, w, c) representing image A
        img_b (numpy array): of shape (h, w, c) representing image B
        stich_direc (int): 0 when stiching vertically and 1 when stiching horizontally

    Returns:
        x_start (int): the x pixel-cordinate to start the crop on the stiched image
        y_start (int): the x pixel-cordinate to start the crop on the stiched image
        x_end (int): the x pixel-cordinate to end the crop on the stiched image
        y_end (int): the y pixel-cordinate to end the crop on the stiched image          
    """
    img_a_h, img_a_w, _ = img_a.shape
    img_b_h, img_b_w, _ = img_b.shape

    # find out where the four corners of the right image are projected with h_mat on the left image pixel coordinate plane
    """
    Arrange the 4 corners (with a column of ones) of image B into a matrix so that matrix multiplication can be used to transform all at once
    corners = [top_left_x, top_left_y, 1;
               top_right_x, top_right_y, 1;
               bottom_right_x, bottom_right_y, 1;
               bottom_left_x, bottom_left_y, 1]
    """
    orig_corners_img_b = np.array([[0, 0, 1],
                                       [img_b_w - 1, 0, 1],
                                       [img_b_w - 1, img_b_h - 1, 1],
                                       [0, img_b_h - 1, 1]])
    
    transfm_corners_img_b = np.matmul(h_mat, orig_corners_img_b.T).T
    transfm_corners_img_b = transfm_corners_img_b / transfm_corners_img_b[:,2].reshape(-1,1)

    #extract the four corners of transformed corners of image B
    top_lft_x_hat = transfm_corners_img_b[0,0]
    top_lft_y_hat = transfm_corners_img_b[0,1]
    top_rht_x_hat = transfm_corners_img_b[1,0]
    top_rht_y_hat = transfm_corners_img_b[1,1]
    btm_rht_x_hat = transfm_corners_img_b[2,0]
    btm_rht_y_hat = transfm_corners_img_b[2,1]
    btm_lft_x_hat = transfm_corners_img_b[3,0]
    btm_lft_y_hat = transfm_corners_img_b[3,1]

    # initialize the crop points
    x_start = None
    x_end = None
    y_start = None
    y_end = None

    if stich_direc == 1: # 1 is horizontal
        x_start = 0 # since left image (image A) will be original and will have vertical edge
        if (top_lft_y_hat > 0) and (top_lft_y_hat > top_rht_y_hat):
            y_start = top_lft_y_hat
        elif (top_rht_y_hat > 0) and (top_rht_y_hat > top_lft_y_hat):
            y_start = top_rht_y_hat
        else:
            y_start = 0
        
        if (btm_lft_y_hat < img_a_h - 1) and (btm_lft_y_hat < btm_rht_y_hat):
            y_end = btm_lft_y_hat
        elif (btm_rht_y_hat < img_a_h - 1) and (btm_rht_y_hat < btm_lft_y_hat):
            y_end = btm_rht_y_hat
        else:
            y_end = img_a_h - 1

        if (top_rht_x_hat < btm_rht_x_hat):
            x_end = top_rht_x_hat
        else:
            x_end = btm_rht_x_hat
    else: # when stiching images in the vertical direction
        y_start = 0 # since top image (image A) will be original and will have horizontal edge
        if (top_lft_x_hat > 0) and (top_lft_x_hat > btm_lft_x_hat):
            x_start = top_lft_x_hat
        elif (btm_lft_x_hat > 0) and (btm_lft_x_hat > top_lft_x_hat):
            x_start = btm_lft_x_hat
        else:
            x_start = 0
        
        if (top_rht_x_hat < img_a_w - 1) and (top_rht_x_hat < btm_rht_x_hat):
            x_end = top_rht_x_hat
        elif (btm_rht_x_hat < img_a_w - 1) and (btm_rht_x_hat < top_rht_x_hat):
            x_end = btm_rht_x_hat
        else:
            x_end = 0

        if (btm_lft_y_hat < btm_rht_y_hat):
            y_end = btm_lft_y_hat
        else:
            y_end = btm_rht_y_hat
    return int(x_start), int(y_start), int(x_end), int(y_end)

def stich_images(img_a, img_b, stich_direc):
    """Function to stich image B to image A in the mentioned direction

    Args:
        img_a (numpy array): of shape (H, W, C) with opencv representation of image A (i.e C: B,G,R)
        img_b (numpy array): of shape (H, W, C) with opencv representation of image B (i.e C: B,G,R)
        stich_direc (int): 0 for vertical and 1 for horizontal stiching

    Returns:
        stiched_image (numpy array): stiched image with maximum content of image A and image B after cropping
            to remove the black space 
    """
    img_a_gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    matches_a, matches_b = get_matches(img_a_gray, img_b_gray, num_keypoints=1000, threshold=0.8)
    h_mat = compute_homography_ransac(matches_a, matches_b)
    if stich_direc == 0:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1], img_a.shape[0] + img_b.shape[0]))
        canvas[0:img_a.shape[0], :, :] = img_a[:, :, :]
        x_start, y_start, x_end, y_end = get_crop_points(h_mat, img_a, img_b, 0)
    else:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1] + img_b.shape[1], img_a.shape[0]))
        canvas[:, 0:img_a.shape[1], :] = img_a[:, :, :]
        x_start, y_start, x_end, y_end = get_crop_points(h_mat, img_a, img_b, 1)
    
    stiched_img = canvas[y_start:y_end,x_start:x_end,:]
    return stiched_img

def check_imgfile_validity(folder, filenames):
    """Function to check if the files in the given path are valid image files.
    
    Args:
        folder (str): path containing the image files
        filenames (list): a list of image filenames

    Returns:
        valid_files (bool): True if all the files are valid image files else False
        msg (str): Message that has to be displayed as error
    """
    for file in filenames:
        full_file_path = os.path.join(folder, file)
        regex = "([^\\s]+(\\.(?i)(jpe?g|png))$)"
        p = re.compile(regex)

        if not os.path.isfile(full_file_path):
            return False, "File not found: " + full_file_path
        if not (re.search(p, file)):
            return False, "Invalid image file: " + file
    return True, None