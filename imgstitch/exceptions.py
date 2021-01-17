class InsufficientImagesError(Exception):
    """Exception class that can be called when there is insufficient number of images.
    
    Args:
        num_images (int): number of images (this is just used to display in the message)
    """
    def __init__(self, num_images):
        msg = "Expected 2 or more images but got only " +  str(num_images)
        super(InsufficientImagesError, self).__init__(msg)


class InvalidImageFilesError(Exception):
    """Exception class that can be called when files ar invalid image files or they do not exist.
    
    Args:
        msg (str): Error description
    """
    def __init__(self, msg):
        super(InvalidImageFilesError, self).__init__(msg)


class NotEnoughMatchPointsError(Exception):
    """Exception class that can be called when there are not enough matches points between images
        as defined by the mimimum
    
    Args:
        num_match_points (int): number of matches found
        min_match_points_req (int): minimum number of match points between images required 
    """
    def __init__(self, num_match_points, min_match_points_req):
        msg = "There are not enough match points between images in the input images. Required atleast " + \
               str(min_match_points_req) + " matches but could find only " + str(num_match_points) + " matches!"
        super(NotEnoughMatchPointsError, self).__init__(msg)


class MatchesNotConfident(Exception):
    """Exception class that can be called when the outliers matches count to all matches count ratio is
        above a minimum threshold to calculate the homography matrix confidently.
    
    Args:
        confidence (int): percentage indicating the confidence of match points 
    """
    def __init__(self, confidence):
        msg = "The confidence in the matches is less than the defined threshold and hence the stitching operation \
        cannot be performed. Perhaps the input images have very less overlapping content to detect good match points!"
        super(MatchesNotConfident, self).__init__(msg + " Confidence: " + str(confidence))