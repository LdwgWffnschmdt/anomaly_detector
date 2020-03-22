import cv2
import numpy as np

from common import logger

class ImageLocationUtility(object):
    def __init__(self):
        self.P = None
        self.P_inv = None
        self.w = None
        self.h = None
        
    def get_image_transformation_matrix(self, w, h):
        """Calculate the matrix that will transform an input of
        given width and height to relative real world coordinates

        Args:
            w (int): Width of the image
            h (int): Height of the image

        Returns:
            A Matrix that will convert image coordinates to
            relative real world coordinates
        """
        # No need to recalculate
        if self.P is not None and self.w == w and self.h == h:
            return self.P
        
        self.w = w
        self.h = h

        # TODO: This should probably be a Matrix defined by the camera parameters
        # (P=K*Rt) and the camera position and orientation, not this reverse
        # engineered hack. I was unfortunately not able to do properly so this is it.
        
        # The transformation matrix is defined by the for corners of
        # the image and their real world coordinates relative to the camera
        src = np.float32([[ 0, 0], [w, 0], [ 0,   h], [w,   h]])
        dst = np.float32([[-3, 6], [3, 6], [-0.9, 1], [0.9, 1]]) # Measured by hand
        self.P = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        return self.P

    def get_inverse_image_transformation_matrix(self, w, h):
        """Calculate the inverse matrix that will transform an input of
        given relative real world coordinates to image points

        Args:
            w (int): Width of the image
            h (int): Height of the image

        Returns:
            A Matrix that will convert relative real world coordinates
            to image coordinates
        """
        # No need to recalculate
        if self.P_inv is not None and self.w == w and self.h == h:
            return self.P_inv
        
        self.w = w
        self.h = h

        self.P_inv = np.linalg.pinv(self.get_image_transformation_matrix(w, h))
        return self.P_inv

    def span_grid(self, w, h, step=1, offset_x=0, offset_y=0):
        """Creates a numpy array of shape (w / step, h / step, 2)

        Args:
            w (int): Width of the grid
            h (int): Height of the grid
            step (float): Step size
            offset_x (float): Offset in x direction
            offset_y (float): Offset in y direction
        
        Returns:
            Numpy array of shape (w / step, h / step, 2)
        """
        x = np.arange(offset_x, w + offset_x, step)
        y = np.arange(offset_y, h + offset_y, step)
        return np.stack(np.meshgrid(x, y), axis=2)

    def image_to_relative(self, image_coordinate, image_width=None, image_height=None):
        """Transform an image coordinate to a location relative to the camera

        Args:
            image_coordinate (array): Array of shape (w, h, 2) or (2)
                                      containing the image coordinate(s)
            image_width (int): Needs to be specified when input is a single coordinate
            image_height (int): Needs to be specified when input is a single coordinate

        Returns:
            Array of shape (w, h, 2) or (2) containing the relative location(s)
        """
        
        if len(image_coordinate.shape) == 3:
            image_width = image_coordinate.shape[0]
            image_height = image_coordinate.shape[1]

        # Get transformation matrix (3x3)
        P = self.get_image_transformation_matrix(image_width, image_height)

        def _to_relative(p):
            p = np.hstack([p, 1])           # Add one dimension so it matches P (3x3)
            p_trans = P.dot(p)
            p_trans = p_trans / p_trans[2]  # Normalize by third dimension
            return p_trans[:-1]             # Return only the first two dimensions

        if len(image_coordinate.shape) == 3:
            return np.apply_along_axis(_to_relative, 2, image_coordinate)
        elif image_coordinate.shape == (2,):
            return _to_relative(image_coordinate)
        else:
            raise ValueError("Input has to be a an array of shape (w, h, 2) or (2,)")

    def relative_to_image(self, relative_location, image_width, image_height):
        """Transform a location relative to the camera to an image coordinate

        Args:
            relative_location (array): Array of shape (w, h, 2) or (2)
                                       containing the location(s)
                                       relative to the camera
            image_width (int): Image width
            image_height (int): Image height

        Returns:
            Array of shape (w, h, 2) or (2) containing the image coordinate(s)
        """
        
        # Get inverse transformation matrix (3x3)
        P_inv = self.get_inverse_image_transformation_matrix(image_width, image_height)

        def _to_image(p):
            p = np.hstack([p, 1])           # Add one dimension so it matches P (3x3)
            p_trans = P_inv.dot(p)
            p_trans = p_trans / p_trans[2]  # Normalize by third dimension
            return p_trans[:-1]             # Return only the first two dimensions

        if len(relative_location.shape) == 3:
            return np.apply_along_axis(_to_image, 2, relative_location)
        elif relative_location.shape == (2,):
            return _to_image(relative_location)
        else:
            raise ValueError("Input has to be a an array of shape (w, h, 2) or (2,)")

    def relative_to_absolute(self, relative_location, camera_translation, camera_rotation):
        """Transform a relative location to an absolute location

        Args:
            relative_location (array): Array of shape (w, h, 2) or (2,)
                                       containing the location(s)
                                       relative to the camera
            camera_translation (array): [x, y] array with the camera location
            camera_rotation (float): Camera rotation around z-axis

        Returns:
            Array of shape (w, h, 2) or (2,) containing the absolute location(s)
        """

        # Construct an inverse 2D rotation matrix
        _s = np.sin(camera_rotation - np.pi / 2.)
        _c = np.cos(camera_rotation - np.pi / 2.)

        R = np.array([[_c, -_s],
                      [_s,  _c]])

        def _to_absolute(p):
            return camera_translation + R.dot(p)

        if len(relative_location.shape) == 3:
            return np.apply_along_axis(_to_absolute, 2, relative_location)
        elif relative_location.shape == (2,):
            return _to_absolute(relative_location)
        else:
            raise ValueError("Input has to be a an array of shape (w, h, 2) or (2,)")

    def absolute_to_relative(self, absolute_location, camera_translation, camera_rotation):
        """Transform an absolute location to a relative location

        Args:
            absolute_location (array): Array of shape (w, h, 2) or (2,)
                                       containing the absolute location(s)
            camera_translation (array): [x, y] array with the camera location
            camera_rotation (float): Camera rotation around z-axis

        Returns:
            Array of shape (w, h, 2) or (2,) containing the
            location(s) relative to the camera
        """
        
        # Construct an inverse 2D rotation matrix
        _s = np.sin(camera_rotation - np.pi / 2.)
        _c = np.cos(camera_rotation - np.pi / 2.)

        # R is orthogonal --> transpose and inverse are the same
        R_inv = np.array([[_c , _s],
                          [-_s, _c]])

        def _to_relative(p):
            return R_inv.dot(p - camera_translation)

        if len(absolute_location.shape) == 3:
            return np.apply_along_axis(_to_relative, 2, absolute_location)
        elif absolute_location.shape == (2,):
            return _to_relative(absolute_location)
        else:
            raise ValueError("Input has to be a an array of shape (w, h, 2) or (2,)")