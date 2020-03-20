import cv2
import numpy as np
import matplotlib.pyplot as plt

import consts

from common import utils, logger, ImageLocationUtility

class Visualize(object):
    WINDOW_HANDLE = "visualize_image"

    @staticmethod
    def image_write_label(image, label):
        """Write the specified label on an image for debug purposes
        (0: Unknown, 1: No anomaly, 2: Contains an anomaly)
        
        Args:
            image (Image)
        """
        
        text = {
            0: "Unknown",
            1: "No anomaly",
            2: "Contains anomaly"
        }

        colors = {
            0: (255,255,255),   # Unknown
            1: (0, 204, 0),     # No anomaly
            2: (0, 0, 255)      # Contains anomaly
        }

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,50)
        fontScale              = 0.5
        thickness              = 1

        cv2.putText(image,"Label: ",
            bottomLeftCornerOfText,
            font,
            fontScale,
            (255,255,255),
            thickness, lineType=cv2.LINE_AA)
        
        cv2.putText(image, text.get(label, 0),
            (bottomLeftCornerOfText[0] + 50, bottomLeftCornerOfText[1]),
            font,
            fontScale,
            colors.get(label, 0),
            thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def image_add_trackbar(image, index, features):
        """Add trackbar to image
        
        Args:
            image (Image)
            index (int): Current index
            features (FeatureArray)
        """
        
        colors = {
            0: (255,255,255),   # Unknown
            1: (0, 204, 0),     # No anomaly
            2: (0, 0, 255)      # Contains anomaly
        }

        width = image.shape[1]

        trackbar = np.zeros((width, 3), dtype=np.uint8)

        factor = features.shape[0] / float(width)

        for i in range(width):
            # Get label
            label = features[int(i * factor), 0, 0].label
            trackbar[i,:] = colors[label]

        trackbar[int(index / factor),:] = (255, 0, 0)

        # Repeat vertically
        trackbar = np.expand_dims(trackbar, axis=0).repeat(15, axis=0)

        return np.concatenate((image, trackbar), axis=0)

    def __init__(self, features, **kwargs):
        """Visualize features on the source image

        Args:
            features (FeatureArray): Array of features as extracted by a FeatureExtractor
            images_path (str): Path to jpgs (Default: consts.IMAGES_PATH)
            show_grid (bool): Overlay real world coordinate grid (Default: False)
            show_map (bool): Update the position on the map every frame (Default: False)
            show_values (bool): Show values on each patch (Default: False)
            feature_to_color_func (function): Function converting a feature to a color (b, g, r)
            feature_to_text_func (function): Function converting a feature to a string
            pause_func (function): Function converting a feature to a boolean that pauses the video
        """
        self.features    = features
        self.images_path = kwargs.get("images_path", consts.IMAGES_PATH)

        self.show_grid = kwargs.get("show_grid", False)
        self.show_map  = kwargs.get("show_map", False)
        self.show_values  = kwargs.get("show_values", False)
        
        self.feature_to_color_func = kwargs.get("feature_to_color_func", None)
        self.feature_to_text_func  = kwargs.get("feature_to_text_func", None)
        self.pause_func            = kwargs.get("pause_func", None)
        self.key_func              = kwargs.get("key_func", None)

        self.index = 0
        self.pause = False
        
        self.has_locations = features[0,0,0].location is not None

        if self.has_locations:
            # Calculate grid overlay
            self._ilu = ImageLocationUtility()
            self._absolute_locations = self._ilu.span_grid(60, 60, 1, -30, -30)

            # Setup map display
            self.extent = features.get_extent()
            
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111)
            self._ax.set_xlim([self.extent[0], self.extent[2]])
            self._ax.set_ylim([self.extent[1], self.extent[3]])

            plt.ion()
            self._fig.show()
        else:
            self.show_grid = False
            self.show_map = False

        self._window_set_up = False

    def show(self):
        self.__setup_window__()
        self.__draw__()

        while True:
            key = cv2.waitKey(0 if self.pause else cv2.getTrackbarPos("delay", self.WINDOW_HANDLE))
            
            if key == 27:    # [esc] => Quit
                cv2.destroyWindow(self.WINDOW_HANDLE)
                return
            elif key == 32:  # [space] => Pause
                self.pause = not self.pause
                continue
            elif key == 100: # [d] => Seek forward + pause
                self.index += 1
                self.pause = True
            elif key == 97:  # [a] => Seek backward + pause
                self.index -= 1
                self.pause = True
            elif key == 101: # [e] => Seek 20 forward
                self.index += 20
            elif key == 113: # [q] => Seek 20 backward
                self.index -= 20
            elif key == -1:  # No input, continue
                self.index += cv2.getTrackbarPos("skip", self.WINDOW_HANDLE)

            if self.index >= self.features.shape[0] - 1:
                self.index = self.features.shape[0] - 1
                self.pause = True
            
            if self.key_func is not None:
                self.key_func(self, key)

            # Update trackbar and thus trigger a draw
            cv2.setTrackbarPos("index", self.WINDOW_HANDLE, self.index)

    def create_trackbar(self, name, default, max):
        self.__setup_window__()
        cv2.createTrackbar(name, self.WINDOW_HANDLE, default, max, self.__draw__)

    def get_trackbar(self, name):
        return cv2.getTrackbarPos(name, self.WINDOW_HANDLE)
    
    def __setup_window__(self):
        ### Set up window if it does not exist
        if not self._window_set_up:
            cv2.namedWindow(self.WINDOW_HANDLE)

            # Create trackbars
            if self.has_locations:
                cv2.createTrackbar("show_grid", self.WINDOW_HANDLE, int(self.show_grid) , 1, self.__draw__)
                cv2.createTrackbar("show_map",  self.WINDOW_HANDLE, int(self.show_map)  , 1, self.__draw__)
            
            if self.feature_to_text_func is not None:
                cv2.createTrackbar("show_values", self.WINDOW_HANDLE, int(self.show_values),    1, self.__draw__)
            
            cv2.createTrackbar("delay",       self.WINDOW_HANDLE, 1,  1000,      self.__draw__)
            cv2.createTrackbar("overlay",     self.WINDOW_HANDLE, 40, 100,       self.__draw__)
            cv2.createTrackbar("skip",        self.WINDOW_HANDLE, 1,  1000,      lambda x: None)
            cv2.createTrackbar("index",       self.WINDOW_HANDLE, 0,  self.features.shape[0] - 1, self.__index_update__)
            
            self._window_set_up = True

    def __draw__(self, x=None):
        total, height, width = self.features.shape

        feature_2d = self.features[self.index,...]
        cv2.setWindowTitle(self.WINDOW_HANDLE, str(feature_2d[0, 0].time))

        # Get the image
        image = feature_2d[0, 0].get_image()

        # Update parameters
        if self.has_locations:
            self.show_grid = bool(cv2.getTrackbarPos("show_grid", self.WINDOW_HANDLE))
            self.show_map  = bool(cv2.getTrackbarPos("show_map", self.WINDOW_HANDLE))
        
        self.show_values = bool(cv2.getTrackbarPos("show_values", self.WINDOW_HANDLE))
        self.show_thresh = bool(cv2.getTrackbarPos("show_thresh", self.WINDOW_HANDLE))

        # Update map
        if self.show_map:
            self._ax.clear()
            self._ax.set_xlim([self.extent[0], self.extent[2]])
            self._ax.set_ylim([self.extent[1], self.extent[3]])

            # Draw FOV polygon
            self._ax.fill([feature_2d[0 ,  0].location[0],
                          feature_2d[-1,  0].location[0],
                          feature_2d[-1, -1].location[0],
                          feature_2d[0 , -1].location[0]],
                         [feature_2d[0 ,  0].location[1],
                          feature_2d[-1,  0].location[1],
                          feature_2d[-1, -1].location[1],
                          feature_2d[0 , -1].location[1]])

            # self._ax.plot(feature_2d[0 ,  0].location[0], feature_2d[0 ,  0].location[1], "r+", markersize=2, linewidth=2)
            # self._ax.plot(feature_2d[-1,  0].location[0], feature_2d[-1,  0].location[1], "r+", markersize=2, linewidth=2)
            # self._ax.plot(feature_2d[0 , -1].location[0], feature_2d[0 , -1].location[1], "r+", markersize=2, linewidth=2)
            # self._ax.plot(feature_2d[-1, -1].location[0], feature_2d[-1, -1].location[1], "r+", markersize=2, linewidth=2)
            
            # Draw camera position
            self._ax.plot(feature_2d[0, 0].camera_position[0], 
                          feature_2d[0, 0].camera_position[1], "bo")
            
            self._fig.canvas.draw()
        
        font      = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.25
        thickness = 1

        overlay = image.copy()

        patch_size = (image.shape[1] / float(width), image.shape[0] / float(height))
        
        # Loop over all feature maps
        if self.feature_to_color_func is not None or \
            (self.feature_to_text_func is not None and self.show_values) or \
            self.pause_func is not None:
            for x in range(width):
                for y in range(height):
                    feature = feature_2d[y,x]

                    p1 = (int(x * patch_size[0]), int(y * patch_size[1]))
                    p2 = (int(p1[0] + patch_size[0]), int(p1[1] + patch_size[1]))
                    
                    if self.feature_to_color_func is not None:
                        cv2.rectangle(overlay, p1, p2, self.feature_to_color_func(self, feature), -1)

                    if self.feature_to_text_func is not None and self.show_values:
                        text = str(self.feature_to_text_func(self, feature))
                        cv2.putText(overlay, text,
                            (p1[0] + 2, p1[1] + patch_size[1] - 2),
                            font,
                            fontScale,
                            (0, 0, 255),
                            thickness, lineType=cv2.LINE_AA)
                    
                    if self.pause_func is not None and self.pause_func(self, feature):
                        self.pause = True
        
        # Draw grid
        if self.show_grid:
            relative_grid = self._ilu.absolute_to_relative(self._absolute_locations, feature_2d[0,0].camera_position, feature_2d[0,0].camera_rotation)
            image_grid = self._ilu.relative_to_image(relative_grid, image.shape[1], image.shape[0])
        
            for a in range(image_grid.shape[0]):
                for b in range(image_grid.shape[1]):
                    pos = (int(image_grid[a][b][0]), int(image_grid[a][b][1]))
                    if pos[0] < 0 or pos[0] > image.shape[1] or pos[1] < 0 or pos[1] > image.shape[0]:
                        continue
                    cv2.circle(overlay, pos, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    
                    cv2.putText(overlay, "%.1f / %.1f" % (self._absolute_locations[a][b][0], self._absolute_locations[a][b][1]),
                        (pos[0] + 3, pos[1] + 2),
                        font,
                        fontScale,
                        (255, 255, 255),
                        thickness, lineType=cv2.LINE_AA)

        # Blend the overlay
        alpha = cv2.getTrackbarPos("overlay",self.WINDOW_HANDLE) / 100.0  # Transparency factor.
        image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw Trackbar
        image_new = self.image_add_trackbar(image_new, self.index, self.features)
        
        # Draw current label
        self.image_write_label(image_new, feature_2d[0, 0].label)
        cv2.imshow(self.WINDOW_HANDLE, image_new)
    
    def __index_update__(self, new_index=None):
        if new_index != self.index:
            self.pause = True
        self.index = new_index
        self.__draw__()


if __name__ == "__main__":
    from common import FeatureArray
    features = FeatureArray(consts.FEATURES_FILE)

    vis = Visualize(features)
    vis.show()