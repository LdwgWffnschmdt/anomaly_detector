import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import consts

from common import utils, logger, ImageLocationUtility

class Visualize(object):
    WINDOWS_IMAGE = "visualize_image"
    WINDOWS_CONTROLS = "visualize_controls"

    LABEL_TEXTS = {
        0: "Unknown",
        1: "No anomaly",
        2: "Contains anomaly"
    }

    LABEL_COLORS = {
        0: (255,255,255),     # Unknown
        1: (80, 175, 76),     # No anomaly
        2: (54, 67, 244)      # Contains anomaly
    }

    DIRECTION_TEXTS = {
        0: "Unknown",
        1: "CCW",
        2: "CW"
    }

    DIRECTION_COLORS = {
        0: (255,255,255),     # Unknown
        1: (136, 150, 0),     # CCW
        2: (0, 152, 255)      # CW
    }

    @staticmethod
    def image_write_label(image, feature):
        """Write the specified label on an image for debug purposes
        (0: Unknown, 1: No anomaly, 2: Contains an anomaly)
        
        Args:
            image (Image)
        """
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.5
        thickness              = 1

        cv2.putText(image,"Label: ",
            (10,50), # bottomLeftCornerOfText
            font,
            fontScale,
            (255,255,255),
            thickness, lineType=cv2.LINE_AA)
        
        cv2.putText(image, Visualize.LABEL_TEXTS.get(feature.label, 0),
            (60, 50),
            font,
            fontScale,
            Visualize.LABEL_COLORS.get(feature.label, 0),
            thickness, lineType=cv2.LINE_AA)

        if feature.direction != 0:
            cv2.putText(image,"Direction: ",
                (10,80), # bottomLeftCornerOfText
                font,
                fontScale,
                (255,255,255),
                thickness, lineType=cv2.LINE_AA)
            
            cv2.putText(image, Visualize.DIRECTION_TEXTS.get(feature.direction, 0),
                (90, 80),
                font,
                fontScale,
                Visualize.DIRECTION_COLORS.get(feature.direction, 0),
                thickness, lineType=cv2.LINE_AA)

        if feature.round_number != 0:
            cv2.putText(image,"Round: ",
                (10,110), # bottomLeftCornerOfText
                font,
                fontScale,
                (255,255,255),
                thickness, lineType=cv2.LINE_AA)
            
            cv2.putText(image, str(feature.round_number),
                (65, 110),
                font,
                fontScale,
                (255,255,255),
                thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def image_add_trackbar(image, index, features):
        """Add trackbar to image
        
        Args:
            image (Image)
            index (int): Current index
            features (FeatureArray)
        """
        
        width = image.shape[1]
        factor = features.shape[0] / float(width)

        # LABELS
        for i in range(width):
            # Get label
            label = features[int(i * factor), 0, 0].label
            if label != 0: image[210:225, i, :] = Visualize.LABEL_COLORS[label]

        image[210:225, int(index / factor), :] = (1, 1, 1)

        # DIRECTION
        for i in range(width):
            # Get label
            direction = features[int(i * factor), 0, 0].direction
            if direction != 0: image[255:270, i, :] = Visualize.DIRECTION_COLORS[direction]

        image[255:270, int(index / factor), :] = (1, 1, 1)

        # ROUND
        for i in range(width):
            # Get label
            round_number = features[int(i * factor), 0, 0].round_number
            if round_number <= 0:
                continue
            elif round_number % 2 == 0:
                image[300:315, i, :] = (181, 81, 63)
            elif round_number % 2 == 1:
                image[300:315, i, :] = (203, 134, 121)

        image[300:315, int(index / factor), :] = (1, 1, 1)

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
        self.orig_features = features
        self.features      = features
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

        self._assets_path = os.path.join(os.path.dirname(__file__), "assets")

        self._window_set_up = False

    def show(self):
        self.mode = 0 # 0: don't edit, 1: single, 2: continuous
        self._label = -1
        self._direction = -1
        self._round_number = -1
        self._last_index = 0
        self._exiting = False
        self._mouse_down = False

        self.__setup_window__()
        self.__draw__()

        while True:
            key = cv2.waitKey(0 if self.pause else cv2.getTrackbarPos("delay", self.WINDOWS_CONTROLS))
            
            if key == 27:    # [esc] => Quit
                if self._exiting:
                    self._exiting = False
                    self.__draw__()
                elif len(self.orig_features.metadata_changed) > 0:
                    self.pause = True
                    self._exiting = True
                    self.__draw__()
                else:
                    cv2.destroyAllWindows()
                    return
            elif key == 121 and self._exiting: # [y] => save changes
                self.orig_features.save_metadata()
                cv2.destroyAllWindows()
                return
            elif key == 110 and self._exiting: # [n] => save changes
                cv2.destroyAllWindows()
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
                self.index += cv2.getTrackbarPos("skip", self.WINDOWS_CONTROLS)

            if self.index >= self.features.shape[0] - 1:
                self.index = self.features.shape[0] - 1
                self.pause = True
            elif self.index <= 0:
                self.index = 0
                self.pause = True

            if self.key_func is not None:
                self.key_func(self, key)

            
            if key == 115:  # [s]   => Switch single / continuous mode
                self.mode += 1
                if self.mode == 3:
                    self.mode = 0
                    self._label = -1
                    self._direction = -1
                    self._round_number = -1
                self.__draw__()
            
            if self.mode > 0:
                if key == 48: # [0]   => Unknown
                    new_label = 0
                elif key == 49: # [1]   => No anomaly
                    new_label = 1
                elif key == 50: # [2]   => Contains anomaly
                    new_label = 2

                elif key == 35: # [#]   => Direction unknown
                    new_direction = 0
                elif key == 44: # [,]   => Direction is CCW
                    new_direction = 1
                elif key == 46: # [.]   => Direction is CW
                    new_direction = 2

                elif key == 43: # [+]   => Increase round number by 1
                    new_round_number = self._round_number + 1
                elif key == 45: # [-]   => Decrease round number by 1
                    new_round_number = self._round_number - 1
                    if new_round_number <= -1: new_round_number = -1
                elif key == 120:# [x]   => Don't set the round number
                    new_round_number = -1
                
                # If we skipped back
                if self.index < self._last_index:
                    self._last_index = self.index
                
                indices = range(self._last_index, self.index + 1)
                self._last_index = self.index
                
                # Label the last features if in continuous mode
                if self.mode == 2:
                    for i in indices:
                        if self._label != -1:
                            self.features[i, 0, 0].label = self._label
                        
                        if self._direction != -1:
                            self.features[i, 0, 0].direction = self._direction

                        if self._round_number != -1:
                            self.features[i, 0, 0].round_number = self._round_number

                if key == 48 or key == 49 or key == 50:
                    if new_label == self._label:
                        new_label = -1
                    elif new_label != -1:
                        self.features[self.index, 0, 0].label = new_label
                    self._label = new_label
                    self.__draw__()
                    
                if key == 35 or key == 44 or key == 46:
                    if new_direction == self._direction:
                        new_direction = -1
                    elif new_direction != -1:
                        self.features[self.index, 0, 0].direction = new_direction
                    self._direction = new_direction
                    self.__draw__()
                    
                if key == 43 or key == 45 or key == 120:
                    if new_round_number != -1:
                        self.features[self.index, 0, 0].round_number = new_round_number
                    self._round_number = new_round_number
                    self.__draw__()
                
                if self.mode == 1:
                    self._label = -1
                    self._direction = -1
                    self.__draw__()

            # Update trackbar and thus trigger a draw
            cv2.setTrackbarPos("index", self.WINDOWS_CONTROLS, self.index)

    def create_trackbar(self, name, default, max):
        self.__setup_window__()
        cv2.createTrackbar(name, self.WINDOWS_CONTROLS, default, max, self.__draw__)

    def get_trackbar(self, name):
        return cv2.getTrackbarPos(name, self.WINDOWS_CONTROLS)
    
    def __setup_window__(self):
        ### Set up window if it does not exist
        if not self._window_set_up:
            cv2.namedWindow(self.WINDOWS_CONTROLS)
            cv2.setMouseCallback(self.WINDOWS_CONTROLS, self.__mouse__)

            # Create trackbars
            if self.has_locations:
                cv2.createTrackbar("show_grid", self.WINDOWS_CONTROLS, int(self.show_grid), 1, self.__draw__)
                cv2.createTrackbar("show_map",  self.WINDOWS_CONTROLS, int(self.show_map),  1, self.__draw__)
            
            if self.feature_to_text_func is not None:
                cv2.createTrackbar("show_values", self.WINDOWS_CONTROLS, int(self.show_values), 1, self.__draw__)
            
            if self.feature_to_text_func is not None or \
               self.feature_to_color_func is not None:
                cv2.createTrackbar("overlay", self.WINDOWS_CONTROLS, 40, 100, self.__draw__)

            cv2.createTrackbar("label", self.WINDOWS_CONTROLS, -1,  2, self.__change_features__)
            cv2.setTrackbarMin("label", self.WINDOWS_CONTROLS, -1)
            cv2.setTrackbarPos("label", self.WINDOWS_CONTROLS, -1)
            cv2.createTrackbar("direction", self.WINDOWS_CONTROLS, -1,  2, self.__change_features__)
            cv2.setTrackbarMin("direction", self.WINDOWS_CONTROLS, -1)
            cv2.setTrackbarPos("direction", self.WINDOWS_CONTROLS, -1)
            cv2.createTrackbar("round_number", self.WINDOWS_CONTROLS, -1,  30, self.__change_features__)
            cv2.setTrackbarMin("round_number", self.WINDOWS_CONTROLS, -1)
            cv2.setTrackbarPos("round_number", self.WINDOWS_CONTROLS, -1)

            cv2.createTrackbar("delay", self.WINDOWS_CONTROLS, 1,  1000, self.__draw__)
            cv2.createTrackbar("skip",  self.WINDOWS_CONTROLS, 1,  1000, lambda x: None)
            cv2.createTrackbar("index", self.WINDOWS_CONTROLS, 0,  self.features.shape[0] - 1, self.__index_update__)
            
            cv2.namedWindow(self.WINDOWS_IMAGE)

            self._window_set_up = True

    def __mouse__(self, event, x, y, flags, param):
        if (event == cv2.EVENT_LBUTTONDOWN and y > 180) or (event == cv2.EVENT_MOUSEMOVE and self._mouse_down):
            self._mouse_down = True
            cv2.setTrackbarPos("index", self.WINDOWS_CONTROLS, self.features.shape[0] * x / 640)
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False

    def __draw_controls__(self):
        if self._exiting:
            cv2.imshow(self.WINDOWS_CONTROLS, cv2.imread(os.path.join(self._assets_path, "Controls_save.jpg")))
            return

        if self.index < 0 or self.features.shape[0] <= 0:
            cv2.imshow(self.WINDOWS_CONTROLS, cv2.imread(os.path.join(self._assets_path, "Controls_no_frames.jpg")))
            return
        
        image = np.concatenate((
            cv2.imread(os.path.join(self._assets_path, "Controls_1_%i.jpg" % self.mode)),
            cv2.imread(os.path.join(self._assets_path, "Controls_2_%i.jpg" % self._label)),
            cv2.imread(os.path.join(self._assets_path, "Controls_3_%i.jpg" % self._direction)),
            cv2.imread(os.path.join(self._assets_path, "Controls_4_set.jpg" if self._round_number >= 0 else "Controls_4_-1.jpg"))), axis=0)
        
        font      = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.4
        thickness = 1

        cv2.putText(image, str(self._round_number),
            (291, 135), # bottomLeftCornerOfText
            font,
            fontScale,
            (0,0,0),
            thickness, lineType=cv2.LINE_AA)
        
        # Draw Trackbar
        self.image_add_trackbar(image, self.index, self.features)
        cv2.imshow(self.WINDOWS_CONTROLS, image)
        
    def __draw__(self, x=None):
        total, height, width = self.features.shape

        self.__draw_controls__()

        if self.index < 0 or self.features.shape[0] <= 0:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image,"No frames to show",
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                1, lineType=cv2.LINE_AA)
            
            cv2.imshow(self.WINDOWS_IMAGE, image)
            return
        
        feature_2d = self.features[self.index,...]
        cv2.setWindowTitle(self.WINDOWS_IMAGE, str(feature_2d[0, 0].time))

        # Get the image
        image = feature_2d[0, 0].get_image()

        # Update parameters
        if self.has_locations:
            self.show_grid = bool(cv2.getTrackbarPos("show_grid", self.WINDOWS_CONTROLS))
            self.show_map  = bool(cv2.getTrackbarPos("show_map", self.WINDOWS_CONTROLS))
        
        self.show_values = bool(cv2.getTrackbarPos("show_values", self.WINDOWS_CONTROLS))
        self.show_thresh = bool(cv2.getTrackbarPos("show_thresh", self.WINDOWS_CONTROLS))

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
        alpha = cv2.getTrackbarPos("overlay",self.WINDOWS_CONTROLS) / 100.0  # Transparency factor.
        image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw current label
        self.image_write_label(image_new, feature_2d[0, 0])
        cv2.imshow(self.WINDOWS_IMAGE, image_new)
    
    def __index_update__(self, new_index=None):
        if new_index != self.index:
            self.pause = True
        self.index = new_index
        self.__draw__()

    def __change_features__(self, *args):
        label        = cv2.getTrackbarPos("label", self.WINDOWS_CONTROLS)
        direction    = cv2.getTrackbarPos("direction", self.WINDOWS_CONTROLS)
        round_number = cv2.getTrackbarPos("round_number", self.WINDOWS_CONTROLS)

        self.features = self.orig_features
        if label == 0:
            self.features = self.features.unknown_anomaly
        elif label == 1:
            self.features = self.features.no_anomaly
        elif label == 2:
            self.features = self.features.anomaly

        if direction == 0:
            self.features = self.features.direction_unknown
        elif direction == 1:
            self.features = self.features.direction_ccw
        elif direction == 2:
            self.features = self.features.direction_cw

        if round_number != -1:
            self.features = self.features.round_number(round_number)

        cv2.setTrackbarPos("index", self.WINDOWS_CONTROLS, 0)
        cv2.setTrackbarMax("index", self.WINDOWS_CONTROLS, max(0, self.features.shape[0] - 1))

        self.__draw__()


if __name__ == "__main__":
    from common import FeatureArray
    features = FeatureArray(consts.IMAGES_PATH + "Test/")

    vis = Visualize(features)
    vis.show()