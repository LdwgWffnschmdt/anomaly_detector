import os
import cv2
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.ndimage.morphology import generate_binary_structure, grey_erosion, grey_dilation

import consts

from common import utils, logger, ImageLocationUtility, PatchArray

class Visualize(object):
    WINDOWS_IMAGE = "visualize_image"
    WINDOWS_CONTROLS = "visualize_controls"
    WINDOWS_MAHA = "visualize_maha"

    DIRECTION_TEXTS = {
        -1: "Not set",
        0: "Unknown",
        1: "CCW",
        2: "CW"
    }

    DIRECTION_COLORS = {
        -1: (100, 100, 100),   # Not set
        0: (255,255,255),     # Unknown
        1: (136, 150, 0),     # CCW
        2: (0, 152, 255)      # CW
    }

    def image_write_label(self, image, frame):
        """Write the specified label on an image for debug purposes
        (0: Unknown, 1: No anomaly, 2: Contains an anomaly)
        
        Args:
            image (Image)
        """
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.5
        thickness              = 1

        cv2.putText(image, self.title,
            (10,20), # bottomLeftCornerOfText
            font,
            fontScale,
            (255,255,255),
            thickness, lineType=cv2.LINE_AA)
        
        cv2.putText(image,"Label: ",
            (10,50), # bottomLeftCornerOfText
            font,
            fontScale,
            (255,255,255),
            thickness, lineType=cv2.LINE_AA)

        cv2.putText(image, self.metric.names.get(frame[self.metric.label_name], 0),
            (60, 50),
            font,
            fontScale,
            PatchArray.Metric.COLORS.get(frame[self.metric.label_name], 0),
            thickness, lineType=cv2.LINE_AA)

        if self._video_writer is None:
            if frame.directions != 0:
                cv2.putText(image,"Direction: ",
                    (10,80), # bottomLeftCornerOfText
                    font,
                    fontScale,
                    (255,255,255),
                    thickness, lineType=cv2.LINE_AA)
                
                cv2.putText(image, Visualize.DIRECTION_TEXTS.get(frame.directions, 0),
                    (90, 80),
                    font,
                    fontScale,
                    Visualize.DIRECTION_COLORS.get(frame.directions, 0),
                    thickness, lineType=cv2.LINE_AA)

            if frame.round_numbers != 0:
                cv2.putText(image,"Round: ",
                    (10,110), # bottomLeftCornerOfText
                    font,
                    fontScale,
                    (255,255,255),
                    thickness, lineType=cv2.LINE_AA)
                
                cv2.putText(image, str(frame.round_numbers),
                    (65, 110),
                    font,
                    fontScale,
                    (255,255,255),
                    thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def image_add_trackbar(image, index, patches):
        """Add trackbar to image
        
        Args:
            image (Image)
            index (int): Current index
            patches (PatchArray)
        """
        
        width = image.shape[1]
        # factor = patches.shape[0] / float(width)
        
        slots, factor = np.linspace(0, patches.shape[0], num=width, endpoint=False, dtype=np.int, retstep=True)

        def _get_round_number_color(round_number):
            if round_number <= 0:
                return (255, 255, 255)
            elif round_number % 2 == 0:
                return (181, 81, 63)
            elif round_number % 2 == 1:
                return (203, 134, 121)

        image[210:217, ...] = np.array(np.vectorize(PatchArray.Metric.COLORS.get)(patches[slots, 0, 0].labels)).T
        image[217:225, ...] = np.array(np.vectorize(PatchArray.Metric.COLORS.get)(patches[slots, 0, 0].stop)).T
        image[255:270, ...] = np.array(np.vectorize(Visualize.DIRECTION_COLORS.get)(patches[slots, 0, 0].directions)).T
        image[300:315, ...] = np.array(np.vectorize(_get_round_number_color)(patches[slots, 0, 0].round_numbers)).T

        image[210:225, int(index / factor), :] = (1, 1, 1)
        image[255:270, int(index / factor), :] = (1, 1, 1)
        image[300:315, int(index / factor), :] = (1, 1, 1)

    def __init__(self, patches, **kwargs):
        """Visualize patches

        Args:
            patches (PatchArray): Array of patches
            images_path (str): Path to jpgs (Default: consts.IMAGES_PATH)
            show_grid (bool): Overlay real world coordinate grid (Default: False)
            show_map (bool): Update the position on the map every frame (Default: False)
            show_values (bool): Show values on each patch (Default: False)
        """
        self.orig_patches = patches
        self.patches      = patches
        self.images_path = kwargs.get("images_path", consts.IMAGES_PATH)

        self.show_grid = kwargs.get("show_grid", False)
        self.show_map  = kwargs.get("show_map", False)
        self.show_values  = kwargs.get("show_values", False)
        
        self.key_func            = kwargs.get("key_func", None)

        self.model_index = 0 # Index for patch_to_color_func ...

        self.index = 0
        self.pause = True
        
        if self.patches.contains_mahalanobis_distances:
            self._metrics_fig, self._metrics_ax1 = plt.subplots(1, 1)
            self._metrics_ax1.set_yscale("log")
            # self._metrics_ax2 = self._metrics_ax1.twinx()

            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()

            self._histogram_fig, (self._histogram_ax1, self._histogram_ax2) = plt.subplots(2, 1, sharex=True)

            self._histogram_ax1.set_title("No anomaly")
            self._histogram_ax2.set_title("Anomaly")
            
            self._histogram_fig.suptitle("Mahalanobis distances")

            
            self._metrics_fig.show()
            self._histogram_fig.show()
        
        if self.patches.contains_locations:
            # Setup map display
            self.extent = patches.get_extent()
            
            self._map_fig = plt.figure()
            self._map_ax = self._map_fig.add_subplot(111)
            self._map_ax.set_xlim([self.extent[0], self.extent[2]])
            self._map_ax.set_ylim([self.extent[1], self.extent[3]])

            # Calculate grid overlay
            self._ilu = ImageLocationUtility()
            self._absolute_locations = self._ilu.span_grid(self.extent[3] - self.extent[1], self.extent[2] - self.extent[0],
                                                                   offset_y=self.extent[1],         offset_x=self.extent[0])

            self._map_fig.show()
        else:
            self.show_grid = False
            self.show_map = False

        plt.ion()

        self._prev_gray = None
        self._cur_glitch = None

        self._labels = None
        self._scores = None
        self._thresh = None

        self._assets_path = os.path.join(os.path.dirname(__file__), "assets")

        self._window_set_up = False
        self.mode = 0 # 0: don't edit, 1: single, 2: continuous
        self._label = -1
        self._direction = -1
        self._round_number = -1
        self._last_index = 0
        self._exiting = False
        self._mouse_down = False

        self._mouse_image_x = -1
        self._mouse_image_y = -1
        self._image_shape = (1, 1, 3)

        self._video_writer = None

    def show(self):
        """ Main function """
        self.mode = 0 # 0: don't edit, 1: single, 2: continuous
        self._label = -1
        self._direction = -1
        self._round_number = -1
        self._last_index = 0
        self._exiting = False
        self._mouse_down = False


        self.__setup_window__()
        self.__draw__()
        plt.show()#block=False)

        while True:
            # plt.draw()
            plt.pause(0.001)

            key = cv2.waitKey(cv2.getTrackbarPos("delay", self.WINDOWS_CONTROLS))
            
            if key == 27:    # [esc] => Quit
                if self._exiting:
                    self._exiting = False
                    self.__draw__()
                elif len(self.orig_patches.metadata_changed) > 0:
                    self.pause = True
                    self._exiting = True
                    self.__draw__()
                else:
                    return self.close()
            elif key == ord("y") and self._exiting: # [y] => save changes
                if self.orig_patches.save_metadata():
                    return self.close()
            elif key == ord("n") and self._exiting: # [n] => save changes
                return self.close()
            elif key == ord("5"): # [5] => extract current patches
                self.patches.extract_current_patches()
            elif key == ord("r"): # [r] => start/stop recording video
                self.record()
            elif key == ord(" "):  # [space] => Pause
                self.pause = not self.pause
                continue
            elif key == ord("d"): # [d] => Seek forward + pause
                self.index += 1
                self.pause = True
            elif key == ord("a"):  # [a] => Seek backward + pause
                self.index -= 1
                self.pause = True
            elif key == ord("e"): # [e] => Seek 20 forward
                self.index += 20
            elif key == ord("q"): # [q] => Seek 20 backward
                self.index -= 20
            elif key == -1 and not self.pause:  # No input, continue
                self.index += cv2.getTrackbarPos("skip", self.WINDOWS_CONTROLS)

            if self.index >= self.patches.shape[0] - 1:
                self.index = self.patches.shape[0] - 1
                self.pause = True
            elif self.index <= 0:
                self.index = 0
                self.pause = True

            if self.key_func is not None:
                self.key_func(key)

            
            if key == ord("s"):  # [s]   => Switch single / continuous mode
                self.mode += 1
                if self.mode == 3:
                    self.mode = 0
                    self._label = -1
                    self._direction = -1
                    self._round_number = -1
                self.__draw__()
            
            if self.mode > 0:
                if   key == ord("0"): # [0]   => Unknown
                    new_label = 0
                elif key == ord("1"): # [1]   => No anomaly
                    new_label = 1
                elif key == ord("2"): # [2]   => Contains anomaly
                    new_label = 2

                elif key == ord("#"): # [#]   => Direction unknown
                    new_direction = 0
                elif key == ord(","): # [,]   => Direction is CCW
                    new_direction = 1
                elif key == ord("."): # [.]   => Direction is CW
                    new_direction = 2

                elif key == ord("+"): # [+]   => Increase round number by 1
                    new_round_number = self._round_number + 1
                elif key == ord("-"): # [-]   => Decrease round number by 1
                    new_round_number = self._round_number - 1
                    if new_round_number <= -1: new_round_number = -1
                elif key == ord("x"):# [x]   => Don't set the round number
                    new_round_number = -1
                
                # If we skipped back
                if self.index < self._last_index:
                    self._last_index = self.index
                
                indices = range(self._last_index, self.index + 1)
                self._last_index = self.index
                
                # Label the last frames if in continuous mode
                if self.mode == 2:
                    for i in indices:
                        if self._label != -1:
                            self.patches[i, 0, 0].stop = self._label
                        
                        if self._direction != -1:
                            self.patches[i, 0, 0].directions = self._direction

                        if self._round_number != -1:
                            self.patches[i, 0, 0].round_numbers = self._round_number

                if key == ord("0") or key == ord("1") or key == ord("2"):
                    if new_label == self._label:
                        new_label = -1
                    elif new_label != -1:
                        self.patches[self.index, 0, 0].stop = new_label
                    self._label = new_label
                    self.__draw__()
                    
                if key == ord("#") or key == ord(",") or key == ord("."):
                    if new_direction == self._direction:
                        new_direction = -1
                    elif new_direction != -1:
                        self.patches[self.index, 0, 0].directions = new_direction
                    self._direction = new_direction
                    self.__draw__()
                    
                if key == ord("+") or key == ord("-") or key == ord("x"):
                    if new_round_number != -1:
                        self.patches[self.index, 0, 0].round_numbers = new_round_number
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
    
    def __maha__(self, x=None, only_refresh_image=False):
        image = np.zeros((350, 480, 3), dtype=np.uint8)
        if self.model_index > 0 and self.patches.contains_mahalanobis_distances:
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 0.5
            thickness              = 1

            model = sorted(self.patches.mahalanobis_distances.dtype.names)[self.model_index - 1]

            cv2.putText(image,"Model:", (10, 20), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)
            cv2.putText(image, model,    (65, 20), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)

            cv2.putText(image,"Filter:", (10, 50), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)

            if not only_refresh_image:
                self.patches.mahalanobis_distances_filtered[:] = self.patches.mahalanobis_distances[model]

            sigma_0 = (cv2.getTrackbarPos("0_gaussian_0", self.WINDOWS_MAHA),
                       cv2.getTrackbarPos("0_gaussian_1", self.WINDOWS_MAHA),
                       cv2.getTrackbarPos("0_gaussian_2", self.WINDOWS_MAHA))
            if sigma_0 != (0, 0, 0):
                cv2.putText(image, "gaussian (%i, %i, %i)" % sigma_0, (65, 50), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)
                if not only_refresh_image:
                    self.patches.mahalanobis_distances_filtered = utils.gaussian_filter(self.patches.mahalanobis_distances_filtered, sigma=sigma_0)

            erosion_dilation = cv2.getTrackbarPos("1_erosion_dilation", self.WINDOWS_MAHA)
            if erosion_dilation > 0:
                struct = generate_binary_structure(cv2.getTrackbarPos("1_erosion_dilation_structure_rank", self.WINDOWS_MAHA),
                                                   cv2.getTrackbarPos("1_erosion_dilation_structure_connectivity", self.WINDOWS_MAHA))
                if struct.ndim == 2:
                    z = np.zeros_like(struct, dtype=np.bool)
                    struct = np.stack((z, struct, z))
                
                if erosion_dilation == 1:
                    cv2.putText(image, "erosion", (65, 80), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)
                    if not only_refresh_image:
                        self.patches.mahalanobis_distances_filtered = grey_erosion(self.patches.mahalanobis_distances_filtered, structure=struct)
                elif erosion_dilation == 2:
                    cv2.putText(image, "dilation", (65, 80), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)
                    if not only_refresh_image:
                        self.patches.mahalanobis_distances_filtered = grey_dilation(self.patches.mahalanobis_distances_filtered, structure=struct)

                for (z, x, y) in np.ndindex(struct.shape):
                    cv2.putText(image, str(int(struct[z, x, y])), (150 + y * 15 + z * 60, 80 + x * 15), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)
            
            # sigma_2 = (cv2.getTrackbarPos("2_gaussian_0", self.WINDOWS_MAHA),
            #            cv2.getTrackbarPos("2_gaussian_1", self.WINDOWS_MAHA),
            #            cv2.getTrackbarPos("2_gaussian_2", self.WINDOWS_MAHA))
            # if sigma_2 != (0, 0, 0):
            #     cv2.putText(image, "gaussian (%i, %i, %i)" % sigma_2, (65, 140), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)
            #     if not only_refresh_image:
            #         self.patches.mahalanobis_distances_filtered = utils.gaussian_filter(self.patches.mahalanobis_distances_filtered, sigma=sigma_2)
            
            # Add some statistics
            threshold = float(cv2.getTrackbarPos("threshold", self.WINDOWS_MAHA)) / 10000.0

            cv2.putText(image, "                        TPR        FPR        Threshold", (10, 190), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)

            self._metrics_ax1.clear()
            # self._metrics_ax2.clear()

            self._metrics_ax1.set_yscale("log")
            
            for i, metric in enumerate(PatchArray.METRICS):
                labels = metric.get_labels(self.patches)
                scores = metric.get_values(self.patches.mahalanobis_distances_filtered)
                
                if metric.current_threshold == -1:
                    m = np.max(scores)
                    metric.current_threshold = m
                else:
                    m = metric.current_threshold

                thresh = m * threshold

                negavites = scores[labels == 1]
                positives = scores[labels == 2]

                false_negavites = np.count_nonzero(negavites >= thresh)
                true_positives = np.count_nonzero(positives >= thresh)

                tpr = true_positives / float(positives.size) * 100.0 if float(positives.size) > 0 else 0
                fpr = false_negavites / float(negavites.size) * 100.0 if float(negavites.size) > 0 else 0

                if i == cv2.getTrackbarPos("metric", self.WINDOWS_MAHA):
                    self._labels = labels
                    self._scores = scores
                    self._thresh = thresh

                    if not only_refresh_image:
                        if metric.name != "patch":
                            for r in np.reshape(np.diff(np.r_[0, labels == 0, 0]).nonzero()[0], (-1,2)):
                                self._metrics_ax1.axvspan(r[0], r[1], facecolor='black', alpha=0.1)

                            for r in np.reshape(np.diff(np.r_[0, np.logical_and(labels == 2, scores >= thresh), 0]).nonzero()[0], (-1,2)):
                                self._metrics_ax1.axvspan(r[0], r[1], facecolor='g', alpha=0.2)

                            for r in np.reshape(np.diff(np.r_[0, np.logical_and(labels == 1, scores >= thresh), 0]).nonzero()[0], (-1,2)):
                                self._metrics_ax1.axvspan(r[0], r[1], facecolor='r', alpha=0.2)

                            for r in np.reshape(np.diff(np.r_[0, np.logical_and(labels == 0, scores >= thresh), 0]).nonzero()[0], (-1,2)):
                                self._metrics_ax1.axvspan(r[0], r[1], facecolor='g', alpha=0.05)

                            for r in np.reshape(np.diff(np.r_[0, np.logical_and(labels == 2, scores < thresh), 0]).nonzero()[0], (-1,2)):
                                self._metrics_ax1.axvspan(r[0], r[1], facecolor='b', alpha=0.2)

                            for r in np.reshape(np.diff(np.r_[0, np.logical_and(labels == 0, scores < thresh), 0]).nonzero()[0], (-1,2)):
                                self._metrics_ax1.axvspan(r[0], r[1], facecolor='b', alpha=0.05)

                            self._metrics_ax1.set_ylim(0, np.max(scores))
                            self._metrics_ax1.plot(scores, lw=1, label=metric.name, color="black")
                            # self._metrics_ax1.axvline(x=self.index, linewidth=0.5, color="black")
                            self._metrics_ax1.axhline(y=thresh, linewidth=0.5, color="black")
                            self._metrics_fig.suptitle(metric.name)
                            self._metrics_fig.canvas.set_window_title("%s [scores]" % self.title)
                        
                        self._histogram_ax1.clear()
                        self._histogram_ax2.clear()

                        # r = (np.nanmin(self.patches.mahalanobis_distances_filtered), np.nanmax(self.patches.mahalanobis_distances_filtered))

                        self._histogram_ax1.set_title("No anomaly")
                        self._histogram_ax2.set_title("Anomaly")
                        
                        self._histogram_fig.suptitle("Mahalanobis distances")

                        _, bins, _ = self._histogram_ax1.hist(negavites.ravel(), bins=200)
                        self._histogram_ax2.hist(positives.ravel(), bins=bins)

                        # x = np.arange(0, 30, .05)
                        # self._histogram_ax2.plot(x, stats.chi2.pdf(x, df=3), color='r', lw=2)

                        self._histogram_fig.canvas.draw()
                        self._histogram_fig.canvas.set_window_title("%s [histogram]" % self.title)

                cv2.putText(image, metric.name, (40, 220 + i*30), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)
                cv2.putText(image, "%.2f" % tpr, (200, 220 + i*30), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)
                cv2.putText(image, "%.2f" % fpr, (300, 220 + i*30), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)
                cv2.putText(image, "%.2f" % thresh, (400, 220 + i*30), font, fontScale, (255,255,255), thickness, lineType=cv2.LINE_AA)


            self._metrics_fig.canvas.draw()

            self.__draw__()
        cv2.imshow(self.WINDOWS_MAHA, image)

    def patch_to_color(self, patch):
        b = 0#100 if patch in self.normal_distribution else 0
        g = 0
        r = 0
        threshold = cv2.getTrackbarPos("threshold", self.WINDOWS_MAHA) / 10000.0
        if cv2.getTrackbarPos("show_thresh", self.WINDOWS_MAHA):
            anomaly = patch.mahalanobis_distances_filtered > self._thresh
            if anomaly and patch.patch_labels == 2:         # Correct
                g = 100
                r = 0
            elif anomaly and patch.patch_labels != 2:       # False positive
                g = 0
                r = 100
            elif not anomaly and patch.patch_labels == 2:    # False negative
                b = 100
                r = 0
        elif threshold == 0:
            r = 0
        else:
            r = min(255, int(patch.mahalanobis_distances_filtered * (255.0 / self._thresh)))
        return (b, g, r)

    def patch_to_text(self, patch):
        return round(patch.mahalanobis_distances_filtered, 2)

    def __setup_window__(self):
        ### Set up window if it does not exist
        if not self._window_set_up:
            cv2.namedWindow(self.WINDOWS_CONTROLS)
            cv2.setMouseCallback(self.WINDOWS_CONTROLS, self.__mouse__)

            # Create trackbars
            if self.patches.contains_locations:
                cv2.createTrackbar("show_grid", self.WINDOWS_CONTROLS, int(self.show_grid), 1, self.__draw__)
                cv2.createTrackbar("show_map",  self.WINDOWS_CONTROLS, int(self.show_map),  1, self.__draw__)
            
            if self.patches.contains_mahalanobis_distances:
                cv2.createTrackbar("show_values", self.WINDOWS_CONTROLS, int(self.show_values), 1, self.__draw__)
            
            if self.patches.contains_locations or self.patches.contains_mahalanobis_distances:
                cv2.createTrackbar("overlay", self.WINDOWS_CONTROLS, 40, 100, self.__draw__)

            if self.patches.contains_mahalanobis_distances:
                cv2.namedWindow(self.WINDOWS_MAHA)
                
                cv2.createTrackbar("threshold", self.WINDOWS_MAHA, 5000, 10000, lambda x: self.__maha__(only_refresh_image=True))
                cv2.createTrackbar("show_thresh", self.WINDOWS_MAHA, 1, 1, self.__draw__)

                cv2.createTrackbar("0_gaussian_0", self.WINDOWS_MAHA, 0, 10, self.__maha__)
                cv2.createTrackbar("0_gaussian_1", self.WINDOWS_MAHA, 0, 10, self.__maha__)
                cv2.createTrackbar("0_gaussian_2", self.WINDOWS_MAHA, 0, 10, self.__maha__)
                
                cv2.createTrackbar("1_erosion_dilation", self.WINDOWS_MAHA, 0, 2, self.__maha__)
                cv2.createTrackbar("1_erosion_dilation_structure_rank", self.WINDOWS_MAHA, 2, 3, self.__maha__)
                cv2.setTrackbarMin("1_erosion_dilation_structure_rank", self.WINDOWS_MAHA, 2)
                cv2.createTrackbar("1_erosion_dilation_structure_connectivity", self.WINDOWS_MAHA, 1, 3, self.__maha__)
                cv2.setTrackbarMin("1_erosion_dilation_structure_connectivity", self.WINDOWS_MAHA, 1)

                # cv2.createTrackbar("2_gaussian_0", self.WINDOWS_MAHA, 0, 10, self.__maha__)
                # cv2.createTrackbar("2_gaussian_1", self.WINDOWS_MAHA, 0, 10, self.__maha__)
                # cv2.createTrackbar("2_gaussian_2", self.WINDOWS_MAHA, 0, 10, self.__maha__)
                
                cv2.createTrackbar("metric", self.WINDOWS_MAHA, 0, len(PatchArray.METRICS) - 1, lambda x: self.__maha__(only_refresh_image=True))
                cv2.createTrackbar("model", self.WINDOWS_MAHA, 0, len(self.patches.mahalanobis_distances.dtype.names), self.__model__)

            cv2.createTrackbar("optical_flow", self.WINDOWS_CONTROLS, 0, 1, self.__draw__)

            cv2.createTrackbar("label", self.WINDOWS_CONTROLS, -1,  2, self.__change_frames__)
            cv2.setTrackbarMin("label", self.WINDOWS_CONTROLS, -1)
            cv2.setTrackbarPos("label", self.WINDOWS_CONTROLS, -1)

            cv2.createTrackbar("stop_label", self.WINDOWS_CONTROLS, -1,  2, self.__change_frames__)
            cv2.setTrackbarMin("stop_label", self.WINDOWS_CONTROLS, -1)
            cv2.setTrackbarPos("stop_label", self.WINDOWS_CONTROLS, -1)

            cv2.createTrackbar("direction", self.WINDOWS_CONTROLS, -1,  2, self.__change_frames__)
            cv2.setTrackbarMin("direction", self.WINDOWS_CONTROLS, -1)
            cv2.setTrackbarPos("direction", self.WINDOWS_CONTROLS, -1)

            cv2.createTrackbar("round_number", self.WINDOWS_CONTROLS, -1,  30, self.__change_frames__)
            cv2.setTrackbarMin("round_number", self.WINDOWS_CONTROLS, -1)
            cv2.setTrackbarPos("round_number", self.WINDOWS_CONTROLS, -1)

            cv2.createTrackbar("error", self.WINDOWS_CONTROLS, 0,  2, self.__change_frames__)

            cv2.createTrackbar("delay", self.WINDOWS_CONTROLS, 1,  1000, self.__draw__)
            cv2.createTrackbar("skip",  self.WINDOWS_CONTROLS, 1,  1000, lambda x: None)
            cv2.createTrackbar("index", self.WINDOWS_CONTROLS, 0,  self.patches.shape[0] - 1, self.__index_update__)
            
            cv2.namedWindow(self.WINDOWS_IMAGE)
            cv2.setMouseCallback(self.WINDOWS_IMAGE, self.__mouse_image__)

            self._window_set_up = True

    def __mouse__(self, event, x, y, flags, param):
        if (event == cv2.EVENT_LBUTTONDOWN and y > 180) or (event == cv2.EVENT_MOUSEMOVE and self._mouse_down):
            self._mouse_down = True
            cv2.setTrackbarPos("index", self.WINDOWS_CONTROLS, self.patches.shape[0] * x / 640)
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False

    def __mouse_image__(self, event, x, y, flags, param):
        if not self.patches.contains_features:
            return
        if (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON) or event == cv2.EVENT_MBUTTONDOWN:
            iy = int(y / float(self._image_shape[0]) * self.patches.shape[1])
            ix = int(x / float(self._image_shape[1]) * self.patches.shape[2])
            if self._mouse_image_y != iy or self._mouse_image_x != ix:
                self._mouse_image_y = iy
                self._mouse_image_x = ix
                self.__draw__()
                self.click(self.patches[self.index, ...], iy, ix)
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_image_y = -1
            self._mouse_image_x = -1
            self.__draw__()

    def click(self, frame, y, x):
        pass

    def __draw_controls__(self):
        if self._exiting:
            cv2.imshow(self.WINDOWS_CONTROLS, cv2.imread(os.path.join(self._assets_path, "Controls_save.jpg")))
            return

        if self.index < 0 or self.patches.shape[0] <= 0:
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
        self.image_add_trackbar(image, self.index, self.patches)

        if self._video_writer is not None:
            cv2.ellipse(image, (620, 20), (10, 10), 0, 0, 360, (0,0,255), -1, lineType=cv2.LINE_AA)

        cv2.imshow(self.WINDOWS_CONTROLS, image)
    
    
    def __draw_flow__(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    def __draw_hsv__(self, flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,1] = 255
        hsv[...,2] = np.minimum(v*4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def __warp_flow__(self, img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res

    def __draw__(self, x=None):
        if not self._window_set_up:
            return
            
        self.__draw_controls__()

        if self.index < 0 or self.patches.shape[0] <= 0:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image,"No frames to show", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, lineType=cv2.LINE_AA)
            cv2.imshow(self.WINDOWS_IMAGE, image)
            return
        
        frame = self.patches[self.index, ...]
        cv2.setWindowTitle(self.WINDOWS_IMAGE, str(frame.times[0, 0]))

        # Get the image
        image = frame[0, 0].get_image()
        self._image_shape = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self._cur_glitch is None:
            self._cur_glitch = image.copy()
        
        # Calculate optical flow
        if bool(cv2.getTrackbarPos("optical_flow", self.WINDOWS_CONTROLS)) and self._prev_gray is not None:
            hsv = np.zeros_like(image)
            hsv[..., 1] = 255

            flow = cv2.calcOpticalFlowFarneback(self._prev_gray,
                                                gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            image = self.__draw_flow__(gray, flow)
            cv2.imshow('flow HSV', self.__draw_hsv__(flow))
            # self._cur_glitch = self.__warp_flow__(self._cur_glitch, flow)
            # cv2.imshow('glitch', self._cur_glitch)

        self._prev_gray = gray

        # Update map
        if self.patches.contains_locations:
            self.show_grid = bool(cv2.getTrackbarPos("show_grid", self.WINDOWS_CONTROLS))
            self.show_map  = bool(cv2.getTrackbarPos("show_map", self.WINDOWS_CONTROLS))

            if self.show_map:
                self._map_ax.clear()
                self._map_ax.set_xlim([self.extent[0], self.extent[2]])
                self._map_ax.set_ylim([self.extent[1], self.extent[3]])
                self._map_ax.set_aspect(1)
                
                self._map_ax.set_xticks(np.arange(self.extent[0], self.extent[2], 0.2), minor=True)
                self._map_ax.set_yticks(np.arange(self.extent[1], self.extent[3], 0.2), minor=True)
                
                self._map_ax.set_xticks(np.arange(self.extent[0], self.extent[2], 2), minor=False)
                self._map_ax.set_yticks(np.arange(self.extent[1], self.extent[3], 2), minor=False)
                
                self._map_ax.grid(linestyle="-", alpha=0.3, which="both")

                # Draw FOV polygon
                self._map_ax.fill([frame[0 ,  0].locations.tl.x,
                                   frame[0 , -1].locations.bl.x,
                                   frame[-1, -1].locations.br.x,
                                   frame[-1,  0].locations.tr.x],
                                  [frame[0 ,  0].locations.tl.y,
                                   frame[0 , -1].locations.bl.y,
                                   frame[-1, -1].locations.br.y,
                                   frame[-1,  0].locations.tr.y], alpha=0.2)

                for l in [(0, 0), (-1, 0), (-1, -1), (0, -1)]:# np.ndindex(frame.shape):
                    self._map_ax.fill([frame[l].locations.tl.x,
                                       frame[l].locations.bl.x,
                                       frame[l].locations.br.x,
                                       frame[l].locations.tr.x],
                                      [frame[l].locations.tl.y,
                                       frame[l].locations.bl.y,
                                       frame[l].locations.br.y,
                                       frame[l].locations.tr.y], alpha=0.5)

                self._map_ax.plot(frame[0 ,  0].locations.tl.x, frame[0 ,  0].locations.tl.y, "y+", markersize=2, linewidth=2)
                self._map_ax.plot(frame[0 , -1].locations.bl.x, frame[0 , -1].locations.bl.y, "b+", markersize=2, linewidth=2)
                self._map_ax.plot(frame[-1, -1].locations.br.x, frame[-1, -1].locations.br.y, "r+", markersize=2, linewidth=2)
                self._map_ax.plot(frame[-1,  0].locations.tr.x, frame[-1,  0].locations.tr.y, "g+", markersize=2, linewidth=2)
                
                # Draw camera position
                self._map_ax.plot(frame.camera_locations[0, 0].translation.x, 
                              frame.camera_locations[0, 0].translation.y, "bo")
                
                self._map_fig.canvas.draw()
                self._map_fig.canvas.set_window_title("%s [map]" % self.title)
            
        
        # Update parameters
        self.show_values = bool(cv2.getTrackbarPos("show_values", self.WINDOWS_CONTROLS))
        self.show_thresh = bool(cv2.getTrackbarPos("show_thresh", self.WINDOWS_CONTROLS))
        font      = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.25
        thickness = 1

        overlay = image.copy()
        
        # Loop over all patches
        if self.patches.contains_mahalanobis_distances:
            mahas = np.zeros(image.shape[:2], dtype=np.float64)

            patch_size = (float(image.shape[0]) / float(frame.shape[0]),
                          float(image.shape[1]) / float(frame.shape[1]))

            threshold = cv2.getTrackbarPos("threshold", self.WINDOWS_MAHA) / 10000.0
            show_thresh = cv2.getTrackbarPos("show_thresh", self.WINDOWS_MAHA)

            for y, x in np.ndindex(frame.shape):
                patch = frame[y, x]

                p1 = (int(x * patch_size[1]), int(y * patch_size[0]))               # (x, y)
                p2 = (int(p1[0] + patch_size[1]), int(p1[1] + patch_size[0]))       # (x, y)
                
                color = self.patch_to_color(patch)
                if color != (0,0,0):
                    cv2.rectangle(overlay, p1, p2, color, -1)

                if self.show_values:
                    text = "%.2f" % self.patch_to_text(patch)
                    cv2.putText(overlay, text,
                        (p1[0] + 2, p1[1] + int(patch_size[0]) - 2),    # (x, y)
                        font,
                        fontScale,
                        (0, 255, 0),
                        thickness, lineType=cv2.LINE_AA)

        # Draw grid
        if self.show_grid:
            relative_grid = self._ilu.absolute_to_relative(self._absolute_locations, frame.camera_locations[0, 0])
            image_grid = self._ilu.relative_to_image(relative_grid, image.shape[0], image.shape[1])

            in_image_filter = np.all([image_grid[...,0] > 0,
                                      image_grid[...,0] < image.shape[0],
                                      image_grid[...,1] > 0,
                                      image_grid[...,1] < image.shape[1]], axis=0)

            a = self._absolute_locations[in_image_filter]

            for i, p in enumerate(image_grid[in_image_filter]):
                pos = (int(p[1]), int(p[0]))

                cv2.circle(overlay, pos, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                
                cv2.putText(overlay, "%.1f / %.1f" % (a[i, 1], a[i, 0]),
                    (pos[0] + 3, pos[1] + 2),
                    font,
                    fontScale,
                    (255, 255, 255),
                    thickness, lineType=cv2.LINE_AA)

        cv2.ellipse(overlay, (320, 480), (280, 280), 0, 0, 360, (50,50,100), 3, lineType=cv2.LINE_AA)

        # Blend the overlay
        alpha = cv2.getTrackbarPos("overlay",self.WINDOWS_CONTROLS) / 100.0  # Transparency factor.
        image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw receptive field of hovered patch
        if self.patches.contains_features and self._mouse_image_x > -1 and self._mouse_image_y > -1:
            model = sorted(self.patches.mahalanobis_distances.dtype.names)[self.model_index - 1]
            fake = "fake" in model

            if "0.20" in model:
                cell_size = 0.2
            else:
                cell_size = 0.5
            
            cell_size = 0.2

            key = "%.2f" % cell_size
            if fake: key = "fake_" + key

            rf = self.patches.calculate_receptive_field(self._mouse_image_y + 0.5, self._mouse_image_x + 0.5,
                                                        scale_y=image.shape[0] / float(self.patches.image_size),
                                                        scale_x=image.shape[1] / float(self.patches.image_size), fake=fake)

            cv2.rectangle(image_new, (int(rf[0][1]), int(rf[0][0])), (int(rf[2][1]), int(rf[2][0])), (0,0,255), 2)
            
            if self.patches.contains_locations and "SpatialBin" in model:
                patch = frame[self._mouse_image_x, self._mouse_image_y]

                ### Draw projected RF (JUST A TEST OF PROJECTION)
                # rf_relative = self._ilu.image_to_relative(np.array(rf), image_width=image.shape[1], image_height=image.shape[0])
                # rf_absolute = self._ilu.relative_to_absolute(rf_relative, frame.camera_locations[0, 0])

                # rf_relative_calc = self._ilu.absolute_to_relative(rf_absolute, frame.camera_locations[0, 0])
                # rf_image = self._ilu.relative_to_image(rf_relative_calc, image.shape[0], image.shape[1])

                # cv2.rectangle(image_new, (int(rf_image[0][1]), int(rf_image[0][0])), (int(rf_image[2][1]), int(rf_image[2][0])), (255,0,0), 1)

                ### Highlight bins

                x_min, y_min, x_max, y_max = self.patches.get_extent(cell_size, fake=fake)

                # Create the bins
                bins_y = np.arange(y_min, y_max, cell_size)
                bins_x = np.arange(x_min, x_max, cell_size)

                current_bins = patch["bins_" + key]

                indices_y, indices_x = np.unravel_index(current_bins, self.patches.rasterizations[key].shape)

                absolute_locations_y = bins_y[indices_y] + cell_size / 2
                absolute_locations_x = bins_x[indices_x] + cell_size / 2

                absolute_locations = np.array((absolute_locations_y, absolute_locations_x)).T

                relative_grid = self._ilu.absolute_to_relative(absolute_locations, frame.camera_locations[0, 0])
                image_grid = self._ilu.relative_to_image(relative_grid, image.shape[0], image.shape[1])

                for i, p in enumerate(image_grid):
                    pos = (int(p[1]), int(p[0]))

                    color = (int((absolute_locations[i, 1] % (cell_size * 2)) / (cell_size * 2) * 255),
                             int((absolute_locations[i, 0] % (cell_size * 2)) / (cell_size * 2) * 255),
                             0)

                    cv2.circle(image_new, pos, 2, color, -1, lineType=cv2.LINE_AA)
                
                    # cv2.putText(image_new, "%.1f / %.1f" % (absolute_locations[i, 1], absolute_locations[i, 0]),
                    #     (pos[0] + 3, pos[1] + 2),
                    #     font,
                    #     fontScale,
                    #     (255, 255, 255),
                    #     thickness, lineType=cv2.LINE_AA)

        if self.metric.name != "patch" and self._labels is not None:
            cv2.putText(image_new, "Threshold:", (10, 370), font, 0.5, (200,200,200), 1, lineType=cv2.LINE_AA)
            cv2.putText(image_new, "%.2f" % self._thresh, (90, 370), font, 0.5, (200,200,200), 1, lineType=cv2.LINE_AA)

            label = self._labels[self.index]
            score = self._scores[self.index]
            cv2.putText(image_new, "Score:", (10, 400), font, 0.5, (200,200,200), 1, lineType=cv2.LINE_AA)
            color = (150, 150, 150)
            if score >= self._thresh and label == 2:     # Correct
                color = (0, 255, 0)
            elif score >= self._thresh and label == 1:   # False positive
                color = (0, 0, 255)
            elif score >= self._thresh and label == 0:   # Not a real False positive
                color = (140, 160, 140)
            elif score < self._thresh and label == 2:    # False negative
                color = (255, 0, 0)
            elif score < self._thresh and label == 0:    # Not a real False negative
                color = (160, 140, 140)

            cv2.putText(image_new, "%.2f" % score, (90, 400), font, 0.5, color, 1, lineType=cv2.LINE_AA)

        # Draw current label
        self.image_write_label(image_new, frame[0, 0])
        cv2.imshow(self.WINDOWS_IMAGE, image_new)
        
        # Write frame to video
        if self._video_writer is not None:
            self._video_writer.write(image_new)
    
    def __model__(self, new_model_index=None):
        self.model_index = new_model_index
        
        for metric in PatchArray.METRICS:
            metric.current_threshold = -1
        self.__maha__()

    def __index_update__(self, new_index=None):
        if new_index != self.index:
            self.pause = True
        self.index = new_index
        self.__draw__()

    def __change_frames__(self, *args):
        label        = cv2.getTrackbarPos("label", self.WINDOWS_CONTROLS)
        stop_label   = cv2.getTrackbarPos("stop_label", self.WINDOWS_CONTROLS)
        direction    = cv2.getTrackbarPos("direction", self.WINDOWS_CONTROLS)
        round_number = cv2.getTrackbarPos("round_number", self.WINDOWS_CONTROLS)
        error        = cv2.getTrackbarPos("error", self.WINDOWS_CONTROLS)

        self.patches = self.orig_patches

        if error == 1:   # False positive
            self.patches = self.patches[np.logical_and(self._labels == 1, self._scores >= self._thresh)]
        elif error == 2: # False negative
            self.patches = self.patches[np.logical_and(self._labels == 2, self._scores < self._thresh)]

        if label == 0:
            self.patches = self.patches.unknown_anomaly
        elif label == 1:
            self.patches = self.patches.no_anomaly
        elif label == 2:
            self.patches = self.patches.anomaly

        if stop_label == 0:
            self.patches = self.patches.stop_ok
        elif stop_label == 1:
            self.patches = self.patches.stop_dont
        elif stop_label == 2:
            self.patches = self.patches.stop_do

        if direction == 0:
            self.patches = self.patches.direction_unknown
        elif direction == 1:
            self.patches = self.patches.direction_ccw
        elif direction == 2:
            self.patches = self.patches.direction_cw

        if round_number != -1:
            self.patches = self.patches.round_number(round_number)

        cv2.setTrackbarPos("index", self.WINDOWS_CONTROLS, 0)
        cv2.setTrackbarMax("index", self.WINDOWS_CONTROLS, max(0, self.patches.shape[0] - 1))

        self.__maha__(only_refresh_image=True)

        self.__draw__()

    @property
    def metric(self):
        return PatchArray.METRICS[cv2.getTrackbarPos("metric", self.WINDOWS_MAHA)]

    @property
    def title(self):
        if self.patches.filename is None:
            return "Images"
        extractor = os.path.basename(self.patches.filename).replace(".h5", "")
        model = sorted(self.patches.mahalanobis_distances.dtype.names)[self.model_index - 1]

        sigma_0 = (cv2.getTrackbarPos("0_gaussian_0", self.WINDOWS_MAHA),
                   cv2.getTrackbarPos("0_gaussian_1", self.WINDOWS_MAHA),
                   cv2.getTrackbarPos("0_gaussian_2", self.WINDOWS_MAHA))
        if sigma_0 != (0, 0, 0):
            gauss_filter = "(%i, %i, %i)" % sigma_0
        else:
            gauss_filter = None

        other_filter = [None, "erosion", "dilation"][cv2.getTrackbarPos("1_erosion_dilation", self.WINDOWS_MAHA)]

        return "%s - %s + %s (%s, %s)" % (self.metric.name, extractor, model, gauss_filter, other_filter)
    
    def record(self):
        """ Start or stop recording to a video file """

        if self._video_writer is None: # Start recording
            folder = os.path.join(consts.BASE_PATH, "Videos")
            if not os.path.exists(folder):
                os.mkdir(folder)

            self._video_writer = cv2.VideoWriter(os.path.join(folder, "%s_%s.avi" % (self.title.replace("/", "_"), datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))), cv2.VideoWriter_fourcc(*'DIVX'), 6.0, (640, 480))
            self.__draw__()
        else: # Stop recording
            self._video_writer.release()
            self._video_writer = None

    def close(self):
        if self._video_writer is not None: # Stop video recording
            self.record()
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import consts
    patches = PatchArray().training_and_validation
    
    vis = Visualize(patches)
    vis.show()