import os
import logging
import h5py
import cv2

import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import feature_extractor.utils as utils
from IPM import IPM, _DictObjHolder

class AnomalyModelBase(object):
    
    def __init__(self):
        self.NAME = ""          # Should be set by the implementing class
    
    def generate_model(self, metadata, features):
        """Generate a model based on the features and metadata
        
        Args:
            metadata (list): Array of metadata for the features
            features (list): Array of features as extracted by a FeatureExtractor
        """
        raise NotImplementedError
        
    def classify(self, feature_vector):
        """ Classify a single feature vector based on the loaded model """
        raise NotImplementedError
    
    def load_model_from_file(self, model_file):
        """ Load a model from file """
        raise NotImplementedError

    def save_model_to_file(self, output_file):
        """Save the model to output_file
        
        Args:
            output_file (str): Output path for the model file
        """
        raise NotImplementedError

    ########################
    # Common functionality #
    ########################

    def generate_model_from_file(self, features_file, output_file = ""):
        """Generate a model based on the features in features_file and save it to output_file
        
        Args:
            features_file (str) : HDF5 or TFRecord file containing metadata and features (see feature_extractor for details)
            output_file (str): Output path for the model file (same path as features_file if not specified)
        """
        # Check parameters
        if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
            raise ValueError("Specified model file does not exist (%s)" % features_file)
        
        if output_file == "":
            output_file = os.path.abspath(features_file.replace(".h5", "")) + "." + self.NAME + ".h5"
            logging.info("Output file set to %s" % output_file)
        
        # Read file
        metadata, features = utils.read_features_file(features_file)

        # Only take feature vectors of images labeled as anomaly free (label == 1)
        features = features[[m["label"] == 1 for m in metadata]]
        metadata = metadata[[m["label"] == 1 for m in metadata]]
        
        # Generate model
        if self.generate_model(metadata, features) == False:
            logging.info("Could not generate model.")
            return False

        # Save model
        self.save_model_to_file(output_file)
        
        return True

    def reduce_feature_array(self, features_vector_array):
        """Reduce an array of feature vectors of shape to a simple list
        
        Args:
            features_vector_array (object[]): feature vectors array
        """
        # Create an array of only the feature vectors, eg. (25000, 1280)
        return features_vector_array.reshape(-1, features_vector_array.shape[-1])

    def visualize(self, metadata, features, feature_to_color_func, feature_to_text_func=None, pause_func=None):
        total, height, width, depth = features.shape
        w = 640
        h = 480
        src = np.float32([[ 0, 0], [w, 0], [ 0,   h], [w,   h]])
        dst = np.float32([[-3, 6], [3, 6], [-1, 0.7], [1, 0.7]])
        P = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        Pinv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation
        print P
        cv2.namedWindow('image')

        def nothing(x):
            pass

        # create trackbars for color change
        cv2.createTrackbar('f_x',           'image', 619        , 1000, nothing)
        cv2.createTrackbar('f_y',           'image', 619        , 1000, nothing)
        cv2.createTrackbar('camera_height', 'image', 1555       , 3000, nothing)
        cv2.createTrackbar('pitch',         'image', 37         , 360 , nothing)
        cv2.createTrackbar('x',             'image', 640        , 640, nothing)
        cv2.createTrackbar('y',             'image', 480        , 480, nothing)
        cv2.createTrackbar('index',             'image', 1        , 2, nothing)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.3
        thickness              = 1

        tfrecord = None
        tfrecordCounter = 0

        pause = False

        # Find homography
        h, mask = cv2.findHomography(np.array([(0, 0), (640, 0), (0, 480), (640, 480)]), np.array([(-3, 6), (3, 6), (-1, 0.7), (1, 0.7)]), cv2.RANSAC)
        
        for i, feature_3d in enumerate(features):
            meta = metadata[i]
            
            if tfrecord != meta["tfrecord"]:
                tfrecord = meta["tfrecord"]
                tfrecordCounter = 0
                image_dataset = utils.load_dataset(meta["tfrecord"]).as_numpy_iterator()
            else:
                tfrecordCounter += 1

            image, example = next(image_dataset)
            # if meta["label"] == 1:
            #     continue

            overlay = image.copy()

            # if example["metadata/time"] != meta["time"]:
            #     logging.error("Times somehow don't match (%f)" % (example["metadata/time"]- meta["time"]))

            patch_size = (image.shape[1] / width, image.shape[0] / height)
            
            # f = plt.figure()
            # ax = f.gca(projection='3d')
            # ax.set_xlim(0,10);ax.set_ylim(0,10);ax.set_zlim(0,10)
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            # w = 10
            # ax.set_xlim(-w,w);ax.set_ylim(-w,w);ax.set_zlim(-w,w)
            # ax.view_init(elev=5, azim=100)
            # fig, ax = plt.subplots()

            # def ipm(image, f_x, f_y, camera_height, pitch, point):
            #     w = image.shape[0]
            #     h = image.shape[1]

            #     # # Projecion matrix 2D -> 3D
            #     # A = np.array([[1   , 0   , -w/2],
            #     #               [0   , 1   , -h/2],
            #     #               [0   , 0   , 0   ],
            #     #               [0   , 0   , 1   ]])

            #     _cp = np.cos((pitch) * np.pi / 180.)
            #     _sp = np.sin((pitch) * np.pi / 180.)

            #     d = abs((camera_height * (_sp + f_x * _cp) / (f_x * _sp - _cp))) + 1
            #     x = camera_height * ((point[0] * _sp + f_x * _cp) / (- point[1] * _cp + f_x * _sp)) + d
            #     y = camera_height * ((point[1] * _sp + f_x * _cp) / (- point[1] * _cp + f_x * _sp)) + d
            #     print (x, y)
            #     # # R - Rotation matrix
            #     # R = np.array([[1   , 0   , 0   ],
            #     #               [0   , _cp , -_sp],
            #     #               [0   , _sp , _cp ]])
                
            #     R = Rotation.from_euler('x', pitch, degrees=True).as_dcm()
                
            #     # T - translation matrix
            #     t = np.array([[0],
            #                   [0],
            #                   [camera_height]])
                
            #     # K - intrinsic matrix
            #     K = np.array([[f_x, 0  , w/2, 0],
            #                   [0  , f_y, h/2, 0],
            #                   [0  , 0  , 1  , 0]]).astype(np.float) # 4x3 intrinsic perspective projection matrix

            #     # K = np.array([[619.66259765625, 0.0              , 324.51873779296875], 
            #     #               [0.0            , 619.9126586914062, 236.8967742919922 ], 
            #     #               [0.0            , 0.0              , 1.0               ]])

            #     # P = K.dot(np.hstack((R,t)))

            #     # C = np.array([0., 0., camera_height])

            #     # X = np.dot(np.linalg.pinv(P), point)
            #     # X = X / X[3]
                
            #     # xvec = C - (X[:3] * (C[2] / X[2]))

            #     # ax.quiver(C[0], C[1], C[2], xvec[0], xvec[1], xvec[2],color='red')
            #     # ax.quiver(x, y, 0, x, y, 0,color='blue')
            #     ax.plot(x, y, 'o')


            #     return None#round(xvec[cv2.getTrackbarPos('index','image')], 2)#cv2.warpPerspective(image, P, (640, 480), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)

            # https://gist.github.com/anujonthemove/7b35b7c1e05f01dd11d74d94784c1e58
            def ipm(image, f_x, f_y, camera_height, pitch, point):
                w = image.shape[1]
                h = image.shape[0]

                # Projecion matrix 2D -> 3D
                # A = np.array([[1   , 0   , -w/2],
                #               [0   , 1   , -h/2],
                #               [0   , 0   , 0   ],
                #               [0   , 0   , 1   ]])

                # _cp = np.cos((90 - pitch) * np.pi / 180.)
                # _sp = np.sin((90 - pitch) * np.pi / 180.)

                # # R - Rotation matrix
                # R = np.array([[1   , 0   , 0   , 0],
                #               [0   , _cp , -_sp, 0],
                #               [0   , _sp , _cp , 0],
                #               [0   , 0   , 0   , 1]])
                
                # # T - translation matrix
                # T = np.array([[1   , 0   , 0   , 0],
                #               [0   , 1   , 0   , 0],
                #               [0   , 0   , 1   , camera_height],
                #               [0   , 0   , 0   , 1]])
                
                # # K - intrinsic matrix
                # K = np.array([[f_x, 0  , w/2, 0],
                #               [0  , f_y, h/2, 0],
                #               [0  , 0  , 1  , 0]]).astype(np.float) # 4x3 intrinsic perspective projection matrix

                # P = K.dot(T.dot(R.dot(A)))
                # Pinv = np.linalg.pinv(P)
                src = np.float32([[ 0, 0], [w, 0], [ 0,   h], [w,   h]])
                dst = np.float32([[-3, 6], [3, 6], [-1, 0.7], [1, 0.7]])
                P = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
                
                for x in range(width):
                    for y in range(height):
                        feature = feature_3d[y,x,:]
                        p1 = (x * patch_size[0], y * patch_size[1])
                        p2 = (p1[0] + patch_size[0], p1[1] + patch_size[1])

                        point = np.array([p1[0], p1[1], 1.])

                        transfromed_point = P.dot(point)
                        transfromed_point = transfromed_point / transfromed_point[2] # Normalize by third component

                        text = str(round(transfromed_point[cv2.getTrackbarPos('index','image')], 2))
                        cv2.putText(overlay, text,
                            (p1[0] + 2, p1[1] + patch_size[1] - 2),
                            font,
                            fontScale,
                            (0, 0, 255),
                            thickness)

                return cv2.warpPerspective(image, P, (640, 480))

            f_x           = cv2.getTrackbarPos('f_x','image')           # focal length x
            f_y           = cv2.getTrackbarPos('f_y','image')           # focal length y
            camera_height = cv2.getTrackbarPos('camera_height','image') # camera height in `mm`
            pitch         = cv2.getTrackbarPos('pitch','image')         # rotation degree around x
            point   = np.array([cv2.getTrackbarPos('x','image'), cv2.getTrackbarPos('y','image'), ])

            # for x in range(width):
            #     for y in range(height):
            #         feature = feature_3d[y,x,:]
            #         p1 = (x * patch_size[0], y * patch_size[1])
            #         p2 = (p1[0] + patch_size[0], p1[1] + patch_size[1])
            #         cv2.rectangle(overlay, p1, p2, feature_to_color_func(feature), -1)

            #         if feature_to_text_func is not None:
            #             text = str(feature_to_text_func(feature))
            #             cv2.putText(overlay, text,
            #                 (p1[0] + 2, p1[1] + patch_size[1] - 2),
            #                 font,
            #                 fontScale,
            #                 (0, 0, 255),
            #                 thickness)
                    
            #         if pause_func is not None and pause_func(feature):
            #             pause = True

            # plt.show()

            alpha = 0.4  # Transparency factor.

            image_rect = ipm(image, f_x, f_y, camera_height, pitch, point)

            # Following line overlays transparent overlay over the image
            image_new_rect = cv2.addWeighted(overlay, alpha, image_rect, 1 - alpha, 0)
            
            utils.image_write_label(image_new_rect, meta["label"])
            
            # Use homography
            # image_new_rect = cv2.warpPerspective(image_new, h, (640, 480))

            # print image_new_rect.shape

            # fig = plt.figure()
            # ax = fig.add_subplot(211)
            # ax.imshow(image_new)
            # ax = fig.add_subplot(212)
            # ax.imshow(image_new_rect)
            # plt.show()
            cv2.imshow("image", image_new_rect)
            key = cv2.waitKey(0 if pause else 1)
            if key == 27:   # [esc] => Quit
                return None
            elif key != -1:
                pause = not pause
        
        cv2.destroyAllWindows()