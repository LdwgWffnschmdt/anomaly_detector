import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import feature_extractor.utils as utils

class AnomalyModelTest(object):
    
    def __init__(self, model, features_file="/home/ludwig/ros/src/ROS-kate_bag/bags/real/TFRecord/Features/MobileNetV2_Block6.h5",
                              model_file="", add_locations_to_features=False):
        self.model = model

        logging.info("Initializing test for anomaly model: %s" % model.NAME)

        self.file_content_flat = None

        self.mahalanobis_no_anomaly = None
        self.mahalanobis_anomaly = None
        self.mahalanobis_no_anomaly_max = None
        self.mahalanobis_anomaly_max    = None
        self.mahalanobis_max = None
        
        if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
            raise ValueError("Features file does not exist (%s)" % features_file)
                
        # Read file
        self.file_content = utils.read_features_file(features_file)
        
        if model_file == "":
            model_file = os.path.abspath(features_file.replace(".h5", "")) + "." + model.NAME + ".h5"
        
        # Load or generate model if it does not exist yet
        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            if model.generate_model(self.file_content.no_anomaly.metadata, self.file_content.no_anomaly.features) == False:
                # Save model
                model.save_model_to_file(os.path.abspath(model_file))
            else:
                logging.error("Could not generate model.")
                return
        else:
            model.load_model_from_file(model_file)
        
        # Add location to features (AFTER model generation, otherwise they would be added twice!)
        if add_locations_to_features:
            self.file_content.all.features        = utils.add_location_to_features(self.file_content.all.metadata,        self.file_content.all.features)
            self.file_content.no_anomaly.features = utils.add_location_to_features(self.file_content.no_anomaly.metadata, self.file_content.no_anomaly.features)
            self.file_content.anomaly.features    = utils.add_location_to_features(self.file_content.anomaly.metadata,    self.file_content.anomaly.features)

    
    def getFlattened(self):
        """ Calculate the flattened version of file_content """
        logging.info("Flattening metadata and features")
        self.file_content_flat = utils._DictObjHolder({
            "all": utils._DictObjHolder({
                "metadata": utils.flatten(self.file_content.all.metadata),
                "features": utils.flatten(self.file_content.all.features)
            }),
            "no_anomaly": utils._DictObjHolder({
                "metadata": utils.flatten(self.file_content.no_anomaly.metadata),
                "features": utils.flatten(self.file_content.no_anomaly.features)
            }),
            "anomaly": utils._DictObjHolder({
                "metadata": utils.flatten(self.file_content.anomaly.metadata),
                "features": utils.flatten(self.file_content.anomaly.features)
            })
        })

    def calculateMahalobisDistances(self):
        """ Calculate all the Mahalanobis distances """
        if self.file_content_flat is None:
            self.getFlattened()
        
        logging.info("Calculating Mahalanobis distances of %i features with and %i features without anomalies" % \
            (len(self.file_content_flat.no_anomaly.features), len(self.file_content_flat.anomaly.features)))

        self.mahalanobis_no_anomaly = np.array(list(map(self.model._mahalanobis_distance, self.file_content_flat.no_anomaly.features))) # 75.49480115577167
        self.mahalanobis_anomaly    = np.array(list(map(self.model._mahalanobis_distance, self.file_content_flat.anomaly.features)))   # 76.93620254133627

        self.mahalanobis_no_anomaly_max = np.amax(self.mahalanobis_no_anomaly)
        self.mahalanobis_anomaly_max    = np.amax(self.mahalanobis_anomaly)
        self.mahalanobis_max = max(self.mahalanobis_no_anomaly_max, self.mahalanobis_anomaly_max)

        logging.info("Maximum Mahalanobis distance (no anomaly): %f" % self.mahalanobis_no_anomaly_max)
        logging.info("Maximum Mahalanobis distance (anomaly)   : %f" % self.mahalanobis_anomaly_max)

    def showMahalanobisDistribution(self):
        """ Plot the distribution of all Mahalanobis distances """
        logging.info("Showing Mahalanobis distance distribution")
        if self.mahalanobis_no_anomaly is None:
            self.calculateMahalobisDistances()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.set_title("No anomaly (%i)" % len(self.file_content_flat.no_anomaly.features))
        ax1.hist(self.mahalanobis_no_anomaly, bins=50)

        ax2.set_title("Anomaly (%i)" % len(self.file_content_flat.anomaly.features))
        ax2.hist(self.mahalanobis_anomaly, bins=50)

        fig.suptitle("Mahalanobis distances")

        plt.show()

    def visualize(self, threshold=None, feature_to_color_func=None, feature_to_text_func=None, pause_func=None):
        """ Visualize the result of a anomaly model """
        def _default_feature_to_color(feature):
            b = 0#100 if feature in self.model.normal_distribution else 0
            g = 0
            #r = self.model._mahalanobis_distance(feature) * (255 / threshold)
            r = 100 if self.model._mahalanobis_distance(feature) > threshold else 0
            return (b, g, r)

        def _default_feature_to_text(feature):
            return round(self.model._mahalanobis_distance(feature), 2)

        if threshold is None:
            if self.mahalanobis_max is None:
                self.calculateMahalobisDistances()
            threshold = self.mahalanobis_max * 0.9
        
        if feature_to_color_func is None:
            feature_to_color_func = _default_feature_to_color

        if feature_to_text_func is None:
            feature_to_text_func = _default_feature_to_text

        utils.visualize(self.file_content.all.metadata, self.file_content.all.features,
                        feature_to_color_func=feature_to_color_func,
                        feature_to_text_func=feature_to_text_func,
                        pause_func=pause_func)