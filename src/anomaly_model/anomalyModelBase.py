import os

class AnomalyModelBase(object):
    
    def __init__(self):
        self.name = ""          # Should be set by the implementing class
    
    def generate_model(self, features_file, model_file):
        pass  

    def classify(self, feature_vector):
        pass

    def generate_model(self, model_file):
        pass  
