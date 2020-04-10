import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from receptivefield.keras import KerasReceptiveField
from receptivefield.image import get_default_image
from Models.C3D.c3d import C2D
import efficientnet.tfkeras as efn

shape = (224, 224, 3)

model = efn.EfficientNetB0(input_shape=shape, include_top=False, weights=None)
# model = C2D(shape)

model.summary()

def model_build_func(input_shape):
    return model

# compute receptive field
rf = KerasReceptiveField(model_build_func, init_weights=False)

layers = list(filter(lambda y: y.name.endswith("conv"), model.layers))
layers_names = [x.name for x in layers]

rf_params = rf.compute(shape, 'input_1', layers_names)

for i, x in enumerate(rf_params):
    print("%16s %16s: %s" % (layers_names[i], layers[i].output_shape[1:], {"offset": x.rf.offset, "stride": x.rf.stride, "size": (x.rf.size.h, x.rf.size.w)}))

# debug receptive field
# rf.plot_rf_grids(get_default_image(shape, name='doge'))
# plt.show()

### EfficientNetB0 (224, 224, 3), RFs berechnet mit (528, 528, 3)
#stem_conv                      (112, 112, 32): {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (3, 3)}
#block1a_dwconv                 (112, 112, 32): {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (7, 7)}
#block1a_project_conv           (112, 112, 16): {'offset': (0.5, 0.5),     'stride': (2.0, 2.0),   'size': (7, 7)}
#block2a_expand_conv            (112, 112, 96): {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (7, 7)}
#block2a_dwconv                 (56, 56, 96):   {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (11, 11)}
#block2a_project_conv           (56, 56, 24):   {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (11, 11)}
#block2b_expand_conv            (56, 56, 144):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (11, 11)}
#block2b_dwconv                 (56, 56, 144):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (19, 19)}
#block2b_project_conv           (56, 56, 24):   {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (19, 19)}
#block3a_expand_conv            (56, 56, 144):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (19, 19)}
#block3a_dwconv                 (28, 28, 144):  {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (35, 35)}
#block3a_project_conv           (28, 28, 40):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (35, 35)}
#block3b_expand_conv            (28, 28, 240):  {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (35, 35)}
#block3b_dwconv                 (28, 28, 240):  {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (67, 67)}
#block3b_project_conv           (28, 28, 40):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (67, 67)}
#block4a_expand_conv            (28, 28, 240):  {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (67, 67)}
#block4a_dwconv                 (14, 14, 240):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (83, 83)}
#block4a_project_conv           (14, 14, 80):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (83, 83)}
#block4b_expand_conv            (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (83, 83)}
#block4b_dwconv                 (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (115, 115)}
#block4b_project_conv           (14, 14, 80):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (115, 115)}
#block4c_expand_conv            (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (115, 115)}
#block4c_dwconv                 (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (147, 147)}
#block4c_project_conv           (14, 14, 80):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (147, 147)}
#block5a_expand_conv            (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (147, 147)}
#block5a_dwconv                 (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (211, 211)}
#block5a_project_conv           (14, 14, 112):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (211, 211)}
#block5b_expand_conv            (14, 14, 672):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (211, 211)}
#block5b_dwconv                 (14, 14, 672):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (275, 275)}
#block5b_project_conv           (14, 14, 112):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (275, 275)}
#block5c_expand_conv            (14, 14, 672):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (275, 275)}
#block5c_dwconv                 (14, 14, 672):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (339, 339)}
#block5c_project_conv           (14, 14, 112):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (339, 339)}
#block6a_expand_conv            (14, 14, 672):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (339, 339)}
#block6a_dwconv                 (7, 7, 672):    {'offset': (15.5, 15.5),   'stride': (32.0, 32.0), 'size': (403, 403)}
#block6a_project_conv           (7, 7, 192):    {'offset': (15.5, 15.5),   'stride': (32.0, 32.0), 'size': (403, 403)}
#block6b_expand_conv            (7, 7, 1152):   {'offset': (15.5, 15.5),   'stride': (32.0, 32.0), 'size': (403, 403)}
#block6b_dwconv                 (7, 7, 1152):   {'offset': (20.0, 20.0),   'stride': (16.0, 16.0), 'size': (522, 522)}
#block6b_project_conv           (7, 7, 192):    {'offset': (20.0, 20.0),   'stride': (16.0, 16.0), 'size': (522, 522)}
#block6c_expand_conv            (7, 7, 1152):   {'offset': (20.0, 20.0),   'stride': (16.0, 16.0), 'size': (522, 522)}
# Ab hier wird es unplausibel
#block6c_dwconv                 (7, 7, 1152):   {'offset': (81.0, 81.0),   'stride': (0.0, 0.0),   'size': (528, 528)}
#block6c_project_conv           (7, 7, 192):    {'offset': (81.0, 81.0),   'stride': (0.0, 0.0),   'size': (528, 528)}
#block6d_expand_conv            (7, 7, 1152):   {'offset': (81.0, 81.0),   'stride': (0.0, 0.0),   'size': (528, 528)}
#block6d_dwconv                 (7, 7, 1152):   {'offset': (145.0, 145.0), 'stride': (0.0, 0.0),   'size': (528, 528)}
#block6d_project_conv           (7, 7, 192):    {'offset': (145.0, 145.0), 'stride': (0.0, 0.0),   'size': (528, 528)}
#block7a_expand_conv            (7, 7, 1152):   {'offset': (145.0, 145.0), 'stride': (0.0, 0.0),   'size': (528, 528)}
#block7a_dwconv                 (7, 7, 1152):   {'offset': (177.0, 177.0), 'stride': (0.0, 0.0),   'size': (528, 528)}
#block7a_project_conv           (7, 7, 320):    {'offset': (177.0, 177.0), 'stride': (0.0, 0.0),   'size': (528, 528)}
#top_conv                       (7, 7, 1280):   {'offset': (177.0, 177.0), 'stride': (0.0, 0.0),   'size': (528, 528)}

### EfficientNetB6 (528, 528, 3)
# stem_conv             (264, 264, 56):  {'offset': (1.5, 1.5),   'stride': (2.0, 2.0),   'size': (3, 3)}
# block1a_dwconv        (264, 264, 56):  {'offset': (1.5, 1.5),   'stride': (2.0, 2.0),   'size': (7, 7)}
# block1a_project_conv  (264, 264, 32):  {'offset': (1.5, 0.5),   'stride': (2.0, 2.0),   'size': (7, 7)}
# block1b_dwconv        (264, 264, 32):  {'offset': (1.5, 1.5),   'stride': (2.0, 2.0),   'size': (11, 11)}
# block1b_project_conv  (264, 264, 32):  {'offset': (1.5, 1.5),   'stride': (2.0, 2.0),   'size': (11, 11)}
# block1c_dwconv        (264, 264, 32):  {'offset': (1.5, 1.5),   'stride': (2.0, 2.0),   'size': (15, 15)}
# block1c_project_conv  (264, 264, 32):  {'offset': (1.5, 1.5),   'stride': (2.0, 2.0),   'size': (15, 15)}
# block2a_expand_conv   (264, 264, 192): {'offset': (1.5, 1.5),   'stride': (2.0, 2.0),   'size': (15, 15)}
# block2a_dwconv        (132, 132, 192): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (19, 19)}
# block2a_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (19, 19)}
# block2b_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (19, 19)}
# block2b_dwconv        (132, 132, 240): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (27, 27)}
# block2b_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (27, 27)}
# block2c_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (27, 27)}
# block2c_dwconv        (132, 132, 240): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (35, 35)}
# block2c_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (35, 35)}
# block2d_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (35, 35)}
# block2d_dwconv        (132, 132, 240): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (43, 43)}
# block2d_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (43, 43)}
# block2e_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (43, 43)}
# block2e_dwconv        (132, 132, 240): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (51, 51)}
# block2e_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (51, 51)}
# block2f_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (51, 51)}
# block2f_dwconv        (132, 132, 240): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (59, 59)}
# block2f_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (59, 59)}
# block3a_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),   'stride': (4.0, 4.0),   'size': (59, 59)}
# block3a_dwconv        (66, 66, 240):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (75, 75)}
# block3a_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (75, 75)}
# block3b_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (75, 75)}
# block3b_dwconv        (66, 66, 432):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (107, 107)}
# block3b_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (107, 107)}
# block3c_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (107, 107)}
# block3c_dwconv        (66, 66, 432):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (139, 139)}
# block3c_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (139, 139)}
# block3d_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (139, 139)}
# block3d_dwconv        (66, 66, 432):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (171, 171)}
# block3d_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (171, 171)}
# block3e_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (171, 171)}
# block3e_dwconv        (66, 66, 432):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (203, 203)}
# block3e_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (203, 203)}
# block3f_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (203, 203)}
# block3f_dwconv        (66, 66, 432):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (235, 235)}
# block3f_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (235, 235)}
# block4a_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),   'stride': (8.0, 8.0),   'size': (235, 235)}
# block4a_dwconv        (33, 33, 432):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (251, 251)}
# block4a_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (251, 251)}
# block4b_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (251, 251)}
# block4b_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (283, 283)}
# block4b_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (283, 283)}
# block4c_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (283, 283)}
# block4c_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (315, 315)}
# block4c_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (315, 315)}
# block4d_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (315, 315)}
# block4d_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (347, 347)}
# block4d_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (347, 347)}
# block4e_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (347, 347)}
# block4e_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (379, 379)}
# block4e_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (379, 379)}
# block4f_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (379, 379)}
# block4f_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (411, 411)}
# block4f_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (411, 411)}
# block4g_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (411, 411)}
# block4g_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (443, 443)}
# block4g_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (443, 443)}
# block4h_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (443, 443)}
# block4h_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (475, 475)}
# block4h_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (475, 475)}
# block5a_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (475, 475)}
# Ab hier wird es unplausibel
# block5a_dwconv        (33, 33, 864):   {'offset': (22.0, 22.0), 'stride': (8.0, 8.0),   'size': (526, 526)}
# block5a_project_conv  (33, 33, 200):   {'offset': (22.0, 22.0), 'stride': (8.0, 8.0),   'size': (526, 526)}
# block5b_expand_conv   (33, 33, 1200):  {'offset': (22.0, 22.0), 'stride': (8.0, 8.0),   'size': (526, 526)}
# block5b_dwconv        (33, 33, 1200):  {'offset': (53.0, 53.0), 'stride': (0.0, 0.0),   'size': (528, 528)}
# block5b_project_conv  (33, 33, 200):   {'offset': (53.0, 53.0), 'stride': (0.0, 0.0),   'size': (528, 528)}
# ...
# top_conv              (17, 17, 2304):  {'offset': (264.0, 264.0), 'stride': (0.0, 0.0), 'size': (528, 528)}


### C3D (C2D, mit (112, 112, 3))
# conv1           (112, 112, 64):  {'offset': (0.5, 0.5), 'stride': (1.0, 1.0),   'size': (3, 3)}
# pool1           (56, 56, 64):    {'offset': (0.5, 0.5), 'stride': (2.0, 2.0),   'size': (3, 3)}
# conv2           (56, 56, 128):   {'offset': (0.5, 0.5), 'stride': (2.0, 2.0),   'size': (7, 7)}
# pool2           (28, 28, 128):   {'offset': (0.5, 0.5), 'stride': (4.0, 4.0),   'size': (7, 7)}
# conv3a          (28, 28, 256):   {'offset': (0.5, 0.5), 'stride': (4.0, 4.0),   'size': (15, 15)}
# conv3b          (28, 28, 256):   {'offset': (0.5, 0.5), 'stride': (4.0, 4.0),   'size': (23, 23)}
# pool3           (14, 14, 256):   {'offset': (0.5, 0.5), 'stride': (8.0, 8.0),   'size': (23, 23)}
# conv4a          (14, 14, 512):   {'offset': (0.5, 0.5), 'stride': (8.0, 8.0),   'size': (39, 39)}
# conv4b          (14, 14, 512):   {'offset': (0.5, 0.5), 'stride': (8.0, 8.0),   'size': (55, 55)}
# pool4           (7, 7, 512):     {'offset': (0.5, 0.5), 'stride': (16.0, 16.0), 'size': (55, 55)}
# conv5a          (7, 7, 512):     {'offset': (0.5, 0.5), 'stride': (16.0, 16.0), 'size': (87, 87)}
# conv5b          (7, 7, 512):     {'offset': (0.5, 0.5), 'stride': (16.0, 16.0), 'size': (119, 119)}

### VGG16 (etwas merkwürdige Ergebnisse)
# block3_conv3    (56, 56, 256):   {'offset': (-1.5, 0.5), 'stride': (4.0, 4.0),   'size': (35, 37)}
# block4_conv3    (28, 28, 512):   {'offset': (-3.0, 0.5), 'stride': (8.0, 8.0),   'size': (82, 85)}
# block5_conv3    (14, 14, 512):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (181, 181)}
### Manuell korrigiert (plausibler so)
# block3_conv3    (56, 56, 256):   {'offset': (0.5, 0.5),  'stride': (4.0, 4.0),   'size': (37, 37)}
# block4_conv3    (28, 28, 512):   {'offset': (0.0, 0.5),  'stride': (8.0, 8.0),   'size': (85, 85)}
# block5_conv3    (14, 14, 512):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (181, 181)}

### ResNet50V2 (224, 224, 3)
# conv2_block1_out (56, 56, 256):   {'offset': (-1.5, -1.5),  'stride': (4.0, 4.0),   'size': (15, 15)}
# conv2_block2_out (56, 56, 256):   {'offset': (-1.5, -1.5),  'stride': (4.0, 4.0),   'size': (23, 23)}
# conv2_block3_out (28, 28, 256):   {'offset': (-1.5, -1.5),  'stride': (8.0, 8.0),   'size': (31, 31)}
# conv3_block1_out (28, 28, 512):   {'offset': (-1.5, -1.5),  'stride': (8.0, 8.0),   'size': (47, 47)}
# conv3_block2_out (28, 28, 512):   {'offset': (-1.5, -1.5),  'stride': (8.0, 8.0),   'size': (63, 63)}
# conv3_block3_out (28, 28, 512):   {'offset': (-1.5, -1.5),  'stride': (8.0, 8.0),   'size': (79, 79)}
# conv3_block4_out (14, 14, 512):   {'offset': (-1.5, -1.5),  'stride': (16.0, 16.0), 'size': (95, 95)}
# conv4_block1_out (14, 14, 1024):  {'offset': (-1.5, -1.5),  'stride': (16.0, 16.0), 'size': (127, 127)}
# conv4_block2_out (14, 14, 1024):  {'offset': (-1.5, -1.5),  'stride': (16.0, 16.0), 'size': (159, 159)}
# conv4_block3_out (14, 14, 1024):  {'offset': (-1.0, -1.0),  'stride': (15.5, 15.5), 'size': (190, 190)}
# conv4_block4_out (14, 14, 1024):  {'offset': (7.0, 7.0),    'stride': (8.0, 8.0),   'size': (206, 206)}
# conv4_block5_out (14, 14, 1024):  {'offset': (15.0, 15.0),  'stride': (0.0, 0.0),   'size': (222, 222)}
# conv4_block6_out (7, 7, 1024):    {'offset': (31.0, 31.0),  'stride': (0.0, 0.0),   'size': (222, 222)}
# conv5_block1_out (7, 7, 2048):    {'offset': (63.0, 63.0),  'stride': (0.0, 0.0),   'size': (222, 222)}
# conv5_block2_out (7, 7, 2048):    {'offset': (95.0, 95.0),  'stride': (0.0, 0.0),   'size': (222, 222)}
# conv5_block3_out (7, 7, 2048):    {'offset': (111.0, 111.0), 'stride': (0.0, 0.0),  'size': (222, 222)}

### ResNet50V2 (449, 449, 3)
# conv2_block1_out (113, 113, 256): {'offset': (-1.5, -1.5), 'stride': (4.0, 4.0),   'size': (15, 15)}
# conv2_block2_out (113, 113, 256): {'offset': (-1.5, -1.5), 'stride': (4.0, 4.0),   'size': (23, 23)}
# conv2_block3_out (57, 57, 256):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (31, 31)}
# conv3_block1_out (57, 57, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (47, 47)}
# conv3_block2_out (57, 57, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (63, 63)}
# conv3_block3_out (57, 57, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (79, 79)}
# conv3_block4_out (29, 29, 512):   {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (95, 95)}
# conv4_block1_out (29, 29, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (127, 127)}
# conv4_block2_out (29, 29, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (159, 159)}
# conv4_block3_out (29, 29, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (191, 191)}
# conv4_block4_out (29, 29, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (223, 223)}
# conv4_block5_out (29, 29, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (255, 255)}
# conv4_block6_out (15, 15, 1024):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (287, 287)}
# conv5_block1_out (15, 15, 2048):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (351, 351)}
# conv5_block2_out (15, 15, 2048):  {'offset': (-1.5, -1.5), 'stride': (25.5, 25.5), 'size': (415, 415)}
# conv5_block3_out (15, 15, 2048):  {'offset': (13.5, 13.5), 'stride': (7.5, 7.5),   'size': (449, 449)}

### ResNet50V2 (673, 673, 3)
# conv2_block1_out (169, 169, 256): {'offset': (-1.5, -1.5), 'stride': (4.0, 4.0),   'size': (15, 15)}
# conv2_block2_out (169, 169, 256): {'offset': (-1.5, -1.5), 'stride': (4.0, 4.0),   'size': (23, 23)}
# conv2_block3_out (85, 85, 256):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (31, 31)}
# conv3_block1_out (85, 85, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (47, 47)}
# conv3_block2_out (85, 85, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (63, 63)}
# conv3_block3_out (85, 85, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (79, 79)}
# conv3_block4_out (43, 43, 512):   {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (95, 95)}
# conv4_block1_out (43, 43, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (127, 127)}
# conv4_block2_out (43, 43, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (159, 159)}
# conv4_block3_out (43, 43, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (191, 191)}
# conv4_block4_out (43, 43, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (223, 223)}
# conv4_block5_out (43, 43, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (255, 255)}
# conv4_block6_out (22, 22, 1024):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (287, 287)}
# conv5_block1_out (22, 22, 2048):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (351, 351)}
# conv5_block2_out (22, 22, 2048):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (415, 415)}
# conv5_block3_out (22, 22, 2048):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (479, 479)

### MobileNetV2
# block_2_add      (56, 56, 24):    {'offset': (3.5, 3.5), 'stride': (4.0, 4.0), 'size': (19, 19)}
# block_4_add      (28, 28, 32):    {'offset': (7.5, 7.5), 'stride': (8.0, 8.0), 'size': (43, 43)}
# block_5_add      (28, 28, 32):    {'offset': (7.5, 7.5), 'stride': (8.0, 8.0), 'size': (59, 59)}
# block_7_add      (14, 14, 64):    {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (107, 107)}
# block_8_add      (14, 14, 64):    {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (139, 139)}
# block_9_add      (14, 14, 64):    {'offset': (15.5, 15.5), 'stride': (16.0, 16.0), 'size': (171, 171)}
# block_11_add     (14, 14, 96):    {'offset': (21.0, 21.0), 'stride': (5.0, 5.0), 'size': (224, 224)}
# block_12_add     (14, 14, 96):    {'offset': (37.0, 37.0), 'stride': (0.0, 0.0), 'size': (224, 224)}
# block_14_add     (7, 7, 160):     {'offset': (101.0, 101.0), 'stride': (0.0, 0.0), 'size': (224, 224)}
# block_15_add     (7, 7, 160):     {'offset': (112.0, 112.0), 'stride': (0.0, 0.0), 'size': (224, 224)}

### MobileNetV2 (673, 673, 3)
# block_2_add      (169, 169, 24):  {'offset': (0.5, -0.5), 'stride': (4.0, 4.0), 'size': (19, 19)}
# block_4_add      (85, 85, 32):    {'offset': (0.5, 0.5), 'stride': (8.0, 8.0), 'size': (43, 43)}
# block_5_add      (85, 85, 32):    {'offset': (-1.5, 0.5), 'stride': (8.0, 8.0), 'size': (59, 59)}
# block_7_add      (43, 43, 64):    {'offset': (0.5, 0.5), 'stride': (16.0, 16.0), 'size': (107, 107)}
# block_8_add      (43, 43, 64):    {'offset': (0.5, 0.5), 'stride': (16.0, 16.0), 'size': (139, 139)}
# block_9_add      (43, 43, 64):    {'offset': (0.5, 0.5), 'stride': (16.0, 16.0), 'size': (171, 171)}
# block_11_add     (43, 43, 96):    {'offset': (0.5, 0.5), 'stride': (16.0, 16.0), 'size': (235, 235)}
# block_12_add     (43, 43, 96):    {'offset': (0.5, 0.5), 'stride': (16.0, 16.0), 'size': (267, 267)}
# block_14_add     (22, 22, 160):   {'offset': (0.5, 0.5), 'stride': (32.0, 32.0), 'size': (363, 363)}
# block_15_add     (22, 22, 160):   {'offset': (0.5, 0.5), 'stride': (32.0, 32.0), 'size': (427, 427)}