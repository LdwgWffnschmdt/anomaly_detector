import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from receptivefield.keras import KerasReceptiveField
from receptivefield.image import get_default_image
from Models.C3D.c3d import C2D
import efficientnet.tfkeras as efn

shape = (449, 449, 3)

model = tf.keras.applications.VGG16(input_shape=shape, include_top=False)
# model = efn.EfficientNetB6(input_shape=shape, include_top=False, weights=None)
# model = C2D(shape)

model.summary()

def model_build_func(input_shape):
    return model

# compute receptive field
rf = KerasReceptiveField(model_build_func)

# layers = list(filter(lambda y: (y.name.endswith("add")) and y.output_shape[1] < 65 and y.output_shape[1] > 1, model.layers))
layers = model.layers[1:]#  list(filter(lambda y: y.name.endswith("project_BN") or y.name.endswith("add"), model.layers))
layers_names = [x.name for x in layers]

rf_params = rf.compute(shape, 'input_1', layers_names)

for i, x in enumerate(rf_params):
    print("%16s %16s: %s" % (layers_names[i], layers[i].output_shape[1:], {"offset": x.rf.offset, "stride": x.rf.stride, "size": (x.rf.size.h, x.rf.size.w)}))

# debug receptive field
# rf.plot_rf_grids(get_default_image(shape, name='doge'))
# plt.show()

### EfficientNetB0 (224, 224, 3), RFs berechnet mit (528, 528, 3)
# stem_conv             (112, 112, 32): {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (3, 3)}
# block1a_dwconv        (112, 112, 32): {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (7, 7)}
# block1a_project_conv  (112, 112, 16): {'offset': (0.5, 0.5),     'stride': (2.0, 2.0),   'size': (7, 7)}
# block2a_expand_conv   (112, 112, 96): {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (7, 7)}
# block2a_dwconv        (56, 56, 96):   {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (11, 11)}
# block2a_project_conv  (56, 56, 24):   {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (11, 11)}
# block2b_expand_conv   (56, 56, 144):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (11, 11)}
# block2b_dwconv        (56, 56, 144):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (19, 19)}
# block2b_project_conv  (56, 56, 24):   {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (19, 19)}
# block2b_add           (56, 56, 24):   {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (19, 19)} ###  -> EfficientNetB0_Block2  (75264) ? Keine guten Ergebnisse zu erwarten
# block3a_expand_conv   (56, 56, 144):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (19, 19)}
# block3a_dwconv        (28, 28, 144):  {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (35, 35)}
# block3a_project_conv  (28, 28, 40):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (35, 35)}
# block3b_expand_conv   (28, 28, 240):  {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (35, 35)}
# block3b_dwconv        (28, 28, 240):  {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (67, 67)}
# block3b_project_conv  (28, 28, 40):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (67, 67)}
# block3b_add           (28, 28, 40):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (67, 67)} ###  -> EfficientNetB0_Block3  (31360)
# block4a_expand_conv   (28, 28, 240):  {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (67, 67)}
# block4a_dwconv        (14, 14, 240):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (83, 83)}
# block4a_project_conv  (14, 14, 80):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (83, 83)}
# block4b_expand_conv   (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (83, 83)}
# block4b_dwconv        (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (115, 115)}
# block4b_project_conv  (14, 14, 80):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (115, 115)}
# block4b_add           (14, 14, 80):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (115, 115)}
# block4c_expand_conv   (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (115, 115)}
# block4c_dwconv        (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (147, 147)}
# block4c_project_conv  (14, 14, 80):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (147, 147)}
# block4c_add           (14, 14, 80):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (147, 147)} ### -> EfficientNetB0_Block4 (15680)
# block5a_expand_conv   (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (147, 147)}
# block5a_dwconv        (14, 14, 480):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (211, 211)}
# block5a_project_conv  (14, 14, 112):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (211, 211)}
# block5b_expand_conv   (14, 14, 672):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (211, 211)}
# block5b_dwconv        (14, 14, 672):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (275, 275)}
# block5b_project_conv  (14, 14, 112):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (275, 275)}
# block5b_add           (14, 14, 112):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (275, 275)}
# block5c_expand_conv   (14, 14, 672):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (275, 275)}
# block5c_dwconv        (14, 14, 672):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (339, 339)}
# block5c_project_conv  (14, 14, 112):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (339, 339)}
# block5c_add           (14, 14, 112):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (339, 339)} ### -> EfficientNetB0_Block5 (21952)
# block6a_expand_conv   (14, 14, 672):  {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (339, 339)}
# Ab hier mit (1024, 1024, 3) berechnet
# block6a_dwconv        (7, 7, 672):    {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (403, 403)}
# block6a_project_conv  (7, 7, 192):    {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (403, 403)}
# block6b_expand_conv   (7, 7, 1152):   {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (403, 403)}
# block6b_dwconv        (7, 7, 1152):   {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (531, 531)}
# block6b_project_conv  (7, 7, 192):    {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (531, 531)}
# block6b_add           (7, 7, 192):    {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (531, 531)}
# block6c_expand_conv   (7, 7, 1152):   {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (531, 531)}
# block6c_dwconv        (7, 7, 1152):   {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (659, 659)}
# block6c_project_conv  (7, 7, 192):    {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (659, 659)}
# block6c_add           (7, 7, 192):    {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (659, 659)}
# block6d_expand_conv   (7, 7, 1152):   {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (659, 659)}
# block6d_dwconv        (7, 7, 1152):   {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (787, 787)}
# block6d_project_conv  (7, 7, 192):    {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (787, 787)}
# block6d_add           (7, 7, 192):    {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (787, 787)} ### -> EfficientNetB0_Block6 (9408)
# block7a_expand_conv   (7, 7, 1152):   {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (787, 787)}
# block7a_dwconv        (7, 7, 1152):   {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (851, 851)}
# block7a_project_conv  (7, 7, 320):    {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (851, 851)}
# top_conv              (7, 7, 1280):   {'offset': (31.5, 31.5),   'stride': (32.0, 32.0), 'size': (851, 851)} ### -> EfficientNetB0        (62720)

### EfficientNetB6 (528, 528, 3)
# stem_conv             (264, 264, 56):  {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (3, 3)}
# block1a_dwconv        (264, 264, 56):  {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (7, 7)}
# block1a_project_conv  (264, 264, 32):  {'offset': (1.5, 0.5),     'stride': (2.0, 2.0),   'size': (7, 7)}
# block1b_dwconv        (264, 264, 32):  {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (11, 11)}
# block1b_project_conv  (264, 264, 32):  {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (11, 11)}
# block1b_add           (264, 264, 32):  {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (11, 11)} ###
# block1c_dwconv        (264, 264, 32):  {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (15, 15)}
# block1c_project_conv  (264, 264, 32):  {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (15, 15)}
# block1c_add           (264, 264, 32):  {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (15, 15)} ###
# block2a_expand_conv   (264, 264, 192): {'offset': (1.5, 1.5),     'stride': (2.0, 2.0),   'size': (15, 15)}
# block2a_dwconv        (132, 132, 192): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (19, 19)}
# block2a_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (19, 19)}
# block2b_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (19, 19)}
# block2b_dwconv        (132, 132, 240): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (27, 27)}
# block2b_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (27, 27)}
# block2b_add           (132, 132, 40):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (27, 27)} ###
# block2c_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (27, 27)}
# block2c_dwconv        (132, 132, 240): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (35, 35)}
# block2c_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (35, 35)}
# block2c_add           (132, 132, 40):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (35, 35)} ###
# block2d_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (35, 35)}
# block2d_dwconv        (132, 132, 240): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (43, 43)}
# block2d_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (43, 43)}
# block2d_add           (132, 132, 40):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (43, 43)} ###
# block2e_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (43, 43)}
# block2e_dwconv        (132, 132, 240): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (51, 51)}
# block2e_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (51, 51)}
# block2e_add           (132, 132, 40):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (51, 51)} ###
# block2f_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (51, 51)}
# block2f_dwconv        (132, 132, 240): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (59, 59)}
# block2f_project_conv  (132, 132, 40):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (59, 59)}
# block2f_add           (132, 132, 40):  {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (59, 59)} ###
# block3a_expand_conv   (132, 132, 240): {'offset': (3.5, 3.5),     'stride': (4.0, 4.0),   'size': (59, 59)}
# block3a_dwconv        (66, 66, 240):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (75, 75)}
# block3a_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (75, 75)}
# block3b_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (75, 75)}
# block3b_dwconv        (66, 66, 432):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (107, 107)}
# block3b_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (107, 107)}
# block3b_add           (66, 66, 72):    {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (107, 107)} ###
# block3c_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (107, 107)}
# block3c_dwconv        (66, 66, 432):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (139, 139)}
# block3c_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (139, 139)}
# block3c_add           (66, 66, 72):    {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (139, 139)} ###
# block3d_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (139, 139)}
# block3d_dwconv        (66, 66, 432):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (171, 171)}
# block3d_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (171, 171)}
# block3d_add           (66, 66, 72):    {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (171, 171)} ###
# block3e_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (171, 171)}
# block3e_dwconv        (66, 66, 432):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (203, 203)}
# block3e_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (203, 203)}
# block3e_add           (66, 66, 72):    {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (203, 203)} ###
# block3f_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (203, 203)}
# block3f_dwconv        (66, 66, 432):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (235, 235)}
# block3f_project_conv  (66, 66, 72):    {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (235, 235)}
# block3f_add           (66, 66, 72):    {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (235, 235)} ###
# block4a_expand_conv   (66, 66, 432):   {'offset': (7.5, 7.5),     'stride': (8.0, 8.0),   'size': (235, 235)}
# block4a_dwconv        (33, 33, 432):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (251, 251)}
# block4a_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (251, 251)}
# block4b_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (251, 251)}
# block4b_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (283, 283)}
# block4b_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (283, 283)}
# block4b_add           (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (283, 283)} ###
# block4c_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (283, 283)}
# block4c_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (315, 315)}
# block4c_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (315, 315)}
# block4c_add           (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (315, 315)} ###
# block4d_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (315, 315)}
# block4d_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (347, 347)}
# block4d_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (347, 347)}
# block4d_add           (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (347, 347)} ###
# block4e_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (347, 347)}
# block4e_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (379, 379)}
# block4e_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (379, 379)}
# block4e_add           (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (379, 379)} ###
# block4f_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (379, 379)}
# block4f_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (411, 411)}
# block4f_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (411, 411)}
# block4f_add           (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (411, 411)} ###
# block4g_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (411, 411)}
# block4g_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (443, 443)}
# block4g_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (443, 443)}
# block4g_add           (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (443, 443)} ###
# block4h_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (443, 443)}
# block4h_dwconv        (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (475, 475)}
# block4h_project_conv  (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (475, 475)}
# block4h_add           (33, 33, 144):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (475, 475)} ###
# block5a_expand_conv   (33, 33, 864):   {'offset': (15.5, 15.5),   'stride': (16.0, 16.0), 'size': (475, 475)}
# Ab hier wird es unplausibel TODO: Nochmal mit dem dicken Rechner und absurd großer input shape rechnen (3000, 3000, 3) oder so?
# block5a_dwconv        (33, 33, 864):   {'offset': (22.0, 22.0),   'stride': (8.0, 8.0),   'size': (526, 526)}
# block5a_project_conv  (33, 33, 200):   {'offset': (22.0, 22.0),   'stride': (8.0, 8.0),   'size': (526, 526)}
# block5b_expand_conv   (33, 33, 1200):  {'offset': (22.0, 22.0),   'stride': (8.0, 8.0),   'size': (526, 526)}
# block5b_dwconv        (33, 33, 1200):  {'offset': (53.0, 53.0),   'stride': (0.0, 0.0),   'size': (528, 528)}
# block5b_project_conv  (33, 33, 200):   {'offset': (53.0, 53.0),   'stride': (0.0, 0.0),   'size': (528, 528)}
# block5b_add           (33, 33, 200):   {'offset': (53.0, 53.0),   'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block5c_add           (33, 33, 200):   {'offset': (85.0, 85.0),   'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block5d_add           (33, 33, 200):   {'offset': (117.0, 117.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block5e_add           (33, 33, 200):   {'offset': (149.0, 149.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block5f_add           (33, 33, 200):   {'offset': (181.0, 181.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block5g_add           (33, 33, 200):   {'offset': (213.0, 213.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block5h_add           (33, 33, 200):   {'offset': (245.0, 245.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block6b_add           (17, 17, 344):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block6c_add           (17, 17, 344):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block6d_add           (17, 17, 344):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block6e_add           (17, 17, 344):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block6f_add           (17, 17, 344):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block6g_add           (17, 17, 344):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block6h_add           (17, 17, 344):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block6i_add           (17, 17, 344):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block6j_add           (17, 17, 344):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block6k_add           (17, 17, 344):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block7b_add           (17, 17, 576):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# block7c_add           (17, 17, 576):   {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)} ###
# ...
# top_conv              (17, 17, 2304):  {'offset': (264.0, 264.0), 'stride': (0.0, 0.0),   'size': (528, 528)}


### C3D (C2D, mit (112, 112, 3))
# conv1           (112, 112, 64):  {'offset': (0.5, 0.5), 'stride': (1.0, 1.0),   'size': (3, 3)}
# pool1           (56, 56, 64):    {'offset': (0.5, 0.5), 'stride': (2.0, 2.0),   'size': (3, 3)}
# conv2           (56, 56, 128):   {'offset': (0.5, 0.5), 'stride': (2.0, 2.0),   'size': (7, 7)}
# pool2           (28, 28, 128):   {'offset': (0.5, 0.5), 'stride': (4.0, 4.0),   'size': (7, 7)}
# conv3a          (28, 28, 256):   {'offset': (0.5, 0.5), 'stride': (4.0, 4.0),   'size': (15, 15)}
# conv3b          (28, 28, 256):   {'offset': (0.5, 0.5), 'stride': (4.0, 4.0),   'size': (23, 23)}     ### -> C3D_Block3   (200704) !
# pool3           (14, 14, 256):   {'offset': (0.5, 0.5), 'stride': (8.0, 8.0),   'size': (23, 23)}
# conv4a          (14, 14, 512):   {'offset': (0.5, 0.5), 'stride': (8.0, 8.0),   'size': (39, 39)}
# conv4b          (14, 14, 512):   {'offset': (0.5, 0.5), 'stride': (8.0, 8.0),   'size': (55, 55)}     ### -> C3D_Block4   (100352)
# pool4           (7, 7, 512):     {'offset': (0.5, 0.5), 'stride': (16.0, 16.0), 'size': (55, 55)}
# conv5a          (7, 7, 512):     {'offset': (0.5, 0.5), 'stride': (16.0, 16.0), 'size': (87, 87)}
# conv5b          (7, 7, 512):     {'offset': (0.5, 0.5), 'stride': (16.0, 16.0), 'size': (119, 119)}   ### -> C3D          (25088)

### VGG16 (224, 224, 3) (etwas merkwürdige Ergebnisse)
# block3_conv3    (56, 56, 256):   {'offset': (-1.5, 0.5), 'stride': (4.0, 4.0),   'size': (35, 37)}
# block4_conv3    (28, 28, 512):   {'offset': (-3.0, 0.5), 'stride': (8.0, 8.0),   'size': (82, 85)}
# block5_conv3    (14, 14, 512):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (181, 181)}
### Manuell korrigiert (plausibler so)
# block3_conv3    (56, 56, 256):   {'offset': (0.5, 0.5),  'stride': (4.0, 4.0),   'size': (37, 37)}
# block4_conv3    (28, 28, 512):   {'offset': (0.0, 0.5),  'stride': (8.0, 8.0),   'size': (85, 85)}
# block5_conv3    (14, 14, 512):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (181, 181)}

# block1_conv1    (449, 449, 64):  'stride': (1.0, 1.0),   'size': (2, 1)}
# block1_conv2    (449, 449, 64):  'stride': (1.0, 1.0),   'size': (4, 4)}
# block1_pool     (224, 224, 64):  'stride': (2.0, 2.0),   'size': (4, 4)}
# block2_conv1    (224, 224, 128): 'stride': (2.0, 2.0),   'size': (8, 8)}
# block2_conv2    (224, 224, 128): 'stride': (2.0, 2.0),   'size': (12, 12)}
# block2_pool     (112, 112, 128): 'stride': (4.0, 4.0),   'size': (12, 12)}
# block3_conv1    (112, 112, 256): 'stride': (4.0, 4.0),   'size': (21, 20)}
# block3_conv2    (112, 112, 256): 'stride': (4.0, 4.0),   'size': (29, 29)}
# block3_conv3    (112, 112, 256): 'stride': (4.0, 4.0),   'size': (37, 35)}    ### -> VGG16_Block3 (802816) !
# block3_pool     (56, 56, 256):   'stride': (8.0, 8.0),   'size': (37, 35)}
# block4_conv1    (56, 56, 512):   'stride': (8.0, 8.0),   'size': (53, 51)}
# block4_conv2    (56, 56, 512):   'stride': (8.0, 8.0),   'size': (69, 68)}
# block4_conv3    (56, 56, 512):   'stride': (8.0, 8.0),   'size': (85, 82)}    ### -> VGG16_Block4 (401408) !
# block4_pool     (28, 28, 512):   'stride': (16.0, 16.0), 'size': (85, 82)}
# block5_conv1    (28, 28, 512):   'stride': (16.0, 16.0), 'size': (117, 116)}
# block5_conv2    (28, 28, 512):   'stride': (16.0, 16.0), 'size': (149, 148)}
# block5_conv3    (28, 28, 512):   'stride': (16.0, 16.0), 'size': (181, 181)}  ### -> VGG16        (100352)
# block5_pool     (14, 14, 512):   'stride': (32.0, 32.0), 'size': (181, 181)}

### ResNet50V2 (224, 224, 3), RFs mit (673, 673, 3) berechnet
# conv2_block1_out (56, 56, 256):   {'offset': (-1.5, -1.5), 'stride': (4.0, 4.0),   'size': (15, 15)}
# conv2_block2_out (56, 56, 256):   {'offset': (-1.5, -1.5), 'stride': (4.0, 4.0),   'size': (23, 23)}
# conv2_block3_out (28, 28, 256):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (31, 31)}   ### -> ResNet50V2_Stack2 (200704) !
# conv3_block1_out (28, 28, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (47, 47)}
# conv3_block2_out (28, 28, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (63, 63)}
# conv3_block3_out (28, 28, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (79, 79)}
# conv3_block4_out (14, 14, 512):   {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (95, 95)}   ### -> ResNet50V2_Stack3 (100352)
# conv4_block1_out (14, 14, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (127, 127)}
# conv4_block2_out (14, 14, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (159, 159)}
# conv4_block3_out (14, 14, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (191, 191)}
# conv4_block4_out (14, 14, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (223, 223)}
# conv4_block5_out (14, 14, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (255, 255)}
# conv4_block6_out (7, 7, 1024):    {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (287, 287)} ### -> ResNet50V2_Stack4 (50176)
# conv5_block1_out (7, 7, 2048):    {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (351, 351)}
# conv5_block2_out (7, 7, 2048):    {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (415, 415)}
# conv5_block3_out (7, 7, 2048):    {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (479, 479)}
# post_bn          (7, 7, 2048):    {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (479, 479)}
# post_relu        (7, 7, 2048):    {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (479, 479)} ### -> ResNet50V2        (100352) ? (2048 is a lot)

### ResNet50V2 (449, 449, 3), RFs mit (673, 673, 3) berechnet
# conv2_block1_out (113, 113, 256): {'offset': (-1.5, -1.5), 'stride': (4.0, 4.0),   'size': (15, 15)}
# conv2_block2_out (113, 113, 256): {'offset': (-1.5, -1.5), 'stride': (4.0, 4.0),   'size': (23, 23)}
# conv2_block3_out (57, 57, 256):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (31, 31)}   ### -> ResNet50V2_Stack2_LargeImage (831744) !
# conv3_block1_out (57, 57, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (47, 47)}
# conv3_block2_out (57, 57, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (63, 63)}
# conv3_block3_out (57, 57, 512):   {'offset': (-1.5, -1.5), 'stride': (8.0, 8.0),   'size': (79, 79)}
# conv3_block4_out (29, 29, 512):   {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (95, 95)}   ### -> ResNet50V2_Stack3_LargeImage (430592) !
# conv4_block1_out (29, 29, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (127, 127)}
# conv4_block2_out (29, 29, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (159, 159)}
# conv4_block3_out (29, 29, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (191, 191)}
# conv4_block4_out (29, 29, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (223, 223)}
# conv4_block5_out (29, 29, 1024):  {'offset': (-1.5, -1.5), 'stride': (16.0, 16.0), 'size': (255, 255)}
# conv4_block6_out (15, 15, 1024):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (287, 287)} ### -> ResNet50V2_Stack4_LargeImage (230400) !
# conv5_block1_out (15, 15, 2048):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (351, 351)}
# conv5_block2_out (15, 15, 2048):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (415, 415)}
# conv5_block3_out (15, 15, 2048):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (479, 479)}
# post_bn          (15, 15, 2048):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (479, 479)}
# post_relu        (15, 15, 2048):  {'offset': (-1.5, -1.5), 'stride': (32.0, 32.0), 'size': (479, 479)} ### -> ResNet50V2_LargeImage        (460800) !

### MobileNetV2 (224, 224, 3), RFs mit (673, 673, 3) berechnet
# expanded_conv_project_BN   (112, 112, 16): {'offset': (0.5, 0.5),  'stride': (2.0, 2.0),   'size': (7, 7)}
# block_1_project_BN         (56, 56, 24):   {'offset': (-0.5, 0.5), 'stride': (4.0, 4.0),   'size': (11, 11)}
# block_2_project_BN         (56, 56, 24):   {'offset': (0.5, -0.5), 'stride': (4.0, 4.0),   'size': (19, 19)}
# block_2_add                (56, 56, 24):   {'offset': (0.5, -0.5), 'stride': (4.0, 4.0),   'size': (19, 19)}
# block_3_project_BN         (28, 28, 32):   {'offset': (0.5, 0.5),  'stride': (8.0, 8.0),   'size': (27, 27)}    ### -> MobileNetV2_Block3     (25088)
# block_4_project_BN         (28, 28, 32):   {'offset': (0.5, 0.5),  'stride': (8.0, 8.0),   'size': (43, 43)}
# block_4_add                (28, 28, 32):   {'offset': (0.5, 0.5),  'stride': (8.0, 8.0),   'size': (43, 43)}
# block_5_project_BN         (28, 28, 32):   {'offset': (0.5, 0.5),  'stride': (8.0, 8.0),   'size': (59, 59)}
# block_5_add                (28, 28, 32):   {'offset': (0.5, 0.5),  'stride': (8.0, 8.0),   'size': (59, 59)}
# block_6_project_BN         (14, 14, 64):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (75, 75)}    ### -> MobileNetV2_Block6     (12544)
# block_7_project_BN         (14, 14, 64):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (107, 107)}
# block_7_add                (14, 14, 64):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (107, 107)}
# block_8_project_BN         (14, 14, 64):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (139, 139)}
# block_8_add                (14, 14, 64):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (139, 139)}
# block_9_project_BN         (14, 14, 64):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (171, 171)}
# block_9_add                (14, 14, 64):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (171, 171)}  ### -> MobileNetV2_Block9     (12544)
# block_10_project_BN        (14, 14, 96):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (203, 203)}
# block_11_project_BN        (14, 14, 96):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (235, 235)}
# block_11_add               (14, 14, 96):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (235, 235)}
# block_12_project_BN        (14, 14, 96):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (267, 267)}
# block_12_add               (14, 14, 96):   {'offset': (0.5, 0.5),  'stride': (16.0, 16.0), 'size': (267, 267)}  ### -> MobileNetV2_Block12    (18816)
# block_13_project_BN        (7, 7, 160):    {'offset': (0.5, 0.5),  'stride': (32.0, 32.0), 'size': (299, 299)}
# block_14_project_BN        (7, 7, 160):    {'offset': (0.5, 0.5),  'stride': (32.0, 32.0), 'size': (363, 363)}
# block_14_add               (7, 7, 160):    {'offset': (0.5, 0.5),  'stride': (32.0, 32.0), 'size': (363, 363)}  ### -> MobileNetV2_Block14    (7840)
# block_15_project_BN        (7, 7, 160):    {'offset': (0.5, 0.5),  'stride': (32.0, 32.0), 'size': (427, 427)}
# block_15_add               (7, 7, 160):    {'offset': (0.5, 0.5),  'stride': (32.0, 32.0), 'size': (427, 427)}
# block_16_project_BN        (7, 7, 320):    {'offset': (0.5, 0.5),  'stride': (32.0, 32.0), 'size': (491, 491)}  ### -> MobileNetV2_Block16    (15680)
# Conv_1_bn                  (7, 7, 1280):   {'offset': (0.5, 0.5),  'stride': (32.0, 32.0), 'size': (491, 491)}
# out_relu                   (7, 7, 1280):   {'offset': (0.5, 0.5),  'stride': (32.0, 32.0), 'size': (491, 491)}  ### -> MobileNetV2            (62720)