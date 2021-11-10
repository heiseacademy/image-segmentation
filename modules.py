import base64
import numpy as np
import io
import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from io import BytesIO
from functools import reduce
import operator
import math
import json

#from google.colab import drive
import skimage as sk
from skimage import io
from skimage.util import img_as_float
import skimage.transform as trans
from skimage.transform import resize, rotate, AffineTransform, ProjectiveTransform, warp
from sklearn.model_selection import train_test_split
from skimage.util import img_as_float, img_as_ubyte
from skimage.morphology import reconstruction
from skimage import measure


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras import backend as keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras import applications
from tensorflow.keras.layers import Input
from collections import defaultdict, OrderedDict
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Conv2D, concatenate, UpSampling2D, BatchNormalization, Activation, Cropping2D, ZeroPadding2D
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from numpy.random import randint, uniform

from scipy import ndimage

import base64
import numpy as np
import io
import os
from PIL import Image
import cv2
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from matplotlib import pyplot as plt
from io import BytesIO
import base64
from skimage import io
import skimage as sk
import json
