import os
from skimage.feature import local_binary_pattern
import cvutils
import glob
import sys
import shutil
import scipy.io as sio
from matplotlib.pyplot import *
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
import scipy.misc as sim
from StringIO import StringIO
import dlib
import cv2
from scipy.misc import imread
import dtcwt
import dtcwt.compat as compat
import pylab
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from skimage.feature import hog
from sklearn.metrics import roc_curve, auc
data_txt = "data.txt"
data_txt2 = "data2.txt"