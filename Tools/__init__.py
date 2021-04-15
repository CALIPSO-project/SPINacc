#=============================================================================================
# MLacc - Machine-Learning-based acceleration of spin-up
#
# Copyright Laboratoire des Sciences du Climat et de l'Environnement (LSCE)
#           Unite mixte CEA-CNRS-UVSQ
# 
# Code manager:
# Daniel Goll, LSCE, <email>
# 
# This software is developed by Yan Sun, Yilong Wang and Daniel Goll.......
# 
# This software is governed by the XXX license
# XXXX <License content>
#
#=============================================================================================

# Ready-made
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import calendar
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import json
import sys

from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from mpl_toolkits.basemap import Basemap

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy import stats 
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.simplefilter(action='ignore')

# Home-made
from classes import pack,auxiliary
import check
import genMask
import extract_X
import extract_Y
import encode
from readvar import readvar
import Cluster
import train
import mapGlobe
import MLeval
import eval_plot
import ML_som_litter
import ML_biomass
