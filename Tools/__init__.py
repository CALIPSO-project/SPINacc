# =============================================================================================
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
# =============================================================================================

import calendar

# Ready-made
import os

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

matplotlib.use("Agg")
import itertools
import json
import random
import sys
from collections import Counter

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from mpl_toolkits.basemap import Basemap
from pandas import DataFrame, Series
from scipy import stats
from sklearn.cluster import Birch, KMeans
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeRegressor

import warnings

warnings.simplefilter(action="ignore")
sys.path.append(os.path.dirname(__file__))

# Home-made
import check
import genMask
import extract_X
import encode
from readvar import readvar
import Cluster
import train
import mapGlobe
import MLeval
import eval_plot_un
import eval_plot_loocv_un
import ML
import forcing
