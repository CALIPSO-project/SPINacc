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
import re
import os
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from copy import deepcopy
from netCDF4 import Dataset

matplotlib.use("Agg")
import itertools
import json
import random
import sys
from collections import Counter
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    ProcessPoolExecutor,
)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from mpl_toolkits.basemap import Basemap
from pandas import DataFrame, Series
from pathlib import Path
from scipy import stats
from sklearn.cluster import Birch, KMeans
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Lasso, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    LeaveOneOut,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import warnings

warnings.simplefilter(action="ignore")
sys.path.append(os.path.dirname(__file__))

# Home-made
import check
import genmask
import extract_x
import encode
from readvar import readvar
import cluster
import train
import mapglobe
import mleval
import eval_plot_un
import eval_plot_loocv_un
import ml
import forcing
