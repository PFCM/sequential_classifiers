"""Load up an addition model and have a look at what's going on"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import mrnn
import rnndatasets.addition as data

import sequential_model as sm
import addition
