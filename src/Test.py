# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:28:23 2023

@author: qmc
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pdb
from alive_progress import alive_bar
from time import sleep
import sympy as sym
from sympy import symbols, solve,cos,sin,Matrix, Inverse
from scipy import optimize
from scipy.special import wofz
from scipy import integrate
import scipy.linalg as LA
from mpi4py import MPI
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
import Operators as op
import CrysFieldExplorer as crs
import cma
import unittest
import math

class Test:
    def __init__(self,exp_e, exp_I,magnetization, susceptibility):
        '''Organzing tests with test classes and specific test cases from published papers (work in progress)'''
        
        self.exp_e=exp_e
        self.exp_I=exp_I
        self.magnetization=magnetization
        self.susceptibility=susceptibility
        
    def energy_level_test(self, ev):
        if len(ev) !=len(self.exp_e):
            print("test fail. (Make sure experimental and calculated energy levels has the same length!)")
        
        for i in len(range(ev)):
            if math.isclose(ev[i], self.exp_e[i]):
                print(f"energy level {i} matches with experiment")
            else:
                print(f'test fail. Energy level {i} does not match')
        
    def Intensity_test(self, I):
        if len(I) !=len(self.exp_I):
            print("test fail. (Make sure experimental and calculated intensity has the same length!)")
        
        for i in len(range(I)):
            if math.isclose(I[i], self.exp_I[i]):
                print(f"The intensity of level {i} matches with experiment")
            else:
                print(f'test fail. Intensity level {i} does not match')
        