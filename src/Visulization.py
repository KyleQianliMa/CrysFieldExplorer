# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:06:05 2023

@author: qmc
"""

import numpy as np
from numpy import sqrt
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import Operators as op
import CrysFieldExplorer as crs
import sympy as sym
from sympy import symbols, solve,cos,sin,Matrix, Inverse
from scipy import optimize
from scipy.special import wofz
from scipy import integrate
import scipy.linalg as LA

class vis:
    def __init__(self,font_size,marker_size):
        self.font_size=font_size
        self.marker_size=marker_size
    
    def susceptibility(self,susceptibility):
        plt.plot(susceptibility[0], 1/susceptibility[1],'.',markersize=self.marker_size)
        plt.xlabel('Temperature (K)',fontsize=self.font_size)
        plt.xticks(fontsize=self.font_size)
        plt.ylabel(r'$4\pi\chi\ (emu\ /cm^3 \times Oe^{-1})$',fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        plt.title('Inverse Susceptibility')
        plt.show()
    
    def magnetization(self,magnetization):
        plt.plot(magnetization[0],magnetization[1],'.',markersize=self.marker_size)
        plt.xlabel('Field (T)',fontsize=self.font_size)
        plt.xticks(fontsize=self.font_size)
        plt.ylabel('Magnetization (\mu_B)',fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        plt.title('Magneization')
        plt.show()
    
    def neutron_spectrum(self, E, I, resolution):
        x_E=np.linspace(0.8*min(E),1.2*max(E),100)
        y_I=0
        width=resolution
        for i in range(len(E)):
            y_I+=crs.Utilities.lorentzian(x_E, I[i], width, E[i])
        plt.plot(x_E,y_I)
        plt.xlabel('Energy (meV)',fontsize=self.font_size)
        plt.ylabel('Intensity (arb.unit)',fontsize=self.font_size)
        plt.title('Neutron Spectrum')
        
        plt.show()