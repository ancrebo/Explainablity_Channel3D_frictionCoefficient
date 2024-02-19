# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:00:40 2023

@author: andres cremades botella

Plot rms in grid points
"""
import ann_config as ann
import get_data_fun as gd

start = 5000
dy = 1
dz = 1
dx = 1
CNN = ann.convolutional_residual()# fileddbb='../P125_21pi_vu/P125_21pir2')#
CNN.readrms()
normdata = gd.get_data_norm('../P125_21pi_vu/P125_21pir2')
normdata.geom_param(start,dy,dz,dx)
normdata.read_Urms()
CNN.plotrms(normdata)

