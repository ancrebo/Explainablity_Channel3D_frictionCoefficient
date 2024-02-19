# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:11:23 2023

@author: andres cremades botella

File for segmenting the domain in structures
"""

import get_data_fun as gd
import numpy as np

#%% Prepare data for training
start =  1200 #  
end =  1201 # 
normdata = gd.get_data_norm()#'../P125_21pi_vu/P125_21pir2') #'P125_21pi_vu'
normdata.geom_param(start,1,1,1)
uv_struc = normdata.Q_perc(start,end,fileQ='../P125_21pi_vu_Q_divide/P125_21pi_vu') #'../../../data2/cremades/P125_21pi_vu_Q_divide')

#../../data2/cremades/