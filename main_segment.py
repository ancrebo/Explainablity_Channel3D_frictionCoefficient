# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:11:23 2023

@author: andres cremades botella

File for segmenting the domain in structures
"""

import get_data_fun as gd
import numpy as np

#%% Prepare data for training
start =  2967 #  
end =  9999 # 
delta = 1
normdata = gd.get_data_norm(file_read='../../../data/cremades/P125_21pi_vu/P125_21pi_vu')#'../P125_21pi_vu/P125_21pir2') #'P125_21pi_vu'
normdata.geom_param(start,1,1,1)
'''normdata.eval_dz(start,end,1,fileQ='../../../data2/cremades/P125_21pi_vu_Q_divide/P125_21pi_vu')
normdata.eval_volfilter(start,end,delta,fileQ='../../../data2/cremades/P125_21pi_vu_Q_divide/P125_21pi_vu')
normdata.eval_filter(1200,1201,1,fileQ='../../../data2/cremades/P125_21pi_vu_Q_divide/P125_21pi_vu') #fileQ='../P125_21pi_vu_Q_divide/P125_21pi_vu')
normdata.calc_Umean(start,end)
normdata.save_Umean()
#normdata.read_Umean()
normdata.plot_Umean()
normdata.calc_rms(start,end)
normdata.save_Urms()
#normdata.read_Urms()
normdata.plot_Urms()
normdata.calc_norm(start,end)
normdata.save_norm()
#normdata.read_norm'''
uv_struc = normdata.calc_uvstruc(start,end,fileQ='./P125_21pi_vu_Q_divide/P125_21pi_vu',\
                                 fold='./P125_21pi_vu_Q_divide') #'../../../data2/cremades/P125_21pi_vu_Q_divide')

