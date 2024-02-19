# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:30:46 2023

@author: andres cremades botella

function for calculating the shap values
"""

import shap_config as sc
start = 7000
end = 7001 #1001
step = 1
shap = sc.shap_conf()
shap.calc_shap_kernel(start,end,step,\
                      file='../P125_21pi_vu_SHAP_ann4_backmod/P125_21pi_vu',\
                      fileQ='../P125_21pi_vu_Q_divide/P125_21pi_vu',backgroundrms=True)
                      
#shap.eval_shap(start=start,end=end,step=step,\
#               fileuvw='../P125_21pi_vu/P125_21pi_vu',\
#               fileQ='../../../data2/cremades/P125_21pi_vu_Q_divide/P125_21pi_vu',\
#               fileshap='../../../data2/cremades/P125_21pi_vu_SHAP_ann4_divide/P125_21pi_vu')
# '../../../data2/cremades/P125_21pi_vu_SHAP/P125_21pi_vu'
# fileQ='../../../data2/cremades/P125_21pi_vu_Q/P125_21pi_vu')