# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:33:47 2023

@author: andres cremades botella

File for evaluating the results of the CNN
"""
import ann_config as ann
import get_data_fun as gd


start = 1000 # 1143 #
end = 9999 # 1144 #
step = 1 #50 #  
dy = 1
dz = 1
dx = 1
CNN = ann.convolutional_residual()#fileddbb='../P125_21pi_vu/P125_21pir2')# )#
CNN.load_model()
#CNN.eval_model(start,down_y=1,down_z=2,down_x=2,start=start)
CNN.pred_rms(start,end,step=step,down_y=dy,down_z=dz,down_x=dx)
normdata = gd.get_data_norm()#'../P125_21pi_vu/P125_21pir2')# 
normdata.geom_param(start,dy,dz,dx)
#normdata.read_Urms()
#CNN.saverms()
#CNN.plotrms_sim(normdata)
#CNN.plotrms_simlin(normdata)
CNN.mre_pred(normdata,start,end,step)
CNN.savemre()