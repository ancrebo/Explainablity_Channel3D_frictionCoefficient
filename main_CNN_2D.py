# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:10:03 2023

@author: andres cremades botella

File containing the configuration of the CNN model and the training process
"""
import ann_config_2D as ann
import numpy as np
import get_data_fun as gd


CNN = ann.convolutional_residual(ngpu=1,
                                 fileddbb='/data/cremades/P125_21pi_vu/P125_21pi_vu',
                                 file_cf='/media/nils/Elements/P125_21pi_vu_cf/P125_21pi_vu'
                                 # fileddbb='./P125_21pi_vu/P125_21pi_vu',
                                 # file_cf='./P125_21pi_vu_cf/P125_21pi_vu'
                                 )
dy = 1
dz = 1
dx = 1
shpy = int((201-1)/dy)+1
shpz = int((96-1)/dz)+1
shpx = int((192-1)/dx)+1
CNN.define_model(shp=(shpy,shpz,shpx,3),learat=1e-2) 
CNN.train_model(1000,7000,delta_t=5,delta_e=20,max_epoch=2e2,
                 batch_size=2,down_y=dy,down_z=dz,down_x=dx) #,fileddbb='../P125_21pi_vu/P125_21pir2') # 1000,7000

print('fin')
