#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:46:04 2024

@author: benedikt_n
"""

''' Friction coefficient calculations without having to instantiate a 
shap_conf object. The code saves an h5 file with the friction coefficients 
of every snapshot'''

import get_data_fun as gd
import numpy as np
import h5py
from time import time

def friction_coefficient_high_precision(input_field, normdata):
    '''
    Calculate the friction coefficient for a certain snapshot with 
    compact finite differences.

    '''
    avg_u_norm = np.mean(
                 input_field[0,:,:,:,0],
                axis=(1,2)          
                 )
    avg_U = normdata.UUmean + normdata.uumin + \
                 avg_u_norm*(normdata.uumax-normdata.uumin)
    grad_U_y = np.linalg.solve(normdata.A, np.dot(normdata.B, avg_U))           
    grad_U_wall = 0.5*(grad_U_y[0]-grad_U_y[-1])
    
    UUmean_shift = np.zeros(len(normdata.UUmean)+1)
    UUmean_shift[1:] = normdata.UUmean
    UUmean_shift = UUmean_shift[:-1]
    
    '''U_bulk = np.sum(np.mean(
             np.multiply(normdata.UUmean,normdata.dy.reshape(-1,1,1)),axis=(1,2))
             )/np.sum(normdata.dy) # the integral doesn't work that way'''
    U_bulk = 0.5*np.dot(np.gradient(normdata.y_h), 
                        0.5*(normdata.UUmean+UUmean_shift)) # /self.delta_y
    c_f = 2*grad_U_wall/(U_bulk*normdata.re_bulk) # *self.delta_y
    
    return c_f


def friction_coefficient(input_field, normdata):
    '''
    Calculate the friction coefficient for a certain snapshot.

    '''
    # average normalized fluctuating u at both walls
    avg_u_norm = np.zeros(3)
    avg_u_norm[0] = np.mean(
                 np.vstack(
                           [input_field[0,1,:,:,0], 
                            input_field[0,-2,:,:,0]]
                           )
                        )
    avg_u_norm[1] = np.mean(
                 np.vstack(
                           [input_field[0,2,:,:,0], 
                            input_field[0,-3,:,:,0]]
                           )
                        )
    avg_u_norm[2] = np.mean(
                 np.vstack(
                           [input_field[0,3,:,:,0], 
                            input_field[0,-4,:,:,0]]
                           )
                        )
    avg_U = 0.5*normdata.UUmean[1:4]+0.5*np.flip(normdata.UUmean[-4:-1])\
        + normdata.uumin + avg_u_norm*(normdata.uumax-normdata.uumin)
    avg_U = avg_U.reshape([1,3])
    grad_U_wall = np.dot(avg_U, normdata.fd_coeffs)
    
    
    UUmean_shift = np.zeros(len(normdata.UUmean)+1)
    UUmean_shift[1:] = normdata.UUmean
    UUmean_shift = UUmean_shift[:-1]
    
    '''U_bulk = np.sum(np.mean(
             np.multiply(normdata.UUmean,normdata.dy.reshape(-1,1,1)),axis=(1,2))
             )/np.sum(normdata.dy)'''
    U_bulk = 0.5*np.dot(np.gradient(normdata.y_h), 
                        0.5*(normdata.UUmean+UUmean_shift)) # /self.delta_y
    c_f = 2*grad_U_wall/(U_bulk*normdata.re_bulk) # *self.delta_y
    
    return c_f


def calc_cf(fileuvw,
            # fileQ,
            file_output,
            start, 
            end, 
            step=1, 
            fileUmean="Umean.txt",
            filenorm="norm.txt",
            filerms="Urms.txt",
            padpix=15,
            high_precision=False):
    '''
    Calculate the friction coefficient for a portion of the dataset.

    '''
    
    normdata = gd.get_data_norm(file_read=fileuvw)
    normdata.geom_param(start,1,1,1)
    try:
        normdata.read_Umean(file=fileUmean)
    except:
        normdata.calc_Umean(start,end)
    try:
        normdata.read_norm(file=filenorm)
    except:
        normdata.calc_norm(start,end)
    try:
        normdata.read_Urms(file=filerms)
    except:
        normdata.calc_rms(start,end)
    
    normdata.read_cfd_matrices()
    normdata.read_fd_coeffs()
        
    c_f = np.zeros(int(np.floor((end-start)/step)))
    for ii in range(start,end,step):

        '''uv_struc = normdata.read_uvstruc(ii,fileQ=fileQ,padpix=padpix)
        
        # Get array of the velocity fields
        uvmax = np.max(uv_struc.mat_segment)
        self.segmentation = uv_struc.mat_segment-1
        self.segmentation[self.segmentation==-1] = uvmax'''
        uu_i,vv_i,ww_i = normdata.read_velocity(ii,padpix=padpix)
        input_field = normdata.norm_velocity(uu_i,vv_i,ww_i,padpix=padpix)[0,:,:,:,:]
        input_field_reshaped = input_field.reshape(-1,
                                                 input_field.shape[0],
                                                 input_field.shape[1],
                                                 input_field.shape[2],
                                                 3)
        if high_precision:
            c_f[ii-start] = friction_coefficient_high_precision(
                                        input_field_reshaped, normdata)
        else:
            c_f[ii-start] = friction_coefficient(
                                        input_field_reshaped, normdata)
        
    
    # Write to file    
    
    if high_precision:
        hf = h5py.File(file_output+'_'+str(start)+'_'+str(end)+'_hp'+'.h5.cf', 'w')
    else:
        hf = h5py.File(file_output+'_'+str(start)+'_'+str(end)+'.h5.cf', 'w')
    hf.create_dataset('c_f', data=c_f)