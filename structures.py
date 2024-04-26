#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:32:50 2024

@author: benedikt_n
"""

import numpy as np
import get_data_fun as gd
import h5py
from tqdm import tqdm

def calc_Q_and_Delta_fields(start, 
                            end,
                            step,
                            file_read='./P125_21pi_vu/P125_21pi_vu',
                            file_grad='./P125_21pi_vu/grad/P125_21pi_vu',
                            file_Q_Delta='./P125_21pi_vu/hunt_chong/P125_21pi_vu'):
    
    normdata = gd.get_data_norm(file_read=file_read,
                                file_grad=file_grad,
                                file_Q_Delta=file_Q_Delta)
    normdata.geom_param(start,1,1,1)
    
    for ii in range(start, end, step):
    
        G = normdata.read_gradients(ii)
        Omega = 0.5*(G-G.transpose((1,0,2,3,4)))
        S = 0.5*(G+G.transpose((1,0,2,3,4)))
    
        Q = 0.5*(np.linalg.norm(Omega, axis=(0,1))**2
             -np.linalg.norm(S, axis=(0,1))**2)
        
        SijSjkSki = np.einsum('ijlmn,jklmn,kilmn->lmn',S,S,S)
        WijWjkSki = np.einsum('ijlmn,jklmn,kilmn->lmn',Omega,Omega,S)
        R = -1/3*(SijSjkSki+3*WijWjkSki)
        
        Delta = 27/4*R**2+Q**3
    
        normdata.write_Q_Delta(ii, Q, Delta)
        
        
        
def calc_std_Delta_plus(start,
                        end,
                        step,
                        file_read='./P125_21pi_vu/P125_21pi_vu',
                        file_grad='./P125_21pi_vu/grad/P125_21pi_vu',
                        file_Q_Delta='./P125_21pi_vu/hunt_chong/P125_21pi_vu'):

    
    normdata = gd.get_data_norm(file_read=file_read,
                                file_grad=file_grad,
                                file_Q_Delta=file_Q_Delta)
    normdata.geom_param(start,1,1,1)
    
    Deltas = [normdata.read_chong_Delta_matrix(ii) 
              for ii in range(start, end, step)]
    
    std_Delta = np.std(np.hstack(Deltas), axis=(1,2))
    std_Delta_plus = std_Delta*(normdata.ny/(normdata.vtau**2))**6
    
    stdD = h5py.File(file_Q_Delta+f'.{start}_{end}_{step}.h5.stdD', 'w')
    stdD.create_dataset('stdD', data=std_Delta_plus)
    
    
    
def calc_std_Delta_by_variances(start,
                                end,
                                step,
                                file_read='./P125_21pi_vu/P125_21pi_vu',
                                file_grad='./P125_21pi_vu/grad/P125_21pi_vu',
                                file_Q_Delta='./P125_21pi_vu/hunt_chong/P125_21pi_vu'):
    
    normdata = gd.get_data_norm(file_read=file_read,
                                file_grad=file_grad,
                                file_Q_Delta=file_Q_Delta)
    normdata.geom_param(start,1,1,1)
    
    variances = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    means = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    
    for ii in tqdm(range(start, end, step)):
        Delta = normdata.read_chong_Delta_matrix(ii)
        Delta_augmented = np.hstack([Delta, np.flip(Delta, axis=0)])
        mean = np.mean(Delta_augmented, axis=(1,2))
        variance = np.mean(Delta_augmented**2, axis=(1,2))
        normdata.write_Delta_mean_var(ii, mean, variance)
        
        variances[:, int(np.floor((ii-start)/step))] = variance 
        means[:, int(np.floor((ii-start)/step))] = mean
        
    std_Delta = np.sqrt(np.mean(variances, axis=1)-np.mean(means, axis=1)**2)
    std_Delta_plus = std_Delta*(normdata.ny/(normdata.vtau**2))**6
    
    stdD = h5py.File(file_Q_Delta+f'.{start}_{end}_{step}.h5.stdD_var', 'w')
    stdD.create_dataset('stdD', data=std_Delta_plus)



def find_Delta_outlier(start,
                       end,
                       step,
                       file_read='./P125_21pi_vu/P125_21pi_vu',
                       file_grad='./P125_21pi_vu/grad/P125_21pi_vu',
                       file_Q_Delta='./P125_21pi_vu/hunt_chong/P125_21pi_vu'):
    
    normdata = gd.get_data_norm(file_read=file_read,
                                file_grad=file_grad,
                                file_Q_Delta=file_Q_Delta)
    normdata.geom_param(start,1,1,1)
    
    max_Delta = 0
    ii_max = 0
    
    for ii in tqdm(range(start, end, step)):
        Delta = normdata.read_chong_Delta_matrix(ii)
        if np.max(Delta) > max_Delta:
            max_Delta = np.max(Delta)
            ii_max = ii
    
        
    print(max_Delta, ii_max)
        
        
        
def calc_std_Q_plus(start,
                    end,
                    step,
                    file_read='./P125_21pi_vu/P125_21pi_vu',
                    file_grad='./P125_21pi_vu/grad/P125_21pi_vu',
                    file_Q_Delta='./P125_21pi_vu/hunt_chong/P125_21pi_vu'):

    
    normdata = gd.get_data_norm(file_read=file_read,
                                file_grad=file_grad,
                                file_Q_Delta=file_Q_Delta)
    normdata.geom_param(start,1,1,1)
    
    Qs = [normdata.read_hunt_Q_matrix(ii) 
              for ii in range(start, end, step)]
    
    std_Q = np.std(np.hstack(Qs), axis=(1,2))
    std_Q_plus = std_Q*(normdata.ny/(normdata.vtau**2))**6
    
    stdQ = h5py.File(file_Q_Delta+f'.{start}_{end}_{step}.h5.stdQ', 'w')
    stdQ.create_dataset('stdQ', data=std_Q_plus)
    
        
    
    
    
    
