#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:32:50 2024

@author: benedikt_n
"""

import numpy as np
import get_data_fun as gd


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
    
        # Delta = 27/4*np.linalg.det(G.transpose((2,3,4,0,1)))**2+Q**3
        
        # Alternative
        SijSjkSki = np.einsum('ijlmn,jklmn,kilmn->lmn',S,S,S)
        WijWjkSki = np.einsum('ijlmn,jklmn,kilmn->lmn',Omega,Omega,S)
        R = -1/3*(SijSjkSki+3*WijWjkSki)
        
        Delta = 27/4*R**2+Q**3
    
        normdata.write_Q_Delta(ii, Q, Delta)
    