#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:09:46 2024

@author: benedikt_n
"""

import shap_config as sc

shaps = sc.shap_conf()

shaps.calc_cf(fileuvw='../../../data/cremades/P125_21pi_vu/P125_21pi_vu',
              fileQ='./P125_21pi_vu_Q_divide/P125_21pi_vu',
              file_output='./P125_21pi_vu_cf/P125_21pi_vu',
              start=2967,
              end=9999)