#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:09:46 2024

@author: benedikt_n
"""

import cf_utils

cf_utils.calc_cf(fileuvw='/data/cremades/P125_21pi_vu/P125_21pi_vu',
              #fileuvw='./P125_21pi_vu/P125_21pi_vu',
              #fileQ='./P125_21pi_vu_Q_divide/P125_21pi_vu',
              file_output='/data2/nils/P125_21pi_vu_cf/P125_21pi_vu',
              start=2967,
              #start=4544,
              #end=4555,
              end=9999
              #high_precision=True
              )
