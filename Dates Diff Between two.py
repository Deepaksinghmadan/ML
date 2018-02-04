# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 00:45:12 2018

@author: deepak
"""

from datetime import date
f_date = date(2014, 7, 2)
l_date = date(2014, 7, 11)
delta = l_date - f_date
print(delta.days)