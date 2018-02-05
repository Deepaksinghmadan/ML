# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:48:26 2018

@author: deepak
"""

from datetime import date
n1 = date(input("Enter starting date"))
n2 = date(input("Enter End date"))
days = n2-n1
print(days.days)


from datetime import date
f_date = date(2014, 7, 2)
l_date = date(2014, 7, 11)
delta = l_date - f_date
print(delta.days)