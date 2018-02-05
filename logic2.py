# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:04:35 2018

@author: deepak
"""

def a(x,y,z):
    sum = x+y+z
    if x==y==z:
        sum = sum*3
    return sum
print(a(5,5,5))
print(a(4,5,3))
