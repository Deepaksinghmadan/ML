# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:11:10 2018

@author: deepak
"""

# "Is" has been added to the front

def a(s):
    if len(s) >= 2 and s[ :2]=="is":
        return s
    else:
        return "is" + s
    
print(a("ideepak"))
print(a("iskl"))
