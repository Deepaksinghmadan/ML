# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:29:50 2018

@author: deepak
"""

#get a string which is n (non-negative integer) copies of a given string
def multiplystring(str,n):
    a = ""
    for i in range(n):
        a=a + str
    return a

print(multiplystring("deepak",3))
