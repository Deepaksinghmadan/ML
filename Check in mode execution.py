# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 18:36:38 2018

@author: deepak
"""

#1_python executing in 32 or 64 bit shell
import struct
print(struct.calcsize("P") * 8)

import sys
#2_python executing in 32 or 64 bi shell
print(sys.version) # easy to remember .


#3_mode cheacking
import platform
print(platform.architecture()[0])