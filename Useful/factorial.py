# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:01:47 2020

@author: Mariusz
"""


def fact(n):
    if n == 1:
        return 1
    else:
        return n * fact(n - 1)
    
print (fact(5))