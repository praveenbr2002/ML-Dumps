#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:15:24 2021

@author: bhajji
"""


def minCost(str1, str2, n):  
  
    cost = 0
  
    # For every character of str1 
    for i in range(n):  
  
        # If current character is not  
        # equal in both the strings  
        if (str1[i] != str2[i]):  
  
            # If the next character is also different in both  
            # the strings then these characters can be swapped  
            if (i < n - 1 and str1[i + 1] != str2[i + 1]):  
                swap(str1[i], str1[i + 1])  
                cost += 1
              
            # Change the current character  
            else:  
                cost += 1
              
    return cost  
  
# Driver code  
if __name__ == '__main__':  
  
    str1 = "Quantom"
    str2 = "Quantum"
    n = len(str1)  
  
    print(minCost(str1, str2, n))  
