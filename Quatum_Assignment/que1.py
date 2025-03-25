#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:33:24 2021

@author: bhajji
"""


def findMinElement(ar, low, high): 
    # condition to handle the case when array is not rotated
    if high < low: 
        return ar[0] 
  
    # If there is only one element left in array
    if high == low: 
        return ar[low] 
  
    # Find mid value
    mid = int((low + high)/2) 
  
    # Check if element (mid+1) is minimum element. Consider 
    # the cases like [3, 4, 5, 1, 2] 
    if mid < high and ar[mid+1] < ar[mid]: 
        return ar[mid+1] 
  
    # Check if mid itself is minimum element 
    if mid > low and ar[mid] < ar[mid - 1]: 
        return ar[mid] 
  
    # Decide whether we need to go to left half or right half 
    if ar[high] > ar[mid]: 
        return findMinElement(ar, low, mid-1) 
    return findMinElement(ar, mid+1, high) 
  
# Driver program to test above functions 
arr1 = [5, 6, 1, 2, 3, 4] 
n1 = len(arr1) 
print(str(findMinElement(arr1, 0, n1-1)))
