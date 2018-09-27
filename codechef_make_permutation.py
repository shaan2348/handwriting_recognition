# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 23:09:03 2018

@author: Shantanu
"""

t = int(input())
list1= []
list2 = []
list3 = []
for i in range(t):
    n = int(input())
    list1 = list(map(int, input()))
    
    for k in range(1,len(list1)+1):
        for j in list1:
            if j not in list2:
                list2.append(j)
                list1.remove(j)
    list3.append(len(list1))
for i in list3:
    print(i)