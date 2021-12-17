#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:52:26 2019

@author: gao
"""

#--------T vs X0-----------------
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

#-----------------------------------------------------------------------------
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

#---background---
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.4,alpha=0.4)

'''parameter initial values'''
x = np.linspace(1, 7, 7)
a_range=[0.4,0.7,1, 1.4,1.8]
N=3
a_num=5
##___-t function
t=[1,1,1,1,1,1,1]

def CHI_equal0(item):
	t=np.log((item+1)/item)	
	return t

def CHI_equal(item):
	t=[1,1,1,1,1,1,1]	
	return t
#==============================fig===================
#--color---------------------
c = np.arange(1, 5 + 1)
color1=['#d7191c','#fdae61','#ffffbf','#abdda4','#2b83ba']
line_style=[':','-.','','--','-']		
maker_list=[".","P",'',"*","x"]	
marker="H"
marker_p="8"
marker_size=80
color_line='#377eb8'

#-----fig1  non-neutral one-----------


for j in range(len(a_range)):
	a=a_range[j]
	t=CHI_equal(x)
	t[N-1]=t[N-1]*a
	if a!=1:
		ax.plot(x, t,label=r"$t_3/t_3^{neutral}$="+str(round(a_range[j],1)),linestyle='--', linewidth=1, c=color_line,alpha=0.2)
		ax.scatter(x, t,label=r"$t_3/t_3^{neutral}$="+str(round(a_range[j],1)), marker=marker_p,s=marker_size-5,c=color_line)
		
#-----bacground ---neutral-----------
ax.scatter(x, CHI_equal(x),label=r"Size neutral", c='grey', marker=marker,s=marker_size,alpha=1)
ax.plot(x, CHI_equal(x),label=r"Size neutral", c='grey', linestyle="-",linewidth=1,alpha=0.6)
ax.annotate("Neutral growth", (5., t[6]+0.06), fontsize=12,color='grey',alpha=1)	
	
##-----add notation-----------
pos_x=N
x_po =[round(a,2) for  a in a_range]
y_po=[t[pos_x]*i+0.04 for  i in a_range]
#
label=[]
for lc in range(a_num):
	label.append(r"$\chi_3$="+str(round(x_po[lc],1)))
	
for i, txt in enumerate(label):
	if i!=2:
		   ax.annotate(txt, (pos_x-0.45, y_po[i]+0.), fontsize=12,color=color_line)	
	
#------remove ticks and top and right frames
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none') 

plt.xlabel('Organism size $n$',fontsize=16)
plt.ylabel('Normalised cell increment component $\chi_{n}$',fontsize=16)

plt.ylim(0.28,1.85)

plt.show()

#fig.savefig('../figure/Figure_2A.pdf',
#			 bbox_inches = 'tight')   # save figures
#

