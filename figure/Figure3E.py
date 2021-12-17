#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:40:40 2018
@author: gao
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib as mpl
import os
from matplotlib.colors import LinearSegmentedColormap

with open("..simulation/LC.txt", "r") as file:
    lcs = eval(file.readline())                          # read lc list

#--------define a function to transform [1+2] to 1+2--------------------
def lc_add(x):                                            # x is the number for life cycles
	lcs[x].sort(reverse=True)
	off_num=len(lcs[x])
	ini=str(lcs[x][0])                                     #initial lcs
	for i in range(1,off_num):
		ini=ini+'+'+str(lcs[x][i])
	return ini

#-------------------------------------------------------------------------------
"build para[b,c] grids"
grid_num=101                                              # grid size in figure, so check figure.py file first

b_range=np.array([1,16])
m_range=np.array([0,0.1])
k_range=np.array([1,2,3,4,5,6,7,8])                        # first take 1,2,3, 
k_cluster_list=np.array([0,1,2,3,4,5,6])

grid_b=np.linspace(b_range[0],b_range[1],num=grid_num,endpoint=True)

cell_number=[3,4,5,6,7,8]
end_lc_list=[3,7,13,23,37,58]                             # 7[4], 13[5], 23[6], 37[7]; 58[8]; 87[9]; 128[10]     
result=[]                                                  # to store all data of b,m,k,lc
maturity_size=8
for i in range(maturity_size-3,maturity_size-2):
	n=cell_number[i]
	end_lc=end_lc_list[i]

	#-------------------------------------------------------------------------------
	"read max and min dps from data into result []"
	num_k=len(k_cluster_list)

	for b_cluster in range(0,grid_num):                  # how many figures or cell divisions
		for i_th in range(0,end_lc):                    # read all lcs data = growth rate
			
			with open('../data/data_figure3E.zip/%d_%d.txt'%(b_cluster,i_th), "r") as file:
				nan=float(np.nan)
				inf=np.inf
				grate = eval(file.readline())          # read growth rate
				result.append(np.array([b_cluster,i_th,grate]) )

result1=np.array(result)                                 # [b, m, k, lc_i, lambda]
	
#================= draw figures====================================================
fig, ax = plt.subplots(1, 1, figsize=(4, 4.5))
fig.subplots_adjust(top=0.89)

c = np.arange(1, 58 + 1)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])
 
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.4,alpha=0.3)

binary_lc=[0,2,6,12,22,36,57]

#---------linear plot------scatter plot------------
s_size=5
for i in binary_lc:
	line_data=result1[np.where(result1[:,1]==i)]
	ax.plot(grid_b,line_data[:,-1],c=cmap.to_rgba( i+ 1),linewidth=3,alpha=0.9)
	ax.scatter(grid_b, line_data[:,-1], s=s_size,marker="o",
			label=lc_add(i),c=cmap.to_rgba(i + 1),alpha=0.9)
lgnd=ax.legend(frameon=False,loc='upper center', bbox_to_anchor=(0.2, 1.05), shadow=True, ncol=1, fontsize=10)
for handle in lgnd.legendHandles:
    handle.set_sizes([60.0])

	# draw xy labels
ax.set_xlabel(r"Ratio of benefit to cost $b/c$",fontsize=12)
ax.set_ylabel(r"Population growth rate $\lambda$",fontsize=12)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none') 

plt.show();

