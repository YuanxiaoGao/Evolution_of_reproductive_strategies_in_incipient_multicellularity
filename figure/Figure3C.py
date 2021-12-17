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
#-------------------------------------------------------------------------------
"build para[b,c] grids"

grid_num=31                                   # grid size in figure, so check figure.py file first
grid_m_num=11                                        # 13[5], 23[6], 37[7]; 58[8]; 87[9]; 128[10]

b_range=np.array([1,16])
m_range=np.array([0,0.1])
k_range=np.array([1,2,3,4,5,6,7,8])          # first take 1,2,3, 
k_cluster_list=np.array([0,1,2,3,4,5,6])

grid_b=np.linspace(b_range[0],b_range[1],num=grid_num,endpoint=True)
grid_m=np.linspace(m_range[0],m_range[1],num=grid_m_num,endpoint=True)
np.shape(grid_m)
#grid_b[22]--b11

cell_number=[3,4,5,6,7,8]
end_lc_list=[3,7,13,23,37,58]             # 7[4], 13[5], 23[6], 37[7]; 58[8]; 87[9]; 128[10]     # how many lcs we care


maturity_size=6
for i in range(maturity_size-3,maturity_size-2):
	n=cell_number[i]
	end_lc=end_lc_list[i]

	#-------------------------------------------------------------------------------
	"read max and min dps from data into result []"
	num_k=len(k_cluster_list)
	result=[]                                                  # to store all data of b,m,k,lc

	for b_cluster in range(0,grid_num):                        # how many figures or cell divisions
		for m_cluster in range(0,grid_m_num):                    # y axes points = a
			for k_cluster in k_cluster_list:
#				all_list=[]                                    # algin all lcs' growth rates
				for i_th in range(0,end_lc):                   # read all lcs data = growth rate
					
					with open('../data/data_threshold_effects/%d_%d_%d_%d.txt'%(b_cluster,m_cluster,k_cluster,i_th), "r") as file:
						nan=float(np.nan)
						inf=np.inf
						grate = eval(file.readline())          # read growth rate
						result.append(np.array([b_cluster,m_cluster,k_cluster+1,i_th,grate]) )

result1=np.array(result)           # [b, m, k, lc_i, lambda]
#-------------------------------------------------------------------------------
"data for fixed b/c=11"
b_index=18
data_b11=result1[np.where(result1[:,0]==b_index)]                                       # take the data with m_grid=15

	#for picked m values
m_scatter_index =[ 1 ]                                                                    # choose five m values to check the effects of k

data_m_values=[]                                                                           # put data into five block based on m values     
for i in range(1):
	data_each_m=data_b11[np.where(data_b11[:,1]==m_scatter_index[i])]                      # take the data for each m value
	data_each_m = data_each_m[data_each_m[:,2].argsort()]                                  # sort by k then by i_th: First sort doesn't need to be stable.
	data_each_m = data_each_m[data_each_m[:,3].argsort(kind='mergesort')]
	data_each_lc=np.vsplit(data_each_m,end_lc)                                           # split for each life cycle acorss this k values
	data_each_lc=np.array(data_each_lc)
	data_m_values.append(data_each_lc)


##%%	
#================= draw figures====================================================
c = np.arange(1, 58 + 1)

norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])
 
fig, ax = plt.subplots(1, 1, figsize=(2.5, 5))
fig.subplots_adjust(top=0.82)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.6,alpha=0.4)

k_list=[1,2,3,4,5,6,7]

data_m_values[0]
s_size=50
for i in range(end_lc):
			
	ax.scatter(k_list, data_m_values[0][i][:,-1], s=s_size,marker="o",c=cmap.to_rgba(i + 1),alpha=0.9)
	ax.plot(k_list, data_m_values[0][i][:,-1], linewidth=0.15,c=cmap.to_rgba(i + 1),alpha=0.95)
	# draw xy labels
ax.set_xlabel(r"Contribution threshold $k$",fontsize=12)
ax.set_xticks([i for i in range(1,8)], minor=False)
ax.set_ylim([0.85, 2.5])
ax.set_ylabel(r"Population growth rate $\lambda$",fontsize=12)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none') 

plt.show();

