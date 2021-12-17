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

cell_number=[3,4,5,6,7,8]
end_lc_list=[3,7,13,23,37,58]             # 7[4], 13[5], 23[6], 37[7]; 58[8]; 87[9]; 128[10]     # how many lcs we care

k=5
k_cluster=k-1

maturity_size=8
for i in range(maturity_size-3,maturity_size-2):
	n=cell_number[i]
	end_lc=end_lc_list[i]

	#-------------------------------------------------------------------------------
	"read max and min dps from data into result []"
	num_k=len(k_cluster_list)
	result=[]                                                  # to store all data of b,m,k,lc

	for b_cluster in range(0,grid_num):                        # how many figures or cell divisions
		for m_cluster in range(0,grid_m_num):                    # y axes points = a
			for i_th in range(0,end_lc):                   # read all lcs data = growth rate
				
				with open('../data/data_threshold_effects/%d_%d_%d_%d.txt'%(b_cluster,m_cluster,k_cluster,i_th), "r") as file:
					nan=float(np.nan)
					inf=np.inf
					grate = eval(file.readline())          # read growth rate
					result.append(np.array([b_cluster,m_cluster,k_cluster+1,i_th,grate]) )

result1=np.array(result)           # [b, m, k, lc_i, lambda]
#-------------------------------------------------------------------------------
"data for fixed b/c=10"
b_index=18
data_b11=result1[np.where(result1[:,0]==b_index)]               # take the data with m_grid=15

	#for picked m values
m_scatter_index =[ 1 ]                                               # choose five m values to check the effects of k

data_m_values=[]                                                     # put data into five block based on m values     
for i in range(1):
	data_each_m=data_b11[np.where(data_b11[:,1]==m_scatter_index[i])]          # take the data for each m value
	data_each_m = data_each_m[data_each_m[:,2].argsort()]                      # sort by k then by i_th: First sort doesn't need to be stable.
	data_each_m = data_each_m[data_each_m[:,3].argsort(kind='mergesort')]
	data_each_lc=np.vsplit(data_each_m,end_lc)                                  # split for each life cycle acorss this k values
	data_each_lc=np.array(data_each_lc)
	data_m_values.append(data_each_lc)
type(data_m_values[0][0])
##%%
#================= draw figures====================================================
c = np.arange(1, 58 + 1)

norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])
 
fig, ax = plt.subplots(1, 1, figsize=(2.5, 5))
fig.subplots_adjust(top=0.82)
#fig.suptitle("Maximum size n=%s, benefit to cost ratio b/c=%s"%(n,round(grid_b[b_index],2)),fontsize=20)                   # whole title

ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.6,alpha=0.4)

k_list=[1,2,3,4,5,6,7]
s_size=50
i_col=0
marker0="o"

for i in data_m_values[0]:
	if i[0][3]==0:
		for item in range(2,9):
			ax.scatter([item], i[0][4], s=s_size,marker=marker0, c=cmap.to_rgba(i_col + 1),alpha=0.9)
		
	elif i[0][3]>0 and i[0][3]<	end_lc_list[0]:
		for item in range(3,9):
			ax.scatter([item], i[0][4], s=s_size,marker=marker0, c=cmap.to_rgba(i_col + 1),alpha=0.9)

	elif i[0][3]>=end_lc_list[0] and i[0][3]<	end_lc_list[1]:
		for item in range(4,9):
			ax.scatter([item], i[0][4], s=s_size,marker=marker0, c=cmap.to_rgba(i_col + 1),alpha=0.9)

	elif i[0][3]>=end_lc_list[1] and i[0][3]<	end_lc_list[2]:
		for item in range(5,9):
			ax.scatter([item], i[0][4], s=s_size,marker=marker0, c=cmap.to_rgba(i_col + 1),alpha=0.9)

	elif i[0][3]>=end_lc_list[2] and i[0][3]<	end_lc_list[3]:
		for item in range(6,9):
			ax.scatter([item], i[0][4], s=s_size,marker=marker0, c=cmap.to_rgba(i_col + 1),alpha=0.9)

	elif i[0][3]>=end_lc_list[3] and i[0][3]<	end_lc_list[4]:
		for item in range(7,9):
			ax.scatter([item], i[0][4], s=s_size,marker=marker0, c=cmap.to_rgba(i_col + 1),alpha=0.9)

	elif i[0][3]>=end_lc_list[4] and i[0][3]<	end_lc_list[5]:
		for item in range(8,9):
			ax.scatter([item], i[0][4], s=s_size,marker=marker0, c=cmap.to_rgba(i_col + 1),alpha=0.9)

	i_col+=1		


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none') 
ax.tick_params(labelleft=False) 

	# draw xy labels
ax.set_xlabel(r"Maximum maturity size $N$",fontsize=12)
ax.set_xticks([i for i in range(2,9)], minor=False)
ax.set_ylim([0.85, 2.5])

plt.show();

