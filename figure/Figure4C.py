#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:09:18 2020

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
"read size data at n=3, only size effects"

grid_chi=101                                               # grid size in figure
grid_n=7                                                 # 13[5], 23[6], 37[7]; 58[8]; 87[9]; 128[10]

chi_ratio_num=grid_chi
chi_ratio_log = np.linspace(-0.4,0.4, chi_ratio_num)    # log scale for chi/chi_neutral
chi_ratio_list=[np.power(10,i) for i in chi_ratio_log]       # chi/chi_neutral

with open("../simulation/LC.txt", "r") as file:
    lcs = eval(file.readline())                          # read lc list

#--------define a function to transform [1+2] to 1+2--------------------
def lc_add(x):                                           # x is the number for life cycles
	lcs[x].sort(reverse=True)
	off_num=len(lcs[x])
	ini=str(lcs[x][0])                                   #initial lcs
	for i in range(1,off_num):
		ini=ini+'+'+str(lcs[x][i])
	return ini

#----------------------------------------
N=3
lc_list=[5,56]

"add a list [[maturity size=3],[],[],;;;]"
data_list=[ ]
	
for chi_cluster in range(13,14):                        # how many figures or cell divisions
	for N_cluster in range(N-1,N):                     # y axes points = a
		all_list=[]                                    # algin all lcs' growth rates
		for i_th in lc_list:                           # read all lcs data = growth rate
				
			with open('../data/data_size_effect/%d_%d_%d.txt'%(N_cluster,chi_cluster,i_th), "r") as file:
				nan=float(np.nan)
				inf=np.inf
				grate = eval(file.readline())          # read growth rate
			data_list.append(np.array([chi_cluster,i_th,grate]))
			
data_size=np.array(data_list)

#-------------------------------------------------------------------------------
"read data of the threshold effects across k, only threshold effects"

grid_num=31                                   # grid size in figure, so check figure.py file first
grid_m_num=11                                        # 13[5], 23[6], 37[7]; 58[8]; 87[9]; 128[10]

b_range=np.array([1,16])
m_range=np.array([0,0.1])
k_range=np.array([1,2,3,4,5,6,7,8])          # first take 1,2,3, 
k_cluster_list=np.array([0,1,2,3,4,5,6])

grid_b=np.linspace(b_range[0],b_range[1],num=grid_num,endpoint=True)
grid_m=np.linspace(m_range[0],m_range[1],num=grid_m_num,endpoint=True)

cell_number=[3,4,5,6,7,8]
end_lc_list=[3,7,13,23,37,58]               # 13[5], 23[6], 37[7]; 58[8]; 87[9]; 128[10]     # how many lcs we care

"add a list [[maturity size=3],[],[],;;;]"
data_game=[]
max_size=8
for i in range(5,max_size-2):
	n=cell_number[i]
	end_lc=end_lc_list[i]
	#-------------------------------------------------------------------------------
	"read max and min dps from data into result []"
	num_k=len(k_cluster_list)
	for b_cluster in range(18,19):                             # b=10
		for m_cluster in range(1,2):                           # y axes points = a
			for k_cluster in k_cluster_list:
				for i_th in lc_list:                            # read all lcs data = growth rate
					with open('../data/data_threshold_effect/%d_%d_%d_%d.txt'%(b_cluster,m_cluster,k_cluster,i_th), "r") as file:
						nan=float(np.nan)
						inf=np.inf
						grate = eval(file.readline())          # read growth rate
					data_game.append([b_cluster,m_cluster,k_cluster,i_th,grate])
data_game=np.array(data_game)
	
#-------------------------------------------------------------------------------
"read data of both effects across k,"
grid_num=7                                                # grid size in figure, so check figure.py file first
t_ratio=0.4

result_list=[[],[]]                                      # to store data of each k value
optimal_matrix_list=[np.zeros(shape=(grid_num,grid_num))*np.nan for i in range(2)]

"add a list [[maturity size=3],[],[],;;;]"
data_both=[]
for t_pterb_cluster in range(2,3):                        # perturbation at n=3
	for k_cluster in range(0,grid_num):                    # y axes points = a
		all_list=[]                                    # algin all lcs' growth rates
		for i_th in lc_list:                           # read all lcs data = growth rate
										
			with open('../data/data_size_threshold_chi_%s/%d_%d_%d.txt'%(t_ratio,t_pterb_cluster,k_cluster,i_th), "r") as file:
				nan=float(np.nan)
				inf=np.inf
				grate = eval(file.readline())          # read growth rate
			data_both.append(np.array([t_ratio,t_pterb_cluster,k_cluster,i_th,grate]))

data_both=np.array(data_both)

#================= draw figures====================================================
c = np.arange(1, 58 + 1)

norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])
 
fig, ax = plt.subplots(1, 1, figsize=(4, 4.5))
fig.subplots_adjust(top=0.82)
ax.set_axisbelow(True)
k_list=[1,2,3,4,5,6,7]

"size figure"
s_size=1
for i in lc_list:
	y=data_size[np.where(data_size[:,1]==i)][0,2]
	ax.scatter(k_list, [y for i in range(7)], s=s_size,marker="^",c=cmap.to_rgba(i + 1),alpha=0.9)
	if s_size==1:
		ax.plot(k_list, [y for i in range(7)], ':',label="Size effects",linewidth=1.5,c=cmap.to_rgba(i + 1),alpha=0.95)
	else:
		ax.plot(k_list, [y for i in range(7)], ':',linewidth=1.5,c=cmap.to_rgba(i + 1),alpha=0.95)
	
	"game figure"	
	y_game=data_game[np.where(data_game[:,3]==i)][:,4]
	ax.scatter(k_list, y_game, s=s_size,marker="^",c=cmap.to_rgba(i + 1),alpha=0.9)
	if s_size==1:
		ax.plot(k_list, y_game, '--',label="Threshold effets",linewidth=2,c=cmap.to_rgba(i + 1),alpha=0.95)
	else:
		ax.plot(k_list, y_game, '--',linewidth=2,c=cmap.to_rgba(i + 1),alpha=0.95)
		
	"size and game figure"	
	y_both=data_both[np.where(data_both[:,3]==i)][:,4]
	ax.scatter(k_list, y_both, s=s_size,marker="^",c=cmap.to_rgba(i + 1),alpha=0.9)
	
	if  s_size==1:
		ax.plot(k_list, y_both, '-',label="Combined effects",linewidth=2,c=cmap.to_rgba(i + 1),alpha=0.75)
	else:
		ax.plot(k_list, y_both, '-',linewidth=2,c=cmap.to_rgba(i + 1),alpha=0.75)

	s_size+=1
	
leg=ax.legend(frameon=False,loc='upper center', bbox_to_anchor=(0.7, 0.98), shadow=True, ncol=1, fontsize=10)
for i in range(3):
	leg.legendHandles[i].set_color('k')
	
ax.annotate(lc_add(lc_list[0]), (1.89, 1.8),color=cmap.to_rgba(5 + 1))	
ax.annotate(lc_add(lc_list[1]), (4.5, 2.2),color=cmap.to_rgba(56 + 1))	

ax.set_xlabel(r"Contribution threshold $k$",fontsize=12)
ax.set_xticks([i for i in range(1,8)], minor=False)
ax.set_ylim([0.65, 3.5])
ax.set_ylabel(r"Population growth rate $\lambda$",fontsize=12)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('left')

plt.show();

			
			





