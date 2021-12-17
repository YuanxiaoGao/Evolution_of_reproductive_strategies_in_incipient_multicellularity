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
grid_num=31                                          # grid size in figure, so check figure.py file first
grid_m_num=11                                        # 13[5], 23[6], 37[7]; 58[8]; 87[9]; 128[10]

b_range=np.array([1,16])
m_range=np.array([0,0.1])
k_range=np.array([1,2,3,4,5,6,7,8])          # first take 1,2,3, 
k_cluster_list=np.array([0,1,2,3,4,5,6])

grid_b=np.linspace(b_range[0],b_range[1],num=grid_num,endpoint=True)
grid_m=np.linspace(m_range[0],m_range[1],num=grid_m_num,endpoint=True)

cell_number=[3,4,5,6,7,8]
end_lc_list=[3,7,13,23,37,58]             # 13[5], 23[6], 37[7]; 58[8]; 87[9]; 128[10]     # how many lcs we care

"set values for the m,b/c"
max_size=8

m_ind=1    #0.01
b_ind=18   #10
b_value=grid_b[b_ind]
m_value=grid_m[m_ind]
n_k_size=max_size-1
"add a list [[maturity size=3],[],[],;;;]"

result0=[]
for i in range(0,max_size-2):
	n=cell_number[i]
	end_lc=end_lc_list[i]
	#-------------------------------------------------------------------------------
	"read max and min dps from data into result []"
	
	num_k=len(k_cluster_list)
	result=[[] for i in range(num_k)]                          # to store data of each k value
	
	for b_cluster in range(b_ind,b_ind+1):                        # how many figures or cell divisions
		for m_cluster in range(m_ind,m_ind+1):                    # y axes points = a
			for k_cluster in k_cluster_list:
				all_list=[]                                    # algin all lcs' growth rates
				for i_th in range(0,end_lc):                   # read all lcs data = growth rate
					
					with open('../data/data_threshold_effects/%d_%d_%d_%d.txt'%(b_cluster,m_cluster,k_cluster,i_th), "r") as file:
						nan=float(np.nan)
						inf=np.inf
						grate = eval(file.readline())          # read growth rate
					all_list.append(grate)
					result0.append([b_cluster,m_cluster,k_cluster,i_th,grate])           # read all data
		            
result0=np.array(result0)

lambda_matrix=np.zeros(shape=(n_k_size,n_k_size))*np.nan 
for k_values in range(max_size-1):  #num_k                               # for the k we consider here is 2,3,4
	data_k=result0[np.where(result0[:,2]==k_values)]
	data_k1=np.unique(data_k,axis=0)

	for i in range(0,max_size-1):		               # n 
		if  i==0:	
			max_lambda=data_k1[:1,:][0][-1]			
		else:			
			end=end_lc_list[i-1]		
			max_lambda=np.max(data_k1[:end,:][:,-1])
			
		lambda_matrix[i][k_values]=max_lambda
		
#===================================mfig2==============================================

"add a list [[maturity size=3],[],[],;;;]"
opt_matrix=[]
#max_size=8
for i in range(0,max_size-2):
	n=cell_number[i]
	end_lc=end_lc_list[i]

	#-------------------------------------------------------------------------------
	"read max and min dps from data into result []"
	
	num_k=len(k_cluster_list)
	result=[[] for i in range(num_k)]                          # to store data of each k value
	
	for b_cluster in range(0,grid_num):                        # how many figures or cell divisions
		for m_cluster in range(0,grid_m_num):                    # y axes points = a
			for k_cluster in k_cluster_list:
				all_list=[]                                    # algin all lcs' growth rates
				for i_th in range(0,end_lc):                   # read all lcs data = growth rate
														
					with open('../data/data_threshold_effects/%d_%d_%d_%d.txt'%(b_cluster,m_cluster,k_cluster,i_th), "r") as file:
						nan=float(np.nan)
						inf=np.inf
						grate = eval(file.readline())          # read growth rate
					all_list.append(grate)
		            
				# if all results are np.nan				
				all_list_f = np.array(all_list, dtype=np.float128)
				all_lc=all_list_f.tolist()
				if all(i is nan for i in all_lc):
					for idx in range(len(k_cluster_list)):
					
						if k_cluster==idx:
							result[idx].append(np.nan)          # k=2
				else:
					max_index=all_lc.index(np.nanmax(all_lc))  # the max lc_ID
					
					for idx in range(len(k_cluster_list)):
					
						if k_cluster==idx:
							result[idx].append([b_cluster,m_cluster,max_index])           # k=2
	#-------------------------------------------------------------------------------
	"get figure matri"
	
	optimal_matrix=[np.zeros(shape=(grid_m_num,grid_num))*np.nan for i in range(num_k)]
	
	for k_values in range(num_k):                                 # for the k we consider here is 2,3,4
		for bmi in result[k_values]:		                          # for [b,m,max_index] under this k value
			for row in range(grid_num):                           # b
				for col in range(grid_m_num):                       # m
	
					if abs(bmi[0]-row)<10**(-6) and abs(bmi[1]-col)<10**(-6):
						optimal_matrix[k_values][col][row]=bmi[2]     # max_lc_ID and min_lc_ID
	
	opt_matrix.append(optimal_matrix)
#-------------------------------------------------------------------------------
"set values for the m,b/c"
m_ind=1    #0.01
b_ind=18   #10
b_value=grid_b[b_ind]
m_value=grid_m[m_ind]

n_k_size=max_size-1

opt_lc=[]
n_k_matrix=np.zeros(shape=(n_k_size,n_k_size))*np.nan 
for row in range(1,n_k_size):
	for col in range(1,n_k_size):
		if row>=col:
			n_k_matrix[row][col]=opt_matrix[row-1][col][m_ind,b_ind]
			opt_lc.append(opt_matrix[row-1][col][m_ind,b_ind])

n_k_matrix[0][0]=0		
optimal_matrix1=[]		
optimal_matrix1.append(n_k_matrix)

opt_lc_uni=np.unique(np.array(opt_lc))                                   # appeared one
others=np.array(list(set([i for i in range(end_lc)])-set(opt_lc_uni)))   # nonappeared one
pos=np.array([i for i in range(end_lc)])                                 # position of all

##%%
#================plot figure==============================-----
"draw figures"

# -- read tick labels---------
cwd = os.getcwd()                                        # Get the current working directory (cwd)
files = os.listdir(cwd)                                  # Get all the files in that directory
with open("../simulation/LC.txt", "r") as file:
    lcs = eval(file.readline())                          # read lc list

#--------define a function to transform [1+2] to 1+2--------------------
def lc_add(x):                                          # x is the number for life cycles
	lcs[x].sort(reverse=True)
	off_num=len(lcs[x])
	ini=str(lcs[x][0])                                   #initial lcs
	for i in range(1,off_num):
		ini=ini+'+'+str(lcs[x][i])
	return ini

#----------------------------------------

# ----set fig frame--------	
fig, ax = plt.subplots(1, 1, figsize=(4.7, 4.7))

color1=['#ffffe5','#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#993404','#662506']
cmap1 = LinearSegmentedColormap.from_list('mycmap', color1) 
norm1 = plt.Normalize(0.98, 2.52)	
	
im1 = ax.imshow(lambda_matrix, interpolation=None, origin='lower',cmap=cmap1, norm=norm1)

#---create text annotations.
text = ax.text(0, 0, "1+1",ha="center", va="center", color="darkgrey",fontsize=10)
for row in range(1,n_k_size):
	for col in range(1,n_k_size):
		if row>=col:	
			if lambda_matrix[row, col]<2:
				text = ax.text(col, row, lc_add(int(n_k_matrix[row, col])),ha="center",
			 va="center", color="dimgray",fontsize=10)
			else:
				text = ax.text(col, row, lc_add(int(n_k_matrix[row, col])),ha="center",
			 va="center", color="silver",fontsize=10)
			
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#---xy label-----
test=np.linspace(0,1,7,endpoint=True)
# artifical x and y ticks
size_num=n_k_size-1
y_num=n_k_size-1

x_lable_f=[i for i in range(1,8)]
x_label=[str(i) for i in x_lable_f]

ax.set_xticks([i*size_num for i in test], minor=False)
ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(x_label))

y_lable_f=[i for i in range(2,9)]
y_label=[str(i) for i in y_lable_f]

ax.set_yticks([i*size_num for i in test], minor=False)
ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(y_label))

# ---colorbar-----
cbar_ax1 = fig.add_axes([0.92, 0.25, 0.02, 0.5])
cbar1=fig.colorbar(im1, cax=cbar_ax1, orientation="vertical",norm=norm1, boundaries=None)
cbar1.ax.set_ylabel(r'Population gorwth rate $\lambda$', rotation=90,fontsize=12)
cbar1.ax.tick_params(labelsize=10,length=3,direction='in')
# x and y label
ax.set_xlabel(r'Contribution threshold $k$',fontsize=14)
ax.set_ylabel(r'Maximum maturity size $N$',fontsize=14)

plt.show()

