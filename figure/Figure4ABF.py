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

grid_num=7                                   # grid size in figure, so check figure.py file first
t_ratio_list=[0.4,1.5]

result_list=[[],[]]                          # to store data of each k value
optimal_matrix_list=[np.zeros(shape=(grid_num,grid_num))*np.nan for i in range(2)]
end_lc=58

"add a list [[maturity size=3],[],[],;;;]"
max_size=8

#=====check the first column which are co-optimal one =========================				
all_list=[]                                    # algin all lcs' growth rates
for fig_num in range(1,2):
	
	for t_pterb_cluster in range(6,7):                        # how many figures or cell divisions
		for k_cluster in range(6,7):                         # y axes points = a
			for i_th in range(0,end_lc):                     # read all lcs data = growth rate
											
				with open('../data/data_size_threshold_chi%s/%d_%d_%d.txt'%(t_ratio_list[fig_num],t_pterb_cluster,k_cluster,i_th), "r") as file:
					nan=float(np.nan)
					inf=np.inf
					grate = eval(file.readline())          # read growth rate
				all_list.append(np.array([t_pterb_cluster,i_th,grate]))

for fig_num in range(2):
	
	for t_pterb_cluster in range(0,grid_num):                # how many figures or cell divisions
		for k_cluster in range(0,grid_num):                 # y axes points = a
			all_list=[]                                    # algin all lcs' growth rates
			for i_th in range(0,end_lc):                   # read all lcs data = growth rate
											
				with open('../data/data_size_threshold_chi%s/%d_%d_%d.txt'%(t_ratio_list[fig_num],t_pterb_cluster,k_cluster,i_th), "r") as file:
					nan=float(np.nan)
					inf=np.inf
					grate = eval(file.readline())          # read growth rate
				all_list.append(np.array([i_th,grate]))
			all_list_f = np.array(all_list, dtype=np.float128)
			all_lc=all_list_f.tolist()
			all_lc=np.array(all_lc)
			if all(i is nan for i in all_lc[:,1]):
				result_list[fig_num].append(np.nan)          # k=2
			else:
				max_v=np.nanmax(all_lc[:,1])
				id_th=np.where(all_lc[:,1]==max_v)
	#			id_th[0][0]
				max_index=all_lc[id_th[0][0]][0]  # the max lc_ID
				result_list[fig_num].append([t_pterb_cluster,k_cluster,int(max_index)])           # k=2
	#-------------------------------------------------------------------------------
	"get figure matri"
	for item in result_list[fig_num]:                                 # for the k we consider here is 2,3,4
		optimal_matrix_list[fig_num][item[0]][item[1]]=item[2]

#================plot figure==============================-----
"draw figures"
from matplotlib import cm

# -- read tick labels---------
cwd = os.getcwd()                                        # Get the current working directory (cwd)
files = os.listdir(cwd)                                  # Get all the files in that directory
with open("/Users/gao/Desktop/life-cycles-with-multiplayer-game/SimulationCode/partition/LC.txt", "r") as file:
    lcs = eval(file.readline())                          # read lc list

#--------define a function to transform [1+2] to 1+2--------------------
def lc_add(x):                                            # x is the number for life cycles
	lcs[x].sort(reverse=True)
	off_num=len(lcs[x])
	ini=str(lcs[x][0])                                   #initial lcs
	for i in range(1,off_num):
		ini=ini+'+'+str(lcs[x][i])
	return ini

# ----set fig frame--------
fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1,0.1,1]},figsize=(10, 5))

#-----colormap--------------
cmap = cm.get_cmap('jet', 58)
norm = plt.Normalize(0, end_lc)	
cmap.set_bad('grey',alpha=0.4)	                         # color for g3=1 i.e. s_t=nan
cmap.set_under('grey',alpha=0.6)
pos=np.array([i+0.5 for i in range(end_lc)])                                 # position of all

#----figure------------------
im = ax[0].imshow(optimal_matrix_list[0], interpolation=None, origin='lower',cmap=cmap, norm=norm)
for i in range(7):
	optimal_matrix_list[1][i][0]=np.nan
im = ax[2].imshow(optimal_matrix_list[1], interpolation=None, origin='lower',cmap=cmap, norm=norm)

ax[0].set_title(r'$\chi_n=%s$'%t_ratio_list[0],fontsize=12,y=1.03) #Beneficial size perturbation 
ax[2].set_title(r'$\chi_n=%s$'%t_ratio_list[1],fontsize=12,y=1.03) #Adverse size perturbation 


# x and y label
ax[0].set_xlabel(r'Contribution threshold $k$',fontsize=14)
ax[0].set_ylabel(r'Organism size under perturbation',fontsize=14)

# artifical x and y ticks
size_num=grid_num-1
y_num=grid_num-1

test=np.linspace(0,1,7,endpoint=True)
ax[0].set_xticks([i*size_num for i in test], minor=False)
x_lable_f=[i for i in range(1,8)]
x_label=[str(i) for i in x_lable_f]
ax[0].xaxis.set_major_formatter(mpl.ticker.FixedFormatter(x_label))

ax[0].set_yticks([i*size_num for i in test], minor=False)
y_lable_f=[i for i in range(1,8)]
y_label=[str(i) for i in y_lable_f]
ax[0].yaxis.set_major_formatter(mpl.ticker.FixedFormatter(y_label))

# Hide the right and top spines
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)

#---create text annotations.------------------------

"""The matrix0 and matrix1 are the two matrice in optimal_matrix_list, here to save time,
 we ran once and save the data. We add the coresponding lcs by hand based LC.txt.
 We add the reproductive strategy in area not in each grid,
thus, we masked some by -1. """

matrix0=np.array([[ -1,  -1,  -1,  0.,  -1,  -1,  -1],
        [ 2.,  6., -1., 57., 55.,  -1,  2.],
        [ 5., 12., 22., 56., -1., -1,  5.],
        [11., 21., -1., 57., -1., -1., 11.],
        [19., 35., 56., -1., 56., -1., 19.],
        [33., -1., -1., -1., -1., 55., 33.],
        [51., 55., -1., -1., -1., -1, 51.]])
matrix1=np.array([[ -1, -1., -1., -1., -1., -1., -1],
       [ -1., -1., -1., -1., -1., -1., -1.],
       [ -1., -1., -1., -1., -1., -1., -1.],
       [-1.,  6., 56., 57., 56., 55., 51.],
       [ -1., 12, -1., -1., -1., -1., -1.],
       [ -1, 22, -1., -1., -1., -1., -1.],
       [-1, -1., 36, -1., -1., -1., -1.]])

#	
for row in range(0,grid_num):
	for col in range(0,grid_num):
#		if row>=col:	
		if matrix0[row, col]>=0:
			if 	int(matrix0[row, col])>=13 and int(matrix0[row, col])<=51:
				text = ax[0].text(col, row, lc_add(int(matrix0[row, col])),ha="center",
					 va="center", color="dimgray",fontsize=10)
			else:				
				text = ax[0].text(col, row, lc_add(int(matrix0[row, col])),ha="center",
				 va="center", color="darkgrey")
		if matrix1[row, col]>=0:
			if 	int(matrix1[row, col])>=13 and int(matrix1[row, col])<=51:
				text = ax[2].text(col, row, lc_add(int(matrix1[row, col])),ha="center",
					 va="center", color="dimgray",fontsize=10)
			else:				
				text = ax[2].text(col, row, lc_add(int(matrix1[row, col])),ha="center",
			     va="center", color="darkgrey")

#---------------fig2_-------------------------------
# x and y label
ax[2].set_xlabel(r'Contribution threshold $k$',fontsize=14)
ax[2].set_ylabel(r'Organism size under perturbation',fontsize=14)

ax[2].set_xticks([i*size_num for i in test], minor=False)
ax[2].set_yticks([i*size_num for i in test], minor=False)
ax[2].xaxis.set_major_formatter(mpl.ticker.FixedFormatter(x_label))
ax[2].yaxis.set_major_formatter(mpl.ticker.FixedFormatter(y_label))

# Hide the right and top spines
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[1].axis("off")
		
#=====-colorbar size and position-===========================
cbar_ax = fig.add_axes([0.1, -0.8, 0.83, 0.03])
cbar=fig.colorbar(im, cax=cbar_ax, orientation="horizontal",norm=norm, boundaries=None,
				  	  ticks=pos)
#---- the laebl list---------------------------------------------------------
all_labels=[lc_add(i) for i in range(58)]		

# ticklabel size and font
font1 = {'family': 'serif',
        'color':  'grey',
        'weight': 'light',   
        'size': 8,
        }	
cbar.ax.set_xticklabels(all_labels,fontdict=font1,rotation=90, ha='center')   #rotation_mode="anchor",		  
font_size = 8             # -- tick label size and tick length    
cbar.ax.tick_params(labelsize=font_size,direction='in',length=3,color="grey", grid_alpha=0.5)

#-----the colorbar frame------------
cbar.outline.set_edgecolor('k')
cbar.outline.set_linewidth(0.3)


#----size binary populations-----------
pop_size=[0,2,5,11,19,33,51]
pop_game=[51,55,56,57]
pop_new=[6,12,21,35,22,36]

tick_color=['#386cb0','#e41a1c','purple']
ind1=0
for item in cbar.ax.xaxis.get_ticklabels():
	if ind1 in pop_size:	
		item.set_color(tick_color[0])
		item.set_weight('bold')
		item.set_size(9)
	elif ind1 in pop_game:	
		item.set_color(tick_color[1])
		item.set_weight('bold')
		item.set_size(9)
	elif ind1 in pop_new:	
		item.set_color(tick_color[2])
		item.set_weight('bold')
		item.set_size(9)
	ind1+=1

cbar.ax.set_ylabel('Reproductive strategy', rotation=0, labelpad=-315, y=-9 )
font = mpl.font_manager.FontProperties(family='times new roman', style='normal', size=12)
cbar.ax.yaxis.label.set_font_properties(font)

plt.show()

