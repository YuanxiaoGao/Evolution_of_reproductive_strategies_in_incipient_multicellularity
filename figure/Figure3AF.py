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

cell_number=[3,4,5,6,7,8]
end_lc_list=[3,7,13,23,37,58]             # 13[5], 23[6], 37[7]; 58[8]; 87[9]; 128[10]     # how many lcs we care

"add a list [[maturity size=3],[],[],;;;]"
opt_matrix=[]
max_size=8
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
#					#---search the files which are cannot be calculated---
									
					with open('../data/data_threshold_effect/%d_%d_%d_%d.txt'%(b_cluster,m_cluster,k_cluster,i_th), "r") as file:
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
n_k_matrix[:,0]=-1
n_k_matrix[0][0]=0


opt_lc_uni=np.unique(np.array(opt_lc))                                   # appeared one
others=np.array(list(set([i for i in range(end_lc)])-set(opt_lc_uni)))   # nonappeared one
pos=np.array([i+0.5 for i in range(end_lc+1)])                                 # position of all


##%%
#================plot figure==============================-----
"draw figures"
from matplotlib import cm

# -- read tick labels---------
cwd = os.getcwd()                                        # Get the current working directory (cwd)
files = os.listdir(cwd)                                  # Get all the files in that directory
with open("../simulation/LC.txt", "r") as file:
    lcs = eval(file.readline())                          # read lc list
lcs[51]
	
#--------define a function to transform [1+2] to 1+2--------------------
def lc_add(x):   # x is the number for life cycles
	lcs[x].sort(reverse=True)
	off_num=len(lcs[x])
	ini=str(lcs[x][0])       #initial lcs
	for i in range(1,off_num):
		ini=ini+'+'+str(lcs[x][i])
	return ini
#----------------------------------------
			
# ----set fig frame--------
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]},figsize=(10, 5))

#-----colormap--------------
cmap = cm.get_cmap('jet', 58)
norm = plt.Normalize(0, end_lc)	
#optimal_matrix1=np.ma.masked_where(n_k_matrix==np.nan,n_k_matrix)
cmap.set_bad('lightgrey',alpha=0.4)	                         # color for g3=1 i.e. s_t=nan
cmap.set_under('grey',alpha=0.6)

#----figure------------------
im = ax[0].imshow(n_k_matrix, interpolation=None, origin='lower',cmap=cmap, norm=norm)

# x and y label
ax[0].set_xlabel(r'Contribution threshold $k$',fontsize=14)
ax[0].set_ylabel(r'Maximum maturity size $N$',fontsize=14)

# artifical x and y ticks
size_num=n_k_size-1
y_num=n_k_size-1

test=np.linspace(0,1,7,endpoint=True)
ax[0].set_xticks([i*size_num for i in test], minor=False)
x_lable_f=[i for i in range(1,8)]
x_label=[str(i) for i in x_lable_f]
ax[0].xaxis.set_major_formatter(mpl.ticker.FixedFormatter(x_label))

ax[0].set_yticks([i*size_num for i in test], minor=False)
y_lable_f=[i for i in range(2,9)]
y_label=[str(i) for i in y_lable_f]
ax[0].yaxis.set_major_formatter(mpl.ticker.FixedFormatter(y_label))

# Hide the right and top spines
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)

#---create text annotations.
text = ax[0].text(0, 0, "1+1",ha="center", va="center", color="darkgrey",fontsize=10)
for row in range(1,n_k_size):
	for col in range(1,n_k_size):
		if row>=col:	
			if 	int(n_k_matrix[row, col])>=13 and int(n_k_matrix[row, col])<=51:
				text = ax[0].text(col, row, lc_add(int(n_k_matrix[row, col])),ha="center",
					 va="center", color="dimgray",fontsize=10)
			else:				
				 text = ax[0].text(col, row, lc_add(int(n_k_matrix[row, col])),ha="center",
				   va="center", color="silver",fontsize=10)
			
#--colorbar size and position------------------------------------
			
cbar_ax = fig.add_axes([0.1, -0.91, 0.9, 0.03])
cbar=fig.colorbar(im, cax=cbar_ax, orientation="horizontal",norm=norm, boundaries=None,
				  	  ticks=pos)
#---- the laebl list-------------------------------------------------
all_labels=[lc_add(i) for i in range(58)]	                # change list into strings
			# ticklabel size and font
font1 = {'family': 'serif',
        'color':  'grey',
        'weight': 'light',   
        'size': 8,
        }	
cbar.ax.set_xticklabels(all_labels,fontdict=font1,rotation=90, ha='center')   #rotation_mode="anchor",		  
font_size = 8             # -- tick label size and tick length    
cbar.ax.tick_params(labelsize=font_size,direction='in',length=3,color="grey", grid_alpha=0.5)

#-----the colorbar frame----------------------------------------------------
cbar.outline.set_edgecolor('k')
cbar.outline.set_linewidth(0.3)

	#---tick weight and size	
	
#----size binary populations-----------
pop_game=[51,55,56,57]                                             # the golbal optimal

othe_pop=[0,2,5,11,19,21,33,51,6,12,22,35,36]                            # the local optimal strategies 
ind0=0
for item in cbar.ax.xaxis.get_ticklabels():
	if ind0 in othe_pop:	
		item.set_color("k")
		item.set_size(9)
	ind0+=1


tick_color=['#386cb0','#e41a1c','#b35806']
ind1=0
for item in cbar.ax.xaxis.get_ticklabels():
	if ind1 in pop_game:	
		item.set_color(tick_color[1])
		item.set_weight('bold')
		item.set_size(9)
	ind1+=1	
	

cbar.ax.set_ylabel('Reproductive strategy', rotation=0, labelpad=-335, y=-9 )
font = mpl.font_manager.FontProperties(family='times new roman', style='normal', size=12)
cbar.ax.yaxis.label.set_font_properties(font)
#======================second figure===============================================
ax[1].axis('off')

plt.show()
