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

grid_chi=101                                                  # the number of points for chi
grid_n=7                                                       # the number of cells

chi_ratio_num=grid_chi
chi_ratio_log = np.linspace(-0.4,0.4, chi_ratio_num)         # log scale for chi/chi_neutral
chi_ratio_list=[np.power(10,i) for i in chi_ratio_log]       # chi/chi_neutral

with open("../simulation/LC.txt", "r") as file:
    lcs = eval(file.readline())                               # read lc list
		
#--------define a function to transform [1+2] to 1+2--------------------
def lc_add(x):                                               # x is the number for life cycles
	lcs[x].sort(reverse=True)
	off_num=len(lcs[x])
	ini=str(lcs[x][0])                                       #initial lcs
	for i in range(1,off_num):
		ini=ini+'+'+str(lcs[x][i])
	return ini
#----------------------------------------

end_lc=58
N=3
"add a list [[maturity size=3],[],[],;;;]"
data_list=[ ]
	
for chi_cluster in range(0,grid_chi):                        # how many figures or cell divisions
	for N_cluster in range(N-1,N):                           # y axes points = a
		all_list=[]                                          # algin all lcs' growth rates
		for i_th in range(0,end_lc):                         # read all lcs data = growth rate
				
			with open('../data/data_size_effect/%d_%d_%d.txt'%(N_cluster,chi_cluster,i_th), "r") as file:
				nan=float(np.nan)
				inf=np.inf
				grate = eval(file.readline())          # read growth rate
			data_list.append(np.array([chi_cluster,i_th,grate]))
			
data_list=np.array(data_list)
	
#================plot figure==============================-----
from matplotlib import cm

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]},figsize=(12, 5))

#------color-----------------
c = np.arange(1, 58 + 1)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap_legend = cm.get_cmap('jet', 58)
cmap_legend.set_bad('lightgrey',alpha=0.4)	                         # color for g3=1 i.e. s_t=nan
cmap_legend.set_under('w',alpha=0.6)

cmap.set_array([])

#------color-----------------
cmap =cm.jet
norm = plt.Normalize(0, end_lc)	
cmap.set_bad('lightgrey',alpha=0.4)	                         # color for g3=1 i.e. s_t=nan
cmap.set_under('w',alpha=0.6)

#==========fig 1========================================
n_k_matrix=np.array([[-1,-57],[-1,-2]])
im = ax[0].imshow(n_k_matrix, interpolation=None, origin='lower',cmap=cmap_legend, norm=norm)

#--colorbar size and position------------------------------------
cbar_ax = fig.add_axes([0.1, -1.65, 0.82, 0.03])

pos=np.array([i+0.5 for i in range(end_lc+1)])                                 # position of all
cbar=fig.colorbar(im, cax=cbar_ax, orientation="horizontal",norm=norm, boundaries=None,
				  	  ticks=pos)
#---- the laebl list-------------------------------------------------
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

#-----the colorbar frame----------------------------------------------------
cbar.outline.set_edgecolor('k')
cbar.outline.set_linewidth(0.3)
	#---tick weight and size
co_optimal_lc=[0,1,2,57]
	
optimal_lc=[0,2,5,11,19,33,51]
tick_color=['#386cb0','#e41a1c','#b35806']
ind1=0
for item in cbar.ax.xaxis.get_ticklabels():
	if ind1 in optimal_lc:	
		item.set_color(tick_color[0])
		item.set_weight('bold')
		item.set_size(9)
	ind1+=1

cbar.ax.set_ylabel('Reproductive strategy', rotation=0, labelpad=-355, y=-9 )
font = mpl.font_manager.FontProperties(family='times new roman', style='normal', size=12)
cbar.ax.yaxis.label.set_font_properties(font)

ax[0].axis('off')

#==========fig 2========================================
#---color----------------
c1 = np.arange(1, 58 + 1)

norm1 = mpl.colors.Normalize(vmin=c1.min(), vmax=c1.max())
cmap1 = mpl.cm.ScalarMappable(norm=norm1, cmap=mpl.cm.jet)
cmap1.set_array([])

#---background---
ax[1].set_axisbelow(True)
ax[1].yaxis.grid(color='gray', linestyle='dashed', linewidth=0.4,alpha=0.3)

#-------------linear plot------------------------
for j in range(58):
	new_data=data_list[np.where(data_list[:,1]==j)]
	ax[1].plot(chi_ratio_list, new_data[:,-1],c=cmap1.to_rgba( (j+ 1)),linewidth=3,alpha=0.9)

#-----add nonoate- 2 populations-----------------------
text_list=[]

for chi in [0,grid_chi-1]:	
	data_test=data_list[np.where(data_list[:,0]==chi)]
	if chi<2:
		data00=data_test[data_test[:,2].argsort()][[-1,-2],:]  # the first two largest
		text_list.append(list(data00))
		
	elif chi>2:
		data00=data_test[data_test[:,2].argsort()][[0,1],:]   # the least two largest
		text_list.append(list(data00))
		
text_list1 = [item for sublist in text_list for item in sublist]
text_list1[-1][2]=0.65

for sth in text_list1:
	if sth[0]<1:
		ax[1].annotate(lc_add(int(sth[1])), (chi_ratio_list[int(sth[0])]+0.03, sth[-1]), 
				 color="gray",fontsize=10,alpha=1)	              #cmap.to_rgba( int(sth[1])+ 1)
	else:
		ax[1].annotate(lc_add(int(sth[1])), (chi_ratio_list[int(sth[0])]-0.38, sth[-1]), 
				 color="gray",fontsize=10,alpha=1)	              #cmap.to_rgba( int(sth[1])+ 1)
	
ax[1].set_xlabel(r"Perturbation ratio $\chi_3$ ",fontsize=16)
ax[1].set_ylabel(r"Population growth rate $\lambda$",fontsize=16)

ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].yaxis.set_ticks_position('none') 

plt.show()
fig.savefig('./fig/figure_2B.pdf', bbox_inches='tight')   # save figures

