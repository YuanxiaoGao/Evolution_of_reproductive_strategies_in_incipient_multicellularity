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
import pickle
#-------------------------------------------------------------------------------
# set display width
np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=400)

#----read lc list--------------------------------------------------
with open("/Users/gao/Desktop/life-cycles-with-multiplayer-game/SimulationCode/partition/LC.txt", "r") as file:
    lcs = eval(file.readline())                          # read lc list
lcs0=np.array(lcs)

#--------define a function to transform [1+2] to 1+2
def lc_add(x):   # x is the number for life cycles
	lcs[x].sort(reverse=True)
	off_num=len(lcs[x])
	ini=str(lcs[x][0])       #initial lcs
	for i in range(1,off_num):
		ini=ini+'+'+str(lcs[x][i])
	return ini


data_pathway="/Users/gao/Desktop/life-cycles-with-multiplayer-game/SimulationCode/v6/v2_VD_V0/v12_sizeti/code_general_size_effect/test_lines10000"

##----ramdom time list -------------
#with open('%s/size_tsn_line10000_v0.txt'%data_pathway, 'rb') as fp:
#	t_data0 = pickle.load(fp)
#	
#test_number=10000                       # test the results under 1000 lines
#t_data=t_data0[0:test_number]	
#
#binary_lc=np.array([1,3,6,7,12,13,20,22,23,34,36,37,52,56,57,58])-1
#
##---read data-----------------------
#'''proved that all optimal lc are binary splitting lcs '''
#"collect the line and its corresponding optimal lcs"
#
#end_lc=58                        # 13[5], 23[6], 37[7]; 58[8]; 87[9]; 128[10]     # how many lcs we care
#
#line_opLCs=[]                    # collect the lines and its optimal lcs
#
#for T_cluster in range(0,test_number):                       # how many figures or cell divisions
#	all_list=[]
#	for i_th in range(0,end_lc):                            # read all lcs data = growth rate
#		
#		with open(data_pathway+'/data/%d_%d.txt'%(T_cluster,i_th), "r") as file:
#			nan=float(np.nan)
#			inf=np.inf
#			grate = eval(file.readline())                  # read growth rate
#		all_list.append(np.array([i_th, grate]))
#            
#	all_list_f = np.array(all_list, dtype=np.float128)
#	max_value=np.amax(all_list_f[:,1])
#	lc_ith=int(all_list_f[np.where(all_list_f[:,1]==max_value)][0][0])
#	"check if all optimal lcs are the binary splitting lcs"
##	if lc_ith not in binary_lc:                       
##		print('Wrong!!!! \n')
##		print('The line is %s'%T_cluster)
#	"save the line and optimal lcs"	
#	line_opLCs.append(np.array([T_cluster,lc_ith]))
#		
#line_opLCs=np.array(line_opLCs)		
#oplc_list=set(line_opLCs[:,1])	
#
#fre_list=[]
#for lc in binary_lc:
#	num=np.shape(np.where(line_opLCs[:,1]==lc))[1]
#	fre_list.append(np.array([lc,num]))
#fre_list=np.array(fre_list)	
#
#fre_list_sort=fre_list[fre_list[:,1]. argsort()[::-1] ]         # 
"""Due to the long reason, we ran the above code only once and save the data of fre_list_sort."""

fre_list_sort=np.array([[   0, 2818],
       [   6, 1168],
       [   2, 1131],
       [  22,  683],
       [  57,  645],
       [   5,  431],
       [  12,  409],
       [  11,  363],
       [  33,  321],
       [  19,  312],
       [  51,  308],
       [  56,  306],
       [  55,  296],
       [  21,  284],
       [  35,  281],
       [  36,  244]])
	
#------fig 0--bar figure--------------

fig, ax = plt.subplots(1,1, figsize=(11.5, 2.3))
fig.subplots_adjust(left=None, bottom=-0.05, right=None, top=None, wspace=None, hspace=None)

label_size=20
heat_size=14
y_small_size=10
rotation_degree=0

# color for each reproductive strategies
c1 = np.arange(1, 58 + 1)
norm1 = mpl.colors.Normalize(vmin=c1.min(), vmax=c1.max())
cmap1 = mpl.cm.ScalarMappable(norm=norm1, cmap=mpl.cm.jet)
cmap1.set_array([])

#--background dashed lines
ax.set_axisbelow(True)
ax.yaxis.grid(color='grey', linestyle='dashed', linewidth=0.4)

width1 = 0.5
x = np.arange(16)
ax.bar(x, fre_list_sort[:,1]/10000,width1,color=[cmap1.to_rgba( (j+ 1)) for j in fre_list_sort[:,0]],alpha=1) # #bababa  #8c510a #e0c49e

##--x labels------
ax.set_xticks(np.arange(16))
x_label=[lc_add(i) for i in fre_list_sort[:,0] ]


ax.set_xticklabels(x_label)
ax.tick_params(axis='x', which='major', labelsize=10)#0,**csfont)
ax.tick_params(axis='y', which='major', labelsize=y_small_size)#,**csfont)

#---x-label rotation-------
plt.setp(ax.get_xticklabels(), rotation=rotation_degree, ha="center",position=(0,0.005),
         rotation_mode="anchor")
plt.setp(ax.get_yticklabels(),position=(-0.005,0))
ax.set_ylabel("Frequency of the optimal \n reproductive strategy",fontsize=16)
ax.set_xlabel("Binary-splitting reproductive strategy",fontsize=16)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', which='both',length=0)
ax.tick_params(axis='x', which=u'both',length=2)

plt.show()

