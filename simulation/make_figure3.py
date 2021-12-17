#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:04:34 2019

@author: gao
"""

grid_num=31		 	         # how many pixels grids (grid_num-1) in the final B-C map
grid_m=11
end_lc=58                     # the total number of life cycles & the number of reproductive strategies

fp = open("run.sh", "w")

for b_cluster in range(0,grid_num):                       # how many figures or cell divisions: grid_num
	for m_cluster in range(0,grid_m):                         # m =m_cluster/(grid_num-1)
		for k_cluster in range(0,7):                         # m =m_cluster/(grid_num-1)
			for i_th in range(end_lc):
	
				jobname = "grate_VD_%d_%d_%d_%d" % (b_cluster,m_cluster,k_cluster,i_th)
				fname = "script/grate_VD_%d_%d_%d_%d.sh" % (b_cluster,m_cluster,k_cluster,i_th)
				fp.write("sbatch %s\n" % fname)
				bashfp = open(fname, "w")
	
	
				bashfp.write("#!/bin/sh\n")
				bashfp.write("#SBATCH --time=00-12:00:00\n")
				bashfp.write("#SBATCH --job-name %s\n" % jobname)
				bashfp.write("#SBATCH -o out/%s\n" % jobname)
				bashfp.write("#SBATCH -e err/%s\n" % jobname)
				bashfp.write("python grate_VD.py %d %d %d %d\n" % (b_cluster,m_cluster,k_cluster,i_th) )
