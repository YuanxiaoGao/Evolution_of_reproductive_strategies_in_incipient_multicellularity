#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:56:54 2019

@author: gao
"""

#-*-encoding:utf-8 -*-
################################################################
# 2018-06-25												   #
################################################################

""" code description:
Aim: get one given life cycle's growth rate-grate'

Parameters and Variables:
	C: [# of germ cells, # of soma cells]=[defectors,ccooperactors] (np array: int elements)
	T: time for one times disvision, totally depends on cell composition of last stepself.
		essentially based on payoff on colony level.
	P: probability for each cell type to divide.
		np.array([pi,pj]), which decided by cell payoff (composition).
	m: type switching probability.
	b: benefits for gemrs.
	c: costs for somas.
	w: synergy or discounting effects.
	W_i: intensity of selection.
	z: simulation times for each trajectory.
	grate: growth rate lambda.
	x0: the initial guessing root.
 ---------------- """
import numpy as np
import operator as op
from functools import reduce
from scipy.misc import derivative
import sys

#------------------------------------------------------------------------------------------------------------
'''import all lcs; which is total 128 lcs for M <=10'''

with open("./LC.txt", "r") as file:
    lcs = eval(file.readline())          # read lc list
num_lcs=len(lcs)                         # number of lcs

#------------------------------------------------------------------------------------------------------------
'''exhaustive cluster parameters'''

N_cluster=int(sys.argv[1])              # size under perturbation from 1-7
chi_cluster=int(sys.argv[2])		   # perturbation degree of t_sn/t_sn^neutral
i_th=int(sys.argv[3])                   # i_th lc in lcc     len(lc_data)
                                        # i_th (0,7) from 0 to 6 = M<=4

'''transform cluster parameters into local parameters'''
N_range=np.linspace(1, 7, 7)
N=N_range[N_cluster]

chi_ratio_num=51
chi_ratio_log = np.linspace(-0.4,0.4, chi_ratio_num)    # log scale for chi/chi_neutral
chi_ratio_list=[np.power(10,i) for i in chi_ratio_log]       # chi/chi_neutral
chi_ratio=chi_ratio_list[chi_cluster]

lc=lcs[i_th]

'''constant parameter'''
z=int(5000)                               # simulation times for each trajectory
Wi=0                                    # fixed intensity of selection
m=0.01
#------------------------------------------------------------------------------------------------------------
'''find each lc's newborn compositions and crutial size for fragment;
	Return:
	1-- number of newborn state (int)
	2-- newborn Composition (np.ndarray),
	3-- group max size (int) for reproduction,
	4-- offspring number of offspring group size(list): [#of 1 cell, #of 2 cells,....]
'''

def Newborn(lc):                                # lc is a life cycle in list form such as [1,1,2]
	size_lc=int(sum(lc))                        # max group size M = fragemnt size

	#------- composition of all newborn states
	offtype=list(set(lc))                       # how many d
	newborn=[]                                  # newborn composition
	for i in range(len(offtype)):
		for j in range(offtype[i]+1):
			newborn.append(np.array([offtype[i]-j,j]))
	num_newbornstate=len(newborn)               # number of newborn state

	#------- offspring number of every offspring types,e.g how many 1 cells produced....
	num_offtype=[]
	for i in range(1,size_lc):
		num_offtype.append(lc.count(i))
	off_num_cell=np.sum(np.vstack(newborn),axis=1)

	return num_newbornstate,np.vstack(newborn),size_lc,num_offtype,off_num_cell

num_newbornstate,newbornstate,size_lc,num_offtype,num_cell_newborn = Newborn(lc)

#------------------------------------------------------------------------------------------------------------
'''Probability P for each possible division;
	Return np.array([p_a*(n**2,2*m*n,m**2),p_b*(n**2,2*m*n,m**2)]) with shape (1,6)
	crossponding to            [g->2g, g->g+s, g->2s,   s->2s, s->s+g, s->2g]
	compositions changes with  [[1,0], [0,1],  [-1,2],  [0,1], [1,0],  [2,-1]]
	C--colony composition
	m--mutation rate
'''

def P(C,m):

	f_g,f_s=1,1                                     # cell fitness

	ratio_f_g=C[0]*f_g/(C[0]*f_g+C[1]*f_s)          # proba for germs ~ f_g

	ratio_f_s=C[1]*f_s/(C[0]*f_g+C[1]*f_s)          # proba for somas ~ f_s

	muta=np.array([(1.0-m)**2,2*m*(1.0-m),m**2])    # mutation order: no-half-both

	proba=np.hstack((ratio_f_g*muta,ratio_f_s*muta))
												    # proba * random mutation
	return proba

#------------------------------------------------------------------------------------------------------------
'''Division time T=K/<average(f)>;     Return - growth time for one step'''

def CHI_equal(item):
	t=np.log((item+1)/item)	
	return t

def T(C):
	num_cell=(C[0]+C[1])
	if num_cell==N:
		coef=chi_ratio*CHI_equal(num_cell)                     # netural coefficient ln[i+j+1]/[i+j]
	else:
		coef=CHI_equal(num_cell)                     # netural coefficient ln[i+j+1]/[i+j]

	f_g,f_s=1,1                                       # cell fitness
	
	time=coef*(num_cell)/(C[0]*f_g+C[1]*f_s)             # C[k]=0 makes sense to the non-exist Fitness ----linear with size effects
	time_s=time

	return time_s

#------------------------------------------------------------------------------------------------------------
'''One times division function;     Return - next cell composition np.array([g,s])'''
'''here is the only random thing we code in this file!!!!!'''

def Division(C):                                   # a tuple after calling

	#---------- which cell type to divide
	p=P(C,m).tolist()                              # call probability and convert into list

	divi_id=np.random.multinomial(1, p, size=1)    # divide ID or direction

	index=np.nonzero(divi_id)[1]

	c_delta=np.array([[1,0],[0,1],[-1,2],[0,1],[1,0],[2,-1]])
	                                               # composition changes with P(C,m)
	next_c=C+c_delta[int(index)]                   # composition after division

	return next_c                       	           # next cell composition && probability for this division

#------------------------------------------------------------------------------------------------------------
'''One trajectory for a given nrebornstate;
	Return - final C(compositon), cumulative T(time).
	One_tra{Fragment[ncr]}, so structure is the following
	ncr() ->Fragment() -> One trajectory()
'''
	#---------- step 1 ---------
'''combination function'''

def ncr(n, r):
	if r>n:
		return 0.0
	else:
	    r = min(r, n-r)                                      # take the smaller
	    numer = reduce(op.mul, range(n, n-r, -1), 1)         # op.mul: operator.mul(a, b)¶
	    denom = reduce(op.mul, range(1, r+1), 1)
	    return numer//denom

	#---------- step 2 ---------
'''fragment function; partition composition into offspring type(newbornstate);
     Return a list [#of type 1, #of type 2,....];
	 read more in notebook: fragment analysis
'''

def Fragment(comp):                                 # a given colony cell composition

	off_dis=[]
	for i in range(num_newbornstate):               # for example lc [1,2] -> 1 and 2
		offsize=np.sum(newbornstate[i])             # for example above 1->[1,0] or [0,1], while 2->[2,0],[1,1] or [0,2]
		i_cell=newbornstate[i][0]                   # for example above [1,0]->1
		j_cell=newbornstate[i][1]                   # for example above [1,0]->0
		off_i=ncr(comp[0],i_cell)*ncr(comp[1],j_cell)/ncr(np.sum(comp),offsize)
											        # probability for comp to get i cells offspring newbornstate[i]
		off_dis.append(num_offtype[offsize-1]*off_i)
											        # number of getting the offspring newbornstate[i]
	return off_dis

	#---------- step 3 ---------
'''one trajectory from newborn to final possible offsprings.
	Give one a newbornstate: np.array([g,s]);
	Return
	1: []--final offspring number of each newborn type;
	2: float--growth time
'''

def One_tra(C_newbron):                             # C_newbron: newborn cell composition
	cum_t=0.0                                       # count growth time

	newbron_size=C_newbron[0]+C_newbron[1]          # size of newborn
	division_times=size_lc-newbron_size             # how many division times left

	i=0                                             # count division_times
	while i<division_times:                         # division_times

		next_c=Division(C_newbron)
		cum_t+=T(C_newbron)
		C_newbron=next_c
		i+=1

	offspring=Fragment(C_newbron)              # call fragment function to get offspring

	return offspring, cum_t

#-------------------------------------------------------------------------------------------------------------------------

'''COLLECT all matrix data; Return offtype+T for z times simulation;
	M_data()=[], with length newbornstates; in which each element is a np.array with shape(z,newbornstates+1);
	and in each np.array, columns corresponds to -[#of newbornstate1, #of newbornstate2,...., t]
'''
def M_data():

	Matrix=[]
	for new_bron in newbornstate:

		#--------- one row's data with shape z*(num_newbornstate+1)
		z_off=[]                                    # list of each offspring for z-th simulations and time T
		for i in range(int(z)):
			offspring, cum_t=One_tra(new_bron)
			offspring.insert(len(offspring),cum_t) # insert the T at the end of offtype size z*(offtype+1)
			z_off.append(offspring)                # put offtype+T into a list; size z*(offtype+1)

		row=np.array(z_off)                        # convert each row data into a np.array
		Matrix.append(row)                         # collect all row data; size (num_newbornstate*z*(offtype+1))

	return Matrix                                  # a list containning np.array, each array is a matrix of z trajectories
#-------------------------------------------------------------------------------------------------------------------------
''' Construct Q by using the simulated data above. Return rooting function
	 grate ----- growth rate i.e. lambda
	 Warning: here we use the mass of the population i.e. the number of the whole cells
'''
data = M_data()                                    # save the simulated data in case of changing when recall afterwards

def F(grate):

	Q=[]
	for i in range(num_newbornstate):             # i means each newbornstate

		#------construct e^(-grate*T)             # z is simulation times i.e. trajectories lines
		e1=np.full((1,int(z)),np.exp(-1.0))       # construct [e^-1,e^-1,e^-1]
		e2=np.power(e1,data[i][:,-1])             # construct [e^-T,e^-T,e^-T]
		e3=np.ones((1,z))*grate                   # construct z [grate,grate,...]
		e4=np.power(e2,e3)                        # construct Z [e^(-grate*T),...]

		#----- get N*e^(-grate*T)
		off_time=np.multiply(data[i][:,:-1],e4.reshape((z,1)))
												  # each simulated line * t
		#----sigma all column of off_time= sigma-tao(=z) N*e^(-grate*T)
		row=(np.sum(off_time,axis=0))/float(z)    # get a row of Q with shape(1,num_newbornstate)

		Q.append(row.tolist())                    # collect all rows

	Q_np=np.array(Q)                              # change row list into np.array()

	Q1=Q_np-np.eye(num_newbornstate)              # ndarray Q-I

	expr=np.linalg.det(Q1)                        # convert into matrix for calculating det

	return expr
##------------------------------------------------------------------------------------------------------------
'''Solve equation to find growth rate; Return growth rate'''

	#---------- step 1 ---------
''' Estimate the max lambda by finding the minimum time '''

t_row_min=[]
t_row_max=[]
for i in range(num_newbornstate):
	t_row_min.append(np.amin(data[i][:,-1]))
	t_row_max.append(np.amax(data[i][:,-1]))

T_min=min(t_row_min)                               # min time
T_max=max(t_row_max)                               # max time

x0=(np.log(sum(lc)))/T_min+0.1                     # the first root guess -- right boundary
x_mini=(np.log(2))/T_max-0.1
root_step=1e-3                                     # sign of the right boundary
step=(x0-x_mini)/root_step +1                      # for later check the f0 f1 having the same sign or not

	#---------- step 2 ---------
''' methods1: Fine single roots by using Bisection'''
''' here the bisection cannot work because the maximum roots are the double roots!!!!!'''
def Find_single_root(func,x):                      # x0 is the first root guess

	#--find the root left and right boundaries by setting the right first
	f0=np.sign(func(x))                            # sign of the first try
	f1=np.sign(func(x-root_step))

	#------find the max root boundary to the left
	n=0
	while f0*f1>0 and (x-n*root_step)>=x_mini:
		f0=np.sign(func(x-n*root_step))                # right
		f1=np.sign(func(x-(n+1)*root_step))            # left
		n+=1
	#---- cannot find the single roots
	if (x-n*root_step)<=x_mini:
		return None, None

	#----- can find the single roots
	else:
		if f0*f1 !=0:
			left=x-n*root_step
			right=x-(n-1)*root_step

			#------find the root between boundary (left, right) by bisection
			while abs(left-right)>10**(-14):
				left_sign=np.sign(func(left))

				mean=(left+right)/2
				mean_sign=np.sign(func(mean))

				if left_sign*mean_sign>0:
					left=mean
				else:
					right=mean
		elif f0==0:
			mean=x-(n-1)*root_step               # since n add extra 1 after f0 anf f1, here should remove it
		elif f1==0:
			mean=x-n*root_step

		return mean, n

''' methods2: Fine double roots by using derivative '''
#--first derivative
def F_d(x):                                            # derivative of f
	f_d=derivative(F, x, dx=1e-6)
	return f_d

def Find_double_root(x):                               # x0 is the first root guess
	single_root,n=Find_single_root(F_d,x)

	root0=1
	while single_root is not None:
		n0=n

		if abs(F(single_root))<10**(-5):
			break
		else:
			new_single_root,new_n=Find_single_root(F_d,x-n0*root_step)

			if new_single_root is None:
				root0=0
				break
			else:
				single_root,n=new_single_root,new_n+n0
	if root0==1:
		return 	single_root
	else:
		return None
#------------------------------------------------------------------------------------------------------------
'''output result'''
single_root,n=Find_single_root(F,x0)

if single_root is not None:
	root=single_root
else:
	double_root=Find_double_root(x0)
	root=double_root

with open('data/%d_%d_%d.txt'%(N_cluster,chi_cluster,i_th), 'w') as f:
    f.write(str(single_root))

#------------------------------------------------------------------------------------------------------------

