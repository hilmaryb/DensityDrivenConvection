#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:32:06 2020

@author: hilmaryb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sp
from scipy.sparse import linalg as splin


from DDC_PACKAGE.Field import *
from DDC_PACKAGE.DDC_classes import *
from DDC_PACKAGE.FileSystem import *




class ADI_Simulation:
    def __init__(self,DIR):
        self.dir = DIR
        
    def initialize_new(self,Nx,Ny,X,Y,Ra,dt,f):
        print(".setting new parameters:   Nx={}, Ny={}, X={}, Y={}, Ra={}, dt={}".format(Nx,Ny,X,Y,Ra,dt))
        self.Nx = Nx; self.Ny = Ny
        self.X = X; self.Y = Y
        self.Ra = Ra; self.dt = dt
        self.it_count = 0
        self.time = 0

        
        #Create Fields and init. them
        print("..initializing fields")
        self.C = Field(self.Nx,self.Ny,self.X,self.Y)
        self.C.make_diff_lists()
        self.F = Field_Like(self.C)
        self.C.init_zero()
        self.F.init_from_func(f)
        
        #Make stream function solver and factorize it
        print("...making stream solver")
        self.SS = StreamSolver(self.F)
        self.SS.factorize()
        
        #Construct the ADI method handler
        print("....preparing FVM")
        self.M = FVM_ADI(self.C,self.dt)
        self.M.make_pt_lists()
        #Construct diffusion operator bands
        self.diff_diags = self.M.make_diff_op()
        
        #Init the filesystem
        print(".....making FS")
        self.FS=FileSystem(self.dir,ow=1)
        self.FS.write_params(self)
        self.FS.comment("{:>14}, {:>15}, {:>15}, {:>30}, {:>30}".format('it','t','m','C-path','Psi-path'))
        
        
    def initialize_cont(self):       
        #Get parameters
        Nx,Ny,X,Y,Ra,dt = get_params(self.dir)
        print(".loading parameters from log:   Nx={}, Ny={}, X={}, Y={}, Ra={}, dt={}".format(Nx,Ny,X,Y,Ra,dt))
        self.Nx = Nx; self.Ny = Ny
        self.X = X; self.Y = Y
        self.Ra = Ra; self.dt = dt
        i,t,C_path = get_last(self.dir)
        
        #Initialize Fields
        self.it_count = i
        self.time = t
        print("..loading fields from: ({}) and ({})".format(self.dir+"/F.npy",C_path))
        self.C = Field(self.Nx,self.Ny,self.X,self.Y)
        self.C.make_diff_lists()
        self.C.from_array(np.load(C_path))
        self.F = Field_Like(self.C)
        self.F.from_array(np.load(self.dir+"/F.npy"))
        
        #Make stream function solver and factorize it
        print("...making stream solver")
        self.SS = StreamSolver(self.F)
        self.SS.factorize()
        
        #Construct the ADI method handler
        print("....preparing FVM")
        self.M = FVM_ADI(self.C,self.dt)
        self.M.make_pt_lists()
        #Construct diffusion operator bands
        self.diff_diags = self.M.make_diff_op()        
        #Init the filesystem
        print(".....making FS for append")
        self.FS=FileSystem(self.dir,ow=0)


    def loop(self,maxit):
        print("system is at it={}, and t={}".format(self.it_count,self.time))
        DL1, DDX, DU1, DLN, DDY, DUN = self.diff_diags
        for k in range(0,maxit):
            Ctop = 1*np.ones(self.Nx)#-1e-6*np.random.rand(self.Nx)
            #Differentiate the concentration field to get the RHS of the Psi eq.
            rhs = self.Ra*self.C.ddx()
            #Solve for the stream function
            self.Psi = self.SS.solve(rhs)
            #Construct the advection operator
            l1,dx,u1,lN,dy,uN = self.M.make_advect_op(self.Psi)
            #ADI Step 1:
            self.C.band_mul(DLN+lN,1+DDY+dy,DUN+uN,self.Nx)                   #Explicit in Y
            self.M.apply_dirichlet_top_vec(self.C,Ctop)                       #Update BC for explicit Y
            self.C.vector = ADI_TRI_X_VEC(-DL1-l1,1-DDX-dx,-DU1-u1, self.C)   #Implicit X, tridiag
            #ADI Step 2:
            self.C.band_mul(DL1+l1,1+DDX+dx,DU1+u1,1)                         #Explicit in X
            self.M.apply_dirichlet_top_vec(self.C,Ctop)                       #Update BC for implicit Y
            self.C.vector = ADI_BAND_Y_VEC(-DLN-lN,1-DDY-dy,-DUN-uN, self.C)  #Implicit Y, banded
            
            self.it_count+=1
            self.time+=self.dt
            if self.it_count%50 ==0:
                print(self.it_count)
                self.FS.write_data(self,fields=1)
        self.FS.close()



def make_new(name,Ra,dt):
    def f(x,y):
        return 1
    sim = ADI_Simulation(name)
    sim.initialize_new(200,200,1,1,Ra,dt,f)
    return sim

def cont(name):
    sim = ADI_Simulation(name)
    sim.initialize_cont()
    


