#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:32:25 2020

@author: hilmaryb
"""

import numpy as np
import matplotlib.pyplot as plt

class Field:
    # This class is intended to be used as a generalized 2D field, which is for computational purposes
    # stored as a single column vector.
    # With this object, a scalar N is associated, and for now the 2D field is assumed to be a N x N grid
    # (subject to change: Nx x Ny)
    
    # CONSTRUCTOR
    def __init__(self,Nx,Ny,X,Y):
        #Default constructor.
        self.Nx = Nx
        self.Ny = Ny
        self.X = X
        self.Y = Y
        
        self.hx = X/Nx
        self.hy = Y/Ny
        self.numel = Nx*Ny
        
    # BASIC INITS
    def init_zero(self):
        self.vector = np.zeros(self.numel)
    def init_empty(self):
        self.vector = np.empty(self.numel)
    def from_array(self,arr):
        if len(arr) == self.numel:
            self.vector = arr
        else:
            print("Couldn't init from array - length mismatch")
            print("This Field is ({},{}) or {} long, and was handed a {} long array".format(self.Nx,self.Ny,self.numel,len(arr)))
    
    def init_from_func(self,f):
        a = self.hx*(0.5 + np.arange(0,self.Nx))
        b = self.hy*(0.5 + np.arange(0,self.Ny))
        x,y = np.meshgrid(a,b)
        x=x.reshape(-1)
        y=y.reshape(-1)
        self.vector = np.empty(self.numel)
        for i in range(0,self.numel):
            self.vector[i] = f(x[i],y[i])
            
    # BASIC OPERATORS    
    def __add__(self,o):
        tmp = Field(self.Nx,self.Ny,self.X,self.Y)
        tmp.from_array(self.vector+o)
        return tmp
        
    def __mul__(self,o):
        tmp = Field(self.Nx,self.Ny,self.X,self.Y)
        tmp.from_array(self.vector*o)
        return tmp
    
    def __rmul__(self,o):
        tmp = Field(self.Nx,self.Ny,self.X,self.Y)
        tmp.from_array(self.vector*o)
        return tmp

        
    def __setitem__(self,I,v):
        #Set index
        self.vector[I]=v
    def __getitem__(self,I):
        #Get index
        return self.vector[I]
            
    def show(self):
        #For quick plotting
        tmp = self.vector.reshape((self.Ny,self.Nx))
        plt.imshow(tmp,extent=[0,self.X,self.Y,0])
        plt.xlabel('x')
        plt.ylabel('y')
        
    def make_transpose_list(self):
        T = np.empty(self.numel,dtype=int)
        for r in range(0,self.Ny):
            for c in range(0,self.Nx):
                #T[normal index] = transpose index
                T[r*self.Nx+c] = c*self.Ny+r
        self.Tlist = T
    
    def make_diff_lists(self):
        self.r = np.zeros((self.Nx-1)*(self.Ny-1),dtype=int)
        self.c = np.zeros((self.Nx-1)*(self.Ny-1),dtype=int)
        for I in range(0,(self.Nx-1)*(self.Ny-1)):
            self.r[I] = int(I/(self.Nx-1))
            self.c[I] = int(I%(self.Nx-1))
            
    def ddx(self):
        d = (self.vector[(self.r+1)*self.Nx+self.c+1] + self.vector[self.r*self.Nx+self.c+1] - self.vector[(self.r)*self.Nx+self.c] - self.vector[(self.r+1)*self.Nx+self.c])/(2*self.hx)
        tmp = Field(self.Nx-1,self.Ny-1,self.X,self.Y)
        tmp.from_array(d)
        return tmp
    
    def ddy(self):
        d = (self.vector[(self.r+1)*self.Nx+self.c+1] - self.vector[self.r*self.Nx+self.c+1] - self.vector[(self.r)*self.Nx+self.c] + self.vector[(self.r+1)*self.Nx+self.c])/(2*self.hy)
        tmp = Field(self.Nx-1,self.Ny-1,self.X,self.Y)
        tmp.from_array(d)
        return tmp
    
    def avg_grain(self):
        d = (self.vector[(self.r+1)*self.Nx+self.c+1] + self.vector[self.r*self.Nx+self.c+1] + self.vector[(self.r)*self.Nx+self.c] + self.vector[(self.r+1)*self.Nx+self.c])/4
        tmp = Field(self.Nx-1,self.Ny-1,self.X,self.Y)
        tmp.from_array(d)
        return tmp
        
    def transpose(self):
        #Returns the transpose of this field.
        #Check if list is ready
        if hasattr(self,'Tlist') is not true:
            self.make_transpose_list()
        #Transpose to a temporary field
        tmp = Field(self.Ny,self.Nx,self.Y,self.X)
        tmp.init_empty()
        for i in range(0,self.numel):
            tmp.vector[i] = self.vector[self.Tlist[i]]
        #Return transpose
        return tmp
    
    def t(self):
        #Transposes this field. Returns nothing
        #Check if list is ready
        if hasattr(self,'Tlist') is not True:
            self.make_transpose_list()
        #Transpose to a temporary field
        tmp = Field(self.Ny,self.Nx,self.Y,self.X)
        Nx = self.Nx
        Ny = self.Ny
        X = self.X
        Y = self.Y
        hx = self.hx
        hy = self.hy
        tmp = np.empty_like(self.vector)
        for i in range(0,self.numel):
            tmp[i] = self.vector[self.Tlist[i]]
        self.Nx = Ny
        self.Ny = Nx
        self.hx = hy
        self.hy = hx
        self.X = Y
        self.Y = X
        self.from_array(tmp)
        
#    def band_mul(self,l,d,u,O):
#        tmp = np.empty_like(self.vector)
#        for i in range(0,O):
#            #First O elements, only d and u band
#            tmp[i] = d[i]*self.vector[i] + u[i+O]*self.vector[i+O]
#        for i in range(O,self.numel-O):
#            tmp[i] = l[i-O]*self.vector[i-O] + d[i]*self.vector[i] + u[i+O]*self.vector[i+O]
#        for i in range(self.numel-O,self.numel):
#            tmp[i] = l[i-O]*self.vector[i-O] + d[i]*self.vector[i]
#        self.vector=tmp
        
    def band_mul(self,l,d,u,O):
        #Multiplies THIS field by a matrix (from the left) with bands l, d and u,
        #where l and u are offset by O from the diagonal
        tmp = np.empty_like(self.vector)
        tmp[0:O] = d[0:O]*self.vector[0:O] + u[O:2*O]*self.vector[O:2*O]
        tmp[O:self.numel-O] = l[0:self.numel-2*O]*self.vector[0:self.numel-2*O] + d[O:self.numel-O]*self.vector[O:self.numel-O] + u[2*O:self.numel]*self.vector[2*O:self.numel]
        tmp[self.numel-O:self.numel] = l[self.numel-2*O:self.numel-O]*self.vector[self.numel-2*O:self.numel-O] + d[self.numel-O:self.numel]*self.vector[self.numel-O:self.numel]
        self.vector=tmp
        

def Field_Like(F):
    tmp = Field(F.Nx,F.Ny,F.X,F.Y)
    return tmp