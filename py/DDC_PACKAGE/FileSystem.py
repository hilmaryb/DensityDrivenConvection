#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:15:25 2020

@author: hilmaryb
"""
import numpy as np
import os


class FileSystem:
    #Class for handling writing to a log file
    def __init__(self,DIR,ow=0):
        self.dir=DIR
        if ow==1:
            try:
                os.mkdir(DIR)
                print("created directory ({}) for data".format(DIR))
            except FileExistsError:
                print("DIRECTORY ({}) ALREADY EXCISTS".format(DIR))
                
            self.dat_count=1
            self.fh = open(DIR+"/run.log",'w+')
        else:
            _,_,Cpath = get_last(DIR)
            self.dat_count=int(Cpath.split("/C")[-1][:-4])
            self.fh = open(DIR+"/run.log",'a+')
        
    def comment(self,st):
        self.fh.write("# "+st+"\n")
    def write_data(self,sim,fields=1):
        #Define data
        m = np.sum(sim.C.vector)*sim.C.hx*sim.C.hy
        dat = "{:15d}, {:15f}, {:15f},".format(sim.it_count,sim.time,m)
        files = ''
        if fields==1:
            #Write fields to npy file
            cname = self.dir+"/C{:04d}.npy".format(self.dat_count)
            psiname = self.dir+"/PSI{:04d}.npy".format(self.dat_count)
            np.save(cname,sim.C.vector)
            np.save(psiname,sim.Psi.vector)
            files = "{:>30}, {:>30},".format(cname,psiname)
            self.dat_count+=1
        self.fh.write(dat + files + "\n")
        #Make sure the buffer is written to file in case of
        #sudden death
        self.fh.flush()
        os.fsync(self.fh.fileno())
        
    def write_params(self,sim):
        #Write the system parameters to the log file
        l1 = "$Nx={}, Ny={}, X={}, Y={}, ".format(sim.Nx,sim.Ny,sim.X,sim.Y)
        l2 = "$Ra={}, dt={}, \n".format(sim.Ra,sim.dt)
        self.fh.write(l1+l2)
        #Write the F-field to the data dir
        np.save(self.dir+"/F.npy",sim.F.vector)
    def close(self):
        self.fh.close()

def get_params(DIR):
    #Used to read system parameters from a log file
    f = open(DIR+"/run.log","r")
    lines = f.readlines()
    for line in lines:
        if line.startswith("$"):
            p = line[1:].strip()
    params = [x.partition('=')[2][:-1] for x in p.split()]
    Nx = int(params[0])
    Ny = int(params[1])
    X,Y,Ra,dt = [float(x) for x in params[2:]]
    f.close()
    return Nx,Ny,X,Y,Ra,dt

def get_last(DIR):
    #Used to find the data needed to continue a run.
    #This includes the F-field, the last C-field
    #and the associated iteration count and time
    fh = open(DIR+"/run.log")
    lines = fh.readlines()
    for line in lines:
        if not (line.startswith("#") or line.startswith("$")):
            if len(line.split())>3:
                #If the line has mroe than 3 elts, it has file paths
                last_save_line=line
    #Assume the following layout, i, t, _, C-file
    i = int(last_save_line.split()[0][:-1])
    t = float(last_save_line.split()[1][:-1])
    C_path = last_save_line.split()[3][:-1]
    fh.close()
    return i,t,C_path

def load_data(DIR):
    #Used to find the data needed to continue a run.
    #This includes the F-field, the last C-field
    #and the associated iteration count and time
    fh = open(DIR+"/run.log")
    lines = fh.readlines()
    it = []
    t = []
    m = []
    C = []
    PSI = []
    for line in lines:
        if not (line.startswith("#") or line.startswith("$")):
            if len(line.split())>3:
                #If the line has mroe than 3 elts, it has file paths
                it.append(int(line.split()[0][:-1]))
                t.append(float(line.split()[1][:-1]))
                m.append(float(line.split()[2][:-1]))
                C.append(line.split()[3][:-1])
                PSI.append(line.split()[4][:-1])
    fh.close()
    return it,t,m,C,PSI