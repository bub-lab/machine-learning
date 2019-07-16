#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:09:26 2019

@author: marc
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import matplotlib.animation as animation

par = {
    'data_type'         : 'synthetic',

    # cellular automatat params
    'ca_x_size'         : 32,
    'ca_y_size'         : 32,
    'ca_jiggle'         : 2,
    'ca_radius'         : 10,
    'ca_active'         : 4,
    'ca_refractory'     : 5,
    'ca_th'             : 0.2,
    'ca_prob'           : 0.001,
    'batch_size'        : 1,
    'frames_per_image'  : 200,
    
}

px1=20
py1=20
px2=80
py2=80
px3=60
py3=100

p1clock = np.random.randint(15,40)
p2clock = np.random.randint(15,40)
p3clock = np.random.randint(15,40)

print(p1clock);
print(p2clock);
print(p3clock);

class Stimulus:
    
    def __init__(self):

        if par['data_type'] == 'synthetic':
            self.ca = [pyCA(par['ca_x_size'], par['ca_y_size'], par['ca_jiggle'], par['ca_radius']) for _ in range(par['batch_size'])]
        else:
            pass
    def get_cells(self):
        return self.ca.getcells()
    def reset_state(self):

        for i in range(par['batch_size']):
            self.ca[i].reset_states()

    def get_batch_rl(self, stimulation = None):

        input_data = []

        for j in range(par['frames_per_image']):
            data = []
            for i in range(par['batch_size']):
                current_stimulation = stimulation[i,:,:] if j == 0 else None
                self.ca[i].iterate(par['ca_active'], par['ca_refractory'], par['ca_th'], par['ca_prob'], current_stimulation)
                data.append(self.ca[i].return_state())
            data = np.stack(data, axis = 0)
            input_data.append(data)
        input_data = np.stack(input_data, axis = -1)

        return input_data


    def get_batch(self):

        input_data = []
        target_data = []

        for _ in range(par['frames_per_image']):
            data = []
            for i in range(par['batch_size']):
                self.ca[i].iterate(par['ca_active'], par['ca_refractory'], par['ca_th'], par['ca_prob'])
                data.append(self.ca[i].return_state())
            data = np.stack(data, axis = 0)
            input_data.append(data)
        input_data = np.stack(input_data, axis = -1)

        for _ in range(par['frames_per_image']):
            data = []
            for i in range(par['batch_size']):
                self.ca[i].iterate(par['ca_active'], par['ca_refractory'], par['ca_th'], par['ca_prob'])
                data.append(self.ca[i].return_state())
            data = np.stack(data, axis = 0)
            target_data.append(data)
        target_data = np.stack(target_data, axis = -1)


        return input_data, target_data


class pyCA:
    #initiation of pyCa object, contains a jiggle, x&y max, radius, array of cells
    def __init__(self,xdim,ydim,jiggle,radius):
        self.numIter=0
        self.radius=radius
        self.xdim=xdim
        self.ydim=ydim
        self.jiggle=jiggle
        self.numcells=xdim*ydim
        self.cells=[]
        #for each (x,y) pair, generate two independent random numbers between
        #-0.5 and 0.5. multiply by some specified jiggle amount and then append
        #the cell to the location in the 2d cell array specified by dimy*x+x
        #using the append.
        for y, x in product(range(ydim), range(xdim)):
            rx=(np.random.rand()-0.5)*jiggle
            ry=(np.random.rand()-0.5)*jiggle
            if (x==px1 and y==py1):
                pmIndex=1
            elif (x==px2 and y==py2):
                pmIndex=2
            elif (x==px3 and y==py3):
                pmIndex=3
            else:
                pmIndex=0
                
            self.cells.append(cell(x+rx,y+ry,pmIndex))
        #determine neighbours of the cells
        self.findNeighbours(radius)
        #iterate for a burn time
        for _ in range(20):
            self.iterate(par['ca_active'], par['ca_refractory'], par['ca_th'], par['ca_prob'])
    #cells is an array, this returns the cell located at the x,y coordinate given
    #which is stored in a flattened array.  Returns cell object.
    def get(self,x,y):
        return self.cells[self.xdim*y+x]
    
    def getcells(self):
        return self.cells
    #returns array of 0-1 values determining whether a cell is active or not.
    def return_state(self):
        state = np.zeros((par['ca_x_size'], par['ca_y_size']), dtype = np.float32)
        for x, y in product(range(par['ca_x_size']), range(par['ca_y_size'])):
            state[y, x] = np.float32(self.cells[self.xdim*y+x].state > 0)
        return state
    def reset_states(self):
        for c in self.cells:
            c.state = 0
            c.statenext = 0
        """
        for _ in range(20):
            self.iterate(par['ca_active'], par['ca_refractory'], par['ca_th'], par['ca_prob'])
        """
    #gets the euclidean distance between two points on the array
    def dist(self,x1,y1,x2,y2):
        c1=self.get(x1,y1)
        c2=self.get(x2,y2)
        d=((c1.xloc-c2.xloc)**2+(c1.yloc-c2.yloc)**2)**0.5
        return d




    #creates a list of neighbouring cells within radius.
    def findNeighbours(self,radius):

        self.radius=radius
        r=self.radius+self.jiggle+0.01
        ri=int(np.ceil(r))

        for y, x in product(range(self.ydim), range(self.xdim)):
            #get cell at x,y in flattened cells array
            c=self.get(x,y)
            c.neighbours=[]
            for yy, xx in product(range(y-ri,y+ri), range(x-ri,x+ri)):
                #if the x,y index is inside the acceptable range and within the radius
                if (xx>=0) and (xx<self.xdim) and (yy>=0) and (yy<self.ydim):
                    #get the cell value
                    c2=self.get(xx,yy)
                    #if the distance between the current cell and the new cell 
                    #is less than the radius, add the neighbouring cells to the
                    #neighbours array of the cell object
                    if self.dist(x,y,xx,yy)<self.radius:
                        c.neighbours.append(c2)
    #one forward step iteration for the CA
    def iterate(self,A,R,threshold,p, stimulation=None):
        for y, x in product(range(self.ydim), range(self.xdim)):
            #get cell at (x,y).
            c=self.get(x,y)
            #if state is 0, with some probability p, set statenext to 1 
            if c.state==0:
                if np.random.rand()<p:
                    c.statenext=1
                active=0.0
                #look through the neighbours list
                for i in range(len(c.neighbours)):
                    #grab cell at index i
                    nb=c.neighbours[i]
                    #if the state is greater than 0 but less than A
                    if nb.state>0 and nb.state<A:
                        #increment active number (set to 0 at start)
                        active+=1.0
#PACEMAKER
########################################################################               
                if (c.pacemakerIndex==1 and c.state==0):
                    if(self.numIter%p1clock==0):
                        for i in range(len(c.neighbours)):
                            c.neighbours[i].state=1
                            c.neighbours[i].statenext=1
                        c.state=1
                        c.statenext=1
                if (c.pacemakerIndex==2 and c.state==0):
                    if(self.numIter%p2clock==0):
                        for i in range(len(c.neighbours)):
                            c.neighbours[i].state=1
                            c.neighbours[i].statenext=1
                        c.state=1
                        c.statenext=1
                if (c.pacemakerIndex==3 and c.state==0):
                    if(self.numIter%p3clock==0):
                        for i in range(len(c.neighbours)):
                            c.neighbours[i].state=1
                            c.neighbours[i].statenext=1
                        c.state=1
                        c.statenext=1
##########################################################################                        
                        
                #if the total activity for the neighbors is greater than some value
                #set the next state of that cell to 1
                if (active/len(c.neighbours))>threshold:
                    c.statenext=1
            #if the cell state is gt zero, add 1 to the state value
            if c.state>0:
                c.statenext=c.state+1
            #if the state value is greater than A + R, reset to 0
            if c.state>A+R:
                c.statenext=0

        #loop over all the cells and set the states to the next states add 0. to cast into float maybe?
        for i in range(len(self.cells)):
            self.cells[i].state = self.cells[i].statenext + 0.
        #if stimulation is set to something, then iterate over all x,y positions
        #if the stimulation for that point is higher is not 0/False and the state
        #for that cell is at 0 then set the state and the next state to 1
        if stimulation is not None:
            for y, x in product(range(self.ydim), range(self.xdim)):
                if stimulation[x,y] and self.cells[y*par['ca_x_size'] + x].state==0:
                    self.cells[y*par['ca_x_size'] + x].state = 1
                    self.cells[y*par['ca_x_size'] + x].statenext = 1
        #increment counter
        self.numIter+=1
class cell:
    def __init__(self,x,y,pacemakerIndex):
        self.xloc=x
        self.yloc=y
        self.state=0
        self.statenext=0
        self.pacemakerIndex = pacemakerIndex

stim = Stimulus()
iData,oData = stim.get_batch()

fig = plt.figure()
im = plt.imshow(iData[0,:,:,0])


def iterate(ca,ma,A,R,T,p):
    ca.iterate(A,R,T,p)

print(np.shape(iData))
print(np.shape(oData))

def init():
    im.set_data(iData[0,:,:,0])
    return [im]

def animate(n):
    im.set_array(iData[0,:,:,n])
    if(n%5==0):
        print("anim",n)
    return [im]

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=par["frames_per_image"], interval=50, repeat=True)
anim.save('CA.mp4',fps=40,extra_args=['-vcodec','libx264'])
plt.show()
print("done")

fig,ax = plt.subplots()
cmap = mpl.colors.ListedColormap(['white','red'])
bounds = [0,0.5,1.5]
norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
img = plt.imshow(iData[0,:,:,0],interpolation="nearest",origin="lower",cmap=cmap, norm=norm)
plt.colorbar(img,cmap=cmap,norm=norm,boundaries=bounds,ticks=[0,5,10])
plt.show()
cs = stim.ca[0].cells
circles=[]
for c in cs:
    circ = plt.Circle((c.xloc,c.yloc),par["ca_radius"],color='b',fill=False)
    ax.add_artist(circ)
plt.show()


















 