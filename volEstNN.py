#!/bin/python
#
# Neural Network for Volume Estimation
# Copyright (C) 2021 Ettore Chimenti <ek5.chimenti@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
from scipy.integrate import quad, dblquad
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import TypedDict
from pickle import load,dump
from os import path
from sys import argv

class Bounds(TypedDict):
    xmax: float
    ymax: float
    xmin: float
    ymin: float

def generatePointGrid(bounds,npoints):
    xmax = bounds["xmax"]
    xmin = bounds["xmin"]
    ymax = bounds["ymax"]
    ymin = bounds["ymin"]

    xstep = (xmax-xmin)/npoints
    ystep = (ymax-ymin)/npoints

    points = []
    for xn in range(0,npoints):
        for yn in range(0,npoints):
            points.append((xn*xstep, yn*ystep))

    return points

class SurfaceFunFactory:

    def __init__(self, bounds, npoint):
        self.bounds = bounds
        self.npoint = npoint
        self.points = generatePointGrid(bounds,npoint)
        self.funList = []
    
    def add(self, fun, **kwargs ):
        # store surfaces
        self.funList.append(SurfaceFun(fun, **kwargs))

    def generateExample(self,func,params):

        data = []
        for x,y in self.points:
            data.append( func.z(x,y,params) )

        target,error = func.volume(self.bounds, params)

        return data,target

    def generateDataSet(self,nsamples,shuffle=1):
        
        fl = self.funList

        funs = [ fl[i]
            for i in [ x%len(fl) for x in range(0,nsamples) ]]

        params = [ fl[i].shuffle() 
            if shuffle else fl[i].reference() 
            for i in [ x%len(fl) for x in range(0,nsamples) ]]

        results = [ self.generateExample(fun,param) 
                for fun,param in zip(funs,params) ]

        X = [result[0] for result in results]
        y = [result[1] for result in results]

        return X,y

    def referenceDataSet(self):
        return self.generateDataSet(len(self.funList),shuffle=0)

    def randomSurfacePlot(self):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator
        import random

        bd = self.bounds
        fun = random.choice(self.funList)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X = np.arange(bd['xmin'], bd['xmax'], 0.25)
        Y = np.arange(bd['ymin'], bd['ymax'], 0.25)
        X, Y = np.meshgrid(X, Y)
        param = fun.shuffle()
        print("Parameters:", param)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, fun.fun(X,Y,**param), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-2.01, 2.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

class SurfaceFun:

    def __init__(self, fun, **kwargs):
        self.fun = fun
        self.params = kwargs

    def reference(self):
        # output mean parameters

        newparam = dict()
        for parameter in self.params:
            for mean, variance in self.params[parameter]:
                # get a random param with given mean and variance 
                n = mean
                # fill a dict
                newparam[parameter] = n

        return( newparam )

    def shuffle(self):
        # output randomized parameters

        newparam = dict()
        for parameter in self.params:
            for mean, variance in self.params[parameter]:
                # get a random param with given mean and variance 
                n = np.random.normal(mean,variance)
                # fill a dict
                newparam[parameter] = n

        return( newparam )

    def z(self, x, y, params):
        # calculate function with given parameters
        f = self.fun
        return(f(x,y,**params))

    def volume(self, bounds, params):
        # calculate surface
        p = params
        # reinstantiating lambda with fixed parameters
        f = lambda x, y: self.fun(x,y,**p)
        # calculate double integral
        return dblquad( f, bounds['xmin'],
                           bounds['xmax'],
                           lambda x: bounds['ymin'],
                           lambda x: bounds['ymax']) 

def init(npoints):
    bounds: Bounds = { 
            'xmin': 0,
            'xmax': 20,
            'ymin': 0,
            'ymax': 20
    }
    funFact = SurfaceFunFactory(bounds,npoints)

    fun = lambda x,y,z0,A : A*np.sin(x+y)+z0
    funFact.add(fun, z0=[(1,0.1)],
                    A=[(1.0/30,0.02)])

    fun = lambda x,y,z0,A,B,C,D,E : A*(x+B)*(y+C)*np.sin(D*(x+E*y))+z0
    funFact.add(fun, z0=[(1,0.1)],
                    A=[(1.0/1200,0.0001)],
                    B=[(20,0.02)],
                    C=[(10,0.02)],
                    D=[(1.0/5,0.02)],
                    E=[(1.0/2,0.02)])

    fun = lambda x,y,z0,A,B : A*np.sin(B*(x+y))+z0
    funFact.add(fun, z0=[(1,0.1)],
                    A=[(1.0/3,0.02)],
                    B=[(1.0/10,0.02)])

    fun = lambda x,y,z0,A,B,C : A*(B*x*x+C*x*y-y*y)+z0
    funFact.add(fun, z0=[(1,0.1)],
                    A=[(1.0/400,0.002)],
                    B=[(2,0.02)],
                    C=[(3,0.02)])

    return funFact

def savedataset(X,y,namefile):
    with open(namefile, 'wb') as output:
        dump((X,y), output)

def loaddataset(namefile):
    with open(namefile, 'rb') as output:
        X,y = load(output)
        return X,y

def savenetwork(net,namefile):
    with open(namefile, 'wb') as output:
        dump(net, output)

def loadnetwork(namefile):
    with open(namefile, 'rb') as output:
        r = load(output)
        return r

def savelog(save,namefile):
    import csv

    with open(namefile, mode='w') as f:
        csvwriter = csv.writer(f, delimiter=',', quotechar='"',
                quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(save)

def main(argv):
    if len(argv) < 3:
        print("not enough args: npoints nexamples layers")
        exit(1)

    npoints = int(argv[1]) if int(argv[1]) > 0 else 0
    nexamples = int(argv[2]) if int(argv[2]) > 0 else 0

    layers_str = argv[3:]
    layers = tuple( [ int(x) for x in argv[3:] ] )

    if not npoints:
        print("invalid args")
        exit(2)
    
    print("Initializing...")
    funFact = init(npoints)

    # plot a random surface
    if not nexamples:
        while [ 1 ]:
            funFact.randomSurfacePlot()
        exit(0)

    print("Creating dataset...")
    Xref,yref = funFact.referenceDataSet()

    X = []
    y = []

    namefile = 'dataset_n{}_ex{}.bin'.format(npoints,nexamples)

    if path.exists(namefile):
        print( "Found dataset, loading...")
        X,y = loaddataset(namefile)
        print("Loaded {} datapoints".format(len(X)))
    else:
        X,y = funFact.generateDataSet(nexamples)
        print( "Saving dataset...")
        savedataset(X,y,namefile)

    if not layers:
        print("Skipping training")
        exit(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("dim trainset", len(X_train))
    print("dim testset", len(X_test))

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    X_dataref_scaled = sc.transform(Xref)

    netnamefile = 'net_n{}_ex{}_l{}.bin'.format(npoints,nexamples,"_".join(layers_str))
    if path.exists(netnamefile):
        print( "Found net, loading...")
        regr = loadnetwork(netnamefile)
    else:
        print("Training network...")
        from joblib import parallel_backend
        with parallel_backend('threading', n_jobs=8):
            regr = MLPRegressor(hidden_layer_sizes=layers,
                                random_state=1,
                                max_iter=5000
                ).fit(X_train_scaled, y_train)
        print( "Saving network...")
        savenetwork(regr,netnamefile)

    print(regr)

    log = []

    for dataref, targetref in zip(X_dataref_scaled,yref):
        predicted = regr.predict([dataref])
        err = predicted - targetref
        print("Data analyzed: ", predicted)
        print("Real volume  : ", targetref)
        print("Error        : ", err)

        log.append(err[0])

    rscore = regr.score(X_test_scaled, y_test)
    log.append(rscore)
    print("score on test set is", rscore)

    log.extend([npoints,nexamples])
    log.extend(layers_str)

    lognamefile = 'net_n{}_ex{}_l{}.csv'.format(npoints,nexamples,"_".join(layers_str))
    savelog(log,lognamefile)

main(argv)
