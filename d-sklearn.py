#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 22:03:16 2023

@author: nuohaoliu
"""

import numpy as np
# from Node import Node
import re
from sklearn import tree
import matplotlib.pyplot as plt

my_file = open("data/D128.txt", "r")
data1 = my_file.read()
# data1 = data1.split('\n')
data1 = re.split('[ \t\n]',data1)
data1 = list(filter(None, data1))
data2 = list()

my_file = open("data/D128-test.txt", "r")
datat1 = my_file.read()
# data1 = data1.split('\n')
datat1 = re.split('[ \t\n]',datat1)
datat1 = list(filter(None, datat1))
datat2 = list()

node_counter = 0
treeinfo = np.zeros((0,4))

for i in range(0, len(data1)):
    data2.append(float(data1[i]))

data = np.asarray(data2).reshape((int(len(data2)/3),3))


for i in range(0, len(datat1)):
    datat2.append(float(datat1[i]))
datat = np.asarray(datat2).reshape((int(len(datat2)/3),3))
# data = data3.reshape((5,3))
    
# import graphviz
# DOT data

    
# Testing
if __name__ == "__main__":

    # from sklearn import svm, datasets
    # from DecisionTree import DecisionTree

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


    X_train = data[:,:2]
    y_train = data[:,2:3].flatten()
    y_train = np.asarray(y_train, dtype = 'int')
    
    X_test = datat[:,:2]
    y_test = datat[:,2:3].flatten()
    y_test = np.asarray(y_test, dtype = 'int')

    
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X_train, y_train)
    node = clf.tree_.node_count
    tree.plot_tree(clf)
    
    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy
    
    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out
    fig, ax = plt.subplots()
# title for the plots
    title = ('Sklearn Decision boundary')
    # Set-up grid for plotting.
    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    xmax = 1.5
    xmin = -xmax
    ymax = 1.5
    ymin = -ymax
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, vmin=0, vmax = 1, s=20, edgecolors='k')
    ax.set_ylabel('feature x2')
    ax.set_xlabel('feature x1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.set_aspect(1)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(np.linspace(xmin,xmax,5))
    plt.yticks(np.linspace(ymin,ymax,5))
    # plt.xticks(range(min(valueX), max(valueX)+1))
    ax.legend()
    plt.show()
    
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print('n:',node,'errn:',1-acc)
        
    # length of data as indices of n_data
# len data as indices of n_data

    # dot_data = tree.export_graphviz(clf, out_file=None, 
    #                                 feature_names=iris.feature_names,  
    #                                 class_names=iris.target_names,
    #                                 filled=True)

    # Draw graph
    # graph = graphviz.Source(dot_data, format="png") 
    # graph
    # print(clf.node.feature)
    
    # best = clf._best_split
    # print(best)
    # tree = clf._build_tree
    # print(tree)
    
    # y_pred = clf.predict(X_test)
    # acc = accuracy(y_test, y_pred)
    # print('y_pred:',y_pred)
    # print('y_true:',y_test)
    # print("Accuracy:", acc)