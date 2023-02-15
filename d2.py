#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 22:03:16 2023

@author: nuohaoliu
Some ideas were inspired by Github author marvin
"""

# create the node class to stroe the information of the tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _stopping_criteria(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        return False
    
    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def _gainratio(self, X, y, thresh):
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._split_create(X, thresh)
        n, left_n, right_n = len(y), len(left_idx), len(right_idx)
        

        if left_n == 0 or right_n == 0: 
            return 0,parent_loss,0
        
        child_loss = (left_n / n) * self._entropy(y[left_idx]) + (right_n / n) * self._entropy(y[right_idx])
        inforgain = parent_loss - child_loss
        # print(left_n,right_n)
        splitinformation = - (left_n/n) * np.log2(left_n/n) - (right_n/n) * np.log2(right_n/n)
        # print(splitinformation)
        gainratio = inforgain / splitinformation
        return gainratio, parent_loss,splitinformation
    
    def _information_gain(self, X, y, thresh):
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._split_create(X, thresh)
        n, left_n, right_n = len(y), len(left_idx), len(right_idx)

        if left_n == 0 or right_n == 0: 
            return 0
        
        child_loss = (left_n / n) * self._entropy(y[left_idx]) + (right_n / n) * self._entropy(y[right_idx])
        inforgain = parent_loss - child_loss
        return inforgain
    
    
    def _split_create(self, X, thresh):
        left_idx = np.argwhere(X < thresh).flatten()
        right_idx = np.argwhere(X >= thresh).flatten()
        return left_idx, right_idx

    def _split_best(self, X, y, features):
        split = {'gain_ratio':- 1, 'feat': None, 'thresh': None, 'inforgain': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                gain_ratio,pl,splitinfo = self._gainratio(X_feat, y, thresh)
                gain_ratio = float(gain_ratio)
                splitinfo = float(splitinfo)
                # print(splitinfo)
                
                inforgain = self._information_gain(X_feat, y, thresh)
                # print(splitinfo)
                # print('Candidate split [X%d < %.6f] ; Gainratio:%.6f,inforgain:%.6f, splitinfo:%.6f' 
                       # %(feat+1,thresh, gain_ratio, inforgain, splitinfo))
                # print('x',feat,'thresholds',thresh,'gainratio',gr,'inofrgain',gain_ratio)
                # print('x',feat,thresh,gr,gain_ratio,pl,cl)
                # print(gr,gain_ratio,pl,cl)


                if gain_ratio > split['gain_ratio']:
                    split['gain_ratio'] = gain_ratio
                    split['feat'] = feat
                    split['thresh'] = thresh
                    split['inforgain'] = inforgain

        return split['feat'], split['thresh'], split['gain_ratio'], split['inforgain']
        # print('feat',feat,'thresh',thresh)
        
    def _build_tree(self, X, y, treeinfo,depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))
    
        # judge wether to stop
        if self._stopping_criteria(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)
    
        # get the best split tree
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        feat_best, thresh_best, gainratio, infogain= self._split_best(X, y, rnd_feats)
        left_idx, right_idx = self._split_create(X[:, feat_best], thresh_best)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], treeinfo, depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], treeinfo, depth + 1)
        # print(1)
    
        info = np.reshape(np.asanyarray([depth,feat_best+1,thresh_best,gainratio,infogain],float),(1,5))
        # print(left_idx[0],right_idx)
        # print(treeinfo.shape)
        # print(info.shape)
        # treeinfo = np.append(treeinfo,info,axis=0)
        tree = str(info)
        # print(left_idx, right_idx)
        left = str(left_idx)
        right = str(right_idx)
        # print(info)
        
        with open('d2-output.txt', 'a') as f:
            # f.write(" ".join(map(str, tree)))
            f.write(tree)
            f.write(',')
            f.write(left)
            f.write(',')
            f.write(right)
            # f.write()
            # f.write('left from %d to %d  ' %(left_idx[0],left_idx[-1]))
            # f.write('right from %d to %d' %(right_idx[0],right_idx[-1]))
            # f.write("%s %s\n" % (int("0xFF" ,16), int("0xAA", 16)))
            f.write('\n')
            f.close

        # print('\n--------------------')
        print('Best Split in depth%d: [X%d $<$ %.3f] ; Gainratio:%.3f' %(depth,feat_best+1,thresh_best,infogain))
        # print('    left: [X%d < %.3f]' % (feat_best+1,thresh_best))
        x1 = np.array(X[left_idx, :])
        y1 = np.array(y[left_idx].astype(int))
        y1 = np.reshape(y1,(x1.shape[0],1))   
        xyl = np.concatenate((x1,y1),axis=1)    
        # print('\n'.join(['        '+str(row) for row in xyl]))
        
        # print('    right: [X%d > %.3f]' % (feat_best+1,thresh_best))
        x2 = np.array(X[right_idx, :])
        y2 = np.array(y[right_idx].astype(int))
        y2 = np.reshape(y2,(x2.shape[0],1))   
        xyr = np.concatenate((x2,y2),axis=1)    
        # print('\n'.join(['        '+str(row) for row in xyr]))
    
        return Node(feat_best, thresh_best, left_child, right_child)

    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def fit(self, X, y, treeinfo):
        self.root = self._build_tree(X, y,treeinfo)

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
    
import re
import numpy as np
my_file = open("data/D2.txt", "r")
data1 = my_file.read()
# data1 = data1.split('\n')
data1 = re.split('[ \n]',data1)
data1 = list(filter(None, data1))
data2 = list()

for i in range(0, len(data1)):
    data2.append(float(data1[i]))

data = np.asarray(data2).reshape((int(len(data2)/3),3))
treeinfo = np.zeros((0,4))

# data = data3.reshape((5,3))
    
# import graphviz
# DOT data

    
# Testing
if __name__ == "__main__":


    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


    X = data[:,:2]
    y = data[:,2:3].flatten()
    y = np.asarray(y, dtype = 'int')
    
    # data = datasets.load_breast_cancer()
    # X, y = data.data, data.target
    # X, y = data.data, data.target

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=1
    # )
    X_train = X
    y_train = y

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train,treeinfo)
    
    # length of data as indices of n_data
# len data as indices of n_data
    n_data = data.shape[0]
    # n_samples based on percentage of n_data
    n_samples = int(n_data * .8)
    # m_samples as the difference (the rest)
    m_samples = n_data - n_samples
    # slice n_samples up until the last column as feats
    train_feats = data[:n_samples, :-1]
    # slice n_samples of only the last column as trgts
    train_trgts = data[:n_samples, -1:]
    # slice n_samples up until the last column as feats
    test_feats = data[:m_samples, :-1]
    # slice n_samples of only the last column as trgts
    test_trgts = data[:m_samples, -1:]
    
    # dot_data = tree.export_graphviz(clf, out_file=None, 
    #                                 feature_names=iris.feature_names,  
    #                                 class_names=iris.target_names,
    #                                 filled=True)

    # Draw graph
    # graph = graphviz.Source(dot_data, format="png") 
    # graph
    # print(clf.node.feature)
    
    # best = clf._split_best
    # print(best)
    # tree = clf._build_tree
    # print(tree)
    
    # y_pred = clf.predict(X_test)
    # acc = accuracy(y_test, y_pred)
    # print('y_pred:',y_pred)
    # print('y_true:',y_test)
    # print("Accuracy:", acc)