# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

happy=pd.read_csv('happy.csv')
angry=pd.read_csv('angry.csv')
sad=pd.read_csv('sad.csv')

happy['expr']='happy'
angry['expr']='angry'
sad['expr']='sad'


dataset=happy.append(angry,ignore_index = True).append(sad,ignore_index = True)

x=dataset.iloc[:,0:6]
y=dataset.iloc[:,-1]


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.15, random_state = 89)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(xtrain,ytrain)

import graphviz

dot_data = tree.export_graphviz(clf, out_file=None,feature_names=list(x.columns),
                                class_names=['angry','happy','sad'],
                                filled=True, rounded=True,special_characters=True)

graph = graphviz.Source(dot_data) 


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = list(x.columns),class_names=['angry','sad','happy'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('dt.png')
Image(graph.create_png())