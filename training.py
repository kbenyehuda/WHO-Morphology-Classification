import pandas as pd
from glob import glob
import numpy as np
import cv2
from model import *
import copy
from training_funcs import get_weighted_loss,calculating_class_weights


import pickle
pickle_in = open("NewXandY.pickle","rb")
dic = pickle.load(pickle_in)
Y = dic['Y']
X = dic['X']

'''
Splitting into train val and test

'''
print('train val test split')
from sklearn.model_selection import train_test_split
X_train, X_rest, Y_train, Y_rest = train_test_split(X, Y, test_size=0.2)
X_val, X_test, Y_val, Y_test = train_test_split(X_rest, Y_rest, test_size=0.5)


'''
Creating new model
'''
print('creating model')
new_model_obj = NN(X,200,200,6)
new_model = new_model_obj.model_cnn()
opt =tf.keras.optimizers.Adam(learning_rate=0.001)
weight1=calculating_class_weights(Y_train)
new_model.compile(loss=get_weighted_loss(weight1), optimizer=opt,metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy_score'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')])
'''

'''
