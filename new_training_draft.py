import pandas as pd
from glob import glob
import numpy as np
import cv2
from model import *
import copy
from training_funcs import get_weighted_loss,calculating_class_weights


# '''
# Uploading the Labels and Image Names
# '''
# print('started')
# label_path = 'donors_and_feats_filtered.xlsx'
# Y_df = pd.read_excel(label_path,na_values=['nan','N'])
# Y_5 = np.array(Y_df)[:,-5:]
# mask = np.all(np.isnan(Y_5.astype(np.float32)), axis=1)
# Y_5 = Y_5[~mask]
# Y_6th = (np.nanmean(Y_5,axis=1)==1.0)*1.0
# Y_6 = np.concatenate((Y_5,Y_6th[:,np.newaxis]),axis=1)
#
# Y_6[np.isnan(Y_6.astype(np.float32))]=0.5
#
# Img_Names = np.array(Y_df)[:,4]
# Img_Names = Img_Names[~mask]
# print('got image names')
#
# '''
# Uploading the images
# '''
#
# imgs_path = 'new_cell_pics/Including midpiece, for keren/output1/'
# print('starting to load pics')
# X = []
# is_not_found = []
# for i in range(len(Img_Names)):
#     img_filename = glob(imgs_path+Img_Names[i]+'.png')
#     if len(img_filename)>0:
#         X.append(cv2.imread(img_filename[0]))
#     else:
#         is_not_found.append(i)
#
# X = np.array(X)
#
# for i in range(len(is_not_found)):
#     i_to_del = is_not_found[len(is_not_found)-i-1]
#     Y_6 = np.delete(Y_6,i_to_del,0)
#     # Y_5 = np.delete(Y_ֵ5, i_to_del, 0)
#
# print('finished')
# '''
# Augmentation
# '''
#
# print('augmenting')
# X__flip = np.flip(X, 1)
# X = np.append(X, X__flip, axis=0)
# Y_6 = np.append(Y_6, Y_6, axis=0)
# Y_5 = np.append(Y_5, Y_5, axis=0)
#
# '''
# Balancing the dataset
# '''
#
# print('balancing')
# Y_stats = np.sum(Y_6,axis=0)
# num_of_good_cells = Y_stats[-1]
# inds_of_good_cells = np.where(Y_6[:,-1]==1.0)
# X_good = X[inds_of_good_cells]
# Y_good = Y_6[inds_of_good_cells]
#
# all_inds = list(copy.copy(inds_of_good_cells[0]))
#
# def rand():
#     all_bad_inds = []
#     for i in range(int(num_of_good_cells)):
#         r = None
#         while r == None or r in all_inds:
#             r = np.random.randint(0,np.shape(X)[0]-1)
#         all_inds.append(r)
#         all_bad_inds.append(r)
#     return all_bad_inds
#
# inds_of_bad_cells = rand()
# X_bad = X[inds_of_bad_cells]
# Y_bad = Y_6[inds_of_bad_cells]
#
# X = np.concatenate((X_good,X_bad))[:,:,:,0]
# Y = np.concatenate((Y_good,Y_bad))


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
Creating the old model and loading weights
'''
# model_obj = NN(X)
# old_model = model_obj.model_cnn()
#
# weights_file = 'weights.hdf5'
# model_obj.upload_weights(weights_file)

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
