import numpy as np
import keras.backend as K
K.set_image_data_format('channels_last')
import tensorflow as tf

def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 3])
    for i in range(number_dim):
        if 0.5 in np.unique(y_true[:,i]):
            weights_i = compute_class_weight('balanced', [0.,0.5,1.], y_true[:, i])
            weights[i] = weights_i
        else:
            weights_i = compute_class_weight('balanced', [0., 1.], y_true[:, i])
            weights[i] = np.array([weights_i[0]]+[0]+[weights_i[1]])
    return weights

def calculating_class_weights_new(y_true):
    y_stats = np.mean(y_true,axis=0)
    max_stat = np.max(y_stats)


#0101 we1*1
def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,2]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss



if __name__=="__main__":
    pass
    
