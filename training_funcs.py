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
    # %load_ext tensorboard
    # # Clear any logs from previous runs
    # !rm -rf ./logs/
    #
    #
    # %tensorboard --logdir logs/scalars

    #Model


    # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    #
    # cbs=[tensorboard,tf.keras.callbacks.ModelCheckpoint('gdrive/My Drive/project/bin_acc5_4CLASSES.hdf5', save_weights_only=True, monitor='val_accuracy_score', mode='max', save_best_only=True,),
    #      tf.keras.callbacks.ModelCheckpoint('gdrive/My Drive/project/prec5_4CLASSES.hdf5', save_weights_only=True, monitor='val_precision', mode='max', save_best_only=True),
    #      tf.keras.callbacks.ModelCheckpoint('gdrive/My Drive/project/loss5_4CLASSES.hdf5', save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)]
    #
    # history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_val, y_val),callbacks=cbs,shuffle=True,class_weight={0:class_weights[0],1:class_weights[1],2:class_weights[2],3:class_weights[3]})

