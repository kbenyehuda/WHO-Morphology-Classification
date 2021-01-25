import keras.backend as K
K.set_image_data_format('channels_last')
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers


class NN:
    # OUTPUT EXPLANATION:
    # The output is a vector with 5 probabilites for each image:
    # The first 4 refer to the four criteria:
    # label 1 = regular-Oval Head form
    # label 2 = acrosome 40-70% of head area
    # label 3 = Up to 2 vacuoles in acrosome <20% of head area
    # label 4 = If the midpiece is not too thin or thick, doesn't have bulges and is aligned with the head direction
    # label 5 = Cytoplasmic droplet <1/3 of head area or excess residual cytoplasm
    # A cell with above 0.5 in all these labels is considered good.
    # label 6 = TOTAL classification (regardless of the first 4). A cell is good
    #           if has above 0.5 in this label.
    def __init__(self,data,nrows=70,ncols=90,num_classes=5):
        self.data = data
        self.img_rows = nrows
        self.img_cols = ncols
        self.num_classes = num_classes

    def model_cnn(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same',activation='relu', input_shape=(self.img_rows,self.img_cols,1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1500,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, kernel_regularizer=regularizers.l2(0.001),activation='sigmoid'))
        self.model = model
        return(model)

    def upload_weights(self,weights_file):
        score = self.model.load_weights(weights_file)
