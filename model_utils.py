from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import keras
from keras.layers import *
from attention_keras import *

def transformer_res_model(input_dim,class_num):

    poolsize = 4
    poolstr = 2
    drop = 0.3 

    input1 = Input(shape=(input_dim[0],input_dim[1]), name='input')
    ##x = Reshape((inpim[0], input_dim[2]))(input1)

    ## First convolutional block (conv,BN, relu)
    
    x = Conv1D(filters=32,
               kernel_size=64,
               padding='same',
               strides=2,
               kernel_initializer='he_normal')(input1)                
    x = BatchNormalization()(x)
    x = Activation('relu')(x)  
    x = Dropout(drop)(x)

    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 = Conv1D(filters=32,
               kernel_size=32,
               padding='same',
               strides=2,
               kernel_initializer='he_normal')(x)
    x1 = MaxPooling1D(pool_size=poolsize,
                      padding='same',
                      strides=poolstr)(x1)    
    x1 = BatchNormalization()(x1)    
    x1 = Activation('relu')(x1)
    x1 = Dropout(drop)(x1)

    # Right branch, shortcut branch pooling
    x2 = Conv1D(filters=32,
               kernel_size=1,
               padding='same',
               strides=2,
               kernel_initializer='he_normal')(x)
    x2 = MaxPooling1D(pool_size=poolsize,
                      padding='same',
                  strides=poolstr)(x2)
    x2 = BatchNormalization()(x2)    
    x2 = Activation('relu')(x2)
    x2 = Dropout(drop)(x2)
    # Merge both branches
    x = keras.layers.add([x1, x2])
    x = Dropout(0.3)(x)
    
    x = Position_Embedding(mode='concat')(x)    
    x = Attention(8, 16)([x, x, x])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(128, activation='relu')(x)
    outputs = Dense(class_num, activation='softmax')(outputs)
    model = Model(inputs=input1, outputs=outputs)    

    return model



def merge_conv_models_transformer_res(input_dims, convModelDims, mergingDenseUnits, mergingDO):
    model_inputs = []
    model_fun = []
    curr_models = []
    for i in range(len(input_dims)):
        curr_models.append(transformer_res_model(input_dims[i]))
        model_inputs.append(Input(input_dims[i]))
        model_fun.append(curr_models[i](model_inputs[i]))

    if len(model_fun) > 1:
        merged = concatenate(model_fun)

    else:
        merged = model_fun[0]

    for i in range(len(mergingDenseUnits)):
        merged = Dense(units=mergingDenseUnits[i], kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(merged)
        merged = PReLU(shared_axes=[1])(merged)
        merged = Dropout(mergingDO[i])(merged)
    pred_layer = Dense(units=2, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(merged)
    pred_layer = Activation('softmax')(pred_layer)
    model = Model(inputs=model_inputs, outputs=pred_layer)
    return model
