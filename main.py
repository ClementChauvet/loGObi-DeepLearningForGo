#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
import golois
import gc

print("here")
planes = 31
moves = 361
N = 10000
epochs = 1000
batch = 96
print("here")
input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype ('float32')
print("here")
policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical (policy)
print("here")
value = np.random.randint(2, size=(N,))
value = value.astype ('float32')
print("here")
end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype ('float32')
print("here")
groups = np.zeros((N, 19, 19, 1))
groups = groups.astype ('float32')
print("here")
print ("getValidation", flush = True)
golois.getValidation (input_data, policy, value, end)


# In[31]:


#to test next configure a ration
filters = 256
trunk = 80
blocks = 11
def mixconv_block(t, filters):
    splitted = tf.split(t, num_or_size_splits=3, axis = 3)
    L = []
    for i in range(len(splitted)):
        L.append(layers.DepthwiseConv2D((i*2+1,i*2+1), padding="same",
            kernel_regularizer=regularizers.l2(0.0001),
            use_bias = False)(splitted[i]))
    t = tf.concat(L, axis = 3)
    return t
    
def SE_Block(t , filters , ratio =16): 
    se_shape = (1, 1, filters )
    se = layers.GlobalAveragePooling2D ()( t )
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense( 4,
        activation="swish",
        use_bias=True)(se)
    se = layers.Dense( filters,
        activation="sigmoid" ,
        use_bias=True)(se) 
    x = layers.multiply([t,se])
    return x


def CA_Dense(inputs, name = "CA", ratio=8):
    inputs = layers.Reshape((1,inputs.shape[1],inputs.shape[2],inputs.shape[3]))(inputs)
    w, h, d, out_dim = [int(x) for x in inputs.shape[1:]]
    temp_dim = max(int(out_dim // ratio), ratio)
 
    h_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 3]))(inputs)
    w_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[2, 3]))(inputs)
    d_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2]))(inputs)
 
    x = layers.Concatenate(axis=1)([w_pool, h_pool, d_pool])
    x = layers.Reshape((1, 1, w + h + d, out_dim))(x)
    x = layers.Conv3D(temp_dim, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x_w, x_h, x_d = layers.Lambda(lambda x: tf.split(x, [w, h, d], axis=3))(x)
    x_w = layers.Reshape((w, 1, 1, temp_dim))(x_w)
    x_d = layers.Reshape((1, 1, d, temp_dim))(x_d)
    x_h = layers.Reshape((1, h, 1, temp_dim))(x_h)
 
    x_w = layers.Conv3D(out_dim, 1, activation='sigmoid')(x_w)
    x_h = layers.Conv3D(out_dim, 1, activation='sigmoid')(x_h)
    x_d = layers.Conv3D(out_dim, 1, activation='sigmoid')(x_d)
    x = layers.Multiply()([inputs, x_w, x_h, x_d])
    x = layers.Reshape((inputs.shape[2],inputs.shape[3],inputs.shape[4]))(x)
    return x


def CA(inputs, name = "CA", ratio=8):
    print(inputs.shape)
    w, h,out_dim = [int(x) for x in inputs.shape[1:]]
    temp_dim = max(int(out_dim // ratio), ratio)
 
    h_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 3]))(inputs)
    w_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[2, 3]))(inputs)
 
    x = layers.Concatenate(axis=1)([w_pool, h_pool])
    x = layers.Reshape((1, 1, w + h))(x)
    x = layers.Conv2D(temp_dim, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    print(x,w,h)
    x_w, x_h= layers.Lambda(lambda x: tf.split(x, [w, h], axis=2))(x)
    x_w = layers.Reshape((1,1,w*temp_dim))(x_w)
    x_h = layers.Reshape((1,1, h, temp_dim))(x_h)
 
    x_w = layers.Dense(out_dim, activation='sigmoid')(x_w)
    x_h = layers.Dense(out_dim, activation='sigmoid')(x_h)
    x = layers.Multiply()([inputs, x_w, x_h])
    return x


def bottleneck_block(x, expand=filters, trunk=trunk, squeeze =False, kernel_DW = (3,3)):
    m = layers.Conv2D(expand, (1,1),
            kernel_regularizer=regularizers.l2(0.0001),
            use_bias = False)(x)
    m = layers.BatchNormalization()(m)
    m = layers.Activation("swish")(m)
    m = layers.DepthwiseConv2D((5,5), padding="same",
            kernel_regularizer=regularizers.l2(0.0001),
            use_bias = False)(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation("swish")(m)
    m = CA_Dense(m)
    m = layers.Conv2D(trunk, (1,1),
            kernel_regularizer=regularizers.l2(0.0001),
            use_bias = False)(m)
    m = layers.BatchNormalization()(m)

    return layers.Add()([m, x])
def getModel (blocks):
    input = keras.Input(shape=(19, 19, 31), name="board")
    x = layers.Conv2D(trunk, (5,5), padding="same",
            kernel_regularizer=regularizers.l2(0.0001))(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x1 = layers.Conv2D(trunk, (3,3), padding="same",
            kernel_regularizer=regularizers.l2(0.0001))(input)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation("swish")(x1)
    x = layers.Add()([x,x1])
    for i in range (blocks):
        x = bottleneck_block (x, filters, trunk, squeeze = True)
    policy_head = layers.Conv2D(1, 1, activation="swish", padding="same",
            use_bias = False,
            kernel_regularizer=regularizers.l2(0.0001))(x)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Activation("softmax", name="policy")(policy_head)
    value_head = layers.GlobalAveragePooling2D()(x)
    value_head = layers.Dense(50, activation="swish",
            kernel_regularizer=regularizers.l2(0.0001))(value_head)
    value_head = layers.Dense(1, activation="sigmoid", name="value",
    kernel_regularizer=regularizers.l2(0.0001))(value_head)
    model = keras.Model(inputs=input, outputs=[policy_head, value_head])
    return model


model = getModel(blocks)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})


# In[ ]:


gc.collect()
load = True
restart_from = 20

version = "17MixConvAttention"

if load == True:
    model = tf.keras.models.load_model("loGObiV"+str(version)+str(restart_from)+".h5")
gc.collect()
lr = model.optimizer.learning_rate
lr = lr / 100
keras.backend.set_value(model.optimizer.learning_rate, lr)
for i in range (restart_from + 1, 10001):
    print ('epoch ' + str (i))
    golois.getBatch (input_data, policy, value, end, groups, i* N)

    history = model.fit(input_data,
                        {'policy': policy, 'value': value}, 
                        epochs=1, batch_size=batch)
    gc.collect()
    

    if (i % 10 == 0):
        golois.getValidation (input_data, policy, value, end)
        val = model.evaluate (input_data,
                              [policy, value], verbose = 0, batch_size=batch)
        print ("val =", val)
        if i>50:
            name = "loGObiV"+str(version)+str(i-50)+".h5"
            get_ipython().system('rm $name')
        model.save ("loGObiV"+str(version)+str(i)+".h5")
        gc.collect()

    if i  %200 == 0 and i!= 0:
        lr = lr / 10
        keras.backend.set_value(model.optimizer.learning_rate, lr)

