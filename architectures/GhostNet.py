#https://github.com/YeFeng1993/GhostNet-Keras

import os
import math
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dropout, Lambda, Concatenate, DepthwiseConv2D, Reshape, add, multiply
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import Input, Model
from keras.activations import softmax
import keras.backend as K
import tensorflow as tf

def reshapes(x):
	y = Reshape((1,1,int(x.shape[1])))(x)
	return y

def softmaxs(x):
	y = softmax(x)
	return y

def slices(x,channel):
	y = x[:,:,:,:channel] 
	return y

def squeezes(x):
	y = K.squeeze(x,1)
	return y

def GhostModule(x,outchannels,ratio,convkernel,dwkernel,padding='same',strides=1,data_format='channels_last', use_bias=False,activation=None):
	conv_out_channel = math.ceil(outchannels*1.0/ratio)
	x = Conv2D(int(conv_out_channel),(convkernel,convkernel),strides=(strides,strides),padding=padding,data_format=data_format, activation=activation,use_bias=use_bias)(x)
	if(ratio==1):
		return x
	
	dw = DepthwiseConv2D(dwkernel,strides,padding=padding,depth_multiplier=ratio-1,data_format=data_format, activation=activation,use_bias=use_bias)(x)
	#dw = dw[:,:,:,:int(outchannels-conv_out_channel)]
	dw = Lambda(slices,arguments={'channel':int(outchannels-conv_out_channel)})(dw)
	x = Concatenate(axis=-1)([x,dw])
	return x

def SEModule(x,outchannels,ratio):
	x1 = GlobalAveragePooling2D(data_format='channels_last')(x)
	#squeeze = Reshape((1,1,int(x1.shape[1])))(x1)
	squeeze = Lambda(reshapes)(x1)
	fc1 = Conv2D(int(outchannels/ratio),(1,1),strides=(1,1),padding='same',data_format='channels_last', use_bias=False,activation=None)(squeeze)
	relu= Activation('relu')(fc1)
	fc2 = Conv2D(int(outchannels),(1,1),strides=(1,1),padding='same',data_format='channels_last', use_bias=False,activation=None)(relu)
	excitation = Activation('hard_sigmoid')(fc2)
	#scale = x * excitation
	scale = Lambda(multiply)([x, excitation])
	#scale = multiply([x, excitation])
	return scale

def GhostBottleneck(x,dwkernel,strides,exp,out,ratio,use_se):
	x1 = DepthwiseConv2D(dwkernel,strides,padding='same',depth_multiplier=ratio-1,data_format='channels_last', activation=None,use_bias=False)(x)
	x1 = BatchNormalization(axis=-1)(x1)
	x1 = Conv2D(out,(1,1),strides=(1,1),padding='same',data_format='channels_last', activation=None,use_bias=False)(x1)
	x1 = BatchNormalization(axis=-1)(x1)
	y = GhostModule(x,exp,ratio,1,3)
	y = BatchNormalization(axis=-1)(y)
	y = Activation('relu')(y)
	if(strides>1):
		y = DepthwiseConv2D(dwkernel,strides,padding='same',depth_multiplier=ratio-1,data_format='channels_last', activation=None,use_bias=False)(y)
		y = BatchNormalization(axis=-1)(y)
		y = Activation('relu')(y)
	if(use_se==True):
		y = SEModule(y,exp,ratio)
	y = GhostModule(y,out,ratio,1,3)
	y = BatchNormalization(axis=-1)(y)
	y = add([x1,y])
	return y

class GhostNet(object):
	def __init__(self, classes, input_shape):
		self.classes = classes
		self.input_shape = input_shape
		self.model = None
		self.net()
		self.layers = self.model.layers

	def save_weights(self, name, save_format, overwrite):
		self.model.save_weights(name, save_format = save_format, overwrite = overwrite)
		
	def save(self, name, save_format, overwrite):
		self.model.save(name, save_format = save_format, overwrite = overwrite)

	def compile(self, loss, optimizer, metrics):
		self.model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

	def fit(self, train_sequence, steps_per_epoch, class_weight, epochs, validation_data, validation_steps, callbacks, workers, use_multiprocessing):
		return self.model.fit(
			train_sequence,
			steps_per_epoch = steps_per_epoch,
			class_weight = class_weight,
			epochs = epochs,
			validation_data = validation_data,
			validation_steps = validation_steps,
			callbacks = callbacks,
			workers = workers,
			use_multiprocessing = use_multiprocessing
		)
 
	def net(self):
		inputdata = Input(shape=self.input_shape)
		
		x = Conv2D(16,(3,3),strides=(2,2),padding='same',data_format='channels_last',activation=None,use_bias=False)(inputdata)
		x = BatchNormalization(axis=-1)(x)
		x = Activation('relu')(x)
	
		x = GhostBottleneck(x,3,1,16,16,2,False)
		x = GhostBottleneck(x,3,2,48,24,2,False)
		x = GhostBottleneck(x,3,1,72,24,2,False)
		x = GhostBottleneck(x,5,2,72,40,2,True)
		x = GhostBottleneck(x,5,1,120,40,2,True)
		x = GhostBottleneck(x,3,2,240,80,2,False)
		x = GhostBottleneck(x,3,1,200,80,2,False)
		x = GhostBottleneck(x,3,1,184,80,2,False)
		x = GhostBottleneck(x,3,1,184,80,2,False)
		x = GhostBottleneck(x,3,1,480,112,2,True)
		x = GhostBottleneck(x,3,1,672,112,2,True)
		x = GhostBottleneck(x,5,2,672,160,2,True)
		x = GhostBottleneck(x,5,1,960,160,2,False)
		x = GhostBottleneck(x,5,1,960,160,2,True)
		x = GhostBottleneck(x,5,1,960,160,2,False)
		x = GhostBottleneck(x,5,1,960,160,2,True)
	
		x = Conv2D(960,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation=None,use_bias=False)(x)
		x = BatchNormalization(axis=-1)(x)
		x = Activation('relu')(x)

		x = GlobalAveragePooling2D(data_format='channels_last')(x)
		#x = Reshape((1,1,int(x.shape[1])))(x)
		x = Lambda(reshapes)(x)
		x = Conv2D(1280,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation=None,use_bias=False)(x)
		x = BatchNormalization(axis=-1)(x)
		x = Activation('relu')(x)

		x = Dropout(0.05)(x)
		x = Conv2D(self.classes,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation=None,use_bias=False)(x)
		#x = K.squeeze(x,1)
		#x = K.squeeze(x,1)
		#out = softmax(x)
		x = Lambda(squeezes)(x)
		x = Lambda(squeezes)(x)
		out = Lambda(softmaxs)(x)

		self.model = Model(inputdata, out)
		return self.model