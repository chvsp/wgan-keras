import keras
from keras.layers import Dense, Input
from keras.models import Sequential

def MLP_G(isize, nz, nc, ngf):

	model = Sequential()

	model.add(Dense(ngf, input_dim=nz, activation='relu'))
	model.add(Dense(ngf, activation='relu'))
	model.add(Dense(ngf, activation='relu'))
	model.add(Dense(nc * isize * isize))

	return model


def MLP_D(isize, nz, nc, ndf):

	model = Sequential()

	model.add(Dense(ndf, input_dim= nc*isize*isize, activation='relu'))
	model.add(Dense(ndf, activation='relu'))
	model.add(Dense(ndf, activation='relu'))
	model.add(Dense(1))

	return model

