
from keras import backend as K
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

K.image_dim_ordering = 'th'

import os
import numpy as np

train_data = np.load(open('bottleneck_features/bn_features_train.npy', 'rb'))
train_labels = np.array([0] * 10500 + [1] * 10500)

validation_data = np.load(open('bottleneck_features/bn_features_validation.npy', 'rb'))
validation_labels = np.array([0] * 2000 + [1] * 2000)

def fc_model():
    fc_model = Sequential()
    fc_model.add(Flatten(input_shape=train_data.shape[1:]))
    fc_model.add(Dense(64, activation='relu', name='dense_one'))
    fc_model.add(Dropout(0.5, name='dropout_one'))
    fc_model.add(Dense(64, activation='relu', name='dense_two'))
    fc_model.add(Dropout(0.5, name='dropout_two'))
    fc_model.add(Dense(1, activation='sigmoid', name='output'))

    return fc_model
print('')
print('-'*50)
print('Обучение верхней части модели на массивах, полученных в первом шаге.')
print('-'*50)
epochs = int(input('Количество эпох обучения: '))
print('-'*50)
print('В процессе подгонки массивов для обучения...')
print('-'*50)
model = fc_model()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=epochs, batch_size=32, validation_data=(validation_data, validation_labels))
print('-'*50)
print('Верхняя часть модели обучена.')
print('-'*50)

if not os.path.exists('weights/top_model_weights/'):
    os.makedirs('weights/top_model_weights/')
    print('Directory "weights/top_model_weights/" has been created')

print('Saving weights to weights/top_model_weights/fc_inception_cats_dogs_250.hdf5')
model.save_weights('weights/top_model_weights/fc_inception_cats_dogs_250.hdf5')

print('Finished')
print('-'*50)
loss, accuracy = model.evaluate(validation_data, validation_labels)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
