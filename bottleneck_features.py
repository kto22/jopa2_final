
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
K.image_dim_ordering = 'th'

import numpy as np
import os

print("Импортируем InceptionV3 модель.")
inc_model = InceptionV3(include_top=False,
                      weights='imagenet',
                      input_shape=(150, 150, 3))

bottleneck_datagen = ImageDataGenerator(rescale=1./255)

train_generator = bottleneck_datagen.flow_from_directory('data/img_train/',
                                        target_size=(150, 150),
                                        batch_size=32,
                                        class_mode=None,
                                        shuffle=False)

validation_generator = bottleneck_datagen.flow_from_directory('data/img_val/',
                                                               target_size=(150, 150),
                                                               batch_size=32,
                                                               class_mode=None,
                                                               shuffle=False)


print('-'*50)
print('Извлечение некоторых функций из уже обученной нейронки InceptionV3 в качестве numpy массивов.')
print('-'*50)

if not os.path.exists('bottleneck_features/'):
    os.makedirs('bottleneck_features/')
    print('Директория "bottleneck_features/" создана.')
    print('-'*50)

print('Сохранение bn_features_train.npy в bottleneck_features/')
bottleneck_features_train = inc_model.predict(train_generator, 21000)
np.save(open('bottleneck_features/bn_features_train.npy', 'wb'), bottleneck_features_train)

print('Saving bn_features_validation.npy to bottleneck_features/')
bottleneck_features_validation = inc_model.predict(validation_generator, 4000)
np.save(open('bottleneck_features/bn_features_validation.npy', 'wb'), bottleneck_features_validation)
print("Сохранение завершено.")
