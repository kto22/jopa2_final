
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import load_model
K.image_dim_ordering = 'th'

a = input('valid/train: ')

if a == 'valid':
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = test_datagen.flow_from_directory(
        'data/img_val/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    pred_generator = test_datagen.flow_from_directory('data/img_val/',
                                                      target_size=(150, 150),
                                                      batch_size=100,
                                                      class_mode='binary')

    model = load_model('saved_model')

    loss, accuracy = model.evaluate(pred_generator)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

elif a == 'train':
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'data/img_train/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        'data/img_val/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    pred_generator = test_datagen.flow_from_directory('data/img_val/',
                                                      target_size=(150, 150),
                                                      batch_size=100,
                                                      class_mode='binary')

    epochs = int(input('Количество эпох обуения: '))

    filepath = "weights/complete_model_checkpoint_weights/weights-improvement-{epoch:02d}-{accuracy:.2f}.h5"

    early_stop = EarlyStopping(monitor='accuracy', min_delta=0.001,
                               patience=10, verbose=1, mode='auto', baseline=0.8)

    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, early_stop]

    model = load_model('saved_model')
    print('-' * 50)
    print('В процессе компилирования модели...')

    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(learning_rate=0.0009, momentum=0.9),
                  metrics=['accuracy'])

    print('-' * 50)
    print('Модель завершила компилирование.')
    print('-' * 50)
    print('Начало процесса обучения модели...')

    model.fit_generator(train_generator, epochs=epochs, callbacks=callbacks_list)

    print('-' * 50)
    print('Модель завершила обучение.')
    print('-' * 50)

    loss, accuracy = model.evaluate(pred_generator)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

    model.save('saved_model', include_optimizer=True)
    print('модель сохранена.')
