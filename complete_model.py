
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import EarlyStopping

K.image_dim_ordering = 'th'


weights_filename = 'weights/complete_model_checkpoint_weights/weights-improvement-14-1.00.h5'


def complete_model():
    inc_model = InceptionV3(include_top=False,
                            weights='imagenet',
                            input_shape=(150, 150, 3))

    x = Flatten()(inc_model.output)
    x = Dense(64, activation='relu', name='dense_one')(x)
    x = Dropout(0.5, name='dropout_one')(x)
    x = Dense(64, activation='relu', name='dense_two')(x)
    x = Dropout(0.5, name='dropout_two')(x)
    top_model = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=inc_model.input, outputs=top_model)
    #model.load_weights(weights_filename, by_name=True)

    for layer in inc_model.layers[:205]:
        layer.trainable = False

        return model


if __name__ == "__main__":

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

    pred_generator = test_datagen.flow_from_directory(
        'data/img_val/',
        target_size=(150, 150),
        batch_size=100,
        class_mode='binary')
    
    print('-' * 50)
    epochs = int(input('Количество эпох обучения: '))

    filepath = "weights/complete_model_checkpoint_weights/weights-improvement-{epoch:02d}-{accuracy:.2f}.h5"

    if not os.path.exists('weights/complete_model_checkpoint_weights/'):
        os.makedirs('weights/complete_model_checkpoint_weights/')
        print('Directory "weights/complete_model_checkpoint_weights/" has been created')

    early_stop = EarlyStopping(monitor='accuracy', min_delta=0.001,
                               patience=3, verbose=1, mode='auto')

    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, early_stop]

    model = complete_model()
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
