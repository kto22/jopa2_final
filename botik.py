
from aiogram import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import shutil
from keras.callbacks import EarlyStopping
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.optimizers import SGD
from keras import backend as K

K.image_dim_ordering = 'th'

model = load_model('saved_model')

bot = Bot(token="5315216942:AAFcaDcCah-bTrLWfl-sASCbQPDJqSzEIIM")
dp = Dispatcher(bot)

import decimal
ctx = decimal.Context()

def float_to_str(f, prec = 18):
    ctx.prec = prec
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.answer(f'{message.from_user.first_name}, добро пожаловать!\nотправьте фоточку пжвста')

@dp.message_handler(content_types=['photo'])
async def bandit(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ['да', ' нет']
    markup.add(*buttons)

    await message.photo[-1].download(destination_file='user_photo/img_val/test.jpg')
    datagen = ImageDataGenerator(rescale=1. / 255)
    img_generator = datagen.flow_from_directory('user_photo/',
                                                target_size=(150, 150),
                                                batch_size=1,
                                                class_mode='binary')
    imgs = img_generator
    predictions = model.predict(imgs)
    await message.answer(float_to_str(predictions[0][0]))
    pred_num = predictions[0][0]
    print(float_to_str(pred_num))
    if float(pred_num) <= 0.2:
        await message.reply("это точно кот!")
    elif float(pred_num) > 0.2 and float(pred_num) < 0.5:
        await message.reply('вроде кот')
    elif float(pred_num) > 0.5 and float(pred_num) < 0.8:
        await message.reply('вроде собака')
    elif float(pred_num) == 0.5:
        await message.reply('не знаю даже')
    else:
        await message.reply('это точно собака!')
    if message.from_user.id == 598970197:
        await message.answer('а ты каримбумчик или кот(болван) хз крч. Биполярка корочк...')

    await message.answer('верно ли предсказание?', reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['да', 'нет'])
async def is_right(message):
    if message.text == 'да':
        await message.answer('ура!')
        if os.path.exists('user_photo/img_val/test.jpg'):
            os.remove('user_photo/img_val/test.jpg')
    elif message.text == 'нет':
        await message.answer('для переобучения нейронки введите пароль!')

@dp.message_handler(lambda message: message.text == '122334455667, ну пожалуйста!')
async def training(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ['cats', 'dogs']
    markup.add(*buttons)
    await message.answer('к какому виду принадлежит животное?', reply_markup=markup)

@dp.message_handler(lambda message: message.text in ['cats', 'dogs'])
async def training(message):
    global model
    a = message.text
    shutil.copyfile('user_photo/img_val/test.jpg', f'data00/img_train/{a}/test.jpg')

    await message.answer('Подождите, модель обучается!')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        'data00/img_train/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    if not os.path.exists('weights/complete_model_checkpoint_weights/'):
        os.makedirs('weights/complete_model_checkpoint_weights/')
        print('Directory "weights/complete_model_checkpoint_weights/" has been created')

    model = load_model('saved_model')

    print('-' * 50)
    print('В процессе компилирования модели...')

    early_stop = EarlyStopping(monitor='accuracy', min_delta=0.001,
                               patience=12, verbose=1, mode='auto', baseline=0.8)

    callbacks_list = [early_stop]

    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(learning_rate=0.0001, momentum=0.9),
                  metrics=['accuracy'])
    print('-' * 50)
    print('Модель завершила компилирование.')
    print('-' * 50)
    print('Начало процесса обучения модели...')

    model.fit_generator(train_generator, epochs=10, callbacks=callbacks_list)

    print('-' * 50)
    print('Модель завершила обучение.')
    print('-' * 50)

    model.save('saved_model')
    print('модель сохранена')

    os.remove(f'data00/img_train/{a}/test.jpg')
    os.remove('user_photo/img_val/test.jpg')

    model = load_model('saved_model')

    await message.answer("модель обучилась!")


if __name__ == '__main__':
    executor.start_polling(dp)
