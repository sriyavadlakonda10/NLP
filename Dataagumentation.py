import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import array_to_img


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        rescale=None,
        fill_mode='nearest')

img = load_img(r"D:\Download\pexels-pixabay-45201.jpg")

x = img_to_array(img)
x = x.reshape((1,) + x.shape)


i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=r"C:\Users\Sriya v\VS CODE\nlp\data agumentations", save_prefix='cat', save_format='jpg'):
    i += 1
    if i > 10:
        break  # otherwise the generator would loop indefinitely

