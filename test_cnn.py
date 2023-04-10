import tensorflow as tf
import os
import numpy as np

model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(1000, 1500, 3)),
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(2)
])

checkpoint_path = "../AEye/training_ep3/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

# testing on the test_scrambel
test = os.listdir("test_scrambel")
for image in test:
    print(image)
    image = tf.keras.preprocessing.image.load_img("test_scrambel/" + image, target_size=(1000, 1500))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr).argmax(axis=-1)
    print(predictions)


retina_train = "train"
image_size=(1500, 1000)

val_ds = tf.keras.utils.image_dataset_from_directory(
  retina_train,
  validation_split=0.2,
  subset="validation",
  seed=123,                 # same seed as in training!!!
  image_size=(1000, 1500),
  batch_size = 1)

file_paths = val_ds.file_paths  # my validation images

for path in file_paths:
    if path!=None:
        image = tf.keras.preprocessing.image.load_img(path, target_size=(1000, 1500))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        predictions = model.predict(input_arr).argmax(axis=-1)
        label = -1

        # i can get the correct label just from the image's path
        if "left" in path:
            label = 0
        elif "right" in path:
            label = 1

        # here i print all misclassified val images:
        if label != int(predictions):
            print(path)