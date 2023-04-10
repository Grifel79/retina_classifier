import tensorflow as tf
import os

retina_train = "train"

train_ds = tf.keras.utils.image_dataset_from_directory(
  retina_train,
  validation_split=0.2, seed=123,
  subset="training", image_size=(1000, 1500),
  batch_size = 8)

val_ds = tf.keras.utils.image_dataset_from_directory(
  retina_train,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(1000, 1500),
  batch_size = 8)

num_classes = 2

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
  tf.keras.layers.Dense(num_classes)
])

model.compile(
 optimizer='adam',
 loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
 metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, save_freq='epoch')

history = model.fit(
 train_ds,
 validation_data=val_ds,
 epochs=3,
 callbacks=[cp_callback]
)

