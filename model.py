
from tensorflow import keras
from keras import layers
from keras_preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


train_data = train_datagen.flow_from_directory(
    'Trained Images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


with open('flabels.txt', 'w') as f:
    for label in train_data.class_indices:
        f.write(f"{label}\n")


base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)


for layer in base_model.layers:
    layer.trainable = False


model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_data, epochs=10)


model.save('fmodel.h5')
