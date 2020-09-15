import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

print('Version of Tensorflow: {}'.format(tf.__version__))

# Create an object of ImageDataGenerator
train_datagen = ImageDataGenerator( rescale = 1./255, 
                                    shear_range = 0.2, 
                                    zoom_range = 0.2, 
                                    horizontal_flip = True
                                )
# Generate training set using ImageDataGenerator Object created above
training_set = train_datagen.flow_from_directory(   'yalefaces/train', 
                                                    target_size = (64, 64), 
                                                    batch_size = 2,
                                                    class_mode = 'categorical'
                                                )

# Create another of object of ImageDataGenerator class (for test set)
validation_datagen = ImageDataGenerator(rescale = 1./255)
# Generate validation set using ImageDataGenerator Object created above
validation_set = validation_datagen.flow_from_directory(    'yalefaces/test',
                                                target_size = (64, 64),
                                                batch_size = 2,
                                                class_mode = 'categorical'
                                            )

# Build the CNN
cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(MaxPool2D(pool_size=2, strides=2))
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))
cnn.add(Flatten())
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dense(units=15, activation='sigmoid'))

# Compile the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit the data to the CNN
cnn.fit(x = training_set, validation_data = validation_set, epochs = 25)

# Save the model to the disk
save_model(cnn, './model/')