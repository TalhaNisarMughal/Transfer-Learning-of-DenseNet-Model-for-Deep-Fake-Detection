import time
from tkinter.tix import InputOnly
from wsgiref.types import InputStream
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from tkinter.tix import InputOnly
from tkinter.ttk import Entry
from tensorflow.keras.layers import Input

#from tensorflow.keras.layers import Input
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D, Dense, Add, Dropout
from sklearn.metrics import accuracy_score, mean_squared_error, mean_squared_log_error, classification_report, confusion_matrix, roc_curve, auc
from pathlib import Path



# Define the paths for train and validation data directories
train_data_dir = "C:\\Users\\HP\\OneDrive\\Desktop\\DFDC_DATASET\\train"
validation_data_dir = "C:\\Users\\HP\\OneDrive\\Desktop\\DFDC_DATASET\\validation"

# Set the input image dimensions
img_rows, img_cols = 128, 128
input_shape = (img_rows, img_cols, 3)
model_input = Input(shape=input_shape)

#model_input = InputOnly(shape=input_shape)
print("The Input size is set to", model_input)

# Data Generators
batch_size = 16
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['fake', 'real']
)
validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    classes=['fake', 'real']
)
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
print("Data folders found!")

# Check if the training samples are found
if nb_train_samples == 0:
    print("NO DATA TRAIN FOUND! Please check your train data path and folders!")
else:
    print("Train samples found!")

# Check if the validation samples are found
if nb_validation_samples == 0:
    print("NO DATA VALIDATION FOUND! Please check your validation data path and folders!")
    print("Check the data folders first!")
else:
    print("Validation samples found!")

# Check the class indices
train_generator.class_indices
validation_generator.class_indices
num_classes = len(train_generator.class_indices)
if nb_train_samples and nb_validation_samples > 0:
    print("Generators are set!")
    print("Check if the dataset is complete and has no problems before proceeding.")

# Transfer Learning - DenseNet121-A
def densenet_tiny_A_builder(model_input):
    densenet_tiny_A_builder = DenseNet121(weights='imagenet', include_top=False, input_tensor=model_input)
    # Partial Layer Freezing
    for layer in densenet_tiny_A_builder.layers:
        layer.trainable = False
    # Model Truncation
    x = densenet_tiny_A_builder.layers[-354].output
    model = Model(inputs=densenet_tiny_A_builder.input, outputs=x, name='densenet-tiny-A')
    return model

# Generate the DenseNet121-A model
densenet_tiny_A = densenet_tiny_A_builder(model_input)

# Plot the model structure
print("PLEASE CHECK THE ENTIRE MODEL UP TO THE END")
densenet_tiny_A.summary()
print("DenseNet-Tiny-A successfully built!")

# Transfer Learning - DenseNet121-B
def densenet_tiny_B_builder(model_input):
    densenet_tiny_B_builder = DenseNet121(weights='imagenet', include_top=False, input_tensor=model_input)
    # Re-training all layers (re-naming layers to prevent overlaps)
    for layer in densenet_tiny_B_builder.layers:
        layer.trainable = True
        layer._name = layer._name + str("_mirror")
    # Model Truncation
    x = densenet_tiny_B_builder.layers[-354].output
    model = Model(inputs=densenet_tiny_B_builder.input, outputs=x, name='densenet_tiny-B')
    return model

# Generate the DenseNet121-B model
densenet_tiny_B = densenet_tiny_B_builder(model_input)

# Plot the model structure
print("PLEASE CHECK THE ENTIRE MODEL UP TO THE END")
densenet_tiny_B.summary()
print("DenseNet-Tiny-B successfully built!")

# Concatenate the models as a single pipeline
models = [densenet_tiny_A, densenet_tiny_B]
print("Concatenation success!")
print("Fused-DenseNet-Tiny ready to connect with its ending layers!")

# Build the Fused-DenseNet-Tiny
def fused_densenet_tiny(models, model_input):
    outputs = [m.output for m in models]
    y = Add()(outputs)
    y = GlobalAveragePooling2D()(y)
    y = Dense(512, activation='relu', use_bias=True)(y)
    y = Dropout(0.5)(y)
    prediction = Dense(num_classes, activation='softmax', name='Softmax_Classifier')(y)
    model = Model(model_input, prediction, name='fused_densenet_tiny')
    return model

# Instantiate the ensemble model and report the summary
fused_densenet_tiny = fused_densenet_tiny(models, model_input)
print()
print()
print("PLEASE CHECK THE MODEL UP TO THE END")
print("Fused-DenseNet-Tiny complete and ready for compilation and training!")
print()
print()
print()
fused_densenet_tiny.summary()
print("Building of the Fused-DenseNet-Tiny COMPLETE!")

batch_size = 16
epochs = 10
start_time = time.time()
fused_densenet_tiny.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy']) 

callbacks = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True,
)

history = fused_densenet_tiny.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks,
    validation_steps=nb_validation_samples // batch_size,
    verbose=1
)

elapsed_time = time.time() - start_time
print("Elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

# Save the Fused-DenseNet-Tiny after training completes
fused_densenet_tiny.save("C:\\Users\\HP\\OneDrive\\Desktop\\DFDC_DATASET\\results")

train_accuracy = history.history['accuracy'][-1]
print('Overall Training Accuracy:', train_accuracy)
