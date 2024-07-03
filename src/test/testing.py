import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model
from keras.applications.efficientnet import preprocess_input
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

root_dir = Path().resolve().parents[0]
print('\n\n' , root_dir)
# LOAD THE TRAINED MODEL
model_pth = os.path.join(root_dir, "model/")
model = tf.saved_model.load(model_pth)
print("FUSED-DENSENET-TINY INITIALIZED!")
print("The model consists of", len(model.layers), "layers")
data_dir = Path().resolve().parents[1]
# CHOOSE TEST DATASET
test_data_dir = os.path.join(data_dir, "data/test")
# PREPARE THE DATA INPUT
img_width, img_height = 128,128
img_rows, img_cols = 128, 128
input_shape = (img_rows,img_cols,3)

# DATA GENERATORS
batch_size = 16
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size=(img_rows,img_cols),
batch_size=batch_size,
class_mode='categorical',
shuffle=False,
classes=['fake', 'real']
)


# CHECK THE NUMBER OF SAMPLES
nb_test_samples = len(test_generator.filenames)
if nb_test_samples == 0:
  print("NO DATA VALIDATION FOUND! Please check your validation data path and folders!")
  print("Check the data folders first!")
else:
  print("Validation samples found!")
# CHECK THE CLASS INDICES
print(test_generator.class_indices)
# TRUE LABELS
Y_test = test_generator.classes
num_classes = len(test_generator.class_indices)
if nb_test_samples > 0:
  print("Generators are set!")
  print("Check if the dataset is complete and has no problems before proceeding.")

# Make predictions
y_pred = model.predict(test_generator, nb_test_samples/batch_size, workers=1)


#Evaluate model
accuracy = accuracy_score(Y_test, y_pred.argmax(axis=-1))
print('The accuracy of the model is:', accuracy)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred.argmax(axis=-1))
auc = metrics.auc(fpr, tpr)
print('AUC:', auc)
# PRINT CONFUSION MATRIX
# print(confusion_matrix(Y_test, y_pred.argmax(axis=-1)))
conf = confusion_matrix(Y_test, y_pred.argmax(axis=-1))
disp = ConfusionMatrixDisplay(confusion_matrix=conf)
disp.plot()
# plt.savefig('C:\\Users\\HP\\OneDrive\\Desktop\\DFDC_DATASET\\testing result', dpi=300)

