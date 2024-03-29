#-------------------------Cleaning .DS_Store files------------------------------#
import os
for root, dirs, files in os.walk(""):
    for file in files:
        if file == ".DS_Store":
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

#-----------------Importing libraries & variable definitions--------------------#

from keras.preprocessing.image import img_to_array, load_img
import csv
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, concatenate, Input
from keras.optimizers import Adam
from keras.metrics import Accuracy
from keras.models import Model
import numpy as np

IMAGE_RES = 320
x_coordinates = []
y_coordinates = []
bool_values = []
#-------------------------------------------------------------------------------#

#-------------------------Extracting train values-------------------------------#
train_photo_folders = ["train/dik/img", "train/tura/img", "train/yazi/img"]
train_csv_files = ["train/dik/dik.csv", "train/tura/tura.csv", "train/yazi/yazi.csv"]
x_train = []
y_train = []
bool_train = []
photo_train = []

for i in range(3):
  with open(train_csv_files[i], encoding='utf-8-sig') as file: #used encoding argument is written to prevent Byte Order Mark (BOM) in the first row of a CSV file
    csv_read = csv.reader(file, delimiter=';')
    for row in csv_read:
      x_coordinates.append(row[0])
      y_coordinates.append(row[1])
      bool_values.append(row[2])

  train_photos_directory = sorted(os.listdir(train_photo_folders[i]))
  for j, photo_files in enumerate(train_photos_directory):
    photo_path = os.path.join(train_photo_folders[i], str(photo_files))
    photo_array = img_to_array(load_img(photo_path, target_size=(IMAGE_RES, IMAGE_RES)))
    photo_train.append(photo_array / 255)   #normalizing pixels from 0-255 to 0-1

x_train = np.array(x_coordinates).astype(int)
y_train = np.array(y_coordinates).astype(int)
bool_train = np.array(bool_values).astype(int)
photo_train = np.array(photo_train).astype(float)
#-------------------------------------------------------------------------------#

#----Reseting variables--------#
x_coordinates = []
y_coordinates = []
bool_values = []
#------------------------------#

#--------------------------Extracting valid values------------------------------#
valid_photo_folders = ["valid/dik/img", "valid/tura/img", "valid/yazi/img"]
valid_csv_files = ["valid/dik/dik.csv", "valid/tura/tura.csv", "valid/yazi/yazi.csv"]
x_valid = []
y_valid = []
bool_valid = []
photo_valid = []

for i in range(3):
  with open(valid_csv_files[i], encoding='utf-8-sig') as file: #used encoding argument is written to prevent Byte Order Mark (BOM) in the first row of a CSV file
    csv_read = csv.reader(file, delimiter=';')
    for row in csv_read:
      x_coordinates.append(row[0])
      y_coordinates.append(row[1])
      bool_values.append(row[2])

  valid_photos_directory = sorted(os.listdir(valid_photo_folders[i]))
  for j, photo_files in enumerate(valid_photos_directory):
    photo_path = os.path.join(valid_photo_folders[i], str(photo_files))
    photo_array = img_to_array(load_img(photo_path, target_size=(IMAGE_RES, IMAGE_RES)))
    photo_valid.append(photo_array / 255)   #normalizing pixels from 0-255 to 0-1

x_valid = np.array(x_coordinates).astype(int)
y_valid = np.array(y_coordinates).astype(int)
bool_valid = np.array(bool_values).astype(int)
photo_valid = np.array(photo_valid).astype(float)
#-------------------------------------------------------------------------------#

#----Reseting variables--------#
x_coordinates = []
y_coordinates = []
bool_values = []
#------------------------------#

#--------------------------Extracting test values-------------------------------#
test_photo_folders = ["test/dik/img", "test/tura/img", "test/yazi/img"]
test_csv_files = ["test/dik/dik.csv", "test/tura/tura.csv", "test/yazi/yazi.csv"]
x_test = []
y_test = []
bool_test = []
photo_test = []

for i in range(3):
  with open(test_csv_files[i], encoding='utf-8-sig') as file: #used encoding argument is written to prevent Byte Order Mark (BOM) in the first row of a CSV file
    csv_read = csv.reader(file, delimiter=';')
    for row in csv_read:
      x_coordinates.append(row[0])
      y_coordinates.append(row[1])
      bool_values.append(row[2])

  test_photos_directory = sorted(os.listdir(test_photo_folders[i]))
  for j, photo_files in enumerate(test_photos_directory):
    photo_path = os.path.join(test_photo_folders[i], str(photo_files))
    photo_array = img_to_array(load_img(photo_path, target_size=(IMAGE_RES, IMAGE_RES)))
    photo_test.append(photo_array / 255)   #normalizing pixels from 0-255 to 0-1

x_test = np.array(x_coordinates).astype(int)
y_test = np.array(y_coordinates).astype(int)
bool_test = np.array(bool_values).astype(int)
photo_test = np.array(photo_test).astype(float)
#-------------------------------------------------------------------------------#

#--------------Convolving , dense and model training ----------------#
photo_input = Input(shape=(IMAGE_RES, IMAGE_RES, 3), name="photo_input")

first_conv = Conv2D(32, kernel_size=(3, 3), activation="relu")(photo_input)
first_pool = MaxPooling2D(pool_size=(2, 2))(first_conv)
sec_conv = Conv2D(64, kernel_size=(3, 3), activation="relu")(first_pool)
sec_pool = MaxPooling2D(pool_size=(2, 2))(sec_conv)
flatten = Flatten()(sec_pool)

# x-y outputs
x_output = Dense(1, activation='linear', name='x_output')(flatten)
y_output = Dense(1, activation='linear', name='y_output')(flatten)

# coin orientation output ( 0 for heads, 1 for tails)
output_bool = Dense(1, activation='sigmoid', name="output_bool")(flatten)

# Model creation
model = Model(inputs=photo_input, outputs=[x_output, y_output, output_bool])

# Compiling model with Adam optimization algorithm
model.compile(optimizer=Adam(), loss=['mse', 'mse', 'binary_crossentropy'], metrics={'x_output': 'mae', 'y_output': 'mae', 'output_bool': 'accuracy'})

# Training model with the dataset
model.fit(photo_train, [x_train, y_train, bool_train], epochs=50, batch_size=32, validation_data=(photo_valid, [x_valid, y_valid, bool_valid]))
#------------------------------------------------------------------------------------------------------#

#-------------------Testing-------------------#
eval_metrics = model.evaluate(photo_test, [x_test, y_test, bool_test])

print(f"Test Loss: {eval_metrics[0]}")
print(f"Test MAE for x_output: {eval_metrics[1]}")
print(f"Test MAE for y_output: {eval_metrics[2]}")
print(f"Test Accuracy for output_bool: {eval_metrics[3]}")

predictions = model.predict(photo_test)

# Extract predictions for each output
x_output_pred, y_output_pred, bool_output_pred = predictions

for i in range(len(photo_test)):
    print(f"Photo {i + 1} Predictions:")
    print(f"Predicted x_output: {x_output_pred[i]}")
    print(f"Predicted y_output: {y_output_pred[i]}")
    print(f"Predicted bool_output probability: {bool_output_pred[i]}")
    print(f"Predicted bool_output (thresholded): {int(bool_output_pred[i] > 0.5)}")  # Applying a threshold
#----------------------------------------------#

#--------------------------Drawing prediction tables-------------------------------#
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.scatter(x_test, x_output_pred, color='blue', label='Predicted')
plt.plot(x_test, x_test, color='red', linestyle='--', label='True')
plt.title('x_output Predictions')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(y_test, y_output_pred, color='green', label='Predicted')
plt.plot(y_test, y_test, color='red', linestyle='--', label='True')
plt.title('y_output Predictions')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.subplot(2, 1, 2)
plt.scatter(range(len(bool_test)), bool_output_pred, color='purple', label='Predicted Probabilities', alpha=0.5)
plt.scatter(range(len(bool_test)), bool_test, color='orange', label='True Labels', alpha=0.5)
plt.title('bool_output Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Probability / True Label')
plt.legend()

plt.tight_layout()
plt.show()
#-------------------------------------------------------------------------------#