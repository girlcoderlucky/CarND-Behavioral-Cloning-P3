import csv
import os
import argparse
import json
import cv2
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Cropping2D
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#Function to randomly flip 10% of the images that is 3 images atleast flipped out of 32 images
def flip_image(image, angle):
    image_flipped = cv2.flip(image, 1)
    angle_flipped = -angle
    return image_flipped, angle_flipped

#Function to randomly rotate between 5 to 25 degrees
def rotate_image(image):
    rows, cols, ch = image.shape
    rotate = random.randint(5, 25)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate, 1)
    rimage = cv2.warpAffine(image, M, (cols, rows))
    return rimage

#Function for brightness augmentation
#Reference : https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.9iigd72nq
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

#Function for horizontal and vertical shifts
#Reference : https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.9iigd72nq
def trans_image(image, steer, trans_range):
    rows, cols, ch = image.shape
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))
    return image_tr, steer_ang

#Function to generate training data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                randimg = random.randint(0, 2)
                name = './data/IMG/'+batch_sample[randimg].split('/')[-1]
                image = cv2.imread(name)
                #Random brightness augmentation
                if (random.randint(0, 5) == 0):
                    image = augment_brightness_camera_images(image)
                angle = float(batch_sample[3])
                #Random horizontal and vertical shifts
                image, angle = trans_image(image, angle, 75)
                #Random flip of images
                if (random.randint(0, 2) == 0):
                    image, angle = flip_image(image, angle)
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def get_model(time_len=1):
  ch, row, col = 3, 160, 320  # camera format

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col,ch)))
  #cropping first 50 rows
  model.add(Cropping2D(cropping=((50, 0), (0, 0)), input_shape=(row, col, ch)))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model


def main():
  parser = argparse.ArgumentParser(description='Steering angle model trainer')

  args = parser.parse_args()
  samples = []
  with open('./data/driving_log.csv') as csvfile:
    print("driving log loaded")
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
      samples.append(line)

  #Get comma.ai model
  model = get_model()

  train_samples, validation_samples = train_test_split(samples, test_size=0.2)

  print("Training samples count {0} , Validation samples count {1}".format(len(train_samples), len(validation_samples)))

  train_generator = generator(train_samples, batch_size=32)
  validation_generator = generator(validation_samples, batch_size=32)

  model.compile(loss='mse', optimizer='adam')
  model.fit_generator(train_generator,
                      samples_per_epoch= len(train_samples),
                      validation_data = validation_generator,
                      nb_val_samples = len(validation_samples),
                      nb_epoch = 5)

  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs/"):
      os.makedirs("./outputs/")

  model.save("./outputs/model.h5")
  with open('./outputs/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

if __name__ == "__main__":
    main()
