import csv
import os
import argparse
import json
import cv2
import random
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Cropping2D
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
from keras.optimizers import Adam

#Function to randomly flip 10% of the images that is 3 images atleast flipped out of 32 images
def flip_image(image, angle):
    image_flipped = cv2.flip(image, 1)
    angle_flipped = -angle
    return image_flipped, angle_flipped

#Function to randomly zoom images
#Reference : http://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
def zoom_image(img, zoom_factor):
    h, w = img.shape[:2]
    # width and height of the zoomed image
    zh = int(np.int(zoom_factor * h))
    zw = int(np.int(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out

#Funtion to apply GaussianBlur
def blur_image(image):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    return blur

#Function to randomly rotate between 5 to 25 degrees
def rotate_image(image):
    rows, cols, ch = image.shape
    rotate = random.randint(1, 5)
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
    #tr_y = 40 * np.random.uniform() - 40 / 2
    tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))
    return image_tr, steer_ang

#Function to save images
def save_image(orig, modified, name):
    plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(modified, cv2.COLOR_BGR2RGB))
    plt.title(name), plt.xticks([]), plt.yticks([])
    plt.savefig('./examples/'+ name +'.png')

#Function to generate training data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = shuffle(samples[offset:offset+batch_size])
            images = []
            angles = []
            #To read all 32 center images
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                image = cv2.imread(name)
                original = image
                angle = float(batch_sample[3])

                image = blur_image(image)
                #save_image(original, image, 'blur_image')

                #Random flip of images
                if (random.randint(0, 5) == 0):
                    image, angle = flip_image(image, angle)
                    #save_image(original, image, 'flip_image')

                # Random zoom image if steering angle > abs(0.3)
                if(abs(angle) > 0.3):
                    if (random.randint(0, 2) == 0):
                        image = zoom_image(image, 1.5)
                        #save_image(original, image, 'zoom_image')

                #Random horizontal and vertical shifts for ONLY central images
                image, angle = trans_image(image, angle, 80)
                #save_image(original, image, 'center_trans_image')

                # Random image blur or rotate or brightness augmentation
                ran = random.randint(0, 9)
                if(ran == 5):
                    image = rotate_image(image)
                    #save_image(original, image, 'rotate_image')
                elif (ran == 1) or (ran == 2):
                    image = augment_brightness_camera_images(image)
                    #save_image(original,image,'augment_brightness')

                images.append(image)
                angles.append(angle)

            # To read randomly read 32 left and right  images
            for batch_sample in batch_samples:
                randimg = random.randint(1, 2)
                name = './data/IMG/' + batch_sample[randimg].split('/')[-1]
                image = cv2.imread(name)
                original = image
                angle = float(batch_sample[3])
                if randimg is 1:
                    angle = angle + 0.2
                if randimg is 2:
                    angle = angle - 0.2

                #Blur the image to reduce noise
                image = blur_image(image)

                # Random flip of images
                if (random.randint(0, 5) == 0):
                    image, angle = flip_image(image, angle)

                # Random horizontal and vertical shifts for ONLY central images
                image, angle = trans_image(image, angle, 40)
                #if(randimg == 1):
                    #save_image(original, image, 'left_trans_image')
                #else:
                    #save_image(original, image, 'right_trans_image')

                # Random image blur or rotate or brightness augmentation
                ran = random.randint(0, 9)
                if (ran == 0):
                    image = rotate_image(image)
                elif (ran == 1) or (ran == 2):
                    image = augment_brightness_camera_images(image)

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
  #cropping first 40 rows
  model.add(Cropping2D(cropping=((40, 0), (20, 20)), input_shape=(row, col, ch)))
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

# Function : Plot Angle histogram
def plot_angle_histogram(train, size):
  angles = []
  for value in train:
      for angle in value[1]:
          angles.append(angle)
      if (len(angles) >= size):
          break
  plt.figure()
  plt.hist(angles, bins=100, histtype="step")  # plt.hist passes it's arguments to np.histogram
  plt.title("Angle Distribution")
  plt.xlabel("Value")
  plt.ylabel("Frequency")
  file_name = "./examples/" + "angle_histogram"
  plt.savefig(file_name)

def main():
  samples = []
  with open('./data/driving_log.csv') as csvfile:
    print("driving log loaded")
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
      samples.append(line)

  #Get comma.ai model
  if os.path.exists("./outputs/model.h5"):
      model = load_model("./outputs/model.h5")
      print("Loading existing model.h5")
  else:
      model = get_model()

  train_samples, validation_samples = train_test_split(samples, test_size=0.2)

  print("Training samples count {0} , Validation samples count {1}".format(len(train_samples), len(validation_samples)))

  train_generator = generator(train_samples, batch_size=32)
  validation_generator = generator(validation_samples, batch_size=32)

  plot_angle_histogram(train_generator, len(train_samples))

  model.compile(loss='mse', optimizer=Adam(lr=0.00001))
  model.fit_generator(train_generator,
                      samples_per_epoch= (len(train_samples)*2),
                      validation_data = validation_generator,
                      nb_val_samples = (len(validation_samples)*2),
                      nb_epoch = 3)

  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs/"):
      os.makedirs("./outputs/")

  model.save("./outputs/model.h5")
  with open('./outputs/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

if __name__ == "__main__":
    main()