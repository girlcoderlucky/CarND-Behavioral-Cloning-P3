**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/angle_histogram.png "Steer Angle Visualization"
[image2]: ./examples/augment_brightness.png "Augment Brightness"
[image3]: ./examples/blur_image.png "Guassain Blur Image"
[image4]: ./examples/center_trans_image.png "Center Horizontal Shift"
[image5]: ./examples/center_trans_image.png "Left Horizontal Shift"
[image6]: ./examples/center_trans_image.png "Right Horizontal Shift"
[image7]: ./examples/flip_image.png "Flipped Image"
[image8]: ./examples/rotate_image.png "Slightly Rotated Image"
[image9]: ./examples/zoom_image.png "Zoomed Image"
[image10]: ./examples/shadow_image.png "Shadowed Image"
[image11]: ./examples/model_visualization.png "Model Visualization"
[video]: ./video.mp4 "Autonomous Driving Video"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

As suggested in the forum I used comma.ai model.Reference : https://github.com/commaai/research/blob/master/train_steering_model.py
My model consists of a convolution neural network with 8x8 and 5x5 convolution kernels (model.py : def get_model()) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

I learned that this project is more about training and transfer learning than having a perfect model architecture. Training data was chosen to keep the vehicle driving on the road, I used a combination of center lane, the left and right lane images to train the model. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA Architecture discussed in the classroom. I then read about comma.ai steering_model, I thought this model might be appropriate for this project.

Reference : https://github.com/commaai/research

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I saved the model when whenever it showed the better behavior as model.h5 using model.save('model.h5') and fine tuned the saved model. If it overfit I would vary the number of epochs and learning rate accordingly for same or additional datasets.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.
For example when there was shadow of tree, car was not not detecting left yellow line correctly was taking left turn. To improve this driving behavior in these cases, I introduced random "shadowed images"

###Update : Also tried with NVIDIA model got better results than comma.ai model

At the end of the process, the vehicle is able to drive autonomously 95% of the first track without leaving the road but at 1 location(sharp left turn) car was going out of track. You can see it in the output video.mp4 that I manually altered path, I think my model is little left baised, many need more training to fix it.

####2. Final Model Architecture

Firstly, added a lambda layer which is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in drive.py.

Secondly, I cropped the first 40 rows and 20 left and right columns of the image in the model only by using Keras Cropping2D Layer. Also, by adding the cropping layer, the model will automatically crop the input images when making predictions in drive.py.

Finally, the  model architecture consisted of a convolution neural network with the following layers and layer sizes.

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
  
Here is a visualization of the architecture :

![alt text][image11]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 2 laps on 1st track one using center lane driving. Also recorded reverse direction of track one for second training data set which helped help the model generalize.
Here is an example image of center lane driving and visualization of the steering angle:

![alt text][image4]
![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to to simulate the effect of car wandering off to the side, and recovering. 
Also added random horizontal shift of the images to simulate the effect of car being at different positions on the road, and add an offset corresponding to the shift to the steering angle. Added 0.004 steering angle units per pixel shift to the right, and subtracted 0.004 steering angle units per pixel shift to the left. 

Below images are the examples of horizontal shift:

![alt text][image4]
![alt text][image5]
![alt text][image6]

Reference : https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.9iigd72nq

Also I added brightness augmentation and shadow augmentation from below reference. Brightness augmentation would definetely help for track two, will simulate day and night conditions. I observed that car was off track when there was shadow of a tree so added shadow image function. 

![alt text][image2]
![alt text][image4]
![alt text][image10]

Then I repeated this process on track two in order to get more data points. Observed that track two data didn't help that much for the car drive in track one.

Used generators with batch size 64, which is much more memory-efficient. Firstly I applied Guassain blur to remove noise. Below image is the example:

![alt text][image3]

To augment the data sat, I also flipped images and angles thinking it's a quick way to augment the data. I thought zoomed images at the turns will help if the car veers off to the side, it should recover back to center.
For example, here are the images that has then been flipped, zoomed and slighlty rotated images:

![alt text][image7]
![alt text][image8]
![alt text][image8]

After the collection process, I had around 40,000 number of data points. I then preprocessed this data by mostly by center images car was more baised to drive straight so randomlay processed center, right and left images for each data points to train the car to drive in the center. I finally randomly shuffled the data set and put 10% of the data into a validation set. The validation set helped determine if the model was over or under fitting. The ideal number of epochs were between 3-5 as evidenced by mean squared error was decreasing and was reduced to ~0.012. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The entire training took about 4-5 minutes. However, it took more than 3 days to arrive at the right training data and training parameters. Finally saved my trained model architecture as model.h5 using model.save('model.h5') 

Here is my final output video:

![alt text][video]

Improvements: This project was defineltely the most challenging and fun in this term. I would continue to architect better model and train to drive the car on track 2 100% autonomously
