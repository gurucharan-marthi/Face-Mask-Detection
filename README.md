# Face-Mask-Detection
In order to protect ourselves from the COVID-19 Pandemic, almost every one of us tend to wear a face mask. It becomes increasingly necessary to check if the people in the crowd wear face masks in most public gatherings such as Malls, Theatres, Parks. The development of an AI solution to detect if the person is wearing a face mask and allow their entry would be of great help to the society. In this, a simple Face Mask detection system is built using the Deep Learning technique called as Convolutional Neural Networks (CNN). This CNN Model is built using the TensorFlow framework and the OpenCV library which is highly used for real-time applications. This model can also be used to develop a full-fledged software to scan every person before they can enter the public gathering. Using this model, an accuracy of over 96% is obtained. This can also be used further to achieve even higher levels of accuracy.

Medium Article - [Click Here!](https://towardsdatascience.com/covid-19-face-mask-detection-using-tensorflow-and-opencv-702dd833515b)

### Data - 
I have used the face mask dataset provided by [Prajna Bhandary](https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A6655711815361761280/) available at [Github](https://github.com/prajnasb/observations/tree/master/experiements/data)

### CNN Architecture -
In this proposed method, the Face Mask detection model is built using the Sequential API of the keras library. This allows us to create the new layers for our model step by step. The various layers used for our CNN model is described below.

The first layer is the Conv2D layer with 100 filters and the filter size or the kernel size of 3X3. In this first step, the activation function used is the ‘ReLu’. This ReLu function stands for Rectified Linear Unit which will output the input directly if is positive, otherwise, it will output zero.  The input size is also initialized as 150X150X3 for all the images to be trained and tested using this model

In the second layer, the MaxPooling2D is used with the pool size of 2X2.

The next layer is again a Conv2D layer with another 100 filters of the same filter size 3X3 and the activation function used is the ‘ReLu’. This Conv2D layer is followed by a MaxPooling3=2D layer with pool size 2X2.

In the next step, we use the Flatten() layer to flatten all the layers into a single 1D layer.

After the Flatten layer, we use the Dropout (0.5) layer to prevent the model from overfitting.

Finally, towards the end, we use the Dense layer with 50 units and the activation function as ‘ReLu’.

The last layer of our model will be another Dense Layer, with only two units and the activation function used will be the ‘Softmax’ function. The softmax function outputs a vector which will represent the probability distributions of each of the input units. Here, two input units are used. The softmax function will output a vector with two probability distribution values.

![alt text](https://github.com/mk-gurucharan/Face-Mask-Detection/blob/master/cnn_facemask.png)

After building the model, we compile the model and define the loss function and optimizer function. In this model, we use the ‘Adam’ Optimizer and the ‘Binary Cross Entropy’ as the Loss function for training purpose.

For the face detection, the Haar Feature-based Cascade Classifiers are used in this experiment. It is is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of  features proposed by Paul Viola and Michael Jones. In this, a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

The cascade classifier used for this experiment is the Face Detection Cascade Classifier. In this, a model is pre-trained with frontal facial features is developed and used in this experiment to detect the faces in real-time.

Finally, the CNN model along with the cascade classifier is trained for 30 epochs with two classes, one denoting the class of images with the face masks and the other without face masks. 

### Demo - 
![](demo.gif)
