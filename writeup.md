#**Traffic Sign Recognition Writeup ** 

## - Mukil Kesavan

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images found over the web
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/master/writeup-images/img1.png "Random Training Images"
[image2]: https://raw.githubusercontent.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/master/writeup-images/img2.png "Training Set Histogram"
[image3]: https://raw.githubusercontent.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/master/writeup-images/img3.png "Testing Set Histogram"
[image4]: https://raw.githubusercontent.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/master/writeup-images/img4.png "Grayscale Transformation"
[image5]: https://raw.githubusercontent.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/master/writeup-images/img5.png "New Images From Web"
[image6]: https://raw.githubusercontent.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/master/writeup-images/img6.png "Classified New Images"
[image7]: https://raw.githubusercontent.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/master/writeup-images/img7.png "Softmax Probabilities for classified New Images"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the cell 2 (following Step 1) in the IPython notebook.

I used the numpy shape attribute to calculate summary statistics of the traffic signs data set:

* Number of training examples = 39209
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in cell 22 (follows right after cell 2) of the IPython notebook.  

*Here are 5 random images from the training set along with their corresponding labels:*

![alt text][image1]

*Histogram of number of training set images of each sign type:*

![alt text][image2]

![alt text][image3]

Overall, it can been seen that some sign types are not well represented in the training set compared to others. For example, there is a 10x difference in the number of images for sign-type-0 (~200 samples) vs. sign-type-1 (~2000 samples). Later we will see how this impacts some of the results.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in code cell 23 (right under Step 2) of the IPython notebook. I first convert the training and testing set images from RGB to grayscale (the library implementation I use applies a weighted sum for the RGB channels). I do this so that the neural network relies primarily on detecting shapes and contrast patterns for detecting traffic signs, instead of color. Colors tend to have different shades in different lighting conditions and I figured that this might be a source of noisy input to the neural network. Also, I did try normalization (mean shifting) of the images after grayscale transformation, but it did not improve my accuracy beyond any reasonable error margin. So, I've left that out of my pre-processing phase.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in code cell 5 of the IPython notebook right under the corresponding headline calling this out.

 The pickled data provided with the project template has training and testing data. However, we still need a validation set to compute the loss function that will be used to adjust weights over multiple training epochs of the neural network. I split off 5% of images from the training set into the validation set for this purpose using scikit-learn and shuffled them so that any inherent ordering of images does not affect learning.

The final tally of images in each of my sets were:

* Training Set = 37,249
* Validation Set = 1,960
* Testing Set = 12,630

I have not chosen to augment any existing data with new samples. In hindsight, this would have perhaps been prudent, especially for sign types with low training samples. I will explain the reasoning behind this later in the writeup.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in code cell 7 of the IPython notebook. I have used the LeNet-5 architecture, presented in class for the handwritten character recognition task, as the basis for classifying traffic signs with some modifications. In addition to changing the input layer and output layer shapes to fit the traffic sign dataset, I've also added dropout to convolutional and fully connected layers so as to prevent overfitting the training set. I used less aggressive droput (0.25) at the convolutional layers early on in the network to make sure that a good amount of input information is passed along the network and used more aggressive dropout (0.5) in the fully connected layers to prevent overfitting. This gave me a good 3-4% improvement in accuracy over the testing set.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| DROPOUT					|  keep_prob 0.75								|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution	    | 1x1 stride, VALID padding, outputs 10x10x16      									|
| RELU					|												|
| DROPOUT					|  keep_prob 0.75								|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten     | outputs 400
| Fully connected		| Output 120      									|
| RELU					|												|
| DROPOUT					|  keep_prob 0.5								|
| Fully connected		| Output 84      									|
| RELU					|												|
| DROPOUT					|  keep_prob 0.5								|
| Fully connected (logits)				| Output 43        									|
|	Softmax					| 												|
|						|												|

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in code cells 8 and 9 of the ipython notebook. I used a batch size of 128 images, learning rate of 0.001 (with the more sophisticated adam optimizer) and 25 epochs to train the neural network. I experimented with different values and found that this combination gave me the best tradeoff between system memory usage and preventing overtraining (or overfitting to the training set well past any significant improvements to validation set accuracy).

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in cell 10 of the Ipython notebook.

My final model results were:

* Validation set accuracy = 97.9 %
* Test set accuracy = 92.9 %

My overall approach to finding a solution was an iterative approach that involved a combination of trial and error and the application of techniques from Yann LeCun's paper - Traffic Sign Recognition with Multi-Scale Convolutional Networks. I discuss the specifics below:

* What was the first architecture that was tried and why was it chosen? What were some problems with the initial architecture?

I first started with the LeNet5 architecture and parameter values from the handwritten character recognition lesson. Right out of the bat I got 94% accuracy on the validation set and 87% accuracy on the testing set. 

*Input noise reduction* - I then realized that reducing the noise in the input may lead to the neural net learning only the essentials required to classify the sign. I applied the grayscale transformation to the input images and this resulted in a validation set accuracy of 97% but the testing set accuracy was still well below at 89%.

*Subpar Validation Set* - One problem that I realized after reading LeCun's paper above was that my validation set selection was not ideal. The GTSRB dataset images are derived from frames of 1 second video clips. Therefore there are multiple frames of the same image under the same environmental conditions in the testing set. In the selection of the validation set, a random shuffling and sampling of images from the training set will end up including multiple images from the same 1 second video. This does not help with the diversity required to properly validate learning. Unfortunately, the pickled training set data does not include information on which 1 second video an image was extracted so that I could choose 1 image from each video into the training set. Therefore, there is an inherent limit to how much accuracy we can achieve given the starter project assets.

* How was the architecture adjusted and why was it adjusted? 
* Which parameters were tuned? How were they adjusted and why?

*Preventing Overfitting and Underfitting* - The primary modifications I did was to introduce the idea of dropout and set the number of training epochs appropriately. I use two distinct dropout values for the convolutional and fully connected layers. For the convolutional layer, I use a dropout probability of 0.25 (or equivalently, keep probability of 0.75) so that most of the input and low dimensional features are passed through to the network to learn. If I had used a higher dropout here, I would have risked underfitting the training data. For the fully connected layers, I used a higher dropout probability of 0.5 in order to prevent overfitting and aid in the network learning alternate representations of higher dimensional features for the same traffic sign. After a lot of trial and error, this approach worked well for me. Also, I played around with the epoch value. I noticed that if I set the epoch value very high it was causing my network to overfit the training dataset. The increase in accuracy on the validation set for higher numbered iterations wasn't that much but the impact the resulting model had on the testing set accuracy was a pretty significant drop. For example, with 70 epochs I was able to get close to 99% accuracy on the validation set but the accuracy on the testing set dropped down to 90%. Therefore, I set the number of epochs to 25 which achieved about 98% accuracy on the validation set without risking overfitting the training data. I also had to use a batch size of 128 to keep the memory usage on my machine low.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

This question has been answered above on model selection and parameter tuning. As to why a convolution layer works well for this problem, it is because of translation invariance - as covered in the lesson. Features that uniquely describe a traffic sign may appear in any part of the input image and our network will be able to successfully classify more often than not.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I was able to achieve a validation accuracy of ~98% and testing set accuracy of ~93% at the end. I will discuss some ways I could have improved my solution after discussing the model results on new images I found on the web.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web along with their sign-types above (manually classified by me):

![alt text][image5]

 I have made sure that at least some of the images have an angle to them (translation - e.g. i6, i8 belonging to classes 16, 1 respectively) and have non-optimal lighting (rainy and foggy conditions in i1 and i5 belonging to classes 7, 18 respectively). I've also manually classified these images and recorded them in new-signs/classes.csv.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in code cell 16 of the Ipython notebook. Optional part is discussed in final section.

Here are the results of the prediction - the classes output by the model are on top of each image. Please compare this with the classes I manually classified above.

![alt text][image6]

Overall, the model was able to classify 7 out of 8 signs correctly achieving an accuracy of 87.5% on the new images. This is reasonably close to the testing set accuracy of ~93%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook and the softmax probabilities output code is in Cell 19. Here are the softmax probabiliites for the top 3 predictions made by the model (I have not included 5 because the last two had very minuscule probabilities in each case and as such does not contribute to any discussion).

![alt text][image7]

* *Image 1* presents a 100 kmph speed limit sign in heavy fog where the letters appear pretty washed out. Correspondingly the model was only 55% certain that it was indeed a 100 kmph speed limit was 25% certain that it could be 80 kmph limit. This is a fairly understandable conclusion. But the model was robust enough to pick the correct value in the end.

* *Image 4* presents a children crossing sign. I noticed in the dataset that the imagery in the middle of the sign seemed to vary slightly between different signs. For example, some imagery included stick figures whereas the new sign I found on the web seem to include a stick figure with a bag. Also, the distance of separation between stick figures varied as well between signs. This might explain why the model was only 73% confident that this was a children crossing sign. The next best prediction was actually a bicycles crossing sign with 11% confidence which is pretty close to the correct one. Nonetheless, the model chose the correct value in the end.

* *Image 6* was classified wrongly by the model. The sign was a "trucks over 3.5 metric tons prohibited sign" and it was classified as "Roundabout Mandatory" with a 92% confidence! This model with a slightly different orientation looks pretty clear with ideal lighting conditions and initially I couldn't figure out why it the model classifying this wrongly. However, looking at the training dataset histograms, the number of input images for the "trucks prohibited" sign may not have been enough for the model to learn its unique characteristics properly. This suggests that data augmentation might have helped some. Notice that the second best prediction was indeed the correct class "trucks prohibited" albeit with only a confidence of ~5%.

###4. Discussion and Scope for Improvement

* *Data Augmentation* - Clearly creating more images from the existing ones by transforming them (e.g. change pixel intensity, translating pixels etc.) would have helped with properly classifying underrepresented sign types.

* *Neural Network Architecture from LeCun's paper* - The architecture from the paper where the input is branched out and fed directly to a fully connected layer, so that low level features are also part of the final classification process would have improved the accuracy as well.

* *Regenerating Training, Validation and Testing sets* - As mentioned previously, the validation set choice was not ideal in terms of diversity. I could have downloaded the original dataset from the source website and re-created all the sets used to train, validate and test the model.

* *Incorporating Color Information* - Some of the errors on the test set or new images may have been avoided if I had included the color information as part of the input training set. The right way to encode color information while minimizing input noise could also be another source of improving classification accuracy.

The options for tuning and improving the accuracy were plenty but I stopped in order to focus on the next project after I understood properly the performance of the model and factors that affected its accuracy.
