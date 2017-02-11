## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, created a deep neural network architecture that uses convolutional layers to classify traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model was trained, I have also tried out its classification accuracy on images found freely on the web and extracted from public domain youtube videos of varying quality.

The Project Goals
---
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

The Project Deliverables
---
To meet specifications, the following have been generated:
* the Ipython notebook with the code - [Traffic_Sign_Classifier.ipynb](https://github.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
* the code exported as an html file - [report.html](https://github.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html) or [report.pdf](https://github.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/blob/master/report.pdf)
* a writeup report either as a markdown or pdf file  - [writeup.md](https://github.com/mukilk7/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md)

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
