# Traffic Sign Recognition

In this project, traffic signs are being classified using convoluted neural network

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the [data set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./saved_images_results/DataVisualization.png "Visualization"
[image2]: ./saved_images_results/NormalisedImage.png "Normalized Image"
[image3]: ./test_images/road_work.jpg
[image4]: ./test_images/slippery_road.jpg
[image5]: ./test_images/stop.png
[image6]: ./test_images/speed_limit_120.jpg
[image7]: ././test_images/speed_limit60.jpg

Here is a link to my [project code](https://github.com/vijaysingla/CarNd-TrafficClassifierProject/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

** Visualization of the dataset**

Here is an exploratory visualization of the data set. It is a bar chart showing the no. of training images corresponding to traffic sign class. The total no. of traffic sign classes are 43

![alt text][image1]

### Design and Test a Model Architecture

#### Image Preprocessing

Image data was normalized so that the data has zero mean and equal variance. For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and was used in this project.This is helpful in context of choosing hyperparameters. Normalising leads to more stablility during training and less oscillations . Here is view of traffic sign image after normalising

![alt text][image2]


Grayscale preprocessing method was also tried on the model but it did not increase the valid_accuracy. Fake data was also tried using image rotation but it did not yield better validation_acuration.
 

#### Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Dropout               | Dropout prob =0.75                            |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU           	    | 2x2 stride,  outputs 10x10x16                 |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Dropout               | Dropout prob =0.75                            |
| Fully Connected    	| Size =400        							    |
| Layer1 Neural Network | Output =120        							|
| Layer2 Neural Network | Output =84        							|
| Layer3 Neural Network | Output =43        							|


#### Model Training

I used Adam optimizer algorithm to train the model. I used hperparameters (mu,sigma)=(0,0.1) , Batch Size=10, Epoch= 15, 
and initial learning rate as 0.001 . 

#### Model Training and Testing results

My final model results were:

* validation set accuracy of 93.7
* test set accuracy of 93.7

LeNet-5 architecture was used to model the network.It is a convolutional neural network designed to recognize visual patterns directly from pixel images .This can recognize patterns with extreme variability (such as handwritten characters), and with robustness to distortions and simple geometric transformations.   It was choosen as it is convolution network and it reduces the number of parameters to be trained. 

After trying  Lenet Architecture , my model validation accuracy was coming to be around 87 %. To increase the accuracy, I tuned batch size, and used dropout method. Decresing batch size and using drop_out increased the valid_accuracy to around 94 %. With increment in epochs, the validation accuracy was oscillating a lot around 93 %. So, I decayed learning rate exponentially with epoch to stabilize the validation accuracy. 

### Test a Model on New Images

I found five German traffic signs on web and tried my network on those signs.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

The first image is difficult to predict due to presence of watermark and background object. The second image does not have any background object but still has watermark that might cause issues in prediction . The third image i.e stop sign is of low contrast that might cause some issues in prediction.The fourth image is easy to predict. The fifth image although very clear and easy to detect, still has some characters written at the bottom that seems to be from another language , might cause some issues.

####  Model's predictions on these new traffic signs 
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road_work      		| Wild animals crossing   					    | 
| Slippery_road     	| Slippery_road 								|
| Stop  				| Stop                                          |
| Speed_limit_120	   	| Speed_limit_120					 			|
| Speed_limit_60		| Speed_limit_60      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.7 % 

#### Model Predictions using top 5 softmax probabilities for each image. 

For the first image, the model is relatively sure that this is a wild animal crossing (probability of 0.705), but the image  contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .705         			| Wild animals crossing   					    | 
| .076     				| Double Curve									|
| .052					| Slippey road                                  |
| .043	      			| Right-of-way at the next intersection         |					 				
| .041				    | Dangerous Curve to the right                  |     							


For the second image , the model is absolutely sure that this is a slippery road sign (probability of 0.993), and the image does contain a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .993         			| Slippery Road          					    | 
| .6e-3     			| Dangerous Curve to the right                  |									
| .3e-4					| Children Crossing                             |
| .4e-5	      			| Pedestrians                                   |					 				
| .6e-6				    | Road narrows on the right                     | 


For the third image , the model is relatively sure that this is a stop sign (probability of 0.572), and the image  contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .572        			| Stop                   					    | 
| .348     			    | Road work                                     |									
| .062					| Speed_limit 30(Km/h)                          |
| .0049     			| Speed_limit 60(Km/h)                          |					 				
| .0046				    | Priority Road                                 | 


For the fourth image , the model is absolutely sure that this is a speed_limit 120 km/h  sign (probability of 0.997), and the image does contain a speed_limit 120 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .997         			| Speed_limit 120(Km/h)                         | 
| .2e-2     			| Speed_limit 100(Km/h)                         |									
| .8e-4					| Speed_limit 80(Km/h)                          |
| .5e-4	      			| Speed_limit 50(Km/h)                          |					 				
| .1e-4				    | Speed_limit 60(Km/h)                          | 


For the fifth image , the model is absolutely sure that this is a speed_limit 60 km/h  sign (probability of 0.997), and the image does contain a speed_limit 60 km/h sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Speed_limit 60(Km/h)                          | 
| .1e-4     			| Speed_limit 50(Km/h)                          |									
| .8e-7					| Speed_limit 80(Km/h)                          |
| .2e-10	      		| Speed_limit 30(Km/h)                          |					 				
| .3e-16				| Speed_limit 20(Km/h)                          | 
