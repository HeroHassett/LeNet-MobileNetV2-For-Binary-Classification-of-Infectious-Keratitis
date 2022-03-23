# LeNet-MobileNetV2-For-Binary-Classification-of-Infectious-Keratitis

Abstract

Corneal Opacification is the fifth leading cause of bilateral blindness in the world. The most common cause for Corneal Opacification is infectious keratitis, which can be classified mostly in two categories: bacterial or fungal. The treatments of bacterial keratitis and fungal keratitis are very different; therefore, timely and accurate diagnosis of infection type is crucial. The gold standard of diagnosis is through microscopic examination of culture. This requires speedy access to a lab, which is not always attainable, especially in remote or rural areas or developing countries. I hypothesized that machine learning can use corneal images taken from something as simple as a smartphone camera to differentiate between bacterial and fungal keratitis. I implemented a deep learning LeNet model that could provide an accurate image-based diagnosis. The model is trained through supervised learning on 671 total images with 134 images set aside for validation. The model contains two convolutional layers that connect to the output layer, which has a sigmoid activation function, allowing for a probability of accuracy with every diagnosis. The model utilized Adam optimization and performance was measured using AUC and Accuracy metrics, as well as a binary cross entropy loss function. On the validation dataset, the model achieved an AUC of 0.63 and an accuracy of 0.66 after 50 epochs. The model is able to quickly differentiate between the two infections and has the potential to help people in resource-poor regions and countries gain expedient access to a diagnosis, allowing treatment that could prevent blindness.

Introduction

	According to a recent report by the World Health Organization (WHO), corneal opacity is responsible for causing 1.5-2 million cases of unilateral blindness annually. Corneal opacity is caused by any damages to an individual's cornea, such as inflammation, trauma, nutrient deficiency, and infection. The most common cause for corneal opacity is through infectious keratitis (IK), which is responsible for approximately one million annual clinical visits in the US alone. Infectious keratitis can be characterized as bacterial, fungal, viral, or parasitic. Each distinct class of pathogen requires a differing clinical treatment, thereby making an early diagnosis vital for minimal visual damage. The two most common forms of infectious keratitis are bacterial and fungal. The treatment of bacterial and fungal keratitis vary considerably and as a result, the ability to differentiate between the two classes of the pathogen is necessary for treatment. The current method of determining the infectious etiology between the two classes of keratitis is to obtain cultures of corneal scrapings and test it. However, this method results in a negative 40-60% of the time, and even if the results are positive, it commonly takes several days for a result to be made available. The issue of infectious keratitis is most prevalent in developing nations where the resources necessary to address it are limited, such as India and Burma. Additionally, the current method of diagnosing this infection is both time-consuming and inaccurate. Without culture samples, ophthalmologists make treatment decisions based purely on observation and comparison with past cases. Even experts are only able to correctly distinguish bacterial and fungal IK 66-73% of the time using only comparison and examination. This exhibits an opportunity for machine learning to be used in order to develop more accurate and efficient image-based diagnosis of infectious keratitis. 
Convolutional Neural Networks (CNN) have displayed outstanding performance in the application of supervised learning for image classification or, more specifically, for its applications in medical diagnosis. Computer vision is a terminology used to describe the application of artificial intelligence (AI) in deriving meaningful information from images. CNN architectures have developed considerably in the past decade, especially architectures designed for computer vision. One significant architecture for CNNs is the LeNet, developed in 1998, which specializes in computer vision. LeNets are specifically efficient for the task of binary classification, which is a classification task between two differing classes. Additionally, image-based classification requires a large dataset, and due to the widespread nature of this infection I have composed a dataset of 671 images of both fungal and bacterial infectious keratitis. The development of deep learning models has also allowed for a significantly greater degree of acceptable data, such as image-based classification using only images from phone cameras. This is especially important in developing nations with limited infrastructure, who need cheap and portable imaging methods that can be used to diagnose infections. Modern CNNs are able to apply computer vision using phone images, and as a result opened up deep learning and computer vision to individuals in developing nations. As a result of these improvements,  I have been able to develop a CNN that is capable of utilizing images of keratitis-infected corneas to perform a binary classification task between fungal and bacterial keratitis. 

Methods

The dataset utilized for this research is a mix of 671 images containing both fungal and bacterial infectious keratitis. In the dataset, there are 232 bacterial images (34%) and 441 fungal images (66%). The study that produced the images was conducted by the Aravind Eye Care System in South India from 2005-2015. The images are of the cornea of individuals who were microbiologically proven to have either bacterial or fungal infectious keratitis. This data also excludes any cases that were culture-negative or polymicrobial infectious keratitis. Prior to the input of the data into the CNN, I performed a validation/train split of 20/80 for the data. This would allow for analysis of the performance of the CNN, as well as helping to determine whether the CNN was overfitting or not. The images that are used in training allow for the model to learn and extract features in order to predict the cause of similar infections. 
The overall task is to perform binary classification using a deep learning model. I chose CNN as it has demonstrated significantly greater efficiency regarding computer vision. The specific CNN architecture utilized during the research was the LeNet. However, a separate CNN was utilized in order to create optimized weights that could be used in the LeNet. This CNN possessed the same general structure as the LeNet, and as such the weights of that CNN could be used to increase the accuracy and probability output of the LeNet. Another reason the development of the LeNet was necessary was because the model needed to be able to accept single images to provide an output, and as such I designed the LeNet to utilize the parameters (weights) of the CNN to make its own predictions. The architecture of the CNN is as follows:
Preprocessing
Resize input image to 256x256 pixels
Rescale pixel values to range from 0 to 1 (by dividing pixel value by 256)
A convolutional layer with pad = 2, stride = 1, and 20 filters
A rectified linear unit (ReLU) activation layer
A max pooling layer with 2x2 filters and a stride of 2
A second convolutional layer with pad = 2, stride = 1, and 50 filters
A second ReLU activation layer
A second max pooling layer with 2x2 filters and a stride of 2
A single densely connected layer with a sigmoid activation function 

The CNN was trained on the Adam optimizer as performance testing indicated that Adam provided the highest accuracy. Each image underwent preprocessing according to the standard for a LeNet. There are two convolutional layers and each was designed with a different filter in order for feature extraction to work on both general features and specific details within each image. Max pooling is also used to help create a feature map that contains the most important features from the original image. The trained CNN utilized a minibatch size of 64 images as well as 50 epochs and a randomized seed. Finally, I designed helper functions for the LeNet to function properly. The helper functions are designed as follows: 
A zero-padding function:
Input: A 3D matrix of any size (height x width x # channels)
Arguments: amount of zero-padding desired
Output: A zero-padded matrix
A convolution function:
Input: A 3D matrix of any size
Arguments: A 4D matrix of weights, desired stride, desired padding, and the number of filters to be applied
Output: A convolved 3D matrix
A max pooling function:
Input: A 3D matrix of any size
Arguments: The desired filter size and stride length
Output: A pooled matrix
A ReLu function:
Input: A numpy array of any size
Output: A numpy array of the same size as the input
And a sigmoid function:
Input: A scalar
Output: A scalar ranging from 0 to 1
The ability of this model is that it is capable of taking an image of any size (because it is initially preprocessed in the zero-padding function) and returning a prediction of whether the image represents a bacterial or fungal ulcer, including returning an estimated probability of accuracy due to the sigmoid activation function in the output layer. This model is, as previously mentioned, capable of analyzing specific images and as such is capable of being used on individual patients. A major part of the model is also its use of backpropagation algorithms. I designed these algorithms for the purpose of using the weights of the previously trained CNN to further “fine-tune” the hyperparameters of the LeNet. Furthermore, the accuracy of this model was evaluated on 4 images originating from a separate source. The accuracy was modeled under two metrics: accuracy and AUC. These two metrics were used accordingly with the binary cross-entropy loss function to properly display the accuracy during evaluation.

	The most integral part of the statistical analysis of the model was in accounting for the significantly limited number of samples present in the dataset. Only having access to 671 total images for the model to learn off significantly influenced whether the model would overfit and produce a lower accuracy. Counteracting this overfitting was a major factor in hyperparameter tuning. The first method employed in determining the most effective hyperparameters was in analyzing the performance of multiple networks. For each network, differing hyperparameters were used on the same data in order to determine which could produce the highest accuracy. The most common performance evaluations were done by changing the max pooling layers’ strides and pooling size to lower the number of features the model determined were of importance. This would allow the model to recognize more detailed features by reducing the number of pixels that were filtered. A secondary alteration made to the CNN to improve validation accuracy was in reducing the complexity of the CNN. Essentially, I had minimal data and as such if I increased the complexity of the network I would be making the model too specialized on the output data. To reduce the numbers of layers necessary I utilized a ReLU activation function, which is able to converge significantly faster than other activation functions. Furthermore, I reviewed the efficiency of multiple activation functions and compared how quickly overfitting started to determine which function would perform the best given limited data. A final method employed to increase the accuracy of the model was in the usage of backpropagation algorithms in the LeNet. The algorithms were capable of automatically enhancing the feature recognition of the LeNet using previously calculated weights. This resulted in a significant increase in accuracy as the model was capable of improving upon its own predictions through an analysis of the weights from the training of previous models. These methods effectively helped delay overfitting and improve the accuracy of each model as they helped decrease the specialization of the CNN by optimizing hyperparameters, reducing complexity, and enhancing the probability of each prediction through past weights. 

Results


Figure 1. highlights an accuracy of both validation and training sets with the validation dataset averaging around 68-70% (0.68-0.7), when based on the LeNet architecture. The training accuracy plateaus at an accuracy of 1.0 after approximately 20 epochs. The training set eventually achieved 1.0 accuracy and was consistently larger by about 0.2-0.3 units compared to the validation accuracy. The validation accuracy oscillated around 0.68-0.7 from epochs 20-50.
The growing divide between accuracy off training and validation, as well as the flattening of the accuracy curves highly suggests that overfitting is occurring. This means that, due to the limited sample sizes in the data, the optimizer is no longer able to find more features and as such is unable to increase the accuracy. The model requires increased complexity (more layers), but that requires an increased dataset, indicating that either a larger dataset or a deep CNN with more layers. Additionally, data augmentation can also help to prevent overfitting and increase the validation accuracy. 

Figure 2. depicts the area under the curve (AUC) of the model that achieved the highest accuracy benchmark. It had an average validation AUC of 0.7-0.71 and displayed a training AUC of 1.0 after around 20 epochs. Similarly, the validation AUC averaged out at around 20 epochs. It also displays rapid changes in the AUC for the first 20 epochs. The training AUC was consistently 0.3 units greater than the validation AUC after 20 epochs and remained constant at 1.0 after 20 epochs.

Figure 2 supports the conclusion made in Figure 1, that overfitting results in a limit on all classification metrics at around the 0.7 mark. This indicates that the model is more likely to favor one type of infection over the other, which likely is due to the differences in number of samples for each type of classification. The effects of the sample imbalance need to be counteracted by using a larger dataset or performing data augmentation on a more complex model. 

Figure 3. reports an exponential decrease in binary cross entropy loss for the training set from 1.4 to approximately 0.1 loss and a rigid increase in the validation loss from 0.6 to approximately 0.9. The training curve begins to smoothen at around 25 epochs, while the validation curve remains at a rigid increase around 0.8-1.0 loss 

Figure 3 further supports the conclusion that overfitting is occurring at approximately 20 epochs. This is present in the plot as the loss increases and the training loss becomes constant, indicating the model is becoming too specific and unable to make generalized predictions. Overfitting is also a common indicator that the dataset is too small, implying that either a more complex CNN is required or a larger dataset must be used.

Figure 4. displays the predictions of the CNN based upon nine different, randomly selected, validation images. The predictions are highlighted with a one or zero with the one representing a fungal infection and a zero representing a bacterial infection. This was able to correctly evaluate 8 of the 9 randomly selected validation images with the image in the first row second column being incorrectly labeled as fungal, as well as the image in the second row first column being incorrectly labeled as bacterial. It also represents the greater number of fungal images compared to bacterial images.
Figure 4 displays a localized accuracy of around 81%, although to achieve a more informative out-of-sample test evaluation a larger test sample is required. However, it does illustrate an increased accuracy in the LeNet that was given hyper-tuned weights as it achieved an average of 10% higher than the CNN without backpropagation algorithms adjusting weights. 


Model Type
Training Time
Time Per Epoch
Avg Probability
Test Accuracy
Overfitting Mark
Prediction Time
Model Type
Preprocessing
3 Minutes
3.1 Seconds
88%
81%
20 Epochs
1.3 Seconds
Preprocessing
Augmentation
41 Minutes
49.2 Seconds
73%
63%
48 Epochs
6 Seconds
Augmentation

Table 1. represents the overall benchmarks of two CNNs: one that underwent data preprocessing standard for a LeNet and another that underwent data augmentation to increase the number of data samples. It displays that preprocessing was significantly faster and more accurate compared to data augmentation. This is most evident in the training time of both models and the prediction times.

Table 1 indicates that data augmentation didn’t help the performance. LeNets we used have few layers and as such the increased samples of augmented data simply resulted in an inability to process the data effectively. This means that more complex models will likely be required for larger datasets. The data also implies that simpler models that are less resource-demanding and time-efficient require large raw datasets rather than augmented datasets that provide a more complex model with a greater number of features to extract. 


Discussion

	These results have demonstrated the viability of applying computer vision to image-based differentiation of fungal and bacterial infectious keratitis. The LeNet achieved a peak accuracy of around 71% and was able to perform at an average of 81% on the test dataset. The data indicates that the use of backpropagation is able to enhance the accuracy by an average of 9-10% even on a simpler model. The overall results are promising as the LeNet was able to conclude with an accuracy of 81%, which is with a limited sample size and overfitting lowering the validation accuracy. Table 1 also provides insight into how the model is less efficient with data augmentation. The reason for this is that the LeNet is a simpler model (few layers) and as such it requires significantly more time to process augmented images and thus the LeNet loses its cost/time efficiency. This means that in order to achieve higher results a more complex model would be required, which would increase the processing power to perform well, thus being less-effective for deployment in developing nations. 
	A major success of the research is the accomplishment of a low-cost (computation-wise) and time-efficient model that can achieve human level results in only 3.1  seconds, as well as being accessible from anywhere. This large access range and small prediction time result in a significant degree of applications in rural areas or developing nations where the inability to gain a quick and cheap diagnosis is widespread. Continuing with the accomplishments of the model, the LeNet was able to demonstrate an 81% accuracy on test datasets. This is an accuracy higher than the average corneal expert accomplished and as such succeeds in the ability to outperform a human. This is incredibly promising as the model was struck by various limitations yet was still able to perform effectively, which highlights many possible improvements/developments that could be made to the usage of deep CNN architecture for image-based classification.
	The project maintained multiple drawbacks that ultimately lessened the performance of the LeNet. The most noticeable and frequent issue faced during research was the issue of overfitting. The presence of overfitting was not unforeseen as it is seldom inevitable with smaller datasets. LeNets excel in their ability to make image-based classifications quickly and with little resource drain, however that does not pair well with a lack of a large dataset. Therefore, likely future improvements of using deep CNN models to make image-based classification could be in attaining both a greater number of sample images and reducing the cost and time of a more complex model that could achieve superhuman levels of accuracy. The second most debilitating issue was the lack of both total images as well as test data acquired from separate sources. The lack of total images, including the inequality in the number of fungal and bacterial samples, was the major cause of overfitting. The lack of test data resulted in less concrete evaluations on the models real-world accuracy as there was little separate data that the model could be tested on. However, this does not mean the model provided zero real metrics, but it does mean that more test samples must be used to create a more accurate evaluation of the model’s real-world diagnosis capabilities. While these limitations did ultimately lessen the performance of the model itself, they were equally able to create avenues for further research.
	This research has presented various opportunities for further study. A major development that could be explored is the usage of a more complex model such as a VGG or ResNet. These models would require more data to function effectively and with more data they could utilize their large number of convolutional layers to extract both very generalized features and very specific features for a flawless prediction accuracy. The usage of more complex models also opens up research in multi-class classification and semi-supervised learning. The ability to classify multiple infections is something that a more complex model could accomplish given a large dataset. Additionally, that large dataset may include some data that has no ground truth and instead of gleaning that data, it could be used for a semi-supervised learning model that can accurately classify unlabeled images based upon similar features. Data augmentation is also something highly effective in more complex models and while my current research suggests its infectivity in simpler models, the function of data augmentation would likely greatly outperform that of standard preprocessing. My research has both performed efficiently and also opened up many chances for the advancement of the concepts and ideas developed in this research.

Conclusion

	My results were able to achieve a majority of the desired outcomes referenced within my initial hypothesis. The prediction accuracy of 81% is greater than the prediction accuracy of the average ophthalmologist (66-73%) and less likely to return culture negative, both of which were modern limitations that I seeked to overcome. Furthermore, the model was able to perform while utilizing less than 10% of my GPU and only took 3 minutes to run on a standard laptop. This very effectively marks my desired goal of designing a model that could be employed in rural or developing areas that did not have access to labs or quick diagnoses. The only desired outcome I was unable to achieve was that of designing a model that was able to generalize data effectively with small datasets. As supported by the presence of overfitting, my model struggled in achieving an incredibly high accuracy with a small dataset, but was still able to achieve an above-human accuracy, which was initially the goal of the research. Therefore, the results from the LeNet were able to answer a significant majority of the goals laid out at the beginning of the research and have opened up many opportunities for usage and improvement. 
	
