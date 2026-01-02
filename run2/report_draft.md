In this part, the given images will be trained and classified based on the Bag of Visual Words (BoVW) method. Firstly, the visual vocabulary is learned through K-Means Clustering, and images are represented as visual word frequency histograms. Then, use one-vs-all Logistic Regression as the classifier to finish the classification.

Firstly, all of the input images will be pre-processed into greyscale images in order to reduce the dimension and complexity of computation due to colors. Label each image and convert it into a Numpy array.

Secondly, build vocabulary for the training set, extract 8 * 8 dense patches for every 4 pixels in the x and y directions. Merge all patches and process mean centring and L2 normalisation for every patch. After that, use K-Means Clustering to learn a number of visual words.

Thirdly, during the classifier training process, convert the training images into BoVW histograms using the current vocabulary. Standardise every feature. Construct a binary label for every category and train a list of Logistic Regression classifiers.

Fourthly, validate the performance of trained classifiers using a validation set. Collect the prediction result from classifiers and calculate the prediction accuracy for the validation set.

Finally, train the final model using the entire training dataset and output the prediction result for the testing set.

For evaluation, a 5-fold cross-validation strategy has been used. For every fold (iteration), the training set has been split into a training and a validation set with 8an 2 ratio; the testing set did not participate in the training. As a result, the mean prediction accuracy for the model is 0.6653 (std 0.0223)