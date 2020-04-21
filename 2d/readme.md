# CNN

Here is a comparison of accuracy and loss for the permutated MNIST at the top and the normal MNIST at the bottom (both with learning rate 0.003 and batch size 16 for the training set):

![Accuracy CNN permutated](/2d/accuracy_cnn_60epochs.png) ![Accuracy CNN](/2c/2c_accuracy_plot.png)

The main difference for the accuracy is the accuracy for the validation set. For the two training sets both accuracies keep approaching 100% with higher epoch numbers, while for the validation the accuracy evens out earlier for the permutated MNIST than the regular dataset and also doesn't increase as fast as for the regular dataset. 

![Accuracy CNN permutated](/2d/loss_cnn_60epochs.png) ![Loss CNN](/2c/2c_loss_plot.png)

And with the losses the same is visible; the losses for the permutated validation set decrease slower and have more variance. The other big difference for the losses is the fact that they beginn to increase again with higher epoches for the validation set of the permutated MNIST. All in all the permutation seems to throw the CNN off, I assume because the CNN is dependent on the spacial relation of specific pixels among each other. The best achieved Accuracy we reached with this setup was a test accuracy of 96.65% with an average loss of 0.036.


# MLP

On the other hand we have the MLP model: Here we had the issue that the training process takes quite a while with our implementation, which is why we only have accuracy plots and we let the training on the permutated dataset only run for 30 epochs (again with learning rate 0.003). Here are the plots (inconsistencies in the labling explained in the readme for 2b); the top one for the permutated dataset, the bottom one for the regular:

![Accuracy MLP permutated](/2d/accuracy_mlp_30epochs.png) ![Accuracy MLP regular dataset](/2b/myplot5.png)

It is a bit difficult to see, but the test accuracy for the permutated dataset (96.87%) appears actually to be higher than the accuracy for the regular dataset which levels at around 95%. This is in fact the opposite of the results for the CNN. This is due to the fact that MLP doesn't exploit the spacial relationship between pixels which means that consequently MLP is more robust to the rearangement of pixels. 
