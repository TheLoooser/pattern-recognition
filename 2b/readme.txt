Exercise 2b - MLP

A plot showing the accuracy of our network can be found in myplot5.png (contrary to what it says on the image, it shows the accuracy
and not the error. In the final implementation, the plot now would show the error, however this picture was taken when the implementation
was not 100% finished.) As you can see we reach an accurracy of around 95% in the end (after the 60 epoches), the same also counts for the
test set (not visible in the image). And even after 60 epoches we do not get any overfitting.

Unfortunately, there is a small issue, which prevented us from further optimising the parameters (# of hidden layers, learning rate, # of
epochs) and this is also the reason, why we use this older image. The problem is that our implementation goes by the lecture too closely.
I.e. we used the MSE as criterion, which meant that we could only do batch sizes of size 1. This resulted in very long execution times. For
example, one epoch takes around three minutes to execute and therefore it took somewhere over three hours to get the accuracy plot with 60
epoches. And because it takes so long for one network to train, validate and test, it is quite cumbersome to test with multiple different
parameters.
