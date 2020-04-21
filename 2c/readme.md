# CNN

The code for this exercise can be found under 2c_CNN.py

In order to find a sensible number of epochs for the training of our CNN, we let it run over a somewhat long time span (60 epochs). We used a learning rate of 0.003 and a batch size of 16 for the training set. Here are the plots for both accuracy as well as loss over the different epochs:
![Accuracy Graph](/2c/2c_accuracy_plot.png) ![Loss Graph](/2c/2c_loss_plot.png)

As we can see the accuracy increases fast in the first couple of epochs but then begins to levels out. But while after about 10 epochs the validation accuracy stays more or less at 98% the training accuracy keeps slowly increasing and approaching 100%. The same with the loss: While both training and validation loss decrease rapidly in the beginning, the validation loss levels out while the training loss keeps decreasing. This is why validation is important; if we let it train long enough the neural network will be able to perfectly predict the training data but may fail with the actual data it is supposed to classify. As such, in order to optimize the training time necessary while still getting good accuracy, we looked at the level out of the validation accuracy curve and chose 10 epochs for further training. 

As far as learning rates are concerned, it seems that CNNs with higher learnig rates seem to be more unstable and vary more in accuracy, though this was less structured testing and more of "let's plug in some numbers and see what happens". Here is an example with learning rate 0.03:
![Accuracy Graph](/2c/2c_accuracy_lr03.png) ![Loss Graph](/2c/2c_loss_lr03.png)

The best accuracy we achieved over 10 epochs, with the learning rate 0.003 and batch size 16 is a test accuracy of 98.61% with an average loss of 0.0144.
