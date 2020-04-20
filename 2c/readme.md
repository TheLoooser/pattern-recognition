# CNN

The code for this exercise can be found under 2c_CNN.py

In order to find a sensible number of epochs for the training of our CNN, we let it run over a long time span. Here are the plots for both accuracy as well as loss for each epoch:

![Accuracy Graph](/2c/2c_accuracy_plot.png)
![Loss Graph](/2c/2c_accuracy_plot.png)

As we can see the "Test Accuracy" - which is wrongly labeled and is in fact the validation accuracy - increases fast in the beginning but then levels out at about 99% accuracy. This was achieved with a learning rate of 0.003. Now while the accuracy flattens out for higher epochs, the loss (which as expected decreased rapidly in the first few epochs) starts to increase again at around the 18th epoch. I assume this is due to the fact that with higher accuracy the learning rate has a bigger influence if it is kept the same. As such decreasing the learning rate periodically for higher epochs might be an approach to achieve some further increase in accuracy. But in order to keep things simple, we decided to use 10 epochs for the CNN which is still a low number of epochs with a hight accuracy. 


## todo

• Optimize learning rate (typically in the range [0.001, 0.1]).
• Optimize number of training iterations. Plot a graph showing the accuracy on the training
set and the validation set, respectively, with respect to the training epochs.
• Perform the random initialization several times and choose the best network during vali-
dation.

# expected output
• Small report in PDF / README format on the GitHub containing:
    - Plot showing the accuracy and loss on the training and the validation set with respect
to the training epochs.
    - Test accuracy with the best parameters found during validation.
• Indiviual: Manual forward-pass handed in through ILIAS.
