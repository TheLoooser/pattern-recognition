The code for the main task can be found in [main.ipynb](https://github.com/TheLoooser/pattern-recognition/blob/master/5/main.ipynb) with the methods in [methods.py](https://github.com/TheLoooser/pattern-recognition/blob/master/5/methods.py). The notebook [gridsearch.ipynb](https://github.com/TheLoooser/pattern-recognition/blob/master/5/gridsearch.ipynb) contains all code regaring the grid search and its evaluation.

**Approach**

For the computation of the edit distance between molecules we determined - as proposed in the tast description - the dirac cost matrix and then found the optimal assignment using the Hungarian Algorithm (with linear_sum_assignment from scipy.optimize). At first we had used the library munkres instead, but this doesn't work with numpy, making the process extremely slow even if other parts of the code are optimized. But by switching we were able to get the run time for one hyperparameter set down to less than 20 seconds on average. 

**Results**

We run the code for the hyperparameters Cn=[0.01, 0.1, 0.5, 1, 5, 10] and Ce=[0.01, 0.1, 0.5, 1, 5, 10]. (And based on the results later also with Cn=[0.001, 0.005, 0.01, 0.05, 0.1,] and Ce=[5, 10, 20, 40, 100].) The best accuracy we achieved with the tested hyperparameters was 0.996 (which is the case where 1 of the 250 molecules in the validation set is misclassified). This was however possible with a slew of different hyperparameters. The bigger Cn the more probable are small Ce to decrease the accuracy. For example for Cn = 0.01 Ce in the range between 0.01 and 10 still result in accuracy 0.996 while for Cn = 10 of our tested Ce values only Ce = 10 achieves still the same accuracy. To get more precise information on the optimal hyperparameters the model would have to be run with more data.
As for the nearest neighbours, for good Cn-Ce combinations it was often enough to take only the nearest neighbour. For bigger k the accuracy begins to decrease as can be seen in the following plog generated with Cn=1 and Ce=1. 

![accuracy over knn](https://github.com/TheLoooser/pattern-recognition/blob/master/5/Accuracy_knn.png)
