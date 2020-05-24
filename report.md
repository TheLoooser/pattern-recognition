# Pattern Recognition Report

## Organization

Organizing group work was a little unusual during the Covid-19 pandemic since our group members were not allowed to see each other in person which is usually done at least once when working in groups. However, we managed to organize our group tasks using chat applications fairly well, and for discussions too large discord came in handy.  
Originally, the first exercises we had to hand in were Exercise 2a and 2b (before their deadline was merged with the deadline of Exercises 2c and 2d). We discussed if half of the group should work on 2a while the other half works on 2b, but we gave up on that idea. We decided that as a team we should be collectively responsible for our group exercises and didn’t need to assign different responsibilities to each group member. We believed that each group member could contribute to the exercises by looking at what needs to be done and then work on one of the missing tasks. We maintained this “laissez-faire-philosophy” for the whole semester, i.e. we also didn’t assign personal responsibilities to our team members for exercises 2c, 2d, 3 and 5.
After each exercise we shortly discussed what we should do better for the next exercise (especially if we have to start working earlier on the next exercise). Furthermore, we had to handle that some of our team members were being prevented from contributing to the exercises due to different circumstances. One member was summoned to civil service during the semester and was therefore unable to contribute to exercise 3. Another team member had trouble following the course due to his background being in Management and not Computer Science and therefore could not contribute to the coding exercises. Those circumstances caused the remaining members of the team to get under some pressure, especially during exercise 3.  
The less technical part of the documentation was written by the member that had trouble following the course, while the technical part was written by several different group members.

## Tasks
### Task 2a-d: SVM, CNN, MLP
#### Specialities
**2b**  
One thing which probably sets our solution apart from the others is that we oriented ourselves too closely to the lecture. I.e. we implemented the MSE criterion, which influenced our performance significantly.  
  
There is not much special about the other parts for task 2.

#### Approach
**2a**   
The idea was to use a library for SVM and Cross-validation and then try different techniques to search for the best parameters for two different kernels: RBF and linear. There was only one parameter to optimize for the implementation with the linear kernel, meaning that multiple algorithms could be tried to find the optimal C. Despite those possibilities, there are a lot of local optimums for C, making it difficult to find the real best parameter.
For the RBF kernel, there were two parameters to optimize. To find the best - or at least a good - pair of parameters, a grid search seemed to be the best way. It allowed us to find a pair of parameters giving more than 98% accuracy on the test set. The biggest issue here was certainly the computation time, which already needs a few hours for 25 different pairs of parameters in RBF.

**2b**   
We implemented the MLP as follows. We started by creating the network with the different layers. Then the task is basically to go over each image in the training set for a certain amount of epochs in order to train the network by backpropagating the loss and tuning the weights. Once the network was sufficiently trained and not yet overfitted, we looped once over the validation set and tested the accuracy of our network (simply by checking whether the network predicted the number displayed on the image correctly).

**2c**  
For this task we completed the CNN code provided in the task, which makes use of the library Torch. The rest of the implementation was reused and adapted code from task 2b. The main difference in adaption to 2b was that we were actually able to use batches. This allowed for a better runtime and as such some experimenting with the hyperparameters.

**2d**  
There was not very much to do for this exercise except copying code from 2b and 2c and adapting it to the new task. This could have gone much better with a proper framework.

#### Issues & Successes
One drawback of our implementations was certainly that we did not build a simple framework. With some helperfiles and some basic functionality most tasks could have been implemented faster. Especially overhead like reading in files and preprocessing datasets. This led to a lot of loose code and unstructured files.  
Furthermore we did not use jupyter notebooks for task 2b-d which would have helped with time consuming tasks done in memory.  
Regarding run time not everyone could afford to let the code run for many epochs as not everyone had a CUDA capable graphics card and with a cpu at 100 percent for hours not much work could be done in the meantime.  
Luckily some were able to use CUDA and we reached good accuracies, especially for task 2c. However due to time constraints we weren't able to run a proper grid search to optimize the hyperparameters for this task, instead using the approch of "let's plug in some numbers and see what happens".

----
### Task 3: Dynamic Time Warping (DTW) for keyword detection
#### Approach
First, we prepared the images. I.e. we binarized the whole image and then cut out the polygons with the words on it before normalising the image to a size of 200 times 200 pixels. This was as it turned out later a bad idea as dtw is handling stretching of images by itself. Therefore by normalizing along the width information which could have helped improve accuracy was lost.  
The normalization along the height axis was not necessary either, as this would not help much with performance but also lose possibly helpful data.  
Regarding dtw itself it was clear that we would use a library. We benchmarked 4 different libraries and took the fastest for our project.  
Evaluation for different neighbouring images was done as it is with knn as well, by iteration over the different image neighbourhood sizes k and finding the one with the best accuracy.

#### Specialities
Due to a very bad mistake (normalized images) we made it into the exercise lesson slides under the category "worst mistakes". There is no such thing as bad publicity :)

#### Issues & Successes
At first we chose a very bad feature vector, meaning we misunderstood the sliding window idea and applied dtw to all 200x200 pixels. This of course performed very badly even though we managed to find a library which computes efficiently in C. It took us some experimentation to realise that good feature vectors are based on columns of images and don't directly use pixels but metrics such as upper/lower contour or black/white pixel proportions.  
Secondly we had a hard time understanding what exactly should be done with the different data sets given. Not because we did not understand at all, but we misinterpreted the task more than once. This resulted in our deadline getting close and we could not implement many different feature vectors.  
On a positive note all the experimentation in the end allowed us to understand dtw properly and what pitfalls might be. The result was very bad, but the experience gained and lessons learned certainly valuable.

---
### Task 5: Graph Matching (molecules)
#### Specialities
We just used cost 1 for both node and edge insertion/deletions, which worked surprisingly well for us.

#### Approach
Our approach to this task is quite straight forward. We started by creating a distance matrix between every molecule of the validation set and every molecule of the training set. The distance is simply the approximated graph editing distance for which we used a bipartite graph matching and the hungarian algorithm to calculate the distance. Then the task was a basic KNN classification, i.e. for each validation molecule find the K most similar molecules in the train set and check whether they are active or inactive and based on the majority assign the validation molecule to either class. We then optimized the performance of our code by only using numpy operations, allowing us to run a grid search over all three hyperparameters.

#### Issues & Successes
We only had one issue with this task, namely we had a disagreement on how we should implement the KNN algorithm, but once we sorted that out (with the help of our lovely assistant) we managed to classify the molecules quite successfully. We reached an accuracy of 99.6% extremely fast without having to optimize the parameters very much. We however still implemented a grid search but couldn't find any hyperparameters with better accuracy. But in the process we were able to get the computation time for one set of hyperparameters down to less than 20 seconds on average.

---
## What worked? What did not work?

As already mentioned, our philosophy consisted in not assigning personal responsibilities to different team members. This strategy somehow always worked out in the end in the sense that we were able to hand in solutions and pass all group exercises. But at the beginning we had some problems with our time management. We got into serious time pressure when we had to hand in Exercises 2a-d, meaning that several team members worked several hours before the deadline in order to finish in time. This also meant that we were only able to hand in minimal solutions since we had to set priorities and could not develop more elaborate solutions. Our time management then got better after each exercise because (among other reasons) we appealed to each other that we don’t want to make the same error and do the work for the following exercise a few days/hours before the deadline. For Exercise 3 we still worked on the day we had to hand in our solutions, but it was less stressful. For Exercise 5 we even managed to more or less finish the exercise two days before the deadline. The last exercise was significantly shorter than the previous ones, which was probably also a reason for finishing so much earlier than the previous exercises.
The inherent problem of a “laissez-faire” organization strategy is that in larger groups, members might assume that someone else would do the work. But if everyone assumes that, nobody will end up doing the work. Generally speaking, the success of our way to organize group work depends highly on the different personalities of the group members. We noticed that some group members started relatively early to work on the exercises, while some waited longer and started working a few days before the exercises. Our organization strategy would obviously work very well if all team members started working early on the exercises and very badly if all group members work best when they are under time pressure. But in the end we always managed to get the work done.

## Improvements for next time
One thing that certainly will be improved for the next group project is that everyone should be on the same page regarding used tools. A short FAQ might help. In our group we had members who used jupyter notebooks and some plain python files, which made working together more difficult. Even more so as the ones using python files did not understand jupyter notebooks yet.  
The second thing regarding tools is usage of github. When working together on github everyone should know the basics functions and workflow. For our projects to come we will ensure that those issues are resolved before starting with the exercises.  
Last but not least is performance. In the beginning of each task we underestimated how extremely time consuming even small classifications can become and opted for basic loops and dictionaries. With time we learned that numpy and vectorized functions can save lots of time. At least if no libraries are available. It sure does make sense to get an understanding with basic implementations, but next time we will do so with performance in mind from the beginning.

## General Thoughts & Feedback
One of the best things for the exercises was the closeness to real life applications. We never had to implement tedious algorithms which would perform much worse than any library. This would not have allowed us to work with real data sets. Instead we could use libraries and apply their functionalities to different tasks and get hand on experience with pattern recognition and machine learning.  
This also increased the motivation and fun factor. Let's be honest, it's much cooler to tell someone you wrote code to make your computer read poorly written words in letters from George Washington rather than bragging about how you implemented some basic function with a performance nearly half as fast as any other public implementation.  
It is also worth mentioning that working in groups was very nice. At the university it is not exactly uncommon to work in teams, but this is often just for some projects and students lack the experience on how teams can efficiently work together. This exercise series was at least for some team members a step forward regarding team management and work sharing.  
In conclusion, great exercises!
