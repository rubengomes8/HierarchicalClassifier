# HierarchicalClassifier

## Description

Instead of having one classifier that is trained among four classe, we divide the task into 2 decision levels. 
This results in 3 classifiers that are trained to classifiy a more specific task.

## Dataset Generator - Creation of 2D points

To create the dataset 2D points, we used a Gaussian Distribution for each class:

Class | Mean (x, y) | Covariance Matrix ((c11, c12),(c21, c22))
------------ | ------------- | ----------- 
**1: top-left** | (-6.93, 4.97) | ((3, 3), (4, 4))
**2: top-right** | (4.09, 4.06) | ((4, 4), (3.47, 3.65))
**3: down-right** | (5.42, -5.55) | ((3.06, 3.16), (3.96, 4.28))
**4: down-left** | (-5.44, -4.75) | ((3.45, 3.60), (3.58, 3.55))

<img src="/Pictures/dataset.png" width="420">

## Training Stage

Classifier | Training Set
---------- | ------------
(a) | every point generated (classes **top-left**, **top-right**, **down-left**, **down-right**)
(b) | every top-point (classes **top-left**, **top-right**)
(c) | every down-point (classes **down-left**, **down-right**)
## Inference Stage

<img src="/Pictures/inference_stage.png" width="720">


### Level 1 of Decision
* The first classifier **(a)** predicts if a point belongs to the **top**, or to the **down** class.

### Level 2 of Decision
* The second classifier **(b)** predicts if a point classified as **top** belongs to the **top-left**, or the **top-right** class.
* The third classifier **(c)** predicts if a point classified as **down** belongs to the **down-left**, or the **down-right** class.

# Results

### Classifier (a)

With the first classifier, we predicted the class of every point with *x* belonging to *(-12, 10)* and *y* belonging to *(-12, 12)*. The next image shows the predicted regions - background color - (**top** and **down**), as well as the training points' groundtruth to verify if the pedicted decision regions make sense.


![Classifier (a)](/Pictures/classification_a.png)

### Classifier (b)

This classifier predicted the class of every point classified as **top** by the classifier (a). The next image shows the predicted regions and the training points' groundtruth.

![Classifier (b)](/Pictures/classification_b.png)


### Classifier (c)

This classifier predicted the class of every point classified as **down** by the classifier (a). The next image shows the predicted regions and the training points' groundtruth.

![Classifier (c)](/Pictures/classification_c.png)

### Aggregated results

Combining these images, we get the 4 predicted zones by the **hierarchical classifier**, which make sense according to the training data.

![Classifier (c)](/Pictures/aggregated_results.png)





