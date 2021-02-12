# HierarchicalClassifier

## Description

## Dataset Generator - Creation of 2D points

To create the dataset 2D points, we used a Gaussian Distribution for each class:

Class | Mean (x, y) | Covariance Matrix ((c11, c12),(c21, c22))
------------ | ------------- | ----------- 
**1: top-left** | (-6.93, 4.97) | ((3, 3), (4, 4))
**2: top-right** | (4.09, 4.06) | ((4, 4), (3.47, 3.65))
**3: down-right** | (5.42, -5.55) | ((3.06, 3.16), (3.96, 4.28))
**4: down-left** | (-5.44, -4.75) | ((3.45, 3.60), (3.58, 3.55))

![Dataset](/Pictures/dataset.png)

## Training Stage

## Inference Stage

![Inference Stage](/Pictures/inference_stage.png)

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





