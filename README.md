# HierarchicalClassifier

## Description

## Dataset Generator - Creation of 2D points

To create the dataset 2D points, we used a Gaussian Distribution for each class:
* **top-left:**
* **top-right:**
* **down-left:**
* **down-right:**

![Dataset](/Pictures/dataset.png)

## Model/Architecture

Hierarchical classifier (comprising three classifiers) that classifies 2D points into 4 zones:
* top-left
* top-right
* down-left
* down-right

## Inference Stage

![Inference Stage](/Pictures/inference_stage.png)

The first classifier **(a)** predicts if a point belongs to the **top**, or to the **down** class.
The second classifier **(b)** predicts if a point classified as **top** belongs to the **top-left**, or the **top-right** class.
The third classifier **(c)** predicts if a point classified as **down** belongs to the **down-left**, or the **down-right** class.




