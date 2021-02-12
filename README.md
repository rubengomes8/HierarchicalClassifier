# HierarchicalClassifier

## Description

## Creation of 2D points


## Model/Architecture

Hierarchical classifier (comprising three classifiers) that classifies 2D points into 4 zones:
* top-left
* top-right
* down-left
* down-right

## Inference Stage

Put image here

The first classifier **(a)** predicts if a point belongs to the **top**, or to the **down** class.
The second classifier **(b)** predicts if a point classified as **top** belongs to the **top-left**, or the **top-right** class.
The third classifier **(c)** predicts if a point classified as **down** belongs to the **down-left**, or the **down-right** class.




