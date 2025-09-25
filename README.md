---
title: 'Assignment 04: Linear Regression, Learning Curves and Regularization'
author: 'CSci 574: Machine Learning'
date: ''
---

# Description

In this exercise you will be using what you have learned about linear regression, polynomial regression and
regularization, to explore an artificial dataset.

I have generated a secret dataset.  The dataset uses a polynomial combination of a single parameter.
The unknown function is no less than degree 2, but no more than a degree 20 polynomial.  Also some random
noise has also been added into the function, so that fitting it is not a completely trivial or obvious
exercise.  Since the dataset is generated from a polynomial function, the output labels `y` are
real valued numbers.  And thus you will be performing a regression fitting task in this assignment.

Your task, should you choose to accept it, is to load and explore the data from this function.  Your ultimate
goal is to try your best to determine the degree of the polynomial used, and the values of the parameters
then used in the secret function.  Because of the noise added to the data you are given, you will not be able
to exactly recover the parameters used to generate the artificial data.  You will even find that determining the
exact degree of the generating polynomial function is not possible.  How you apply polynomial fitting and 
regularization techniques can give different and better or worse approximations of the true underlying function.

In the below cells, I give instructions for the tasks you should attempt.  You will need to load the data and
visualize it to begin with.  Then you will be asked to apply polynomial fitting and regularization in an attempt
to fit the data.  But ultimately, at the end, you should keep in mind that there is a true function of some
unknown polynomial degree with some coefficient settings used.  You will later be able to see what the
true function was and compare your model performance to the true function you are exploring in this
exercise.


**Instructions**:

- You need to use the class development environment and make sure that you are pushing your assignments to your
  GitHub classroom and they are successfully passing the autograder.
- Avoid using for-loops and while loops, unless you are explicitly told to do so in the assignment task functions for this assignment.
- Do not modify the `### TESTED FUNCTION [function name]` cells.  These cells call unit tests on the functions
  you are asked to write for these assignments.
- All functions you need to write should be placed into the `src/assignment_tasks.py` file.  Functions that are
  tested and graded are imported from there into this notebook.
  - You have not been given function documentation and stub functions for this and future assignment.  You need to
    add in the function declarations and uncomment the code to import and test your functions.
  - You are required to give NumPy style formatted Pydoc documentation for your functions.
- After coding your function, run the `### TESTED FUNCTION` cell to determine if it is passing the assignment
  unit tests and that your result is correct.

# Objectives

**After this assignment you will**:

- Better understand Linear/Polynomial Regression using standard RMSE cost
- Get experience with what underfit, overfit and good fitting/generalizing
  models look like.
- Understand basics of using linear curves to tune underfit/overfit models
  correctly.
- Learn and use regularization to tune overfit models
  - Use both l-2 Ridge regularization and l-1 Lasso regularization
  - Get a feeling for the difference between these and where one or the other
    might be useful.
- Practice using `scikit-learn` pipelines to create more complex model workflows.
- Experince using grid search in `scikit-learn` to explore a parameter space
  such as tuning `alpha` parameters for regularization.

# Overview and Setup

# Assignment Tasks

# Assignment Submission

# Additional Information

