# Implementing Machine Learning algorithms

This is my lazy atempt to implement basic verisons of Machine Learning algorithm from scratch and without the use of any libraries. However, I may use some libraries to create data and make plots.

## Steps:

There are a few basic steps to implement machine learning algorithms:

1. Initialize weights and parameters
2. Set a learning rate
3. Start the epoch loop
4. Initialize loss and gradients to 0 (here the number of gradients should be the same as parameters)
5. Run loop for the batch
6. Accumulate the loss and accumulate the gradients
7. Update the weights and biases using the accumulated gradients
8. Print your loss and weights to realise that your algorithm works.

One of the most important steps would be to know the loss function and the gradients with respect to each parameter. I haven't mentioned them exclusively in the steps, but we need to calculate the loss function while running the loop for the batch.

## Things to keep in mind:

These are few issues which I personally faced while implementing and I thought I would share them to make life easier.

* It's better to normalize the datapoints and bring them between 0 and 1 (i.e. just the X part)
* Techinically, one should initialize the weights randomly, but setting them to 0.5 works since we've normalized the data
* Play around with the learning rate and number of epochs to find a balance in the results
* If it is the loss you're calculating, then you should minimize it by going in the opposite direction(-ve sign while updating) of the gradient. However, if it is something you're maximizing, then you should go in the direction(+ve sign while updating) of the gradient
* Drawing the decision surface is not mandatory but it is fun to visulaize it. The same can be achieved with some metrics if the decision surface is complex

**Note:** *This repo is not a display of my skillset as a Machine Learning engineer. It was purely made for fun and to understand the algorithms better. Normally, there would be more complex steps involved in an end-to-end ML pipeline and I haven't even scratched the surface in this repository.*

## Algorithms

* [Linear regression](src/linear_regression.py)
* [Logistic regression](src/logistic_regression.py)
* [svm](src/svm.py)
