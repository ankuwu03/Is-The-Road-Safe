# Ankur Kumar
# Roll 210152

# What the program really does:
So my program takes input an array of 3 integers (0 or 1) and it learns to output the first integer in the array
For eg in "[1,0,1]" it has to output 1
Also it plots the error in output of the program while it is learning plotted against the number of iterations
also we can print the weights at end of iterations


# Ans B:
For the dataset, since it just requires an array with 3 numbers each of which is either 0 or one,
this can be created as many times we want using for loop and random int generation function in the code itself
Or we can create a dataset using the same for loop...save it as txt file and then read that txt file to use the dataset
Although the first part is easier...



# Ans C:
I used perceptron neural network for this job...which learns by calulating the error at each iteration and then backpropagating that error to improve the output bit by bit until it can almost always get the correct answer.
Perceptron neural networks work on the thoery of linear algrbra and linear combination of inputs with some weights (coefficients) added and returns the output
so in this case if array is "[ x1 , x2 , x3 ]" then the output can be written as linear combination : 
1*x1 + 0*x2 + 0*x3 = x1 ...where 1,0,0 are the respective weights
So we can arbitrarily start with some random weights (say 0.5 each) and then train the perceptron by calculating the cost function, then backpropagating it (subtracting the gradient(C) factor) which tweaks the weights a bit and iterating this over a large dataset to finally get approx values of weights that do this

HENCE I USED PERCEPTRON NEURAL NETWORK FOR THIS JOB
