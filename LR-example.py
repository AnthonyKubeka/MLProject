import numpy as np
import matplotlib.pyplot as plt

data_file = open('data.txt','r') # open the file containing the data
data_string = data_file.read() # read in the contents of the file as a string
data_string_split = data_string.split(',')[:-1] # split the string by the commas (drop the last index after the split as this is an empty space)
data = np.array(data_string_split) # Turn the data_string_split list into a numpy array
data = data.astype(np.float64) # Turn the string values of numbers in the array into floating point numbers
data = data.reshape(60,2) # reshape the array so that column 0 contains x values and column 1 contains y values
x_values = data[:,0] # create array of x values from column 0 of data
y = data[:,1] # create array of y values from column 1 of data

plt.axhline(0,color='black') # plot horizontal axis at 0
plt.axvline(0,color='black') # plot vertical axis at 0
plt.scatter(x_values,y, color='blue') # create a scatter plot of data points
plt.xlabel('x-values') # label horizontal axis
plt.ylabel('y-values') # label vertical axis
plt.show() # show the plot

def h(x, θ): # Regression function
    return 1/(1+np.exp(-np.dot(x, θ))) # logistic regression using the dot product

x_values = x_values[np.newaxis].T # turn our array of x values into a column vector
X = np.hstack([np.ones(x_values.shape[0])[np.newaxis].T, x_values]) # create the design matrix
α = 1e-4 # define our learning rate
θ = np.ones(X.shape[1]) # initialize our parameters
θ_old = np.zeros(X.shape[1]) # initialize the old parameter values (must be different from the parameter values so we enter the while loop below)
while np.sqrt(np.sum(np.power(θ - θ_old, 2))) > 0.0005: # while euclidean norm > 0.0005 (so ϵ = 0.0005) 
    θ_old = θ # set old parameter values to parameter values before they are updated
    for i in range(X.shape[0]): # loop over each row of the design matrix (each data point)
        θ = θ - α*( (h(X[i], θ) - y[i]) * X[i] ) # update the parameters using the update rule
print("Model Parameters: ", θ) # Print model parameters after convergence

model_predictions = h(X, θ) # for each data point obtain the predicted y value
category1 = np.where(model_predictions < 0.5) # data point with a predicted y value are in class 1
category2 = np.where(model_predictions >= 0.5) # data point with a predicted y value are in class 2

range_of_x = np.arange(-3,15,0.1)[np.newaxis].T # Create an array of x values from -3 to 15 so we can plot our logistic regression
range_of_x_design_matrix = np.hstack([np.ones(range_of_x.shape), range_of_x]) # Create a design matrix so we can plot the logistic regression
logistic_function = h(range_of_x_design_matrix, θ) # for each x in (-3,15) obtain a y-value
dividing_line = np.where(np.abs(logistic_function-0.5) <= 0.01) # find x value where logistic regression = 0.5 (indifferent about the classes, separating line between classes)
plt.axvline(range_of_x[dividing_line[0]],color='purple',label="dividing line between classes") # Draw the line separating the classes
plt.axhline(0,color='black') # plot horizontal axis at 0
plt.axvline(0,color='black') # plot vertical axis at 0
plt.scatter(x_values[category1],y[category1], color='blue') # create a scatter plot of the actual data points in class 1, coloured in blue
plt.scatter(x_values[category2],y[category2], color='red') # create a scatter plot of the actual data points in class 2, coloured in red
plt.plot(range_of_x,logistic_function, color='green', label="logistic regression") # plot the regression function's predicted value for each x value in (-3,15)
plt.legend(loc="right") # Add the legend to the plot
plt.show() # show the plot