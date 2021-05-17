import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#Basic Linear regression in randomly generated dataset

x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
print(y_numpy.shape)
X = torch.from_numpy(x_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)
print(X.shape)
n_samples, n_features = X.shape

#Defining the model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

#loss and optimizer 
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass
    Y_pred = model(X)
    loss = criterion(Y_pred, Y)
    
    #backward pass
    loss.backward()
    
    #update
    optimizer.step()
    
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#plot
predicted = model(X).detach().numpy()
plt.plot(x_numpy, y_numpy, 'bo')
plt.plot(x_numpy, predicted, 'r')
plt.show()