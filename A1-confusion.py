#%% Import library
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#%% 1. Data processing
#Data import
data = pd.read_csv('D:/HeCloud/Master of Data Science/2023 Trimester 3/Deep Learning Fundamentals/A1/data/diabetes.csv')
data.head()
X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness',
                      'Insulin','BMI','DiabetesPedigreeFunction','Age']]
Y = data['Outcome']

# Use Oversampling methods to turn datasets into balanced datasets
smote = SMOTE(random_state=42)
X, Y = smote.fit_resample(X, Y)

# Divide the training and test sets
Y = Y*2-1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1907614)


## Data transformation
# Convert dataframe form to array form
X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_test=np.array(X_test)
Y_test=np.array(Y_test)
# Convert data to PyTorch tensor 
X_train = torch.tensor(X_train, dtype=torch.float32)  
Y_train = torch.tensor(Y_train, dtype=torch.float32)  
X_test = torch.tensor(X_test, dtype=torch.float32)  
Y_test = torch.tensor(Y_test, dtype=torch.float32)
#%% 2. Create a model of an MLP perceptual machine 
class Perceptron(nn.Module):  
    def __init__(self):  
        super(Perceptron, self).__init__() 
        # Defines the fully connected network to be called in the forward function nn.Linear
        # bootstrap each layer with self, d1 is the name of the layer
        self.d1 = nn.Linear(8, 10)  #input_dim: number of neurons, set to 8 because there are 8 parameters.
        self.tanh = nn.Tanh()
        self.d2 = nn.Linear(10, 1) # Output has only one label, so it's 1
        
  
    def forward(self, x):  
        # Call self.d1 network structure block to implement forward propagation from input x to output y
        out = self.d1(x)
        out = self.tanh(out)
        out = self.d2(out)
        return out 
    
#Initail model
model = Perceptron()
print(model)
#%% 3. Configure training methods
#Create DataLoader for batch loading of data
batch_size =50
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Create a stochastic gradient descent (SGD) (stochastic gradient descent) optimiser
Optimizer = optim.SGD(model.parameters(), lr=0.01)#Select the SGD optimiser and set the learning rate to 0.01
#Create a loss function (MSE mean square error)
criterion = nn.MSELoss() 
#%%# 4.Training model
num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_Y in train_loader:
        Optimizer.zero_grad()# Clear the gradient 
        model_outputs = model.forward(batch_X) # Forward propagation
        loss = criterion(model_outputs, batch_Y.view(-1, 1))# Calculated loss 
        loss.backward() # Backpropagation, calculate the gradient 
        Optimizer.step()# Update the weights
    # Print loss values for each training cycle   
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
#%% 5.Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model.forward(X_test)
    predictions = torch.sign(predictions)
    accuracy = (predictions == Y_test.view(-1, 1)).float().mean()
    print("Accuracy:", accuracy.item())

confusion=confusion_matrix(predictions, Y_test)
# Visual confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
            yticklabels=["Predicted Negative", "Predicted Positive"],
            xticklabels=["Actual Negative", "Actual Positive"])
plt.ylabel('Predicted Lable')
plt.xlabel('Actual Lable')
plt.title('Confsion Matrix')
plt.show()


    
    
    
    