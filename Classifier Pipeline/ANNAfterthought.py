#---------------------------------------------
#After running prepdata..
import torch
import numpy as np

# batch_size = 128
#
# num_inputs, num_outputs, num_hiddens = train_X.shape[1], 1, 25
#
# W1 = nn.Parameter(torch.randn(
#     num_inputs, num_hiddens, requires_grad=True) * 0.01)
#
# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
#
# W2 = nn.Parameter(torch.randn(
#     num_hiddens, num_outputs, requires_grad=True) * 0.01)
#
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
#
# params = [W1, b1, W2, b2]
#
# def relu(X):
#     a = torch.zeros_like(X)
#     return torch.max(X,a)
#
# def net(X):
#     X = X.reshape((-1, num_inputs)) #maybe not needed
#     H = relu(X @ W1 + b1)
#     return (H @ W2 + b2)
#
# loss = nn.LogSigmoid()
#
# num_epochs, lr = 10, 0.1
# updater = torch.optim.SGD(params,lr=lr)

#Then the book ses some pretty lengthy fxns to do it by scratch
##basically: it does the forward pass, loss fxn, then the auto grad step to do auto diff


from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 33)

X_train_final_oversample, y_train_oversample = sm.fit_sample(train_X, train_Y)


train_py_X = torch.from_numpy(X_train_final_oversample).float()

train_py_Y = torch.from_numpy(y_train_oversample).float()

test_py_X = torch.from_numpy(valid_X).float()
test_py_Y = torch.from_numpy(valid_Y).float()


#Jal is finding 5 fucking billion things so jut pick one i guess

#Imbalanced data:
#k-fold : ensure for all k that # of 0's match # of 1's in training
#load weights from each previous k iteration




class simpleMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size_fc1, hidden_size_fc2):
        super(simpleMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size_fc1 = hidden_size_fc1
        self.hidden_size_fc2 = hidden_size_fc2

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_fc1)
        self.Bnorm1 = torch.nn.BatchNorm1d(self.hidden_size_fc1)
        self.Relu1 = torch.nn.ReLU()
        self.Dropout1 = torch.nn.Dropout(p=0.25)

        self.fc2 = torch.nn.Linear(self.hidden_size_fc1, self.hidden_size_fc2)
        self.Bnorm2 = torch.nn.BatchNorm1d(self.hidden_size_fc2)
        self.Relu2 = torch.nn.ReLU()
        self.Dropout2 = torch.nn.Dropout(p=0.25)

        self.fc3 = torch.nn.Linear(self.hidden_size_fc2,1)

        # self.fc4 = torch.nn.Linear(25, 18)
        # self.Relu4 = torch.nn.ReLU(self.fc4)
        #
        # self.fc5 = torch.nn.Linear(18,8)
        # self.Relu5 = torch.nn.ReLU(self.fc5)
        #
        # self.fc6 = torch.nn.Linear(8, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.Bnorm1(x)
        x = self.Relu1(x)
        x = self.Dropout1(x)
        x = self.fc2(x)
        x = self.Bnorm2(x)
        x = self.Relu2(x)
        x = self.Dropout2(x)
        x = self.fc3(x)
        # x = self.Relu3(x)
        # x = self.fc4(x)
        # x = self.Relu4(x)
        # x = self.fc5(x)
        # x = self.Relu5(x)
        # x = self.fc6(x)
        output = self.sigmoid(x)
        return output

model = simpleMLP(281, 100,10)
#Set up weights before the criterion
#then put as inputs to the loss fxn
#Then use reduction in the bce loss also

def modelTrainer(data_X,data_Y,model):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1000):

        optimizer.zero_grad()

        yhat = model(data_X)

        loss = criterion(yhat,data_Y.unsqueeze(1))

        loss.backward()
        optimizer.step()

modelTrainer(data_X = train_py_X,data_Y = train_py_Y,model = model)

##Need a accuracy metric and test accuracy metric

from sklearn.metrics import accuracy_score

def validAccuracy(data_X,data_Y,trainedModel):

    #Evaluate model on the test set
    y_hat = trainedModel(data_X)

    y_hat = y_hat.detach().numpy().round().squeeze()

    y_actual = data_Y.detach().numpy()

    acc = accuracy_score(y_actual, y_hat)

    return acc, np.vstack((y_actual,y_hat)).T

accuracyValue, dataDF = validAccuracy(data_X = train_py_X, data_Y= train_py_Y,trainedModel=model)

##Now test it

testAccuracy, testDF = validAccuracy(data_X=test_py_X, data_Y=test_py_Y, trainedModel=model)

# (unique, counts) = np.unique(testDF[:,1], return_counts=True)
# frequencies = np.asarray((unique, counts)).T
# print(frequencies)

from sklearn.metrics import roc_curve
# Compute ROC curve and ROC area for each class

fpr_rf, tpr_rf, _ = roc_curve(testDF[:,0], testDF[:,1])

import matplotlib.pyplot as plt

plt.figure(0)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='Neural Nut')
# plt.plot(dataset.falsePositive, dataset.truePositive, label='Elastic Net')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc=0)
plt.show()

#-----------------------------------------------------------------
#save the model smiley face

torch.save(model,f="C:/Users/jlombardi/Documents/GitLabCode/nnplay/Rtorch/testExample.pt",
           _use_new_zipfile_serialization=True)

