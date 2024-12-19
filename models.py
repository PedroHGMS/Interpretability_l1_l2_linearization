import numpy as np
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

#sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def one_epoch_L1_L2(model, dataloader, optimizer, criterion, L1_weight, L2_weight, device):
        # Inicialização de variáveis úteis
        train_loss = 0
        
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # move inputs and labels to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss += L1_weight*torch.sum(torch.abs(model.fc1.weight))
            loss += L2_weight*torch.sum(torch.square(model.fc2.weight))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        return train_loss

def eval_L1_L2(model, dataloader, criterion, L1_weight, L2_weight, device):
    total_loss = 0
    with torch.no_grad():
            for i, data in enumerate(dataloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # move inputs and labels to the GPU
                inputs, labels = inputs.to(device), labels.to(device)
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss += L1_weight*torch.sum(torch.abs(model.fc1.weight))
                loss += L2_weight*torch.sum(torch.square(model.fc2.weight))
                total_loss += loss.item()
    return total_loss

class Net_L1_L2_class(nn.Module):
    # Rede que implementa regularização L1 e L2
    def __init__(self, hidden_layer_size, input_size, output_size, activation_function='relu'):
        super(Net_L1_L2_class, self).__init__()

        self.activation_function = activation_function

        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, output_size)        

    def forward(self, x):
        if self.activation_function=='relu':
            x = F.relu(self.fc1(x))
        elif self.activation_function=='tanh':
            x = F.tanh(self.fc1(x))
        else:
            raise ValueError('Função de ativação não reconhecida')
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

def NN_L1_L2_training_class(dataloaders, dataset_train, dataset_val, dataset_test, scalers, 
                hidden_layer_size=300, learning_rate=0.0001, L1_weight = 0.0, L2_weight=0.0, activation_function='relu',
                num_epochs_save = 1, num_epochs = 10, 
                print_tqdm=True, print_results=True, calc_test_loss=False):
    # Args retrieval
    train_dataloader, val_dataloader, test_dataloader = dataloaders
    x_train_neural, y_train_neural = dataset_train
    x_val_neural, y_val_neural = dataset_val
    x_test_neural, y_test_neural = dataset_test
    scaler_x, scaler_y = scalers

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if print_results:
        print(f'Using device: {device}')

    # Model instantiation
    input_size = int(dataset_train[0].shape[1])
    output_size = int(dataset_train[1].shape[1])
    model = Net_L1_L2_class(hidden_layer_size=int(hidden_layer_size), input_size=input_size, output_size=output_size, activation_function=activation_function).to(device)

    # Define loss function
    criterion = nn.BCELoss()
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
      
    # Useful variables
    train_loss_array = []
    val_loss_array = []
    test_loss_array = []
    epochs_plot = []
    menor_mse_val = np.inf

    # Check tqdm
    if (print_tqdm==True):
        loop_epoch = tqdm(range(num_epochs))
    else:
        loop_epoch = range(num_epochs)
    for epoch in loop_epoch:
        train_loss = 0
        val_loss = 0

        train_loss = one_epoch_L1_L2(model, train_dataloader, optimizer, criterion=criterion, L1_weight=L1_weight, L2_weight=L2_weight, device=device)

        if epoch % num_epochs_save == 0:
            # Save the epoch
            epochs_plot.append(epoch)
            
            # Save train loss in array
            train_loss_array.append(train_loss / len(train_dataloader))  
                    
            # Evaluate val loss
            val_loss = eval_L1_L2(model=model, dataloader=val_dataloader, criterion=criterion, L1_weight=L1_weight, L2_weight=L2_weight, device=device)
            
            # Save val loss in array
            val_loss_array.append(val_loss / len(val_dataloader))

            # Test loss
            if calc_test_loss:
                # Evaluate test loss
                test_loss = eval_L1_L2(model=model, dataloader=test_dataloader, criterion=criterion, L1_weight=L1_weight, L2_weight=L2_weight, device=device)
                # Save test loss in array
                test_loss_array.append(test_loss / len(test_dataloader))

            # Store best model based on best val error
            if val_loss < menor_mse_val:
                melhor_epoch = epoch
                menor_mse_val = val_loss
                # deepcopy best model
                melhor_modelo = copy.deepcopy(model)
    last_model = copy.deepcopy(model)
    return melhor_modelo, melhor_epoch, last_model, epochs_plot, train_loss_array, val_loss_array, test_loss_array