# Pacotes padrão
import numpy as np

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Sklearn
from sklearn.metrics import log_loss

class Rede_linear_e_nao_linear(nn.Module):
        def __init__(self, camada_linear, camada_nao_linear, segunda_camada, mode=None, activation_function=None):
                '''
                Modo deve ser 'Classification' ou 'Prediction'
                Função de ativação deve ser 'tanh' ou 'relu'
                '''
                super(Rede_linear_e_nao_linear, self).__init__()
                
                self.fc1_linear = camada_linear
                self.fc1_nao_linear = camada_nao_linear
                
                self.fc2 = segunda_camada 

                if not(mode=='Classification' or mode=='Prediction'):
                    raise ValueError('Modo da rede deve ser "Classification" ou "Prediction"')
                if not(activation_function=='tanh' or activation_function=='relu'):
                    raise ValueError('Função de ativação deve ser "tanh" ou "relu"')
                
                self.mode = mode
                self.activation_function = activation_function

        def forward(self, x):
                # Linear
                # Check if there is at least a linear neuron
                if self.fc1_linear is not None:
                    x_linear = self.fc1_linear(x)
                else:
                    # Tensor with one zero
                    x_linear = torch.zeros(x.shape[0], 1).to(device)

                # Non-linear
                # Check if there is at least a non-linear neuron
                if self.fc1_nao_linear is not None:
                    if self.activation_function=='tanh': 
                        x_non_linear = F.tanh(self.fc1_nao_linear(x))
                    elif self.activation_function=='relu':
                        x_non_linear = F.relu(self.fc1_nao_linear(x))
                    x_non_linear = self.fc2(x_non_linear)
                else:
                    x_non_linear = torch.zeros(x.shape[0], 1).to(device)

                # Sum
                x = x_linear + x_non_linear

                # Sigmoid
                if self.mode=='Classification':
                    # print(F.sigmoid)
                    x = F.sigmoid(x)
                return x

def get_equivalent_linear_layer(fc1_weights, fc1_bias, fc2_weights, fc2_bias):
    w_equivalent_array = []
    b_equivalent_array = []

    # for output_idx in range(20):
    for output_idx in range(fc2_weights.shape[0]):  

        w_equivalent_array_parcial = []
        # for i in range(185):
        for i in range(fc1_weights.shape[1]):
            w_equivalent = (fc1_weights[:,i] * fc2_weights[output_idx,:]).sum().item()
            w_equivalent_array_parcial.append(w_equivalent)
        w_equivalent_array_parcial = np.array(w_equivalent_array_parcial)
        w_equivalent_array.append(w_equivalent_array_parcial)

        b_equivalent = ((fc2_weights[output_idx,:] * fc1_bias).sum() + fc2_bias[output_idx]).item()
        b_equivalent = np.array(b_equivalent)
        b_equivalent_array.append(b_equivalent)

    w_equivalent_array = np.array(w_equivalent_array).T
    b_equivalent_array = np.array(b_equivalent_array)
    
    return (w_equivalent_array, b_equivalent_array)
    
def get_if_inside_range(saidas_primeira_camada, desvio, activation_function):
        '''
        Para neurônios com ativação tanh, calcula quais os neurônios tem o range dentro de [-desvio, desvio].
        Para neurônios com ativação relu, calcula quais os neurônios tem o range dentro de [-desvio, inf].
        '''
        if activation_function=='tanh':
            max_array_primeira_camada = saidas_primeira_camada.max(axis=0)
            min_array_primeira_camada = saidas_primeira_camada.min(axis=0)
            range_array = np.append(min_array_primeira_camada.reshape(-1,1), max_array_primeira_camada.reshape(-1,1), axis=1)

            min_array = range_array[:,0]
            max_array = range_array[:,1]
            matrix_in_range = np.append((min_array >= -desvio).reshape(-1,1), (max_array <= desvio).reshape(-1,1), axis=1)
            return np.all(matrix_in_range, axis=1)
        elif activation_function=='relu':
            max_array_primeira_camada = saidas_primeira_camada.max(axis=0)
            min_array_primeira_camada = saidas_primeira_camada.min(axis=0)
            range_array = np.append(min_array_primeira_camada.reshape(-1,1), max_array_primeira_camada.reshape(-1,1), axis=1)

            min_array = range_array[:,0]
            max_array = range_array[:,1]
            matrix_in_range = np.append((min_array >= -desvio).reshape(-1,1), (max_array <= np.inf).reshape(-1,1), axis=1)
            return np.all(matrix_in_range, axis=1)

def lineariza_neuronios(modelo_base, treino_scaled, desvio, mode, activation_function):
        '''
        Lineariza os neurônios de uma rede que tem o range num dataset dentro de um desvio
        '''
        # Cópia do modelo
        modelo_base_copia = copy.deepcopy(modelo_base)
        
        # Saidas da primeira camada
        saidas_primeira_camada = modelo_base_copia.fc1(treino_scaled[0]).detach().cpu().numpy()

        # Calcula quais os neurônios tem o range dentro de [-desvio, desvio]
        neuronios_dentro_do_range = get_if_inside_range(saidas_primeira_camada, desvio, activation_function=activation_function)

        # Definição da nova rede
        # Checa se há neurônios não lineares
        if np.sum(neuronios_dentro_do_range==False):
            # Cria camada apenas com neurônios não lineares, com pesos do modelo_linearidade
            camada_nao_linear = nn.Linear(treino_scaled[0].shape[1], (neuronios_dentro_do_range==False).sum())
            camada_nao_linear.weight = nn.Parameter(modelo_base_copia.fc1.weight[(neuronios_dentro_do_range==False),:])
            camada_nao_linear.bias = nn.Parameter(modelo_base_copia.fc1.bias[(neuronios_dentro_do_range==False)])

            # Organiza segunda camada
            segunda_camada = nn.Linear(modelo_base_copia.fc1.out_features, 1)
            pesos_segunda_camada_organizados = modelo_base_copia.fc2.weight[:,(neuronios_dentro_do_range==False)]
            segunda_camada.weight = nn.Parameter(pesos_segunda_camada_organizados)
            segunda_camada.bias = nn.Parameter(modelo_base_copia.fc2.bias)
            segunda_camada.in_features=np.sum(neuronios_dentro_do_range==False)
        else:
            camada_nao_linear = None
            segunda_camada = None

        # Checa se há neurônios lineares
        if np.sum(neuronios_dentro_do_range==True):
            # Cria camada linear
            w_equivalent_array, b_equivalent_array = get_equivalent_linear_layer(modelo_base_copia.fc1.weight[neuronios_dentro_do_range,:], modelo_base_copia.fc1.bias[neuronios_dentro_do_range], modelo_base_copia.fc2.weight[:,neuronios_dentro_do_range], modelo_base_copia.fc2.bias)
            camada_linear = nn.Linear(treino_scaled[0].shape[1], w_equivalent_array.shape[1])
            camada_linear.weight = nn.Parameter(torch.tensor(w_equivalent_array.T).float())
            camada_linear.bias = nn.Parameter(torch.tensor(b_equivalent_array).float())
        else:
            camada_linear = None
        
        # Cria modelo
        rede_linear_e_nao_linear = Rede_linear_e_nao_linear(camada_linear, camada_nao_linear, segunda_camada, mode=mode, activation_function=activation_function).to(device)
        return rede_linear_e_nao_linear

def calc_erro(modelo, ground_truth_scaled, scalers, mode):
    '''
    Calculate the prediction or classification error of a model.
    '''
    if mode=='Prediction':
        y_hat_teste = modelo(ground_truth_scaled[0]).detach().cpu().numpy().flatten()
        y_hat_teste = scalers[1].inverse_transform(y_hat_teste.reshape((-1,1))).flatten()
        y_gt = ground_truth_scaled[1].detach().cpu().numpy().reshape(-1,1).flatten()
        y_gt = scalers[1].inverse_transform(y_gt.reshape((-1,1))).flatten()
        train_diff = y_hat_teste - y_gt
        mse = np.mean((train_diff)**2)
        return mse
    elif mode=='Classification':
        y_gt = ground_truth_scaled[1].detach().cpu().numpy().reshape(-1,1).flatten()
        y_hat_teste = modelo(ground_truth_scaled[0]).detach().cpu().numpy().flatten()
        loss = log_loss(y_gt, y_hat_teste)
        return loss
    else:
        raise ValueError('Modo deve ser "Classification" ou "Prediction"')

def linearize_ANN(modelo_base, treino_scaled, val_scaled, scalers, activation_function, mode, percent_erro=0.1, num_desvios=10, print_it=True):
    '''
    Linearizes the neurons of a network that has the range in a dataset within a deviation.
    Parameters:
        modelo_base (torch.nn.Module): The base model to be linearized.
        treino_scaled (torch.Tensor): Scaled training data.
        val_scaled (torch.Tensor): Scaled validation data.
        scalers (object): Scalers for data normalization.
        activation_function (callable): Activation function used in the network.
        mode (str): Mode of the network ('classification' or 'prediction').
        percent_erro (float, optional): Maximum added error percentage to accept a model. Default is 0.1.
        num_desvios (int, optional): Number of deviations to be tested. Default is 10.
        print_it (bool, optional): Whether to print the progress. Default is True.
    Returns:
        torch.nn.Module: The best linearized model.
    '''
    # Copy model
    modelo_base_copia = copy.deepcopy(modelo_base)
    
    # Initial val error
    erro_val_base = calc_erro(modelo_base_copia, val_scaled, scalers, mode=mode)
    erro_val_curr = 0
    
    # Calculate the initial deviation (minimun to exclude a neuron)
    saidas_primeira_camada = modelo_base_copia.fc1(treino_scaled[0]).detach().cpu().numpy()
    
    desvio_inicial = np.abs(saidas_primeira_camada).min()
    desvio_maximo = np.abs(saidas_primeira_camada).max()

    menor_erro_val = np.inf
    melhor_modelo = []
    for desvio in np.linspace(desvio_inicial, desvio_maximo, num_desvios):
        # Linearize
        modelo_loop = lineariza_neuronios(modelo_base_copia, treino_scaled, desvio, mode=mode, activation_function=activation_function)
        
        # Calcula erro
        erro_val_curr = calc_erro(modelo_loop, val_scaled, scalers, mode=mode)
        acc = np.mean((modelo_loop(val_scaled[0]).detach().cpu().numpy().flatten() > 0.5) == val_scaled[1].detach().cpu().numpy().flatten())
        if print_it:
            if not (modelo_loop.fc1_nao_linear is None):
                num_neuronios_lineares = modelo_base.fc1.weight.shape[0] - modelo_loop.fc1_nao_linear.weight.shape[0]
            else:
                num_neuronios_lineares = modelo_base.fc1.weight.shape[0]
            print(f'Desvio: {desvio:.2f}, Erro de validação: {erro_val_curr:.4f} - Acc: {acc} - Num neurônios lineares: {num_neuronios_lineares}')
        # Salva melhor modelo
        if (erro_val_curr < erro_val_base*(1+percent_erro)):
            menor_erro_val = erro_val_curr
            melhor_modelo = copy.deepcopy(modelo_loop)
    # print(menor_mse_val)
    return melhor_modelo

def extract_feature_importance_from_linear_non_linear_model(model, inputs):
    '''
    Extrai a importância dos neurônios lineares e não lineares de um modelo linear e não linear
    '''
    # Pesos
    if model.fc1_linear is not None:
        pesos_lineares = model.fc1_linear.weight.detach().cpu().numpy()
        importancia_lineares = np.square(pesos_lineares).mean(axis=0)
    else:
        importancia_lineares = np.zeros((len(inputs)))

    if model.fc1_nao_linear is not None:
        pesos_nao_lineares = model.fc1_nao_linear.weight.detach().cpu().numpy()
        importancia_nao_lineares = np.square(pesos_nao_lineares).mean(axis=0)
    else:
        importancia_nao_lineares = np.zeros((len(inputs)))

    return (importancia_lineares, importancia_nao_lineares)