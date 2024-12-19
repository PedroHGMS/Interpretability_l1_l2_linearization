import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# PyTorch
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_array_of_dat_TEC_Chiang(files):
    '''
    Load an array of dat files from the Tennessee Eastman Process Simulation as stored in the Chiang dataset.
    Also, it adds the name of each variable and a column with the True IDV, containing 0 for normal operation and 1 for the fault.
    Train dataset is all fault, while the test dataset starts the fault at t=160. The d00.dat and d00_te.dat files are always normal operation.
    Parameters:
    files: list of Path objects
        List of files to be loaded

    Returns:
    loaded_data: list of pd.DataFrame
        List of dataframes with the loaded
    '''
    # Load data
    # The d00.dat file should be transposed
    loaded_data = [pd.read_csv(f, sep='\s+', header=None).T for f in files if f.name == 'd00.dat']
    # Load the other files
    loaded_data += [pd.read_csv(f, sep='\s+', header=None) for f in files if f.name != 'd00.dat']

    # Define the header with the names of the variables
    header = [f'XMEAS({i})' for i in range(1,41+1)] + [f'XMV({i})' for i in range(1,11+1)]

    # Attribute the header to the dataframes and add the True IDV column
    for i in range(len(loaded_data)):
        # Add header
        loaded_data[i].columns = header

        # Add True IDV
        num_rows = loaded_data[i].shape[0]
        file_name = files[i].name
        # If it is the d00.dat or d00_te.dat files, all rows are normal operation
        if ('d00' in file_name):
            fault_array = np.zeros(num_rows)
        else:
            # If it is a training file, all rows are fault
            if 'te' not in file_name:
                fault_array = np.ones(num_rows)
            # If it is a test file, only the rows after t=160 are fault
            elif 'te' in file_name:
                fault_array = np.zeros(num_rows)
                fault_array[160:] = 1
        loaded_data[i]['True IDV'] = fault_array

    return loaded_data

def load_dataset_chiang(fault, path=Path('Data/Chiang/')):
    '''
    Load dataset from Chiang's data

    Parameters:
    path (Path): Path to the data
    fault (int): Fault number
    '''

    # Get all dat files names
    dat_files = list(path.glob('*.dat'))

    # Training data
    if f'd{fault:02d}.dat' not in [f.name for f in dat_files]:
        raise ValueError(f'Fault {fault:02d} not found in the dataset')
    train_files = [f for f in dat_files if ((f.name == 'd00.dat') or ((f.name == 'd00_te.dat')) or (f.name == f'd{fault:02d}.dat'))]
    train_data = load_array_of_dat_TEC_Chiang(train_files)
    # Concat pd dfs
    train_data = pd.concat(train_data, axis=0).reset_index(drop=True)

    # Test data
    test_files = [f for f in dat_files if ((f.name == f'd{fault:02d}_te.dat'))]
    test_data = load_array_of_dat_TEC_Chiang(test_files)
    # Concat pd dfs
    if len(test_files) > 0:
        test_data = pd.concat(test_data, axis=0).reset_index(drop=True)

    return train_data, test_data

def load_dataset_Chiang_pca(fault, path=Path('Data/Chiang/')):
    '''
    Load dataset from Chiang's data

    Parameters:
    path (Path): Path to the data
    fault (int): Fault number
    '''

    # Get all dat files names
    dat_files = list(path.glob('*.dat'))

    # Training data
    if f'd{fault:02d}.dat' not in [f.name for f in dat_files]:
        raise ValueError(f'Fault {fault:02d} not found in the dataset')
    train_files = [f for f in dat_files if ((f.name == 'd00.dat'))]
    train_data = load_array_of_dat_TEC_Chiang(train_files)
    # Concat pd dfs
    train_data = pd.concat(train_data, axis=0).reset_index(drop=True)

    # Validation data
    val_files = [f for f in dat_files if ((f.name == 'd00_te.dat'))]
    val_data = load_array_of_dat_TEC_Chiang(val_files)
    # Concat pd dfs
    val_data = pd.concat(val_data, axis=0).reset_index(drop=True)

    # Test data
    test_files = [f for f in dat_files if ((f.name == f'd{fault:02d}_te.dat'))]
    test_data = load_array_of_dat_TEC_Chiang(test_files)
    # Concat pd dfs
    if len(test_files) > 0:
        test_data = pd.concat(test_data, axis=0).reset_index(drop=True)

    return train_data, val_data, test_data

def get_x_y(dataframe, target, feature_set):
    """
    Splits the dataset into x and y according to the target and feature_set.
    Parameters:
    dataframe (pd.DataFrame): The dataset containing the data.
    target (str): The name of the target variable.
    feature_set (list): A list of variables to be included in the output dataset.
    Returns:
    tuple: A tuple containing:
        - x (pd.DataFrame): The input dataset containing the features.
        - y (pd.DataFrame): The output dataset containing the target variable.
    """
    
    dataframe = dataframe.copy()
    # Separação dos dados de treinamento e teste
    x = dataframe[feature_set]
    y = dataframe[target]
    
    return (x, y)

def get_dataloader(dataset_train_df, dataset_test_df, outputs, inputs, val_split, scale_output, dataset_val_df=None, batch_size='num_samples', shuffle_train_val=False):
    """
    Splits the training and testing datasets into dataloaders for use in PyTorch, also separating into training and validation sets.
    Parameters:
        dataset_train_df (pd.DataFrame): DataFrame containing the training data.
        dataset_test_df (pd.DataFrame): DataFrame containing the testing data.
        outputs (list): List of output variable names.
        inputs (list): List of input variable names.
        val_split (float): Proportion of training data to be used for validation.
        scale_output (bool): Whether to scale the output variables.
        dataset_val_df (pd.DataFrame, optional): DataFrame containing the validation data. Defaults to None.
        batch_size (int or str, optional): Batch size. Defaults to 'num_samples'.
        shuffle_train_val (bool, optional): Whether to shuffle the training and validation data. Defaults to False.
    Returns:
    tuple: 
        - (train_dataloader, val_dataloader, test_dataloader): Dataloaders for training, validation, and testing.
        - (x_train_neural, y_train_neural): Training data.
        - (x_val_neural, y_val_neural): Validation data.
        - (x_test_neural, y_test_neural): Testing data.
        - (scaler_x, scaler_y): Scalers for data normalization.
    """
    
    # Define inputs como as variáveis restantes, caso ele seja None
    if inputs==None:
        inputs = dataset_train_df.columns.values[~np.in1d(dataset_train_df.columns.values, outputs)]
    num_outputs = len(outputs)

    # Separa dataframe em x e y para facilitar uso no sklearn
    x_train_neural, y_train_neural = get_x_y(dataset_train_df, outputs, inputs)
    x_test_neural, y_test_neural = get_x_y(dataset_test_df, outputs, inputs)
    
    # Separa treino em treino e validação
    if dataset_val_df is None:
        x_train_neural, x_val_neural, y_train_neural, y_val_neural = train_test_split(x_train_neural, y_train_neural, test_size=val_split, shuffle=shuffle_train_val)
    else:
        x_val_neural, y_val_neural = get_x_y(dataset_val_df, outputs, inputs)
    # Normaliza dados
    # X
    scaler_x = StandardScaler().fit(x_train_neural)
    x_train_neural = scaler_x.transform(x_train_neural)
    x_val_neural = scaler_x.transform(x_val_neural)
    x_test_neural = scaler_x.transform(x_test_neural)
    # Y
    if scale_output:
        scaler_y = StandardScaler().fit(y_train_neural.values.reshape(-1,num_outputs))
        y_train_neural = scaler_y.transform(y_train_neural.values.reshape(-1,num_outputs))
        y_val_neural = scaler_y.transform(y_val_neural.values.reshape(-1,num_outputs))
        y_test_neural = scaler_y.transform(y_test_neural.values.reshape(-1,num_outputs))
    else:
        scaler_y = None
        y_train_neural = y_train_neural.values.reshape(-1,num_outputs)
        y_val_neural = y_val_neural.values.reshape(-1,num_outputs)
        y_test_neural = y_test_neural.values.reshape(-1,num_outputs)
    
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Transforma em tensores e move para a GPU
    x_train_neural = torch.from_numpy(x_train_neural).float().to(device)
    y_train_neural = torch.from_numpy(y_train_neural).float().reshape(-1,num_outputs).to(device)
    x_val_neural = torch.from_numpy(x_val_neural).float().to(device)
    y_val_neural = torch.from_numpy(y_val_neural).float().reshape(-1,num_outputs).to(device)
    x_test_neural = torch.from_numpy(x_test_neural).float().to(device)
    y_test_neural = torch.from_numpy(y_test_neural).float().reshape(-1,num_outputs).to(device)

    # Número de dimensões
    N = x_train_neural.shape[1]
    
    # Batch size
    if batch_size == 'num_samples':
        batch_size = x_train_neural.shape[0]
    else:
        batch_size = batch_size

    # Define dataset
    
    train_dataset = TensorDataset(x_train_neural, y_train_neural)
    val_dataset = TensorDataset(x_val_neural, y_val_neural)
    test_dataset = TensorDataset(x_test_neural, y_test_neural)

    # Define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return (train_dataloader, val_dataloader, test_dataloader), (x_train_neural, y_train_neural), (x_val_neural, y_val_neural), (x_test_neural, y_test_neural), (scaler_x, scaler_y)


def get_dataset_prediction_per_batch(model, dataset, batch_size=2**10):
    '''
    Get the prediction of the dataset per batch
    '''
    # Separate dataset in dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    outputs = []
    for i, batch in enumerate(dataloader):
        inputs = batch
        outputs.append(model(inputs).cpu().detach().numpy())
    return np.concatenate(outputs).reshape(-1,)

def plot_feature_importance(feature_importance, inputs, title='Feature Importance', save_path=None, plot=True):
    '''
    Plot the feature importance as a sorted histogram and the cumulative sum.

    feature_importance: np.array with the feature importance, don't need to be sorted
    inputs: list with the inputs names
    '''

    # Percentual feature importance
    if not np.all(feature_importance==0):
        feature_importance = feature_importance / feature_importance.sum()

    # Sort feature importance
    feature_importance = pd.Series(feature_importance, index=inputs)
    feature_importance = feature_importance.sort_values(ascending=False)

    # Plot feature importance histogram
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=feature_importance.index, y=feature_importance, name='Importance'), secondary_y=False)
    fig.add_trace(go.Scatter(x=feature_importance.index, y=feature_importance.cumsum(), mode='lines', name='Cumulative sum', opacity=0.8), secondary_y=True)
    fig.update_layout(showlegend=False, 
                    title=title, 
                    yaxis_title='Importance (%)', 
                    xaxis_title='Feature', xaxis_tickangle=35, xaxis_tickfont=dict(size=8),
                    yaxis2_title='Cumulative Importance sum (%)',
                    height=600, width=1200)
    fig.update_yaxes(range=[0, 1.05], secondary_y=True)
    if save_path is not None:
        fig.write_image(save_path)
    if plot:
        fig.show()

def plot_feature_importance_array(feature_importance_array, inputs, title='Feature Importance', plot_std=True, save_path=None, plot=True):
    '''
    Plot the feature importance as a sorted histogram and the cumulative sum, and std.

    feature_importance_array: np.array with the feature importance, don't need to be sorted
    inputs: list with the inputs names
    '''

    # Percentual feature importance
    for idx in range(len(feature_importance_array)):
        feature_importance = np.array(feature_importance_array[idx])
        if not np.all(feature_importance==0):
            feature_importance_array[idx] = feature_importance / feature_importance.sum()

    # Calculate the mean and std
    feature_importance_array = np.array(feature_importance_array)
    feature_importance = feature_importance_array.mean(axis=0)
    feature_importance_std = feature_importance_array.std(axis=0)

    # Sort feature importance
    feature_importance = pd.Series(feature_importance, index=inputs)
    feature_importance = feature_importance.sort_values(ascending=False)

    # Plot feature importance histogram
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if plot_std:
        fig.add_trace(go.Bar(x=feature_importance.index, y=feature_importance, error_y=dict(type='data', array=feature_importance_std), name='Importance'), secondary_y=False)
    else:
        fig.add_trace(go.Bar(x=feature_importance.index, y=feature_importance, name='Importance'), secondary_y=False)
    fig.add_trace(go.Scatter(x=feature_importance.index, y=feature_importance.cumsum(), mode='lines', name='Cumulative sum', opacity=0.8), secondary_y=True)
    fig.update_layout(showlegend=False, 
                    title=title, 
                    yaxis_title='Importance (%)', 
                    xaxis_title='Feature', xaxis_tickangle=35, xaxis_tickfont=dict(size=8),
                    yaxis2_title='Cumulative Importance sum (%)',
                    height=600, width=1200)
    
    if save_path is not None:
        fig.write_image(save_path)
    if plot:
        fig.show()


def calculate_FAR_TTD_MDR(y_true, y_pred):
    '''
    Calculate the false alarm rate (FAR), time to detection (TTD) and missed detection rate (MDR)
    '''
    # Convert y_pred to boolean
    y_pred = y_pred >= 0.5

    # False alarm rate (FAR)
    FAR = sum((y_true == 0) & (y_pred == 1)) / sum(y_true == 0)

    # Missed detection rate (MDR)
    MDR = sum((y_true == 1) & (y_pred == 0)) / sum(y_true == 1)

    # Time to detection (TTD)
    # This needs special treatment as the dataset is concatenatet. It can be separated by the date in the index
    # So, first, we need to split the dataset at the moments the index goes down
    idx = y_true.index
    idx_diff = idx[1:] - idx[:-1]
    if type(idx_diff) == pd.TimedeltaIndex:
        idx_diff = idx_diff.total_seconds()
    idx_diff = idx_diff < 0
    idxs_separations = np.where(idx_diff)[0]+1
    y_true_separated = np.split(y_true, idxs_separations)
    y_true_separated = np.array(y_true_separated)
    y_pred_separated = np.split(y_pred, idxs_separations)
    y_pred_separated = np.array(y_pred_separated)

    # Now, we can calculate the TTD
    TTD = []
    for y_true_window, y_pred_window in zip(y_true_separated, y_pred_separated):
        # get idx of start of the fault in the ground truth window
        idx_first_detection_groundtruth = np.where(y_true_window == 1)[0][0]

        # get idx of start of the fault in the prediction window, which is the first time there is a sequence of 3 1s
        idx_first_detection_pred = np.inf
        for i in range(len(y_pred_window)-5):
            if np.all(y_pred_window[i:i+6] == 1):
                idx_first_detection_pred = i
                break

        # Only count if the detection was after the fault started
        time_to_detection = idx_first_detection_pred - idx_first_detection_groundtruth
        if time_to_detection >= 0:
            TTD.append(time_to_detection)
    if len(TTD) != 0:
        TTD = np.mean(TTD)
    else:
        TTD = np.inf

    return pd.Series([FAR*100, TTD, MDR*100], index=['FAR (%)', 'TTD', 'MDR (%)'])

def plot_training_and_val_losses_with_std(epochs_plot, train_loss_mean, train_loss_std, val_loss_mean, val_loss_std, path=None):
    """
    Plots the training and validation losses with standard deviation as shaded areas.
    Parameters:
    epochs_plot (array-like): Array of epoch numbers.
    train_loss_mean (array-like): Mean training loss values.
    train_loss_std (array-like): Standard deviation of training loss values.
    val_loss_mean (array-like): Mean validation loss values.
    val_loss_std (array-like): Standard deviation of validation loss values.
    path (str, optional): Path to save the plot image. If None, the plot will be displayed.
    Returns:
    None
    """

    # Colors
    train_color = [0,0,255]
    val_color = [255,0,0]

    
    # Figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Trainig
    x = epochs_plot
    y = train_loss_mean.tolist()
    y_upper = (train_loss_mean + train_loss_std).tolist()
    y_lower = (train_loss_mean - train_loss_std).tolist()

    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=f'rgb({train_color[0]},{train_color[1]},{train_color[2]})'), mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(
        x=x+x[::-1], # x, then x reversed
        y=y_upper+y_lower[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor=f'rgba({train_color[0]},{train_color[1]},{train_color[2]},.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    # Validation
    x = epochs_plot
    y = val_loss_mean.tolist()
    y_upper = (val_loss_mean + val_loss_std).tolist()
    y_lower = (val_loss_mean - val_loss_std).tolist()

    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=f'rgb({val_color[0]},{val_color[1]},{val_color[2]})'), mode='lines', name='Validation Loss'), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=x+x[::-1], # x, then x reversed
        y=y_upper+y_lower[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor=f'rgba({val_color[0]},{val_color[1]},{val_color[2]},0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ), secondary_y=True)

    # Title
    fig.update_layout({'title': 'Training and Validation Losses'})
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Train Loss", secondary_y=False)
    fig.update_yaxes(title_text="Val loss", secondary_y=True)

    # Save
    if path is not None:
        fig.write_image(path)
    else:
        fig.show()

# Calculate the mean of all dataframes
def calculate_mean_of_dataframe_array(df_array):
    '''
    Calculate the mean of all dataframes in the array
    '''
    plot_df_matrix = np.array([a.values for a in df_array])
    plot_df_array = df_array[0].copy()
    plot_df_array[plot_df_array.columns] = 0
    means = plot_df_matrix.mean(axis=0)
    plot_df_array[plot_df_array.columns] = means

    return plot_df_array

# Calculate the mean and var of all dataframes
def calculate_mean_and_var_of_dataframe_array(df_array):
    '''
    Calculate the mean of all dataframes in the array
    '''
    plot_df_matrix = np.array([a.values for a in df_array])
    stds = plot_df_matrix.astype(np.float64).std(axis=0)
    means = plot_df_matrix.mean(axis=0)

    plot_df_array = df_array[0].copy()
    plot_df_array[plot_df_array.columns] = means
    for i in range(len(plot_df_array.columns)):
        # If column starts with 'RL' or is 'Logistic Regression', skip
        if ('RL' in plot_df_array.columns[i]) or ('Logistic Regression' in plot_df_array.columns[i]):
            continue
        for j in range(len(plot_df_array)):
            plot_df_array.iloc[j,i] = f'{means[j,i]:.3f} ± {stds[j,i]:.3f}'

    return plot_df_array

# Style
def apply_style(column):
    '''
    Format the Dataframe to highlight the lowest ones
    '''
    # Define styles
    best_style = "font-weight: bold; text-decoration: underline; color:red"
    second_style = "font-weight: bold; color:orange"
    others_style = "color:white;"
    best_idx = np.argmin(column.values)
    second_idx = np.argsort(column.values)[1]
    final_collumn_style = [others_style]*len(column)
    final_collumn_style[best_idx] = best_style
    final_collumn_style[second_idx] = second_style
    return final_collumn_style

def apply_style_std(column):
    '''
    Format the Dataframe to highlight the lowest ones
    '''
    # Define styles
    best_style = "font-weight: bold; text-decoration: underline; color:red"
    second_style = "font-weight: bold; color:orange"
    others_style = "color:white;"
    values = [value for value in column.values]
    values = [float(str(value).split('±')[0]) for value in values]
    best_idx = np.argmin(values)
    second_idx = np.argsort(values)[1]
    final_collumn_style = [others_style]*len(column)
    final_collumn_style[best_idx] = best_style
    final_collumn_style[second_idx] = second_style
    return final_collumn_style
