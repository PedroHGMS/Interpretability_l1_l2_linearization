import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import TEP_utils

def calculate_T2(df, pca, mean_and_std):
    """
    Calculate the Hotelling's T-squared statistic for each observation in the dataframe.
    Parameters:
    df (pandas.DataFrame): The input dataframe containing the data to be analyzed.
    pca (sklearn.decomposition.PCA): The PCA object that has been fitted to the data.
    mean_and_std (tuple): A tuple containing the mean and standard deviation used for normalization.
    Returns:
    numpy.ndarray: An array of T-squared values for each observation in the dataframe.
    """

    mean, std = mean_and_std
    norm_df = ((df - mean) / std).iloc[:, :-1]
    transformed_df = pca.transform(norm_df)
    T_2_values = np.sum(((transformed_df)**2)/pca.explained_variance_, axis=1)
    return T_2_values

def get_limit(T2_values, FAR):
    """
    Calculate the limit value for T2 statistics based on the given False Alarm Rate (FAR).
    Parameters:
    T2_values (array-like): An array of T2 statistic values.
    FAR (float): The desired False Alarm Rate, a value between 0 and 1.
    Returns:
    float: The limit value for the T2 statistics corresponding to the given FAR.
    """

    limit = np.percentile(T2_values, 100*(1-FAR))
    return limit

def get_pca(df, explained_variance):
    """
    Perform Principal Component Analysis (PCA) on the given DataFrame.
    Parameters:
    df (pandas.DataFrame): The input data frame containing the data to perform PCA on.
    explained_variance (float): The amount of variance that needs to be explained by the PCA. 
                                Must be a value between 0 and 1. If 1, the number of components 
                                will be set to the number of features minus one.
    Returns:
    tuple: A tuple containing the PCA object and a tuple with the mean and standard deviation 
           of the original DataFrame.
    Raises:
    ValueError: If explained_variance is not between 0 and 1.
    """

    if explained_variance == 1:
        explained_variance = df.shape[1] - 1
    elif (explained_variance > 1 or explained_variance <= 0):
        raise ValueError('explained_variance must be a value between 0 and 1')
    pca = PCA(n_components=explained_variance)
    pca.fit(((df - df.mean()) / df.std()).iloc[:, :-1])
    mean_and_std = (df.mean(), df.std())
    return pca, mean_and_std

def calculate_FAR(ground_truth_faults, fault_prediction):
    """
    Calculate the False Alarm Rate (FAR).
    The False Alarm Rate is the ratio of the number of false alarms (instances where the ground truth indicates no fault, 
    but a fault is predicted) to the total number of predicted faults.
    Parameters:
    ground_truth_faults (array-like): An array or list indicating the ground truth faults (0 for no fault, 1 for fault).
    fault_prediction (array-like): An array or list indicating the predicted faults (0 for no fault, 1 for fault).
    Returns:
    float: The False Alarm Rate (FAR).
    """

    FAR = ((ground_truth_faults==0) & fault_prediction).sum() / (fault_prediction).sum()
    return FAR

def calculate_MDR(ground_truth_faults, fault_prediction):
    '''
    Calculate the missed detection rate (MDR) of the fault prediction.
    '''
    MDR = ((ground_truth_faults==1) & (fault_prediction==0)).sum() / (ground_truth_faults==1).sum()
    return MDR

def calculate_TTD(ground_truth_faults, detected_as_faults):
    '''
    Calculate the time to detection (TTD) of the first fault in the dataset in seconds.
    The first fault is considered the first time there is a sequence of 6 1s in the detected_as_faults array, as described in the original paper.

    Parameters:
    ground_truth_faults (np.ndarray): Array with ground truth fault labels.
    detected_as_faults (np.ndarray): Array with detected fault labels.

    Returns:
    float: Time to detection of the first fault in the dataset.
    '''
    # get idx of start of the fault in the ground truth window
    idx_first_detection_groundtruth = np.where(ground_truth_faults == 1)[0][0]

    # get idx of start of the fault in the prediction window, which is the first time there is a sequence of 3 1s
    idx_first_detection_pred = np.inf
    for i in range(len(detected_as_faults)-5):
        if np.all(detected_as_faults[i:i+6] == 1):
            idx_first_detection_pred = i+1
            break

    # Calculate difference
    time_to_detection = idx_first_detection_pred - idx_first_detection_groundtruth
    # print(idx_first_detection_pred, idx_first_detection_groundtruth)

    # Only count if the detection was after the fault started
    if time_to_detection < 0:
        idx_first_detection_pred = np.nan
        
    return time_to_detection*3 # 3 seconds per sample

def calculate_Q(df, pca, mean_and_std):
    """
    Calculate the Q statistic (squared prediction error) for each observation in df.

    Parameters:
    df (pd.DataFrame): The dataset for which the Q statistic is to be calculated.
    pca (PCA): Fitted PCA object (from sklearn).
    mean_and_std (tuple): Tuple containing (mean, std) of the training dataset for normalization.

    Returns:
    np.ndarray: Array of Q statistics for each point in df.
    """
    # Unpack mean and std for normalization
    mean, std = mean_and_std
    
    # Normalize the dataframe (exclude the last column if needed)
    norm_df = (df - mean) / std

    # Transform the data using the fitted PCA model
    transformed_df = pca.transform(norm_df.iloc[:, :-1])
    
    # Get the PCA loading vectors (principal components)
    P = pca.components_.T  # Transposed to align with the data dimensions

    # Reconstruct the data from the PCA-transformed space
    reconstructed_df = transformed_df.dot(P.T)
    
    # Calculate residuals in the original space
    residuals = norm_df.values[:, :-1] - reconstructed_df
    
    # Calculate the Q statistic for each observation (sum of squared residuals)
    Q = np.sum(residuals**2, axis=1)
    
    return Q

def get_metrics_chiang_PCA_T2_and_Q(explained_variance):
    '''
    Calculate metrics for Chiang dataset using PCA with T^2 and Q values

    Returns:
    - results_MDR: DataFrame with MDR values for each fault
    - results_FAR: DataFrame with FAR values for each fault
    - results_TTD: DataFrame with TTD values for each fault
    '''
    # Load dataset
    df_train_chiang, df_val_chiang, df_test_chiang = TEP_utils.load_dataset_Chiang_pca(fault=1)

    # Calculate PCA from d00.dat
    pca, mean_and_std = get_pca(df_train_chiang, explained_variance=explained_variance)

    # Calculate T^2 values from d00_te.dat
    T2_values_val = calculate_T2(df_val_chiang, pca, mean_and_std)
    limit_99_T2 = get_limit(T2_values_val, 0.01)

    # Calculate Q values from d00_te.dat
    Q_values_val = calculate_Q(df_val_chiang, pca, mean_and_std)
    limit_99_Q = get_limit(Q_values_val, 0.01)

    results_MDR = pd.DataFrame(columns=['PCA (T²)', 'PCA (Q)'], index=range(1, 21+1))
    results_FAR = pd.DataFrame(columns=['PCA (T²)', 'PCA (Q)'], index=range(1, 21+1))
    results_TTD = pd.DataFrame(columns=['PCA (T²)', 'PCA (Q)'], index=range(1, 21+1))

    for fault in range(1, 21+1):
        # Load dataset
        _, _, df_test_chiang = TEP_utils.load_dataset_Chiang_pca(fault=fault)

        # Faults gt
        ground_truth_faults = df_test_chiang['True IDV'].values==1

        # For T^2
        # Detect faults
        Q_values_test = calculate_T2(df_test_chiang, pca, mean_and_std)
        detected_as_faults_Q = Q_values_test >= limit_99_T2

        # Calculate metrics using T^2
        FAR_T2 = calculate_FAR(ground_truth_faults, detected_as_faults_Q)
        MDR_T2 = calculate_MDR(ground_truth_faults, detected_as_faults_Q)
        TTD_T2 = calculate_TTD(ground_truth_faults, detected_as_faults_Q)

        # For Q
        # Detect faults
        Q_values_test = calculate_Q(df_test_chiang, pca, mean_and_std)
        detected_as_faults_T2 = Q_values_test >= limit_99_Q

        # Calculate metrics using Q
        FAR_Q = calculate_FAR(ground_truth_faults, detected_as_faults_T2)
        MDR_Q = calculate_MDR(ground_truth_faults, detected_as_faults_T2)
        TTD_Q = calculate_TTD(ground_truth_faults, detected_as_faults_T2)
        
        # Attributes results
        results_MDR.loc[fault] = [MDR_T2, MDR_Q]
        results_FAR.loc[fault] = [FAR_T2, FAR_Q]
        results_TTD.loc[fault] = [TTD_T2, TTD_Q]
    
    return results_MDR, results_FAR, results_TTD