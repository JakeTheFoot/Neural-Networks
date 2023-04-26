import numpy as np
import pandas as pd
import zipfile
import io
import os
from sklearn.model_selection import train_test_split


def process_stock_data(Num_of_Days, Percent_for_Tests):
    data_matrices = []

    etf_folder_path = "stock_dataset/ETFs/"

    if os.path.exists(etf_folder_path):
        # Iterate through each file in the folder
        for file in os.listdir(etf_folder_path):
            file_path = os.path.join(etf_folder_path, file)

            # Read the CSV file using pandas
            df = pd.read_csv(file_path)

            # Extract the last Num_of_Days days of data and save it as a 2D matrix
            last_n_days = df[-Num_of_Days:].copy()
            data_matrix = last_n_days[["Open", "High", "Low", "Close"]].values

            # Pad the data_matrix with zeros if the number of rows is less than Num_of_Days
            if data_matrix.shape[0] < Num_of_Days:
                data_matrix = np.pad(data_matrix, ((
                    Num_of_Days - data_matrix.shape[0], 0), (0, 0)), mode='constant', constant_values=0)

            # Append data_matrix to data_matrices
            data_matrices.append(data_matrix)
    else:
        # Access the folder labeled "ETF" from the file "stock_dataset.zip"
        with zipfile.ZipFile("stock_dataset.zip", "r") as zip_ref:
            etf_folder = [info for info in zip_ref.infolist(
            ) if info.filename.startswith("ETF/")]

            # Iterate through each file in the folder
            for file in etf_folder:
                with zip_ref.open(file, "r") as csvfile:
                    # Read the CSV file using pandas
                    df = pd.read_csv(io.TextIOWrapper(csvfile))

                    # Extract the last Num_of_Days days of data and save it as a 2D matrix
                    last_n_days = df[-Num_of_Days:].copy()
                    data_matrix = last_n_days[[
                        "Open", "High", "Low", "Close"]].values

                    # Pad the data_matrix with zeros if the number of rows is less than Num_of_Days
                    if data_matrix.shape[0] < Num_of_Days:
                        data_matrix = np.pad(data_matrix, ((
                            Num_of_Days - data_matrix.shape[0], 0), (0, 0)), mode='constant', constant_values=0)

                    # Append data_matrix to data_matrices
                    data_matrices.append(data_matrix)

    # Convert the list of 2D matrices into a 3D array
    data_3d_array = np.array(data_matrices)

    # Shuffle the 3D array along its third axis
    np.random.shuffle(data_3d_array)

    # Normalize the values in the 3D array to a range of -1 to 1
    half_max = np.max(data_3d_array) / 2
    normalized_3d_array = (data_3d_array - half_max) / half_max

    # Split the matrices into training and testing data
    X_all, X_test_all = train_test_split(
        normalized_3d_array, test_size=Percent_for_Tests, random_state=42)

    # Break off the "Close" column from each matrix and append them to separate variables "y" and "y_test"
    y = X_all[:, :, -1]
    y_test = X_test_all[:, :, -1]

    # Remove the "Close" column from each matrix in X_all and X_test_all
    X = np.delete(X_all, -1, axis=2)
    X_test = np.delete(X_test_all, -1, axis=2)

    return X, X_test, y, y_test


X, X_test, y, y_test = process_stock_data(
    Num_of_Days=30, Percent_for_Tests=0.2)

print('\n\n\n\nX:\n\n', X, '\n\n\n\nX_test:\n\n',
      X_test, '\n\n\n\ny:\n\n', y, '\n\n\n\ny_test:\n\n', y_test)
