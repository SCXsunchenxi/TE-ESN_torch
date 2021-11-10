import math
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn
import TimeEncoding
from torchesn.nn import ESN
from torchesn import utils

# cpu/gpu
device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
dtype = torch.double
torch.set_default_dtype(dtype)


if __name__ == "__main__":

    # parameters
    model_dimension=16 # dimension of TE
    hidden_size= 500 # dimension of reservoir
    washout = [2] # number of initial timesteps in reservoir

    # ************************************************************
    data_dir='dataset' # data dir
    test_data = pd.read_csv('dataset/mg_full.csv') # test data
    test_TE = np.expand_dims(TimeEncoding.timeencoding(test_data['t'].to_list()),axis=1)# ['t']改成时间列名
    test_X_data = np.concatenate((test_TE, np.expand_dims([[i] for i in test_data['x'].to_list()], axis=1)), axis=2) # ['x']改成特征列名，可多元变量
    test_Y_data = np.expand_dims([[i] for i in test_data['y'].to_list()], axis=1)  # ['y']改成预测列名
    test_X_data = torch.from_numpy(test_X_data).to(device)
    test_Y_data = torch.from_numpy(test_Y_data).to(device)

    # for each data
    start = time.time()
    ini=1
    listfile = os.listdir(data_dir)
    for file in listfile:
        if (not (file.endswith(".csv"))): continue

        # read data
        print(file)
        data = pd.read_csv(os.path.join(data_dir,file))

        # time encoding
        TE=TimeEncoding.timeencoding(data['t'].to_list()) # ['t']改成时间列名 ******

        # dataset
        TE = np.expand_dims(TE, axis=1)
        X_data = np.expand_dims([[i] for i in data['x'].to_list()], axis=1)  # ['x']改成特征列名，可多元变量 ******
        X_data=np.concatenate((TE,X_data),axis=2)
        Y_data = np.expand_dims([[i] for i in data['y'].to_list()], axis=1)  # ['y']改成预测列名 ******
        X_data = torch.from_numpy(X_data).to(device)
        Y_data = torch.from_numpy(Y_data).to(device)
        # split_n=2*int(len(X_data)/3)
        # trX = X_data[:split_n]
        # trY = Y_data[:split_n]
        # tsX = X_data[split_n:]
        # tsY = Y_data[split_n:]
        trX = X_data
        trY = Y_data
        input_size  = model_dimension+ trX.shape[1]
        output_size= trX.shape[1]
        loss_fcn = torch.nn.MSELoss()

        # initial model
        if (ini==1):
            model = ESN(input_size, hidden_size, output_size,readout_training='inv')
            model.to(device)
            ini =0

        # Training
        trY_flat = utils.prepare_target(trY.clone(), [trY.size(0)], washout)
        model(trX, washout, None, trY_flat)
        model.fit()
        output, hidden = model(trX, washout)
        print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

        # Test
        output, hidden = model(test_X_data, [0], hidden)
        print("Test error:", loss_fcn(output, test_Y_data).item())

    print("Ended in", time.time() - start, "seconds.")











