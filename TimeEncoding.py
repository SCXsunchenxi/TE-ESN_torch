import math
import numpy as np
import time
import matplotlib.pyplot as plt
import torch


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def timeencoding(time_list,d_model=16):

    """
    :param d_model: dimension of the model
    :param time_list: the list of time in time series
    :return: position matrix of (length of time list*d_model)
    """

    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))

    # date to float
    time_list = [time.mktime(time.strptime(i, "%Y-%m-%d %H:%M:%S")) for i in time_list]

    new_time_list=[t-time_list[0] for t in time_list]
    TE = torch.zeros(len(new_time_list), d_model)
    time_list = torch.tensor(new_time_list).unsqueeze(1)

    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))

    TE[:, 0::2] = torch.sin(time_list.float() * div_term)
    TE[:, 1::2] = torch.cos(time_list.float() * div_term)

    return TE


# test
if __name__ == "__main__":

    # para
    model_dimension=128
    time_list = ['2018-05-20 08:30:00', '2018-05-21 07:30:00','2018-05-21 15:12:00']

    # time encoding
    TE = timeencoding(time_list,model_dimension)
    print(TE)

    # show figure
    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(3), TE[:, 0:63:16].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.savefig('TE.jpg')







