import numpy as np
import torch
import random

def generate_white_noise(shape, mean=0, std=1):
    white_noise = np.random.normal(mean, std, shape)
    return white_noise


def generate_red_noise(num_points, mean=0, std=1, correlation_coefficient=0.4):
    white_noise = np.random.normal(mean, std, num_points)
    red_noise = np.zeros(num_points)

    n_size = num_points
    if len(n_size) == 1:
        for i in range(n_size[-1]):
            if i == 0:
                red_noise[i] = white_noise[i]
            else:
                red_noise[i] = correlation_coefficient * red_noise[i - 1] + np.sqrt(1 - np.power(correlation_coefficient, 2)) * white_noise[i]

    if len(n_size) == 2:
        for i in range(n_size[1]):
            if i == 0:
                red_noise[:,i] = white_noise[:,i]
            else:
                red_noise[:,i] = correlation_coefficient * red_noise[:,i - 1] + np.sqrt(1 - np.power(correlation_coefficient, 2)) * white_noise[:,i]

    if len(n_size) == 3:
        for i in range(n_size[1]):
            if i == 0:
                red_noise[:,i,:] = white_noise[:,i, :]
            else:
                red_noise[:,i,:] = correlation_coefficient * red_noise[:,i - 1,:] + np.sqrt(1 - np.power(correlation_coefficient, 2)) * white_noise[:,i,:]
    return red_noise


def generate_cyclical_noise(num_points, amplitude=1, frequency=1):
    n_size = num_points
    time = np.arange(n_size[1])
    cyclical_noise = amplitude * np.sin(2 * np.pi * frequency * time / n_size[1])

    if len(n_size) == 3:
        b_r = n_size[0]*n_size[2]
        cyclical_noise = np.repeat(cyclical_noise, b_r, axis=0)
        cyclical_noise = cyclical_noise.reshape(n_size[1], n_size[0], n_size[2])
        cyclical_noise = np.transpose(cyclical_noise, (1, 0, 2))

    if len(n_size) == 2:
        cyclical_noise = np.repeat(cyclical_noise, n_size[0], axis=0)
        cyclical_noise = cyclical_noise.reshape(n_size[-1], n_size[0])
        cyclical_noise = np.transpose(cyclical_noise, (1, 0))


    return cyclical_noise


def generate_auto_regressive_noise(num_points, ar_order=1, ar_coefficients=[0.5]):
    ar_values = np.zeros(num_points)

    n_size = num_points
    if len(n_size) == 1:
        for i in range(ar_order, num_points):
            for j in range(ar_order):
                ar_values[i] += ar_coefficients[j] * ar_values[i - (j + 1)]
            ar_values[i] += np.random.normal()

    if len(n_size) == 2:
        for i in range(ar_order, num_points[1]):
            for j in range(ar_order):
                ar_values[:,i] += ar_coefficients[j] * ar_values[:,i - (j + 1)]
            ar_values[:,i] += np.random.normal(size=(n_size[0],1))

    if len(n_size) == 3:
        for i in range(ar_order, num_points[1]):
            for j in range(ar_order):
                ar_values[:,i,:] += ar_coefficients[j] * ar_values[:,i - (j + 1),:]
            ar_values[:,i,:] += np.random.normal(size=(n_size[0],n_size[2]))

    return ar_values

def generate_moving_average_noise(num_points, ma_order=5):
    n_size = num_points
    time = np.arange(n_size[1])
    ma_values = np.random.normal(0, 1, num_points)

    n_size = num_points

    if len(n_size) == 1:
        for i in range(ma_order, num_points):
            ma_values[i] = np.mean(ma_values[i - ma_order: i])

    if len(n_size) == 2:
        for i in range(ma_order, num_points[1]):
            ma_values[:,i] = np.mean(ma_values[:,i - ma_order: i])

    if len(n_size) == 3:
        for i in range(ma_order, num_points[1]):
            ma_values[:,i,:] = np.mean(ma_values[:,i - ma_order: i,:])

    return ma_values



def add_noise2(xx, yy, ratio, device):
    x_num = xx.shape[0]
    n_num = int(x_num * ratio)
    if n_num == 0:
        return xx, yy
    x_len = xx.shape[1]
    y_len = yy.shape[1]

    # #方式2, 在原来batch，添加噪声样本
    noiseData = np.random.normal(0.5, 0.8 ** 2, (n_num, x_len + y_len))
    choice_num = np.random.choice(x_num, n_num)
    noise_stamp_x = xx[choice_num, :, 1:]
    noise_stamp_y = yy[choice_num, :, 1:]
    noise_data_xy = torch.hstack((noise_stamp_x, noise_stamp_y))
    noiseData = noiseData.reshape(n_num, x_len + y_len, 1)
    noiseData = torch.tensor(noiseData)
    noiseData = torch.cat((noiseData, noise_data_xy), 2)

    noiseData = torch.tensor(noiseData).to(device)
    noiseData = noiseData.float()
    noiseData_X = noiseData[:, :x_len]
    noiseData_Y = noiseData[:, -y_len:]

    xx = torch.vstack((xx.to(device), noiseData_X))
    yy = torch.vstack((yy.to(device), noiseData_Y))

    rid = torch.randperm(xx.size(0))
    xx = xx[rid, :]
    rid = torch.randperm(yy.size(0))
    yy = yy[rid, :]

    # # #方式1, 选一部分样本，添加扰动
    # if len(xx.shape) >= 3:
    #     f_len = xx.shape[2]
    #     rid = random.sample(range(0, x_num), n_num)
    #     xx[rid, :] = xx[rid, :] + np.random.uniform(-10, 10, (n_num, x_len, f_len))
    #     # xx = torch.hstack((xx, flag))  # 增加1列，标识是否是噪声样本
    #     yy[rid, :] = yy[rid, :] + np.random.uniform(-3, 3, (n_num, y_len, f_len))
    # else:
    #     rid = random.sample(range(0, x_num), n_num)
    #     xx[rid, :] = xx[rid, :] + np.random.uniform(3, 3, (n_num, x_len))
    #     # xx = torch.hstack((xx, flag)) #增加1列，标识是否是噪声样本
    #     yy[rid, :] = yy[rid, :] + np.random.uniform(-10, 10, (n_num, y_len))

    return xx, yy, rid

def add_noise(xx, yy, ratio, batch_size):
    x_num = xx.shape[0]
    n_num = int(x_num * ratio)
    ntimes = x_num - x_num % batch_size
    if n_num == 0:
        return xx, yy, 0
    x_len = xx.shape[1]
    y_len = yy.shape[1]

    # #方式1, 选一部分样本，添加扰动
    if len(xx.shape) >= 3:
        f_len = xx.shape[2]
        rid = random.sample(range(0, x_num), n_num)


        xx[rid,:] = xx[rid,:] + np.random.uniform(-100, 100, (n_num, x_len, f_len))
        yy[rid, :] = yy[rid, :] + np.random.uniform(-300, 300, (n_num, y_len, f_len))

        # xx[rid, :] = xx[rid, :] + generate_red_noise((n_num, x_len, f_len), 0, 2,0.1)
        # yy[rid, :] = yy[rid, :] + generate_red_noise((n_num, y_len, f_len), 0, 1,0.5)

        # xx[rid, :] = xx[rid, :] + generate_cyclical_noise((n_num, x_len, f_len))
        # yy[rid, :] = yy[rid, :] + generate_cyclical_noise((n_num, y_len, f_len),0.3,4)

        # xx[rid, :] = xx[rid, :] + generate_moving_average_noise((n_num, x_len, f_len))
        # yy[rid, :] = yy[rid, :] + generate_moving_average_noise((n_num, y_len, f_len), 20)

        # xx[rid, :] = xx[rid, :] + generate_auto_regressive_noise((n_num, x_len, f_len))
        # yy[rid, :] = yy[rid, :] + generate_auto_regressive_noise((n_num, y_len, f_len), 3,[0.3,0.2,0.1])

    else:
        rid = random.sample(range(0, x_num), n_num)

        xx[rid, :] = xx[rid, :] + np.random.uniform(3, 3, (n_num, x_len))
        yy[rid, :] = yy[rid, :] + np.random.uniform(-10, 10, (n_num, y_len))

        # xx[rid, :] = xx[rid, :] + generate_red_noise((n_num, x_len), 0, 2, 0.1)
        # yy[rid, :] = yy[rid, :] + generate_red_noise((n_num, y_len), 0, 1, 0.5)

        # xx[rid, :] = xx[rid, :] + generate_cyclical_noise((n_num, x_len))
        # yy[rid, :] = yy[rid, :] + generate_cyclical_noise((n_num, y_len), 0.3, 4)

        # xx[rid, :] = xx[rid, :] + generate_moving_average_noise((n_num, x_len))
        # yy[rid, :] = yy[rid, :] + generate_moving_average_noise((n_num, y_len), 20)

        # xx[rid, :] = xx[rid, :] + generate_auto_regressive_noise((n_num, x_len))
        # yy[rid, :] = yy[rid, :] + generate_auto_regressive_noise((n_num, y_len), 3,[0.3,0.2,0.1])

    return xx, yy, rid
    # #方式2, 添加扰动样本
    # if len(xx.shape) >= 3:
    #     f_len = xx.shape[2]
    #     noiseData = np.random.normal(0.5, 0.8 ** 2, (n_num, x_len + y_len, f_len))
    #     # flag = torch.zeros((x_num + n_num, 1, f_len))  # 0是正常，1是噪声
    #     # flag[x_num:, 0, :] = 1
    # else:
    #     noiseData = np.random.normal(0.5, 0.8 ** 2, (n_num, x_len + y_len))
    #     # flag = torch.zeros((x_num + n_num, 1))  # 0是正常，1是噪声
    #     # flag[x_num:, :] = 1
    # noiseData = torch.tensor(noiseData).to(args.device)
    # noiseData = noiseData.float()
    # noiseData_X = noiseData[:, :x_len]
    # noiseData_Y = noiseData[:, -y_len:]
    #
    # new_X = torch.vstack(( noiseData_X, xx.to(args.device)))
    # new_Y = torch.vstack(( noiseData_Y, yy.to(args.device)))
    #
    # ntr = new_X.shape[0]
    # ntimes = ntr - ntr % batch_size
    # b = torch.randperm(ntimes)
    # new_X = new_X[b, :]
    # # bb = torch.randperm(new_Y.size(0))
    # new_Y = new_Y[b, :]
    #
    # return new_X, new_Y, b