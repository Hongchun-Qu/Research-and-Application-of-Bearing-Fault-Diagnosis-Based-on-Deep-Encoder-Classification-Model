'''
https://blog.csdn.net/weixin_44620044/article/details/106877805
'''

import numpy as np
from random import shuffle
import os
from scipy.fftpack import fft
from sklearn import preprocessing  # 0-1编码
from scipy.io import loadmat
import matplotlib.pyplot as plt

def max_min(train_x, test_x):
    # x = (x - Min) / (Max - Min)
    scalar = preprocessing.MinMaxScaler().fit(train_x)
    train_x = scalar.transform(train_x)
    test_x = scalar.transform(test_x)
    return train_x, test_x

# https://blog.csdn.net/qq_27825451/article/details/88553441
def fft_normalize(x):

    fft_x = fft(x)  # 快速傅里叶变换
    abs_x = np.abs(fft_x)  # 取复数的绝对值，即复数的模(双边频谱)
    N = len(x)
    abs_half_x = abs_x[range(int(N / 2))]
    normalization_x = abs_x / N  # 归一化处理（双边频谱）
    normalization_half_x = normalization_x[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

    # plt.figure()
    # plt.plot(half_number, normalization_half_x, 'blue')
    # plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
    # plt.show()

    return normalization_half_x

# https://blog.csdn.net/sinat_24259567/article/details/93889547
def add_noise(signal, SNR=None):
    if SNR==None:
        return signal
    else:
        # noise = np.random.randn(signal.shape[0], signal.shape[1], signal.shape[2])   #产生N(0,1)噪声数据
        noise = np.random.randn(signal.shape[0])
        noise = noise - np.mean(noise)  #均值为0
        signal_power = np.linalg.norm(signal - signal.mean()) ** 2 / signal.size  #此处是信号的std**2
        # linalg=linear（线性）+algebra（代数），norm则表示范数。默认求二范数
        noise_variance = signal_power / np.power(10, (SNR / 10))    # np.power(x,y)计算x的y次方  #此处是噪声的std**2
        noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
        signal_noise = noise + signal

        return signal_noise


'''
所有样本都通过快速傅立叶变换（FFT）转换为频谱。为了规范化数据，通过最大-最小映射将频谱样本缩放到0到1的范围
'''
def cut_samples(SNR=None):
    '''
    每种状况得到400个样本
    '''
    # results = np.zeros(shape=(15, 120, 1000))
    # temporary_s = np.zeros(shape=(120, 2000))
    # fft_temporary_s = np.zeros(shape=(120, 1000))
    results = []

    files = os.listdir('C:\\Users\\LENOVO\\Desktop\\论文\\数据集\\JNU')
    domain = os.path.abspath(
        r'C:\\Users\\LENOVO\\Desktop\\论文\\数据集\\JNU')
    for i in range(len(files)):

        file = os.path.join(domain, files[i])
        s = np.loadtxt(file)
        x = 0
        res_li = []
        print(files[i])
        # 采样频率50khz
        for x in range(500):
            temporary_s = s[1000 * x: 1200 + 1000 * x]
            fft_temporary_s = fft_normalize(temporary_s)
            noise_s = add_noise(fft_temporary_s, SNR)
            res_li.append(noise_s)

            if  x == 100 or x == 150 or x == 200:
                time = np.arange(0,0.02,0.02/1000)
                plt.plot(time, temporary_s[0:1000])
                plt.title(files[i])

                plt.show()

        res = np.array(res_li)
        results.append(res)

    return results



# 划分训练集和测试集
def make_datasets(results):
    '''输入3*x*2000的原始样本'''
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for i in range(12):
        s = results[i]
        # 打乱顺序
        index_s = [a for a in range(len(s))]
        shuffle(index_s)
        s = s[index_s]
        # 对每种类型都划分训练集和测试集
        train_x.append(s[:int(0.8*len(s))])
        test_x.append(s[int(0.8*len(s)):])

        # 填写标签
        label = np.array([i for _ in range(len(s))]).reshape(-1,1)
        train_y.append(label[:int(0.8*len(s))])
        test_y.append(label[int(0.8*len(s)):])

    #将训练集和测试集分别合并并打乱
    x1 = train_x[0]
    y1 = train_y[0]
    x2 = test_x[0]
    y2 = test_y[0]
    for i in range(11):
        x1 = np.row_stack((x1, train_x[i + 1]))
        x2 = np.row_stack((x2, test_x[i + 1]))

        y1 = np.row_stack((y1, train_y[i + 1]))
        y2 = np.row_stack((y2, test_y[i + 1]))


    index_x1 = [i for i in range(len(x1))]
    index_x2 = [i for i in range(len(x2))]

    shuffle(index_x1)
    shuffle(index_x2)

    x1 = x1[index_x1]
    y1 = y1[index_x1]
    x2 = x2[index_x2]
    y2 = y2[index_x2]

    x1, x2 = max_min(x1, x2)


    return x1, y1, x2, y2   #分别代表：训练集样本，训练集标签，测试集样本，测试集标签


# 读取原始数据，处理后保存
if __name__ == "__main__":

    data = cut_samples()
    train_x, train_y, test_x, test_y = make_datasets(data)
    print('train_x:', train_x.dtype, train_x.shape)
    print('train_y:', train_y.dtype, train_y.shape)
    print('test_x:', test_x.dtype, test_x.shape)
    print('test_y:', test_y.dtype, test_y.shape)










