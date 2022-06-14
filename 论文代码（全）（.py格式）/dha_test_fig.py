import shujuchuli
import numpy as np
from DHA_CDA import DHA_CDA
from DHA_SDA import DHA_SDA
from DHA_DDA import DHA_DDA
from paint_figure import Visualization
import os
from sklearn.model_selection import KFold
import time
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import shujuchuli_MFPT
import shujuchuli_JNU


class Bagging_encoder(object):
    # __slots__ = ["forest", "para_forest","n_estimators",""fault_num, "subsamples", "subfeatures", 'probablity_train', 'train_step']
    def __init__(self, train_step, fault_num=10):
        '''
        @:param n_estimators: 树的总数量
        @:param subsamples: 子采样个数
        @:param subfeatures: 随机选取特征个数
        '''

        self.forest = list()
        self.fault_num = fault_num
        self.probablity_train = np.ones(shape=(3,  self.fault_num,  self.fault_num))  # 训练集条件概率
        self.para_forest = []
        self.train_step = train_step

    def __chooseSubset(self, X, y):
        '''
        :param X: 输入训练特征集合
        :param y: 输入标签值集合,y为一维
        :return: 训练子集
        '''
        N, cols = X.shape
        # 随机选取特征 , random.sample从list（即range(0, cols)）中随机选取self.subfeatures个元素，作为一个列表
        # select_cols = list(random.sample(range(0, cols), self.subfeatures))
        # 随机选取训练子集， np.random.permutation(N)对0-N之间的序列随机排序
        select_rows = list(np.random.permutation(N)[:2*X.shape[0]//3])
        # 获取训练子集
        X_subset = X[select_rows, :]
        # X_subset = X_subset[:, select_cols]
        y_subset = y[select_rows]
        return X_subset, y_subset


    def fit(self, X, y):
        '''
        :param X: 输入训练特征集合
        :param y: 输入标签值集合，一维
        :return: 随机森林回归器
        '''
        start_train = time.clock()
        self.pos = {}


        tree_sda1 = DHA_SDA(500, 250, 125, train_step=self.train_step, lr=0.004, fault_num=self.fault_num)
        # tree_dda1 = DHA_DDA(300, 150, 75, train_step=self.train_step, lr=0.006, fault_num=self.fault_num)
        # tree_dda2 = DHA_DDA(300, 150, 75, train_step=self.train_step, lr=0.006, fault_num=self.fault_num)
        # tree_dda3 = DHA_DDA(300, 150, 75, train_step=self.train_step, lr=0.006, fault_num=self.fault_num)
        # tree_cda1 = DHA_CDA(300, 150, 75, train_step=self.train_step, lr=0.006, fault_num=self.fault_num)
        # tree_sda2 = DHA_SDA(300, 150, 75, train_step=self.train_step, lr=0.004, fault_num=self.fault_num)
        tree_sda3 = DHA_SDA(500, 250, 125, train_step=self.train_step, lr=0.004, fault_num=self.fault_num)
        tree_cda2 = DHA_CDA(500, 250, 125, train_step=self.train_step, lr=0.006, fault_num=self.fault_num)
        # tree_cda3 = DHA_CDA(500, 250, 125, train_step=self.train_step, lr=0.006, fault_num=self.fault_num)

        tmp_li = [tree_sda1, tree_cda2, tree_sda3]

        for i, encoder in enumerate(tmp_li):

            X_subset, y_subset = self.__chooseSubset(X, y)  # y_subset：一维

            print('训练第' + str(i+1) + '个分类器')
            da_pre, da_circle, da_R, da_w, da_b, _, _ = encoder.fit(X_subset, y_subset, i)
            cm_estimators = confusion_matrix(y_subset, da_pre)  # 混淆矩阵
            para = [da_circle, da_R, da_w, da_b]

            # 组成随机森林
            self.forest.append(encoder)
            self.para_forest.append(para)

            cm_estimators = cm_estimators.astype(np.float64)
            for j in range(self.fault_num):  # 真实标签为j  :行
                m = len(np.argwhere(y_subset == j).flatten())  # 每种标签的数量
                label_pre = (m + 1) / (y_subset.shape[0] + y_subset.shape[1])  # 每种标签k的先验  laplace校准
                for jj in range(self.fault_num):  # 预测标签为jj  :列
                    cm_estimators[j][jj] = cm_estimators[j][jj] / m  # 混淆矩阵的概率形式 0.01防止分母为0
                    self.probablity_train[i, j, jj] = label_pre * cm_estimators[j][jj]   # 后验概率


        end_train = time.clock()
        print('DHA_BAGGING train time: %s Seconds' % (end_train - start_train))

        return self.forest

    def predict(self, X, y):

        start_test = time.clock()
        assert len(X) > 0


        predict = []
        encoder_test = []
        final_pre = np.zeros(shape=(X.shape[0], 1))   # 最终的预测

        for i in range(3):
            _, y_pre, space_test = self.forest[i].test(
                X, y,self.para_forest[i][0], self.para_forest[i][1], self.para_forest[i][2], self.para_forest[i][3], test=True)   # y_pre：shape：（,1）

            # predict[:, i] = y_pre
            predict.append(y_pre)
            encoder_test.append(space_test)


        index_lrp = []
        index = 0
        final_encoder = np.zeros(shape=[space_test.shape[0], space_test.shape[1]])
        for m in range(X.shape[0]):
            max = 0
            axis_x = y[m][0]  # 实际标签
            for j in range(3):
                axis_y = int(predict[j][m])  # 第j个编码器对第i个样本的预测标签
                t = self.probablity_train[j][axis_x][axis_y]  # 第j个编码器的 后验概率
                if t > max:
                    max = t
                    final_pre[m] = predict[j][m]
                    final_encoder[m] = encoder_test[j][m]
                    index = j      # 使用的哪个分类器的分类结果
            index_lrp.append(index)

        final_pre = final_pre.astype(np.int64)

        dir1 = './DHA_Bagging_jnu/'
        if not os.path.exists(dir1):
            os.mkdir(dir1)
        picturesdir = './DHA_Bagging_jnu/bagging_result_pictures/'
        if not os.path.exists(picturesdir):
            os.mkdir(picturesdir)

        fig = Visualization(y, final_pre, picturesdir)
        acc = fig.plot_confusion_matrix('DHA_bagging test HAR Confusion Matrix')  # 画混淆矩阵
        fig.pca_2D(space_test)
        fig.tsne_3d(space_test)

        fig.plot_roc('DHA_bagging ROC curve')  # 画roc图
        end_test = time.clock()
        print('test time: %s Seconds' % (end_test - start_test))

        return final_pre, index_lrp


if __name__ == '__main__':
    data = shujuchuli_MFPT.cut_samples()
    train_x, train_y, test_x, test_y, = shujuchuli_MFPT.make_datasets(data)
    # train_x = train_x[:300, :]
    # train_y = train_y[:300, :]
    # val_x = test_x[:150, :]
    # val_y = test_y[:150, :]

    suiji_encoder = Bagging_encoder(train_step=500, fault_num=15)
    encoder_li = suiji_encoder.fit(train_x, train_y)
    # 模型验证
    rf_test_pre, index_lrp = suiji_encoder.predict(test_x, test_y)



    if not os.path.exists('./DHA_Bagging_jnu/'):
        os.mkdir('./DHA_Bagging_jnu/')

    dir = './DHA_Bagging_jnu/Partition_dataset/'
    if not os.path.exists(dir):
        os.mkdir(dir)

    # 保存数据集的使用情况
    np.savetxt(dir + 'train_JNU.txt', train_x)
    np.savetxt(dir+ 'train_index_JNU.txt', train_y)
    np.savetxt(dir + 'test_JNU.txt', test_x)
    np.savetxt(dir + 'test_index_JNU.txt', test_y)


    dir1 = './DHA_Bagging_jnu/final_classifier/'
    if not os.path.exists(dir1):
        os.mkdir(dir1)

    # 保存在集成阶段每个样本所使用的分类器
    f1 = open('./DHA_Bagging_jnu/final_classifier/index_JNU.txt', 'w')
    print(index_lrp, file=f1)
    f1.close()

    # 保存在集成阶段确定的分类器组合
    f2 = open('./DHA_Bagging_jnu/final_classifier/DHA_list_JNU.txt', 'w')
    print(encoder_li, file=f2)
    f2.close()

