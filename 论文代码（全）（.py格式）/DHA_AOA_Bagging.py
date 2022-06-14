import shujuchuli
import shujuchuli_MFPT
import numpy as np
from DHA_CDA import DHA_CDA
from DHA_SDA import DHA_SDA
from DHA_DDA import DHA_DDA
from paint_figure import Visualization
from DHA_AOA import AOA
import os
from sklearn.model_selection import KFold
import time
import tensorflow as tf
from sklearn.metrics import confusion_matrix


class Bagging_encoder(object):
    # __slots__ = ["forest", "para_forest", "n_estimators", "subsamples", "subfeatures", 'probablity_train',
    # 'train_step', 'aoa_size', 'aoa_iter_num', 'fault_num']
    def __init__(self, n_estimators, subsamples, subfeatures, train_step,
                 aoa_size=4, aoa_iter_num=20, fault_num=10):
        '''
        :param n_estimators: 分类器的总数量
        :param subsamples: 子采样个数
        :param subfeatures: 随机选取特征个数
        :param train_step: 分类器的迭代次数
        :param pso_size: aoa的粒子数目
        :param pso_iter_num: aoa训练时的迭代次数
        '''

        self.n_estimators = n_estimators
        self.subsamples = subsamples
        self.subfeatures = subfeatures
        self.fault_num = fault_num
        self.forest = list()
        self.probablity_train = np.ones(shape=(self.n_estimators, self.fault_num, self.fault_num))  # 训练集条件概率
        self.aoa_size = aoa_size  # 粒子数目
        self.aoa_iter_num = aoa_iter_num   # 迭代次数
        self.para_forest = []
        self.train_step = train_step

    def __chooseSubset(self, X, y):
        '''
        :param X: 输入训练特征集合
        :param y: 输入标签值集合
        :return: 训练子集
        '''
        N, cols = X.shape
        # 随机选取特征 , random.sample从list（即range(0, cols)）中随机选取self.subfeatures个元素，作为一个列表
        # select_cols = list(random.sample(range(0, cols), self.subfeatures))
        # 随机选取训练子集， np.random.permutation(N)对0-N之间的序列随机排序
        select_rows = list(np.random.permutation(N)[:self.subsamples])
        # 获取训练子集
        X_subset = X[select_rows, :]
        # X_subset = X_subset[:, select_cols]
        y_subset = y[select_rows]
        return X_subset, y_subset

    def select_encoder(self, train_data, train_label, n_folds=3):

        coeff = 2
        kf = KFold(n_splits=n_folds)
        weight_ratio = [1 / 3, 1 / 3, 1 / 3]

        for ii, (train_index, verify_index) in enumerate(kf.split(train_data)):
            print('第', ii+1, '折交叉验证')

            x_train, y_train = train_data[train_index], train_label[train_index]
            verify_x, verify_y = train_data[verify_index], train_label[verify_index]

            dha_sda = DHA_SDA(self.pos['sda'][0], self.pos['sda'][1], self.pos['sda'][2], lr=self.pos['sda'][3],
                              train_step=self.train_step, fault_num=self.fault_num)
            dha_dda = DHA_DDA(self.pos['dda'][0], self.pos['dda'][1], self.pos['dda'][2], lr=self.pos['dda'][3],
                              train_step=self.train_step, fault_num=self.fault_num)
            dha_cda = DHA_CDA(self.pos['cda'][0], self.pos['cda'][1], self.pos['cda'][2], lr=self.pos['cda'][3],
                              train_step=self.train_step, fault_num=self.fault_num)

            sda_pre, sda_circle, sda_R, sda_w, sda_b, _, _ = dha_sda.fit(x_train, y_train, -1)
            acc_sda, _ = dha_sda.test(verify_x, verify_y, sda_circle, sda_R, sda_w, sda_b)

            cda_pre, cda_circle, cda_R, cda_w, cda_b, _, _ = dha_cda.fit(x_train, y_train, -1)
            acc_cda, _ = dha_cda.test(verify_x, verify_y, cda_circle, cda_R, cda_w, cda_b)

            dda_pre, dda_circle, dda_R, dda_w, dda_b, _, _ = dha_dda.fit(x_train, y_train, -1)
            acc_dda, _ = dha_dda.test(verify_x, verify_y, dda_circle, dda_R, dda_w, dda_b)

            acc_verify = [acc_sda, acc_dda, acc_cda]
            alpha = [np.exp(coeff * h) for h in acc_verify]

            sum_w_1 = alpha[0] * weight_ratio[0]
            sum_w_2 = alpha[1] * weight_ratio[1]
            sum_w_3 = alpha[2] * weight_ratio[2]
            sum = sum_w_1 + sum_w_2 + sum_w_3

            for j in range(3):
                weight_ratio[j] = (weight_ratio[j] * alpha[j]) / sum
            print(weight_ratio)

        return weight_ratio

    def fit(self, X, y):
        '''
        :param X: 输入训练特征集合
        :param y: 输入标签值集合，一维
        :return: 随机森林回归器
        '''
        start_train = time.clock()
        self.pos = {}

        # aoa调参
        aoa_sda = AOA(self.aoa_size, self.aoa_iter_num, X, y, sda_para=True,
                      encoder_iternum=self.train_step, fault_num=self.fault_num)
        best_pos_sda = aoa_sda.update()
        print("DHA_SDA最优位置:" + str(best_pos_sda))
        self.pos['sda'] = best_pos_sda

        aoa_dda = AOA(self.aoa_size, self.aoa_iter_num, X, y, dda_para=True,
                      encoder_iternum=self.train_step, fault_num=self.fault_num)
        best_pos_dda = aoa_dda.update()
        print("DHA_DDA最优位置:" + str(best_pos_dda))
        self.pos['dda'] = best_pos_dda

        aoa_cda = AOA(self.aoa_size, self.aoa_iter_num, X, y, cda_para=True,
                      encoder_iternum=self.train_step, fault_num=self.fault_num)
        best_pos_cda = aoa_cda.update()
        print("DHA_CDA最优位置:" + str(best_pos_cda))
        self.pos['cda'] = best_pos_cda

        for i in range(self.n_estimators):
            # 获取构建单棵树的训练子集
            X_subset, y_subset = self.__chooseSubset(X, y) # y_subset：一维
            # 训练单棵树
            print('选择第', i+1, '个树的分类器形式')
            w, pos = self.select_encoder(X_subset, y_subset)   # pos: sda dda cda

            if max(w) == w[0]:
                onetree = DHA_SDA(pos[0][0], pos[0][1], pos[0][2], lr=pos[0][3],
                                  train_step=self.train_step, fault_num=self.fault_num)
                print('第', i+1, '个分类器为DHA_SDA')
                sda_pre, sda_circle, sda_R, sda_w, sda_b, _, _ = onetree.fit(X_subset, y_subset, i)
                cm_estimators = confusion_matrix(y_subset, sda_pre)  # 混淆矩阵
                para = [sda_circle, sda_R, sda_w, sda_b]

            elif max(w) == w[1]:
                onetree = DHA_DDA(pos[1][0], pos[1][1], pos[1][2], lr=pos[1][3],
                                  train_step=self.train_step, fault_num=self.fault_num)
                print('第', i+1, '个分类器为DHA_DDA')
                dda_pre, dda_circle, dda_R, dda_w, dda_b, _, _ = onetree.fit(X_subset, y_subset, i)
                cm_estimators = confusion_matrix(y_subset, dda_pre)  # 混淆矩阵
                para = [dda_circle, dda_R, dda_w, dda_b]

            else:
                onetree = DHA_CDA(pos[2][0], pos[2][1], pos[2][2], lr=pos[2][3],
                                  train_step=self.train_step, fault_num=self.fault_num)
                print('第', i+1, '个编码器为DHA_CDA')
                cda_pre, cda_circle, cda_R, cda_w, cda_b, _, _ = onetree.fit(X_subset, y_subset, i)
                cm_estimators = confusion_matrix(y_subset, cda_pre)  # 混淆矩阵
                para = [cda_circle, cda_R, cda_w, cda_b]


            # 组成随机森林
            self.forest.append(onetree)
            self.para_forest.append(para)
            for i in range(len(self.forest)):
                print('第' + str(i) + '个分类器是：')
                print(self.forest[i])
                print('参数是：')
                print(self.para_forest[i])

            cm_estimators = cm_estimators.astype(np.float64)
            for j in range(self.fault_num):  # 真实标签为j  :行
                m = len(np.argwhere(y_subset == j).flatten())        # 每种标签的数量
                label_pre = (m + 1) / (y_subset.shape[0] + y_subset.shape[1])  # 每种标签k的先验  laplace校准
                for jj in range(self.fault_num):  # 预测标签为jj  :列
                    cm_estimators[j][jj] = cm_estimators[j][jj] / m  # 混淆矩阵的概率形式 0.01防止分母为0
                    self.probablity_train[i, j, jj] = label_pre * cm_estimators[j][jj]  # 后验概率

        print(self.forest)
        end_train = time.clock()
        print(' train time: %s Seconds' % (end_train - start_train))

    def predict(self, X, y):

        start_test = time.clock()
        assert len(X) > 0

        # predict = np.zeros(shape=(X.shape[0], self.n_estimators))  # q个分类器的输出  E（m,q）
        predict = []
        final_pre = np.zeros(shape=(X.shape[0], 1))   # 最终的预测

        for i in range(self.n_estimators):
            _, y_pre = self.forest[i].test(X, y,
                self.para_forest[i][0], self.para_forest[i][1], self.para_forest[i][2], self.para_forest[i][3])    # y_pre：shape：（，1）

            predict.append(y_pre)


        index_lrp = []
        index = 0
        for m in range(X.shape[0]):
            max = 0
            axis_x = y[m][0]  # 实际标签
            for j in range(self.n_estimators):
                axis_y = int(predict[j][m])  # 第j个编码器对第i个样本的预测标签
                t = self.probablity_train[j][axis_x][axis_y]  # 第j个编码器的 后验概率
                if t > max:
                    max = t
                    final_pre[m] = predict[j][m]
                    index = j      # 使用的哪个分类器的分类结果
            index_lrp.append(index)

        final_pre = final_pre.astype(np.int64)


        dir1 = './DHA_AOA_Bagging/'
        if not os.path.exists(dir1):
            os.mkdir(dir1)
        picturesdir = './DHA_AOA_Bagging/bagging_result_pictures/'
        if not os.path.exists(picturesdir):
            os.mkdir(picturesdir)

        fig = Visualization(y, final_pre, picturesdir)
        acc = fig.plot_confusion_matrix('DHA_AOA_bagging test HAR Confusion Matrix' )  # 画混淆矩阵
        # fig.tsne_precl(X, 'DHA_AOA_bagging Initial data')
        # fig.tsne_af(X,  'DHA_AOA_bagging Data after classification')
        fig.plot_roc('DHA_AOA_bagging ROC curve')  # 画roc图
        end_test = time.clock()
        print('test time: %s Seconds' % (end_test - start_test))

        return final_pre, index_lrp


if __name__ == '__main__':
    data = shujuchuli.cut_samples(0)
    train_x, train_y, test_x, test_y = shujuchuli.make_datasets(data)

    # train_x = train_x[:250, :]
    # train_y = train_y[:250, :]
    # val_x = val_x[:150, :]
    # val_y = val_y[:150, :]
    # test_x = test_x[:150, :]
    # test_y = test_y[:150, :]


    suiji_encoder = Bagging_encoder(n_estimators=3,
                                subsamples=2*train_x.shape[0]//3,
                                subfeatures=train_x.shape[1]//2,
                                train_step=300,
                                aoa_size=2, aoa_iter_num=20)
    encoder_li = suiji_encoder.fit(train_x, train_y)

    # 模型验证

    rf_test_pre, index_lrp = suiji_encoder.predict(test_x, test_y)


    if not os.path.exists('./DHA_AOA_Bagging/'):
        os.mkdir('./DHA_AOA_Bagging/')

    dir = './DHA_AOA_Bagging/data/'
    if not os.path.exists(dir):
        os.mkdir(dir)

    np.savetxt(dir + 'train.txt', train_x)
    np.savetxt(dir+ 'train_index.txt', train_y)

    np.savetxt(dir + 'test.txt', test_x)
    np.savetxt(dir + 'test_index.txt', test_y)

    dir1 = './DHA_AOA_Bagging/final_classifier/'
    if not os.path.exists(dir1):
        os.mkdir(dir1)

        # 保存在集成阶段每个样本所使用的分类器
    f1 = open('./DHA_AOA_Bagging/final_classifier/index.txt', 'w')
    print(index_lrp, file=f1)
    f1.close()

    # 保存在集成阶段确定的分类器组合
    f2 = open('./DHA_AOA_Bagging/final_classifier/DHA_list.txt', 'w')
    print(encoder_li, file=f2)
    f2.close()