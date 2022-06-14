from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import *
import random
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Visualization():
    def __init__(self, train_y, predict_y, picturesdir):
        self.train_y = train_y
        self.predict_y = predict_y
        self.picturesdir = picturesdir
        self.labels_name = ['normal', 'outer_25', 'outer_50', 'outer100', 'outer_150', 'outer_200', 'outer_250', 'outer_300',
                       'Inner_0', 'Inner_50', 'Inner_100', 'Inner_150', 'Inner_200', 'Inner_250', 'Inner_300']
        # self.labels_name = ['Ball14', 'Ball21', 'Ball7', 'IR14', 'IR21',
        #                    'IR7', 'Normal', 'OR14', 'OR21', 'OR7']
        # self.labels_name = ['inner_1000', 'inner_600', 'inner_800', 'normal_1000','normal_600','normal_800',
        #                     'outer_1000', 'outer_600', 'outer_800', 'ball_1000', 'ball_600', 'ball_800']

        self.colors = cycle(['aqua', 'orchid', 'yellow', 'blue', 'brown',
                         'green', 'gray', 'orange', 'maroon', 'red'])

    def plot_confusion_matrix(self, title):
        '''
        title：混淆矩阵图的标题
        '''
        from sklearn.metrics import f1_score

        cm = confusion_matrix(self.train_y, self.predict_y)
        print(cm)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(True, cmap='viridis', xticks_rotation=15)
        plt.show()


        # 均方误差是指参数估计值与参数真值之差平方的期望值; 它可以评价数据的变化程度，
        # 当MSE的值越小时也就说明预测模型描述实验数据具有更好的精确度
        mse = mean_squared_error(self.train_y, self.predict_y)
        print('mean_squared_error：', mse)
        acc = accuracy_score(self.train_y, self.predict_y)
        print('accuracy：', acc)
        rec_w = recall_score(self.train_y, self.predict_y, average='macro')
        print('recall score (macro)：', rec_w)
        f1_w = f1_score(self.train_y, self.predict_y, average='macro')
        print('f1_score (macro)：', f1_w)
        precision_w = precision_score(self.train_y, self.predict_y, average='macro')
        print('precision_score (macro)：', precision_w)
        # ' macro '  ： 相当于类间不带权重。不考虑类别数量，不适用于类别不均衡的数据集，其计算方式为： 各类别的P求和/类别数量
        # ' weighted ' : 相当于类间带权重。各类别的P × 该类别的样本数量（实际值而非预测值）/ 样本总数量


        # # 画混淆矩阵图像
        # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
        # plt.figure()
        # plt.imshow(cm_normalized, interpolation='nearest')  # 在特定的窗口上显示图像
        # plt.title(title)  # 图像标题
        # plt.colorbar()
        #
        # num_label = np.array(range(self.fault_num))
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        #
        # # 给图中某个点加标签 https://blog.csdn.net/qq_36982160/article/details/80038380
        # ind_array = np.arange(self.fault_num)  # 生成array
        # x, y = np.meshgrid(ind_array, ind_array)  # 生成网格点坐标矩阵
        # for x_val, y_val in zip(x.flatten(), y.flatten()):  # flatten()返回一个一维数组，默认按行的方向降维
        #     c = cm_normalized[y_val][x_val]
        #     if c > 0.01:
        #         plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
        # plt.savefig(self.picturesdir + title)
        # plt.show()

        # 求四大指标 TP: true positive ：分类正确，预测为正，实际为正  TN : 预测为负，实际为负
        # FP: 分类错误，预测为正，实际为负   FN：预测为负，实际为正
        # https://blog.csdn.net/sihailongwang/article/details/77527970
        return acc


    # 画多分类roc曲线： https://blog.csdn.net/NockinOnHeavensDoor/article/details/83384844
    def plot_roc(self, title):

        # 对标签进行二进制编码
        if self.train_y.shape[1] == 1:
            self.train_y = label_binarize(self.train_y, classes=[i for i in range(len(self.labels_name))])
        if self.predict_y.shape[1] == 1:
            self.predict_y = label_binarize(self.predict_y, classes=[i for i in range(len(self.labels_name))])

        n_classes = self.train_y.shape[1]
        # Compute ROC curve and ROC area for each class
        fpr = dict()   # 假阳性率，分类错误，预测为正，实际为负
        tpr = dict()   # 真阳性率，分类正确， 预测为正，实际为正
        roc_auc = dict()
        # n_classes = self.train_y.shape[1]
        for i in range(n_classes):
            # 取出来的各个类的测试值和预测值， curve：曲线
            # roc_curve： https://blog.csdn.net/u014264373/article/details/80487766
            # auc：利用梯形法则计算曲线下的面积(AUC)。
            fpr[i], tpr[i], _ = roc_curve(self.train_y[:, i], self.predict_y[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 计算微平均ROC曲线（micro-average ROC curve）和ROC面积
        # 类总和的基础上平均的ROC和AUC， ravel：降到一维
        fpr["micro"], tpr["micro"], _ = roc_curve(self.train_y.ravel(), self.predict_y.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # 计算宏平均ROC曲线（macro-average ROC curve）和ROC面积
        # 首先汇总所有的假阳性率
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))  # 拼接数组
        # 然后在这一点上插值所有ROC曲线
        mean_tpr = np.zeros_like(all_fpr)  # 输出和all_fpr形状一样的全零矩阵
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            # interp（待插入数据的横坐标，原始数据的横坐标，原始数据的纵坐标）: 一维线性插值，返回离散数据的一维分段线性插值结果
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw = 2  # linewidth
        plt.figure()
        # plt.plot(fpr["micro"], tpr["micro"],
        #          label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["micro"]),
        #          color='pink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)


        for i, color in zip(range(n_classes), self.colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right", fontsize=8)
        plt.savefig(self.picturesdir + title)
        plt.show()

    # 画 R 训练过程中的变化曲线图
    def plot_R(self, R_history, split_num):
        '''
        :param R_history: 每训练一次，保存一次半径值，保存在列表中
        :param split_num: 每个类别的数据的个数，列表
        :return:
        '''
        R_history = np.array(R_history)
        plt.figure()
        plt.title('radius')
        for i in range(len(split_num)):
            plt.plot(R_history[:, i])

        plt.show()

    # 画 loss 训练过程中的变化曲线图
    def plot_loss(self, loss_list):
        '''
        :param loss_list: 每训练一次，保存一次损失函数值，保存在列表中
        :return:
        '''
        loss_history = np.array(loss_list)
        plt.figure()
        plt.plot(loss_history)
        plt.title('loss history')
        plt.show()





    # https://www.cnblogs.com/mliu222/p/11920380.html
    def pca_2D(self, data):
        pca = PCA(n_components=2)
        pca_dataPre = pca.fit_transform(data)
        plt.figure()


        for i in range(len(self.labels_name)):
            sp = np.argwhere(self.predict_y == i)
            sp = sp[:, 0]
            plt.scatter(pca_dataPre[sp, 0], pca_dataPre[sp, 1], marker='o',  alpha=0.4)
            # alpha:控制点的透明度，s控制点的大小

        plt.legend(self.labels_name,loc="lower right", fontsize=8)
        plt.show()



    def tsne_3d(self, data):  # 训练前的数据可视化

        model = TSNE(n_components=3)
        embs = model.fit_transform(data)  # 进行数据降维，并返回结果


        # '''嵌入空间可视化'''
        # x_min, x_max = embs.min(0), embs.max(0)
        # X_norm = (embs - x_min) / (x_max - x_min)  # 归一化

        # 三维绘图 https://blog.csdn.net/u014636245/article/details/82799573
        ax = Axes3D(plt.figure())
        for i in range(len(self.labels_name)):
            indices = np.argwhere(self.train_y == i).flatten()
            ax.scatter(embs[indices, 0], embs[indices, 1], embs[indices, 2], marker='o', alpha=0.4)

        plt.legend(self.labels_name, loc="lower right", fontsize=8)
        plt.show()





        # 采用PCA画训练后的点
    def pca_aff_2D(self, data, circle, R_list):

        pca = PCA(n_components=2)
        pca_data_aff = pca.fit_transform(data)
        pca = PCA(n_components=2)
        pca_circle = pca.fit_transform(circle)

        theta = np.arange(0, 2 * np.pi, 0.01)
        plt.figure()

        try:
            for i in range(len(self.labels_name)):
                indices = len(np.argwhere(self.train_y == i))
                plt.scatter(pca_data_aff[indices, 0], pca_data_aff[indices, 1])
                x = pca_circle[i][0] + R_list[i] * np.cos(theta)
                y = pca_circle[i][1] + R_list[i] * np.sin(theta)
                plt.plot(x, y, c='b')

        except:
            print('可能颜色不够用了!')
        pass





    def tsne_afcl(self, data, circle, R, title):  # 分类后的数据可视化
        from sklearn.manifold import TSNE

        model = TSNE(n_components=3)
        embs = model.fit_transform(data)  # 进行数据降维，并返回结果
        embs_circle = model.fit_transform(circle)

        '''嵌入空间可视化'''
        x_min, x_max = embs.min(0), embs.max(0)
        X_norm = (embs - x_min) / (x_max - x_min)  # 归一化
        circle_min, circle_max = embs_circle.min(0), embs_circle.max(0)
        circle_norm = (embs_circle - circle_min) / (circle_max - circle_min)  # 归一化


        # https://my.oschina.net/ahaoboy/blog/1827281
        u = np.linspace(0, 2 * np.pi, 100)  # 在0-2pi之间返回均匀间隔的100个数据点
        v = np.linspace(0, np.pi, 100)
        ax = Axes3D(plt.figure())
        for i, color in zip(range(len(self.labels_name)), self.colors):

            # 经过编码器处理后的数据可视化
            indices = np.argwhere(self.predict_y == i).flatten()
            ax.scatter(X_norm[indices, 0], X_norm[indices, 1], X_norm[indices, 2], marker='o', alpha=0.5, c=color, s=20)

            x = R[i] * np.outer(np.cos(u), np.sin(v)) + circle_norm[i][0]
            y = R[i] * np.outer(np.sin(u), np.sin(v)) + circle_norm[i][1]
            z = R[i] * np.outer(np.ones(np.size(u)), np.cos(v)) + circle_norm[i][2]
            ax.plot_wireframe(x, y, z, rstride=10, cstride=10)  # 画线框图

        # ax.legend(labels=self.labels_name, loc="upper left")
        plt.savefig(self.picturesdir + title)
        plt.show()





