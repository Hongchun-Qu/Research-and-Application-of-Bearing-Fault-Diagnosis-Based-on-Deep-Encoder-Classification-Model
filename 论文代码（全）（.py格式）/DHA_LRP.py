import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
import os
import heapq


'''
输入的参数
'''
cda_dir = './DHA_CDA_jnu/model'
sda_dir = './DHA_SDA_jnu/model'
dda_dir = './DHA_DDA_jnu/model'

cda_scope = 'dha_cda'
sda_scope = 'dha_sda'
dda_scope = 'dha_dda'

cda_collection_weights = 'LRP_CDA_weights'
sda_collection_weights = 'LRP_SDA_weights'
dda_collection_weights = 'LRP_DDA_weights'

cda_collection_activations = 'LRP_CDA_activations'
sda_collection_activations = 'LRP_SDA_activations'
dda_collection_activations = 'LRP_DDA_activations'


'''
保存结果的位置
'''
if not os.path.exists('./DHA_Bagging_jnu/'):
    os.mkdir('./DHA_Bagging_jnu/')
if not os.path.exists('./DHA_Bagging_jnu/LRP_Picture/'):
    os.mkdir('./DHA_Bagging_jnu/LRP_Picture/')
if not os.path.exists('./DHA_Bagging_jnu/LRP_data/'):
    os.mkdir('./DHA_Bagging_jnu/LRP_data/')


clrp_dir = './DHA_Bagging_jnu/LRP_Picture/CDA/'
clrp_datadir = './DHA_Bagging_jnu/LRP_data/CDA/'
if not os.path.exists(clrp_dir):
    os.mkdir(clrp_dir)
if not os.path.exists(clrp_dir):
    os.mkdir(clrp_dir)

slrp_dir = './DHA_Bagging_jnu/LRP_Picture/SDA/'
slrp_datadir = './DHA_Bagging_jnu/LRP_data/SDA/'
if not os.path.exists(slrp_dir):
    os.mkdir(slrp_dir)
if not os.path.exists(slrp_dir):
    os.mkdir(slrp_dir)

dlrp_dir = './DHA_Bagging_jnu/LRP_Picture/DDA/'
dlrp_datadir = './DHA_Bagging_jnu/LRP_data/DDA/'
if not os.path.exists(dlrp_dir):
    os.mkdir(dlrp_dir)
    if not os.path.exists(dlrp_dir):
        os.mkdir(dlrp_dir)


class LayerwiseRelevancePropagation:

   def __init__(self, dir, scope, activations, weight, tree_i):

         self.epsilon = 1e-10
         self.tree_i = tree_i
         self.dir = dir
         self.scope = scope
         self.collection_weights = weight
         self.collection_activations = activations


         with tf.Session() as sess:
             sess.run(tf.global_variables_initializer())
             saver = tf.train.import_meta_graph(self.dir + '_' + str(self.tree_i) + '.meta')
             saver.restore(sess, self.dir + '_' + str(self.tree_i))

             self.weights = tf.get_collection(self.collection_weights + str(tree_i))
             self.activations = tf.get_collection(self.collection_activations + str(tree_i))
             self.x = self.activations[0]

         self.relevances = self.get_relevances()


   def get_relevances(self):   # 得到相关性
        relevances = [self.activations[-1], ]
        for i in range(len(self.activations)-2, -1, -1):
            relevances.append(self.backprop_fc(i, self.activations[i], relevances[-1]))

        return relevances

   def backprop_fc(self, name, activation, relevance):   # 全连接层 fc：fully connected layer
          w = self.weights[name]
          w_pos = tf.maximum(np.float64(0), w)   # 返回 0 和 w 的最大值， maximum（x,y） y必须和x具有相同的类型
          z = tf.matmul(activation, w_pos) + self.epsilon
          s = relevance / z
          c = tf.matmul(s, tf.transpose(w_pos))
          return c * activation

   def get_heatmap(self, x, y, digit):

          samples, index = self.get_samples(x, y, digit=digit)
          with tf.Session() as sess:
              sess.run(tf.global_variables_initializer())

              heatmap = sess.run(self.relevances[-1], feed_dict={self.x: samples})
              heatmap /= heatmap.max()

              # mean_heat = np.mean(heatmap)
              # heatmap = np.where(heatmap <= mean_heat, 0, heatmap)

          return heatmap, index

   def get_samples(self, x, y, digit):   # 得到每种标签的样本

          samples_indices = np.argwhere(y == digit).flatten()
          # samples_indices1 = np.where(y == digit)
          # np.argwhere(a):返回非0的数组元组的索引，其中a是要索引数组的条件。
          # np.flatten(): 返回一个折叠成一维的数组，只适用于numpy对象
          shuffle(samples_indices)
          return x[samples_indices], samples_indices
      # np.random.choice(a, size=None, replace=True, p=None):从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
      # a :1-D array-like or int, 如果是ndarray，则从它的元素生成一个随机样本。If an int, the random sample is generated as if a were np.arange(a)
      # replace:True表示可以取相同数字，False表示不可以取相同数字
      # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。


   def test_lrp(self, x, y):
        digit = np.random.choice(15)
        samples, _ = self.get_samples(x, y, digit)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # saver = tf.train.import_meta_graph(self.dir + '_' + str(self.tree_i) + '.meta')
            # saver.restore(sess, self.dir + '_' + str(self.tree_i))

            R = sess.run(self.relevances, feed_dict={self.x: samples})   # R：相关性
            for r in R:   # 每一层的相关性
                print(r.sum())   # 每一层的相关性求和

                # # 记录相关性大于0.001的特征的索引
                # important_feature_index =[]
                # for i in range(r.shape[0]):
                #     for j in range(r.shape[1]):
                #         if r[i][j] > 0.001:
                #             if j not in important_feature_index:
                #                 important_feature_index.append(j)
                # print(important_feature_index)


if __name__ == '__main__':

    train_x = np.loadtxt('./DHA_Bagging_jnu/Partition_dataset/train_JNU.txt')
    train_y = np.loadtxt('./DHA_Bagging_jnu/Partition_dataset/train_index_JNU.txt')
    test_x = np.loadtxt('./DHA_Bagging_jnu/Partition_dataset/test_JNU.txt')
    test_y = np.loadtxt('./DHA_Bagging_jnu/Partition_dataset/test_index_JNU.txt')

    with open('./DHA_Bagging_jnu/final_classifier/index_JNU.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')  # line为字符串
            DHA_bagging_pre = list(eval(line))  # 转换为列表

    with open('./DHA_Bagging_jnu/final_classifier/DHA_list_JNU.txt', 'r') as f:
        f1 = f.read().strip('\n')
        dha_1 = f1.split(',')[0].split('.')[0]
        dha_2 = f1.split(',')[1].split('.')[0]
        dha_3 = f1.split(',')[2].split('.')[0]
        DHA_list = [dha_1, dha_2, dha_3]


    para_cda = [cda_dir, cda_scope, cda_collection_activations, cda_collection_weights]
    para_sda = [sda_dir, sda_scope, sda_collection_activations, sda_collection_weights]
    para_dda = [dda_dir, dda_scope, dda_collection_activations, dda_collection_weights]

    dir_cda = [clrp_dir, clrp_datadir]
    dir_sda = [slrp_dir, slrp_datadir]
    dir_dda = [dlrp_dir, dlrp_datadir]

    heat = []
    heat_index =[]
    for i, dha in enumerate(DHA_list):
        if 'DHA_CDA' in dha:
            lrp_para = para_cda
            lrp_dir = dir_cda
        if 'DHA_SDA' in dha:
            lrp_para = para_sda
            lrp_dir = dir_sda
        if 'DHA_DDA' in dha:
            lrp_para = para_dda
            lrp_dir = dir_dda

        lrp_da = LayerwiseRelevancePropagation(lrp_para[0], lrp_para[1], lrp_para[2], lrp_para[3], i)
        lrp_da.test_lrp(test_x, test_y)

        h = []
        ind = []
        for digit in range(15):
            heatmap, index = lrp_da.get_heatmap(test_x, test_y, digit)
            h.append(heatmap)
            ind.append(index)

        heat.append(h)
        heat_index.append(ind)

    # labels_name = ['Ball14', 'Ball21', 'Ball7', 'IR14', 'IR21', 'IR7', 'Normal', 'OR14', 'OR21', 'OR7']
    # labels_name = ['inner_1000', 'inner_600', 'inner_800', 'normal_1000','normal_600','normal_800',
    #                     'outer_1000', 'outer_600', 'outer_800', 'ball_1000', 'ball_600', 'ball_800']
    labels_name = ['normal', 'outer_25', 'outer_50', 'outer100', 'outer_150', 'outer_200', 'outer_250',
                        'outer_300', 'Inner_0', 'Inner_50', 'Inner_100', 'Inner_150', 'Inner_200', 'Inner_250', 'Inner_300']

    for j in range(15):

        fina_heat = np.zeros(shape=[heat[0][j].shape[0], heat[0][j].shape[1]])
        for jj in range(len(heat[0][j])):
            target_sample = heat_index[0][j][jj]  # 第0个编码器的第j类样本的第jj个样本
            target_class = DHA_bagging_pre[target_sample]  # 用了第几个分类器的分类结果

            target_class_index = np.where(heat_index[target_class][j]==target_sample)  # 找到第num_cla个分类器的第j类故障的index_name在rel_index中的位置
            fina_heat[jj] = heat[target_class][j][target_class_index]  # 样本的排列顺序和第0个编码器的相同

        # 可视化所有的相关性得分
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.axis('off')

        ax.set_title(str(labels_name[j]) + '_1_heatmap')
        # imshow函数用于绘制热图，cmap：将标量数据映射到色彩图，interpolation：str，使用的插值方法
        ax.imshow(fina_heat, cmap='Reds', interpolation='bilinear')
        plt.show()

        resdir = './DHA_Bagging_jnu/LRP_Picture/DHA_bagging/'
        if not os.path.exists(resdir):
            os.mkdir(resdir)
        fig1.savefig(resdir + str(labels_name[j]) + '_ALL_rule.jpg')


        # # 只可视化五十个
        # # 求均值，令小于均值的元素等于0
        # mean_heat = np.mean(fina_heat)
        # fina_heatmap = np.where(fina_heat <= mean_heat, 0, fina_heat)
        #
        # # 取前100个影响样本量最多的特征
        # fina_heatmap = fina_heatmap.transpose()  # 进行转置，则特征变为行
        #
        # # 计算共有多少样本与该特征有关
        # exist = (fina_heatmap > 0) * 1.0
        # factor = np.ones(fina_heatmap.shape[1])
        # num_li = np.dot(exist, factor)
        # num_li = np.argsort(num_li)  # 返回值是数组中的元素升序排列后的原下标,返回一维数组，则后面50个元素为最大值的下标
        # fina_heatmap[num_li[-101::-1]] = 0  # 把指定列的数字变为0
        # fina_heatmap = fina_heatmap.transpose()

        sum_heat = np.sum(fina_heat, axis=0)
        num_li = np.argsort(sum_heat)
        fina_heatmap = fina_heat.transpose()
        fina_heatmap[num_li[-201::-1]] = 0  # 把指定列的数字变为0
        fina_heatmap = fina_heatmap.transpose()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        # ax.set_xlabel('feature', fontsize=8)
        # ax.set_ylabel('data', fontsize=8)
        ax.set_title(str(labels_name[j]) + '_heatmap')
        # imshow函数用于绘制热图，cmap：将标量数据映射到色彩图，interpolation：str，使用的插值方法
        ax.imshow(fina_heatmap, cmap='Reds', interpolation='bilinear')
        plt.show()

        resdir = './DHA_Bagging_jnu/LRP_Picture/DHA_bagging/'
        if not os.path.exists(resdir):
            os.mkdir(resdir)
        fig.savefig(resdir + str(labels_name[j]) + '_MAX100_rule.jpg')




