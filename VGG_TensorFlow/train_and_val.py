import os
import numpy as np
import tensorflow as tf
import csv
from VGG_TensorFlow.tensorflow_vgg import vgg16
from sklearn.preprocessing import LabelBinarizer
from VGG_TensorFlow.tensorflow_vgg import utils
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.misc import imread, imresize

data_dir = './/flower_photos/'
contents = os.listdir(data_dir)  # os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
classes = [each for each in contents if os.path.isdir(data_dir + each)]  # 判断某一路径是否为文件路径
# 首先设置计算batch的值，如果该运算平台的内存越大，这个值可以设置的越高
batch_size = 10
# 用codes_list来存储特征值
codes_list = []
# 用labels来存储花的类别
labels = []
# batch数组用来临时存储图片数据
batch = []

codes = None
with tf.Session() as sess:
    # 构建VGG16模型对象
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope('content_vgg'):
        # 载入VGG16模型
        vgg.build(input_)
    # 对不同种类的花分别用VGG16计算特征值
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        # enumerate是python的内置函数，enumerate在字典上是枚举，列举的意思。对于一个可迭代的对象，enumerate将其组成一个索引序列
        # 利用它可以同时获得索引和值
        '''
        enumerate还可以接收第二个参数，用于指定索引起始值，如：
        list1 = ["这", "是", "一个", "测试"]
        for index, item in enumerate(list1, 1):
            print index, item
        >>>
        1 这
        2 是
        3 一个
        4 测试
        '''
        # 以下得到一个codes数组和一个labels数组，分别存储了所有花朵的特征值和类别
        for ii, file in enumerate(files, 1):
            # 载入图片并放入batch数组中
            img = utils.load_image(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(each)
            # 如果图片数量到了batch_size则开始具体的运算
            if ii % batch_size == 0 or ii == len(files):
                images = np.concatenate(batch)  # 实现多个数组的拼接[n.224,224,3]
                feed_dict = {input_: images}
                # 计算特征值
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)  # 到这里codes_batch的值为[n,4096]

                # 将结果放入到codes数组中
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))

                # 清空准备下一个batch的计算
                batch = []
                print('{} images processed'.format(ii))
# 用如下代码将这两个数组保存到硬盘上
with open('codes', 'w') as f:
    codes.tofile(f)
with open('labels', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)
#############################准备训练集，验证集和测试集#################################
# 将标签矩阵二值化
'''
关于labelBinarizer()的解释可参考
'''
lb = LabelBinarizer()
lb.fit(labels)
labels_vecs = lb.transform(labels)

# 抽取数据，不同类型的花的数据数量并不是完全一样，labels数组中的数据也还没有被打乱，最合适的方法是使用StratifiedShuffleSplit方法来
# 进行分层随机划分，假设我们使用训练集：验证集：测试集=8:1:1
'''
关于StratifiedShuffleSplit的解释参考网址http://blog.csdn.net/m0_38061927/article/details/76180541
'''
ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)  # 参数 n_splits是将训练数据分成train/test对的组数，这里只有一组
train_idx, val_idx = next(ss.split(codes, labels))

half_val_len = int(len(val_idx) / 2)
val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

train_x, train_y = codes[train_idx], labels_vecs[train_idx]
val_x, val_y = codes[val_idx], labels_vecs[val_idx]
test_x, test_y = codes[test_idx], labels_vecs[test_idx]

print('Train shapes(x,y):', train_x.shape, train_y.shape)
print("Validation shapes (x, y):", val_x.shape, val_y.shape)
print("Test shapes (x, y):", test_x.shape, test_y.shape)

###########################训练网络##########################################

# 我们在得到数据集的特征以后，我们又设计使用了一个256维的全连接层，一个5维的全连接层(因为数据集中给出的种类是五种不同种类的花朵)，和一
# 个softmax层，这里即你微调参数发挥的地方，网络结构可以任意修改，可以不断尝试其他的结构以找到最适合的结构

# 输入数据的维度
inputs_ = tf.placeholder(tf.float32, shape=[None, codes.shape[1]])
print(input_.shape)
print(codes.shape[1])
# 标签数据的维度
labels_ = tf.placeholder(tf.int64, shape=[None, labels_vecs.shape[1]])
print(labels_.shape)
print(labels_vecs.shape[1])
# 加入一个256维的全连接的层
fc = tf.contrib.layers.fully_connected(inputs_, 256)
# 加入一个5维的全连接层
logits = tf.contrib.layers.fully_connected(fc, labels_vecs.shape[1], activation_fn=None)
# 计算cross entropy的值
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)
# 计算损失函数
cost = tf.reduce_mean(cross_entropy)
# 采用用的最广泛的AdamOptimizer优化器
optimizer = tf.train.AdamOptimizer().minimize(cost)
# 得到最后的预测分布
predicted = tf.nn.softmax(logits)
# 计算准确度
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_batches(x, y, n_batches=10):
    """ 这是一个生成器函数，按照n_batches的大小将数据划分了小块 """
    batch_size = len(x) // n_batches

    for ii in range(0, n_batches * batch_size, batch_size):
        # 如果不是最后一个batch，那么这个batch中应该有batch_size个数据
        if ii != (n_batches - 1) * batch_size:
            X, Y = x[ii: ii + batch_size], y[ii: ii + batch_size]
            # 否则的话，那剩余的不够batch_size的数据都凑入到一个batch中
        else:
            X, Y = x[ii:], y[ii:]
        # 生成器语法，返回X和Y
        yield X, Y

# 运行的轮次数
epochs = 20
# 统计训练效果的频率
iteration = 0
# 保存模型的保存器
saver=tf.train.Saver()
ckpt_dir = './checkpoints/flowers.ckpt'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        # for x,y in get_batches(train_x,train_y):
        feed={inputs_:train_x,
              labels_:train_y}
        #训练模型
        loss,_=sess.run([cost,optimizer],feed_dict=feed)
        print("Epoch: {}/{-}".format(e + 1, epochs),
              "Iteration: {}".format(iteration),
              "Training loss: {:.5f}".format(loss))
        iteration += 1
        if iteration % 5 == 0:
            feed = {inputs_: val_x,
                    labels_: val_y}
            val_acc = sess.run(accuracy, feed_dict=feed)
            # 输出用验证机验证训练进度
            print("Epoch: {}/{}".format(e, epochs),
                  "Iteration: {}".format(iteration),
                  "Validation Acc: {:.4f}".format(val_acc))
            # 保存模型
        saver.save(sess, ckpt_dir)
#测试网络
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
#
#     feed = {inputs_: test_x,
#             labels_: test_y}
#     test_acc = sess.run(accuracy, feed_dict=feed)
#     print("Test accuracy: {:.4f}".format(test_acc))
#    #

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    img1 = imread('.//test_data/rouse_test.jpeg', mode='RGB')
    img1 = imresize(img1, (224, 224))

    feed = {input_  : [img]}
    prob = sess.run(predicted, feed_dict=feed)[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(classes[p], prob[p])
