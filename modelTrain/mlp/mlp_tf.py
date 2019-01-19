import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
in_units = 784 #输入节点数
h1_units = 300 #隐含层节点数

#************************************************自定义部分开始*************************************************#
#********************参数**********************#
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1)) #初始化隐含层权重W1，服从默认均值为0，标准差为0.1的截断正态分布
b1 = tf.Variable(tf.zeros([h1_units])) #隐含层偏置b1全部初始化为0

W2 = tf.Variable(tf.zeros([h1_units, 10]))   #
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])  #数据输入口
keep_prob = tf.placeholder(tf.float32) #Dropout失活率    ***dropout***
 #******************************************************#

#***************开始画图（计算图）***********#
#定义模型结构 

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)    #第一层  
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)   #加入dropout 
hidden2 = tf.matmul(hidden1_drop, W2) + b2   #第二层
y = tf.nn.softmax(hidden2)                   #softmax    ***softmax***

#训练部分定义
y_ = tf.placeholder(tf.float32, [None, 10])     #标签输入placehold
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  # 交叉熵作为误差的衡量
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)       # 参数优化算法选择与设置

saver = tf.train.Saver()

#*******************计算图结束*********************#
#**********************************************自定义部分结束***********************************************************#
#定义一个InteractiveSession会话并初始化全部变量
sess = tf.InteractiveSession()    # 会话窗口
tf.global_variables_initializer().run()  # 对参数进行初始化
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))   #
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #计算正确率

# 迭代训练
for i in range(3001):  #3000步迭代
    batch_xs, batch_ys = mnist.train.next_batch(100)   #获得要进行训练的数据   获取数据的方式要自己写
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})   # 前向传导+后向传导+更新参数
    # y.run({x:xxx})
    if i % 10 ==0:
		#训练过程每200步在测试集上验证一下准确率，动态显示训练过程
        print(i, 'training_arruracy:', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, 
                             keep_prob: 1.0}))
        saver.save(sess,save_path="./model/model.ckpt")
print('final_accuracy:', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



