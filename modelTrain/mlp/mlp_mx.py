import mxnet as mx
from mxnet import nd,gluon,init,autograd
from mxnet.gluon import loss as gloss,nn
from tensorflow.examples.tutorials.mnist import input_data
import d2lzh as d2l
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

in_units = 784 #输入节点数
h1_units = 300 #隐含层节点数


# w1 = nd.random.normal(scale=0.01,shape=[in_units,256])
# b1 = nd.zero(256)
# w2 = nd.random.norml(scale=0.01,shape=[256,10])
# b2 = nd.zero(10)
# params = [w1,b1,w2,b2]


# for param in params:
#     param.attach_grad()

# def relu(x):
#     return nd.maximum(x,0)

# def net(x):
#     x = x.reshape((-1,in_units))
#     H =relu(nd.dot(x,w1)+b1)
#     return nd.dot(H,w2)+b2


# loss = gloss.SoftmaxCrossEntropyLoss()

# num_epochs,lr = 5,0.5

net = nn.HybridSequential()
net.add(nn.Dense(256,activation="relu"),
        nn.Dense(10)
        )
net.initialize(init.Normal(sigma=0.01))
net.hybridize()
loss = gloss.SoftmaxCrossEntropyLoss()

batch_size = 64


# trainer = gluon.Trainer(net.collect_params(),"sgd",{"learning_rate":0.01})
trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'wd': 0.0005, 'momentum': 0.9, "learning_rate":0.01},
        kvstore='local')
num_epochs = 5000
def train_ch3(net,train_iter,test_iter,loss,num_eopchs,batch_size,params=None,lr=None,trainer=None):
    train_l_sum,train_acc_sum,n = 0.0,0.0,0.0
    for epoch in range(num_epochs):  
        for X,y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
        l.backward()
        if trainer is None:
            d2l.sgd(params,lr,batch_size)
        else:
            trainer.step(batch_size)
        y = y.astype("float32")
        train_l_sum+=l.asscalar()
        # train_acc_sum +=
        n +=y.size
    print("epoch %d, loss %.4f,train_acc %.3f"%(epoch+1,train_l_sum /n))

for epoch in range(num_epochs):
    train_l_sum,train_acc_sum,n = 0.0,0.0,0.0
    X,y  = mnist.train.next_batch(batch_size)
    X,y = nd.array(X),nd.array(y)
    print(X.max())
    # print(y.shape)
    with autograd.record():
        y_hat = net(X)
        l = loss(y_hat,y).sum()
    l.backward()
    if trainer is None:
        d2l.sgd(params,lr,batch_size)
    else:
        trainer.step(batch_size)
    y = y.astype("float32")
    train_l_sum+=l.asscalar()
    n +=y.size
    print("epoch %d, loss %.4f"%(epoch+1,train_l_sum /n))
net.export("my_mlp")
print("ok!")
