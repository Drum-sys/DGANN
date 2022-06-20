import numpy as np
import scipy.io as sio
#import input_data
import pickle as pk
import random
import matplotlib.pyplot as plt
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
sess = tf.InteractiveSession()
Sess = tf.Session()
net_size = 90
fea_size = 240
out_sz = 64
x = tf.placeholder("float", [None, net_size*fea_size])
Apow = tf.placeholder("float", [None, net_size*net_size])
y_ = tf.placeholder("float", [None, 2])
data_dir = r'data/Data_At_multi2_NC_TLE.mat'
#data_dir = 'Huaxim.mat'
emptyTrain = 1

data1 = sio.loadmat(data_dir)
label1 = data1['label'] # size = N*2 N= Number of subjects
data2 = data1['tc_fea_all'] # node features (i.e., FC) size = N*21600
data3 = data1['Apow_all']#  Connections (i.e., SC) size = N*8100
data_dic = {'label':label1,'tc_fea':data2,'Apow': data3}

def weight_variable(shape, lam = 0.2):
    initial = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial)
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lam)(var))
    return var

def bias_variable(shape, lam=0.1):
    initial = tf.constant(0.1, shape=shape)
    var = tf.Variable(initial)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lam)(var))
    return var#tf.Variable(initial)

def data_next_batch(batch_size,batch_num,data_x1,data_x2,data_y):
    if(len(data_x1)!= len(data_y)):
        print ('size not ok!')
        return
    start = batch_size * batch_num
    end = start + batch_size
    if(end>=len(data_x1)):
        return data_x1[start:],data_x2[start:],data_y[start:]
    return data_x1[start:end],data_x2[start:end],data_y[start:end]

def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

# DCNN-step1
tc_fea = tf.reshape(x, [-1, net_size, fea_size])
A_net =  tf.reshape(Apow, [-1, net_size, net_size])
seq_fts = tf.layers.conv1d(tc_fea,out_sz, 1, use_bias=False)
f_1 = tf.layers.conv1d(seq_fts , 1, 1)
f_2 = tf.layers.conv1d(seq_fts , 1, 1)
logits = f_1 + tf.transpose(f_2, [0, 2, 1])
coefs1 = LeakyRelu(logits)
zero_vec = -9e15*tf.ones_like(coefs1)
coefs2 = tf.nn.softmax(tf.where(A_net > 0,coefs1,zero_vec))
combin1 = tf.matmul (coefs2, seq_fts)
Z_test_sp1 = tf.reshape (combin1, [-1, net_size,out_sz])
# DCNN-step2
seq_fts2 = tf.layers.conv1d(tc_fea,out_sz, 1, use_bias=False)
f_1_2 = tf.layers.conv1d(seq_fts2 , 1, 1)
f_2_2 = tf.layers.conv1d(seq_fts2 , 1, 1)
logits2 = f_1_2 + tf.transpose(f_2_2, [0, 2, 1])
coefs1_2 = LeakyRelu(logits2)
zero_vec_2 = -9e15*tf.ones_like(coefs1_2)
coefs2_2 = tf.nn.softmax(tf.where(A_net*A_net > 0,coefs1_2,zero_vec_2))
combin2 = tf.matmul (coefs2_2, seq_fts2)
Z_test_sp2 = tf.reshape (combin2, [-1, net_size,out_sz])
# Bil pooling
Z_test_sp_T = tf.transpose( Z_test_sp1, perm=[0, 2, 1])
phi_I = tf.matmul(Z_test_sp1, Z_test_sp_T)
phi_I = tf.reshape(phi_I, [-1, net_size * net_size])
y_sqrt = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))
z_l2 = tf.nn.l2_normalize(y_sqrt, dim=1)

Z_test_sp_T2 = tf.transpose( Z_test_sp2, perm=[0, 2, 1])
phi_I2 = tf.matmul(Z_test_sp2, Z_test_sp_T)
phi_I2 = tf.reshape(phi_I2, [-1, net_size * net_size])
y_sqrt2 = tf.multiply(tf.sign(phi_I2), tf.sqrt(tf.abs(phi_I2) + 1e-12))
z_l22 = tf.nn.l2_normalize(y_sqrt2, dim=1)
z_all = tf.concat([z_l2,z_l22],1)

# FC

fc_size = 60
fc_size2 = 2

#FC layer
W_fc1 = weight_variable([180*90,fc_size])
b_fc1 = bias_variable([fc_size])

h_pool1_flat = tf.reshape(z_all, [-1, 180*90])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#Output layer
W_fc2 = weight_variable([fc_size, fc_size2])
b_fc2 = bias_variable([fc_size2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#Train and evaluate
saver = tf.train.Saver()
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#Weight reg
tf.add_to_collection("losses",cross_entropy)
loss = tf.add_n(tf.get_collection("losses"))
train_step = tf.train.AdamOptimizer(5e-6).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()
if emptyTrain:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
else:
    res = saver.restore(sess, r'./Modelasdtest/modelxiao.ckpt')
    #print("Model restored. %s" % (res))
Iteration = 50
data_dic
#train_x = scale_to_01(np.array(data_dic['data']))
train_x1 = np.array(data_dic['tc_fea'])
train_x2 = np.array(data_dic['Apow'])

train_y = np.array(data_dic['label'])
#shuffle the train data
train = np.hstack((train_x1,train_x2,train_y))
train_list = train.tolist()
#random.shuffle(train_list)
train = np.array(train_list)
train_acc_fold = []
test_acc_fold = []
val_acc_fold = []
train_loss_fold = []
test_loss_fold = []
val_loss_fold = []
d1_fold =[]
d2_fold =[]
d3_fold =[]
d4_fold =[]
d5_fold =[]
d6_fold =[]
d7_fold =[]
init_op = tf.global_variables_initializer()
fsp = 19
#train = np.delete(train,[k for k in range(210,350)],axis = 0)
#train = np.delete(train,[k for k in range(280,350)],axis = 0)
all_num = len(train)
print (len(train))
fold_num = 10
acc_pend = 0.00
for j in range(fold_num):
    #j = 5
    #os.system("pause")
    if emptyTrain:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
    else:
        res = saver.restore(sess, r'./Modelasdresult/modelxiao.ckpt')
        # print("Model restored. %s" % (res))
    tr_idx = [k for k in range(all_num)]
    te_idx = [k for k in range(j*fsp,(j+1)*fsp)]
    va_idx = [k for k in range((j+1)%fold_num*fsp,(j+2)%fold_num*fsp)]
    tr_idx[j*fsp:(j+1)*fsp]=[]
    tr_idx[(j)%fold_num*fsp:(j+1)%fold_num*fsp]=[]
    if j == (fold_num-2):
        va_idx = [k for k in range((fold_num-1)*fsp,(fold_num)*fsp)]
    if j ==(fold_num-1):
        tr_idx[(j+1)%(fold_num)*fsp:(j+2)%(fold_num)*fsp]=[]
    train_x1 = train[tr_idx,0:21600]
    train_x2 = train[tr_idx, 21600:-2]
    print (len(train_x1))
    train_y = train[tr_idx,-2:]
    test_x1 = train[te_idx,0:21600]
    test_x2 = train[te_idx,21600:-2]
    test_y = train[te_idx,-2:]
    val_x1 = train[va_idx,0:21600]
    val_x2 = train[va_idx,21600:-2]
    val_y = train[va_idx,-2:]
    data_num = len(train_x1)
    batch_size = 150
    batch_num = data_num//batch_size
    train_acc_list=[]
    test_acc_list = []
    val_acc_list = []
    train_loss_list=[]
    test_loss_list = []
    val_loss_list = []
    x_iter_list = []
    d1_list = []
    d2_list = []
    d3_list = []
    d4_list = []
    d5_list = []
    d6_list = []
    d7_list = []
    #merged_summary_op = tf.merge_all_summaries()
    #summary_writer = tf.train.SummaryWriter('/tmp/lung_image_logs', sess.graph)
    ranlist = [i for i in range(0,net_size*net_size)]
    ran_num = 500
    for i in range(Iteration):

        if(i%10000 == 0):
            train_x1_ran = train_x1.copy()
            train_x1_ran[train_x1_ran < random.uniform(0, 0.8)] = 0
            train_x2_ran = train_x2.copy()
            train_x2_ran[train_x2_ran < random.uniform(0, 0.8)] = 0
            batch = data_next_batch(batch_size, i % batch_num, train_x1_ran, train_x2_ran, train_y)
            '''
            train_ran_list = random.sample(ranlist, 500)
            train_x_ran = train_x.copy()
            train_x_ran[:,train_ran_list] = 0
            batch = data_next_batch(batch_size, i % batch_num, train_x_ran, train_y)
            '''
        else:
            batch = data_next_batch(batch_size, i % batch_num, train_x1, train_x2, train_y)
        #batch1 = mnist.train.next_batch(50)
        if i%50 ==0:
            x_iter_list.append(i)
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], Apow:batch[1],y_:batch[2],keep_prob:1.0})
            print ("step %d, fold %d,trainning accuracy %g"%(i, j, train_accuracy))
            train_loss = cross_entropy.eval(feed_dict={
                x: batch[0], Apow: batch[1], y_: batch[2], keep_prob: 1.0})
            print(train_loss)
            train_acc_list.append(train_accuracy)
            train_loss_list.append(train_loss)
            test_accuracy = acc_pend+accuracy.eval(feed_dict={
                x:test_x1,Apow: test_x2, y_:test_y,keep_prob:1.0})
            print ("step %d, test accuracy %g"%(i, test_accuracy))
            test_acc_list.append(test_accuracy)
            test_loss = cross_entropy.eval(feed_dict={
                x: test_x1, Apow: test_x2,y_: test_y, keep_prob: 1.0})
            print(test_loss)
            test_loss_list.append(test_loss)
            val_accuracy = acc_pend+accuracy.eval(feed_dict={
                x:val_x1, Apow: val_x2,y_:val_y,keep_prob:1.0})
            print ("step %d, validation accuracy %g"%(i, val_accuracy))
            val_loss = cross_entropy.eval(feed_dict={
                x: val_x1,Apow: val_x2, y_: val_y, keep_prob: 1.0})
            print(val_loss)
            val_acc_list.append(val_accuracy)
            val_loss_list.append(val_loss)
            d1 = coefs2.eval (feed_dict={
                x:batch[0], Apow:batch[1],y_:batch[2],keep_prob:1.0})
            d2 = Z_test_sp1.eval (feed_dict={
                x:batch[0], Apow:batch[1],y_:batch[2],keep_prob:1.0})
            d3 = coefs2_2.eval (feed_dict={
                x:batch[0], Apow:batch[1],y_:batch[2],keep_prob:1.0})
            d4 = Z_test_sp2.eval (feed_dict={
                x:batch[0], Apow:batch[1],y_:batch[2],keep_prob:1.0})
            d5 = combin1.eval (feed_dict={
                x:batch[0], Apow:batch[1],y_:batch[2],keep_prob:1.0})
            d6 = combin2.eval (feed_dict={
                x:batch[0], Apow:batch[1],y_:batch[2],keep_prob:1.0})
            d7 = y_.eval (feed_dict={
                x:batch[0], Apow:batch[1],y_:batch[2],keep_prob:1.0})
            d1_list.append(d1)
            d2_list.append(d2)
            d3_list.append(d3)
            d4_list.append(d4)
            d5_list.append(d5)
            d6_list.append(d6)
            d7_list.append(d7)
            '''
            d1 = h_conv1.eval(feed_dict={
                x: val_x, y_: val_y, keep_prob: 1.0})
            #print('d1',d1)
            d2 = h_conv2.eval(feed_dict={
                x: val_x, y_: val_y, keep_prob: 1.0})
            d3 = h_fc1.eval(feed_dict={
                x: val_x, y_: val_y, keep_prob: 1.0})
            wf = W_fc2.eval()
            #print('d2', d3)
            #print('d3', d3)
            '''
            #summary_str = sess.run(merged_summary_op)
            #summary_writer.add_summary(summary_str, total_step)
            #saver.save(sess, "mnistnnsave1/model.ckpt")
        train_step.run(feed_dict={x: batch[0], Apow: batch[1], y_: batch[2], keep_prob:0.5})
    '''
    print(wf[1:90,0])
    print(max(wf[1]))
    dic = {'h_conv1_rsp': d1, 'h_conv2_rsp': d2, 'label': val_y,'x': val_x,'h_fc1': d3,'wf':wf}
    fff = open('tsnedatawf.pkl', 'wb')
    pk.dump(dic, fff)
    '''
    d1_fold.append(d1_list)
    d2_fold.append(d2_list)
    d3_fold.append(d3_list)
    d4_fold.append(d4_list)
    d5_fold.append(d5_list)
    d6_fold.append(d6_list)
    d7_fold.append(d7_list)
    train_acc_fold.append(train_acc_list)
    test_acc_fold.append(test_acc_list)
    val_acc_fold.append(val_acc_list)
    train_loss_fold.append(train_loss_list)
    test_loss_fold.append(test_loss_list)
    val_loss_fold.append(val_loss_list)
    saver.save(sess, r'./Modeltest/modelxiao.ckpt')

train_acc_list = []
test_acc_list = []
val_acc_list = []
dic = {'train':train_acc_fold,'test':test_acc_fold,'val':val_acc_fold,'trainl':train_loss_fold,'testl':test_loss_fold,'vall':val_loss_fold,
       'd1':d1_fold,'d2': d2_fold,'d3':d3_fold,'d4':d4_fold,'d5':d5_fold,'d6':d6_fold,'d7':d7_fold}
ff = open('At_NC_TLE.pkl','wb')
pk.dump(dic,ff)
ff.close()
tr_acc = np.array(train_acc_fold[0])
te_acc = np.array(test_acc_fold[0])
for k in range(1,10):
    tr_acc = tr_acc + train_acc_fold[k]
    te_acc = te_acc + test_acc_fold[k]
tr_acc = tr_acc/10.0
te_acc = te_acc/10.0
train_acc_list = tr_acc.tolist()
test_acc_list = te_acc.tolist()
y1 = train_acc_list
y2 = test_acc_list
x1 = x_iter_list
plt.plot(x1,y1,label = 'train acc',color = 'r')
plt.plot(x1,y2,label = 'val acc')
plt.xlabel('iteration num')
plt.ylabel('acc')
plt.legend(loc = 'upper right')
plt.show()

