import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math 
import numpy as np

eta = 0.5

input1 = tf.placeholder(np.float32, (1, 1), name='i1')
input2 = tf.placeholder(np.float32, (1, 1), name='i2')
output1 = tf.placeholder(np.float32, (1, 1), name='o1')
output2 = tf.placeholder(np.float32, (1, 1), name='o2')


w1 =  tf.Variable(0.15, name='w1')
w2 =  tf.Variable(0.20, name='w2')
w3 =  tf.Variable(0.25, name='w3')
w4 =  tf.Variable(0.30, name='w4')
b1 =  tf.Variable(0.35, name='b1')

w5 =  tf.Variable(0.40, name='w5')
w6 =  tf.Variable(0.45, name='w6')
w7 =  tf.Variable(0.50, name='w7')
w8 =  tf.Variable(0.55, name='w8')
b2 =  tf.Variable(0.60, name='b2')


# ## FeedForwared


h1net = tf.multiply(w1, input1) + tf.multiply(w2, input2) + b1
h2net = tf.multiply(w3, input1) + tf.multiply(w4, input2) + b1
h1out = tf.nn.sigmoid(h1net)
h2out = tf.nn.sigmoid(h2net)

o1net = tf.multiply(w5, h1out) + tf.multiply(w6, h2out) + b2
o2net =  tf.multiply(w7, h1out) + tf.multiply(w8, h2out) + b2
o1out = tf.nn.sigmoid(o1net)
o2out = tf.nn.sigmoid(o2net)


Eo1 = 0.5*(output1 - o1out)**2
Eo2 = 0.5*(output2 - o2out)**2
Etotal = Eo1 + Eo2


# ## BackPropagation


del_Etotal_o1out = - (output1 - o1out)
del_o1out_o1net = o1out * (1 - o1out)
del_o1net_w5 = h1out
del_Etotal_w5 = del_Etotal_o1out * del_o1out_o1net * del_o1net_w5

del_o1net_w6 = h2out
del_Etotal_w6 = del_Etotal_o1out * del_o1out_o1net * del_o1net_w6

del_Etotal_o2out = - (output2 - o2out)
del_o2out_o2net = o2out * (1 - o2out)
del_o2net_w7 = h1out
del_Etotal_w7 = del_Etotal_o2out * del_Etotal_o2out * del_o2net_w7

del_o2net_w8 = h2out
del_Etotal_w8 = del_Etotal_o2out * del_Etotal_o2out * del_o2net_w8



del_o1net_h1out = w5
del_Eo1_o1out = - (output1 - o1out)
del_Eo1_o1net = del_Eo1_o1out * del_o1out_o1net
del_Eo1_h1out = del_Eo1_o1net * del_o1net_h1out

del_o2net_h1out = w7
del_Eo2_o2out = - (output2 - o2out)
del_Eo2_o2net = del_Eo2_o2out * del_o2out_o2net
del_Eo2_h1out = del_Eo2_o2net * del_o2net_h1out

del_Etotal_h1out = del_Eo1_h1out + del_Eo2_h1out

del_h1net_w1 = input1
del_h1out_h1net = h1out * (1 - h1out)
del_Etotal_w1 = del_Etotal_h1out * del_h1out_h1net * del_h1net_w1 

del_h1net_w2 = input2
del_Etotal_w2 = del_Etotal_h1out * del_h1out_h1net * del_h1net_w2

del_o1net_h2out = w6
del_Eo1_h2out = del_Eo1_o1net * del_o1net_h2out

del_o2net_h2out = w8
del_Eo2_h2out = del_Eo2_o2net * del_o2net_h2out

del_Etotal_h2out = del_Eo1_h2out + del_Eo2_h2out

del_h2net_w3 = input1
del_h2out_h2net = h2out * (1 - h2out)
del_Etotal_w3 = del_Etotal_h2out * del_h2out_h2net * del_h2net_w3

del_h2net_w3 = input2
del_Etotal_w4 =  del_Etotal_h2out * del_h2out_h2net * del_h2net_w3
 
w1New = w1 - eta*del_Etotal_w1
w2New = w2 - eta*del_Etotal_w2
w3New = w3 - eta*del_Etotal_w3
w4New = w4 - eta*del_Etotal_w4


w5New = w5 - eta*del_Etotal_w5
w6New = w6 - eta*del_Etotal_w6
w7New = w7 - eta*del_Etotal_w7
w8New = w8 - eta*del_Etotal_w8



init = tf.global_variables_initializer()


inp_out_dict = {input1 : np.asarray([0.05]).reshape(1,1),\
 input2 : np.asarray([0.10]).reshape(1,1),
                output1 : np.asarray([0.01]).reshape(1,1),\
                 output2 : np.asarray([0.99]).reshape(1,1)}



def update_WB(sess, w1a, w2a, w3a, w4a, w5a, w6a, w7a, w8a, b1a, b2a):
    global w1, w2, w3, w4, w5, w6, w7, w8, b1, b2
    sess.run(w1.assign(sess.run(w1a , feed_dict=inp_out_dict)[0][0]))
    sess.run(w2.assign(sess.run(w2a , feed_dict=inp_out_dict)[0][0]))
    sess.run(w3.assign(sess.run(w3a , feed_dict=inp_out_dict)[0][0]))
    sess.run(w4.assign(sess.run(w4a , feed_dict=inp_out_dict)[0][0]) )
    sess.run(w5.assign(sess.run(w5a , feed_dict=inp_out_dict)[0][0]))
    sess.run(w6.assign(sess.run(w6a , feed_dict=inp_out_dict)[0][0] ))
    sess.run(w7.assign(sess.run(w7a , feed_dict=inp_out_dict)[0][0] ))
    sess.run(w8.assign(sess.run(w8a , feed_dict=inp_out_dict)[0][0] ))
    sess.run(b1.assign(sess.run(b1a , feed_dict=inp_out_dict)))
    sess.run(b2.assign(sess.run(b2a , feed_dict=inp_out_dict)))


def print_WB(sess):
    global w1, w2, w3, w4, w5, w6, w7, w8, b1, b2
    print('W1 : ',sess.run(w1 , feed_dict=inp_out_dict))
    print('W2 : ',sess.run(w2 , feed_dict=inp_out_dict)) 
    print('W3 : ',sess.run(w3 , feed_dict=inp_out_dict)) 
    print('W4 : ',sess.run(w4 , feed_dict=inp_out_dict)) 
    print('W5 : ',sess.run(w5 , feed_dict=inp_out_dict)) 
    print('W6 : ',sess.run(w6 , feed_dict=inp_out_dict)) 
    print('W7 : ',sess.run(w7 , feed_dict=inp_out_dict)) 
    print('W8 : ',sess.run(w8 , feed_dict=inp_out_dict)) 
    print('B1 : ',sess.run(b1 , feed_dict=inp_out_dict)) 
    print('B2 : ',sess.run(b2 , feed_dict=inp_out_dict))


print('Weights and Biases before Backpropagation ')

with tf.Session() as sess:
    sess.run(init)
    print_WB(sess)
    print('Total error before Backpopagation is %.6f'%\
    (sess.run(Etotal, feed_dict = inp_out_dict)))

    update_WB(sess, w1New, w2New, w3New, w4New, w5New, w6New,\
     w7New, w8New, b1, b2)

    print('Weights and Biases After Backpropagation ')
    print_WB(sess)
    print('Total error after Backpopagation is %.6f'%\
    (sess.run(Etotal, feed_dict = inp_out_dict)))

