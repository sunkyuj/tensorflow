
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import tensorflow as tf

t1 = tf.constant([3,4,5])
t2 = tf.constant([6,7,8])

t3 = tf.constant([[1,2],[3,4]])
t4 = tf.constant([[5,6],[7,8]])

t5=tf.zeros([3,3])
print(t1+t2)
print(t3*t4)
print(tf.matmul(t3,t4)) #행렬곱
print(t5)
#print(t5.value())
w = tf.Variable(1.0)
w.assign(2.7)
print(w.numpy())

'''
a=tf.Variable(5)
b=tf.Variable(3)
c=tf.multiply(a,b)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(c)
'''

 