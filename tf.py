
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

hello = tf.constant('Hello World!')
sess = tf.Session()
print(sess.run(hello))

a=tf.Variable(5)
b=tf.Variable(3)
c=tf.multiply(a,b)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(c)


