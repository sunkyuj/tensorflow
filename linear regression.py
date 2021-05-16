import tensorflow as tf

height = [170,180,175,160]
weight = [260,270,265,255] 

a=tf.Variable(0.1)
b=tf.Variable(0.2)

키 = 170
신발 = 260

# 케라스에서 아담 옵티마이저(최적화 함수) 쓸거
opt = tf.keras.optimizers.Adam(learning_rate=0.1) #learning_rate == 알파값

def loss():
    pred = 키*a +b

    return tf.square(신발 - pred)

for i in range(300):
    opt.minimize(loss,var_list=[a,b]) # 경사하강
    if (i%20==0):
        print(a.numpy(),b.numpy())

print(키*a+b)

'''
키 = 170
신발 = 260

신발 = 키*a +b
'''