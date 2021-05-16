import pandas as pd

data = pd.read_csv("gpascore.csv")
data=data.dropna() #누락된 데이터가 있으면 없애버림
#data.fillna(x) 누락된 곳을 전부 x로 채움

ydata = data['admit'].values
xdata=[]
for i,rows in data.iterrows():
    xdata.append( [ rows['gre'],rows['gpa'],rows['rank'] ] )

import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([ #딥러닝 모델

    #레이어1, 레이어1에 64개 들어감,개수는 실험적으로 해야함, 관습적으로 2의 제곱수
    tf.keras.layers.Dense(64,activation='tanh'), 
    #레이어2, tanh 활성함수는 -1~1
    tf.keras.layers.Dense(128,activation='tanh'),#레이어2, tanh 활성함수는 -1~1
    

    #레이어3, 예측 결과, 확률을 결과로 할거니까 시그모이드 
    tf.keras.layers.Dense(1,activation='sigmoid'),#레이어3, 예측 결과, 확률을 결과로 할거니까 시그모이드 
]) 

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#binary_crossentropy <- 확률문제에서 많이 쓰이는 손실함수

model.fit(np.array(xdata),np.array(ydata),epochs=1000) #알아서 학습(fit)시킴.., x는 학습데이터, y는 답

pred = model.predict([ [750, 3.70, 3] , [400, 2.2, 1] ])
print(pred)
