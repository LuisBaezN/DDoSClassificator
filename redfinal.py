import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd 
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22,10

RANDOM_SEED = 37

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#//////////////////////read data////////////////////////

df = pd.read_csv("01-12/Database_f_v2.csv")#, index_col="timestamp"

#SourceIP,SourcePort,DestinationIP,DestinationPort,Protocol
#FlowDuration,TotalBackwardPackets,TotalLenghtofBwdPackets,BwdPacketLengthStd,Flowbytes/s,FlowPackets/s
#//////FlowIATMean,FwdHeaderLength,AvgBwdSegmentsize,Active Min,idle Mean,Label

print(df.head)

#plt.plot(df['FlowDuration'], label='Flow Duration')
#plt.legend()
#plt.show()

#/////////////////////data sep///////////////////////////

train_size = int(len(df) * 0.97)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)

#///////////////////normalization////////////////////////

scaler = MinMaxScaler()
scaler.fit(train)
MinMaxScaler(copy=True, feature_range=(0,1))

train = scaler.transform(train) #['FlowDuration','FlowPs']

scaler.fit(test)
MinMaxScaler(copy=True, feature_range=(0,1))

test_p = test[['FlowDuration','TotalBackwardPackets','TotalLenghtofBwdPackets','BwdPacketLengthStd','FlowIATMean','FwdHeaderLength','AvgBwdSegmentsize','Active Min','idle Mean','Label']]
print(type(test_p))
print(test_p.head())

test = scaler.transform(test)

train = pd.DataFrame(train, columns=['SourceIP','SourcePort','DestinationIP','DestinationPort','Protocol','FlowDuration','TotalBackwardPackets','TotalLenghtofBwdPackets','BwdPacketLengthStd','Flowbytes/s','FlowPackets/s','FlowIATMean','FwdHeaderLength','AvgBwdSegmentsize','Active Min','idle Mean','Label'])
print(train.head())

test = pd.DataFrame(test, columns=['SourceIP','SourcePort','DestinationIP','DestinationPort','Protocol','FlowDuration','TotalBackwardPackets','TotalLenghtofBwdPackets','BwdPacketLengthStd','Flowbytes/s','FlowPackets/s','FlowIATMean','FwdHeaderLength','AvgBwdSegmentsize','Active Min','idle Mean','Label'])
print(test.head())

scaler.fit(test_p)
MinMaxScaler(copy=True, feature_range=(0,1))


#////////////////////////Sequence/////////////////////////

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


TIME_STEPS = 70

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(
  train[['SourceIP','SourcePort','DestinationIP','DestinationPort','Protocol','FlowDuration','TotalBackwardPackets','TotalLenghtofBwdPackets','BwdPacketLengthStd','Flowbytes/s','FlowPackets/s','FlowIATMean','FwdHeaderLength','AvgBwdSegmentsize','Active Min','idle Mean']],
  train[['FlowDuration','TotalBackwardPackets','TotalLenghtofBwdPackets','BwdPacketLengthStd','FlowIATMean','FwdHeaderLength','AvgBwdSegmentsize','Active Min','idle Mean','Label']],
  TIME_STEPS
)

X_test, y_test = create_dataset(
  test[['SourceIP','SourcePort','DestinationIP','DestinationPort','Protocol','FlowDuration','TotalBackwardPackets','TotalLenghtofBwdPackets','BwdPacketLengthStd','Flowbytes/s','FlowPackets/s','FlowIATMean','FwdHeaderLength','AvgBwdSegmentsize','Active Min','idle Mean']],
  test[['FlowDuration','TotalBackwardPackets','TotalLenghtofBwdPackets','BwdPacketLengthStd','FlowIATMean','FwdHeaderLength','AvgBwdSegmentsize','Active Min','idle Mean','Label']],
  TIME_STEPS
)

print(X_train.shape)
print(y_train.shape)

in_dim = (X_train.shape[1], X_train.shape[2])
out_dim = y_train.shape[1]

#////////////////////Modelo///////////////////////////

model = keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=256, input_shape=(in_dim)))) #128/128/2  
model.add(keras.layers.Dropout(rate=0.2))
#model.add(keras.layers.LeakyReLU(alpha=0.02))
model.add(keras.layers.Dense(units=256))
model.add(keras.layers.Dropout(rate=0.2))
#model.add(keras.layers.LeakyReLU(alpha=0.05))
#model.add(keras.layers.Dense(units=64))
#model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=out_dim))
#model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])  ))

model.compile(loss='mean_squared_error', optimizer='SGD') #loss = mae

history = model.fit(
    X_train, y_train,
    epochs=37,
    batch_size=91,
    validation_split=0.1,
    shuffle=False
) #batch 71

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model.save('LSTM_mul_v2-2.h5')
print('>Model Saved')

#/////////////////////Prediction///////////////////


pred = model.predict(X_test)

print("y1 MSE:%.4f" % mean_squared_error(y_test[:,0], pred[:,0]))
print("y2 MSE:%.4f" % mean_squared_error(y_test[:,1], pred[:,1])) 
print("y3 MSE:%.4f" % mean_squared_error(y_test[:,2], pred[:,2])) 
print("y4 MSE:%.4f" % mean_squared_error(y_test[:,3], pred[:,3])) 
print("y5 MSE:%.4f" % mean_squared_error(y_test[:,4], pred[:,4])) 
print("y6 MSE:%.4f" % mean_squared_error(y_test[:,5], pred[:,5])) 
print("y7 MSE:%.4f" % mean_squared_error(y_test[:,6], pred[:,6]))
print("y8 MSE:%.4f" % mean_squared_error(y_test[:,7], pred[:,7])) 
print("y9 MSE:%.4f" % mean_squared_error(y_test[:,8], pred[:,8])) 
print("y10 MSE:%.4f" % mean_squared_error(y_test[:,9], pred[:,9])) 
#print("y11 MSE:%.4f" % mean_squared_error(y_test[:,10], pred[:,10])) 
#print("y12 MSE:%.4f" % mean_squared_error(y_test[:,111], pred[:,11])) 

print('\nPrediccion normalizada:')
print(pred[5,:])

#//////////////////////////////////////////////////////////////////////////////////////////////////
y_test = scaler.inverse_transform(y_test)

print('\nPrediccion:')
pred = scaler.inverse_transform(pred)
print(pred[5,:])
#//////////////////////////////////////////////////////////////////////////////////////////////////

x_ax = range(len(X_test))

plt.title("Flow Duration")
plt.scatter(x_ax, y_test[:,0],  s=6, label="y1-test")
plt.plot(x_ax, pred[:,0], color = 'red', label="y1-pred")
plt.legend()
plt.show()

plt.title("Total Backward Packets")
plt.scatter(x_ax, y_test[:,1],  s=6, label="y2-test")
plt.plot(x_ax, pred[:,1], color = 'red', label="y2-pred")
plt.legend()
plt.show()

plt.title("Total Lenght of Bwd Packets")
plt.scatter(x_ax, y_test[:,2],  s=6, label="y3-test")
plt.plot(x_ax, pred[:,2], color = 'red', label="y3-pred")
plt.legend()
plt.show()

plt.title("Bwd Packet Length Std")
plt.scatter(x_ax, y_test[:,3],  s=6, label="y4-test")
plt.plot(x_ax, pred[:,3], color = 'red', label="y4-pred")
plt.legend()
plt.show()

plt.title("Flow IAT Mean")
plt.scatter(x_ax, y_test[:,4],  s=6, label="y5-test")
plt.plot(x_ax, pred[:,4], color = 'red', label="y5-pred")
plt.legend()
plt.show()

plt.title("Fwd Header Length")
plt.scatter(x_ax, y_test[:,5],  s=6, label="y6-test")
plt.plot(x_ax, pred[:,5], color = 'red', label="y6-pred")
plt.legend()
plt.show()

plt.title("Avg Bwd Segment size")
plt.scatter(x_ax, y_test[:,6],  s=6, label="y7-test")
plt.plot(x_ax, pred[:,6], color = 'red', label="y7-pred")
plt.legend()
plt.show()

plt.title("Active Min")
plt.scatter(x_ax, y_test[:,7],  s=6, label="y8-test")
plt.plot(x_ax, pred[:,7], color = 'red', label="y8-pred")
plt.legend()
plt.show()

plt.title("idle Mean")
plt.scatter(x_ax, y_test[:,8],  s=6, label="y9-test")
plt.plot(x_ax, pred[:,8], color = 'red', label="y9-pred")
plt.legend()
plt.show()

plt.title("Label")
plt.scatter(x_ax, y_test[:,9],  s=6, label="y10-test")
plt.plot(x_ax, pred[:,9], color = 'red', label="y10-pred")
plt.legend()
plt.show()

#time step a 30, data test 0.997
