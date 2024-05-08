###These codes run on my local jupyter notebook

#Installation
#conda install -c conda-forge keras

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow
from sklearn import datasets, linear_model
from numpy import *  
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from scipy.stats import t
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

import random

from sklearn.metrics import roc_curve, auc

#Read data and print info
df = pd.read_csv("abalone.csv",sep = ',', header=None)
df = df.values
df[:,0] = np.where(df[:,0] == "M", 0, np.where(df[:,0] == "F", 1,0.5))
df = pd.DataFrame(df)
#print(df)
print(df.info())

#Removing Outliers
df = df.values.astype(float)
from scipy import stats
z=np.abs(stats.zscore(df))
threshold = 3
df2= df[(z<= threshold).all(axis=1)]
df2 = pd.DataFrame(df2)
print(df2.info())
df2 = df2.values.astype(float)

#Finding Class distribution
print('Class 1 is 0-7 years, \nClass 2 is 8-10 years, \nClass 3 is 11-15 years, \nClass 4 is >15 years.')
df2[:,8] = df2[:,8].astype(float)
df2[:,8] = np.where(df2[:,8] < 8, 1, np.where(df2[:,8] < 11, 2, np.where(df2[:,8] < 16, 3, 4)))
#Class 1: 0 - 7 years
#Class 2: 8- 10 years
##Class 3: 11 - 15 years
#Class 4: Greater than 15 years
def class_dist(ydata):
    a_set = set(ydata) #to count the number of unique classes, change the list to set)
    for i in a_set:
        valuez = pd.value_counts(ydata)[i]
        print('For Class', i,',there are', valuez, 'number of entries.')

abalone_y2 = df2[:,8] #insert this as data value
class_dist(abalone_y2)

#Finding feature distribution
def hist_plot(name, data, xlabel_name):
    plt.figure()
    plt.hist(data)
    plt.xlabel('feature '+str(xlabel_name))
    plt.ylabel('Frequency')
    plt.savefig(name+'.png', dpi = 500)
    plt.show()
    plt.clf()

df3 = df2[:,1:8]
for i in range(len(df3[0])): 
        name = 'histogram figure'+str(i+1) 
        data = df3[:,i]
        xlabel_name = i+1 
        hist_plot(name,data,xlabel_name)

print('feature 1= Length , feature 2= Diameter, feature 3= Height, feature 4= Whole weight, feature 5= Shucked weight, feature 6= Viscera weight, feature 7= Shell weight')


#One hot encoding
from keras.utils import np_utils
encoder = LabelEncoder()
Y = df2[:,-1]
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)

#Data input
def read_data(run_num):
    
    data_in = df2
    data_inputx = data_in[:,0:8] # all features 0, 1, 2, 3, 4, 5, 6, 7 

    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
    transformer = Normalizer().fit(data_inputx)  # fit does nothing.
    data_inputx = transformer.transform(data_inputx)
    data_inputy = dummy_y 

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=run_num)
    print(y_train[:,1])
    return x_train, x_test, y_train, y_test

def keras_nn(x_train, x_test, y_train, y_test, type_model, hidden_neurons, learn_rate, run_num):
    outputs = 4

    if type_model ==0: #SGD
        
        model = Sequential()
        model.add(Dense(hidden_neurons, input_dim=x_train.shape[1], activation='relu'))
        model.add(Dense(outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd',  metrics=['accuracy'])
    
    elif type_model ==1: #Adam
        
        model = Sequential()
        model.add(Dense(hidden_neurons, input_dim=x_train.shape[1], activation='relu'))
        model.add(Dense(outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    elif type_model ==2: #SGD with 2 hidden layers
        
        model = Sequential()
        model.add(Dense(hidden_neurons, input_dim=x_train.shape[1], activation='relu')) 
        model.add(Dense(hidden_neurons, activation='softmax'))
        model.add(Dense(outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    else:
        print('no model')

    # Fit model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=10, verbose=0)

    # Evaluate the model
    train_lost, acc_train = model.evaluate(x_train, y_train, verbose=2)
    test_lost, acc_test = model.evaluate(x_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (acc_train, acc_test))
    print('Train Lost: %.3f, Test Lost: %.3f' % (train_lost, test_lost))

    '''############################ TURN THIS ON FOR FINAL CONFUSION MATRIX AND ROC/AUC CURVE
    #Predictions & COnfusion matrix
    probability_model = tensorflow.keras.Sequential([model, tensorflow.keras.layers.Softmax()])
    predictions = probability_model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1) #will look through each row (axis=1) and return the column index where value is max in that row
    print(predictions)
    #e = pd.Series(y_pred).unique()  
    #test = list(e.flatten())
    #print(test) #this is to check the min/max values of the predictions, eg: does it include class 3? -- in this case no
    print(y_pred.shape) #just to check that y_true has the same shape
   

   
    y_true = np.argmax(y_test, axis=1)
    print(y_true.shape)
    cm = confusion_matrix(y_true, y_pred, labels = class_name)
    
    normalize=True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    # Plot confusion matrix
    fig = plt.figure(figsize=(16, 14))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); 
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    
    ax.set_ylabel('True', fontsize=20)
    
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix', fontsize=20)
    plt.savefig('ConMat.png')
    plt.show()
    plt.clf()
    
    
    
    #ROC_AUC CURVES
    fpr = dict()
    tpr = dict()
    auroc = dict()
    
    for i in class_name:
        

        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:,i], predictions[:,i], pos_label = 1) 
        auroc[i] = metrics.auc(fpr[i], tpr[i])
        print('LRE',i,'--AUC--->',auroc[i])
    
    %matplotlib inline
    import itertools
    colors = itertools.cycle(['blue', 'green', 'red','darkorange','olive','purple','navy'])
    for i, color in zip(class_name, colors):
        pyplot.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, auroc[i]))
    pyplot.plot([0, 1], [0, 1], 'k--', lw=2)
    pyplot.xlim([-0.05, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel('False Positive Rate',fontsize=12, fontweight='bold')
    pyplot.ylabel('True Positive Rate',fontsize=12, fontweight='bold')
    
    pyplot.legend(loc="lower right")
    
    pyplot.show()
    plt.clf()        
    ##########################################################'''
    
    return acc_test #,acc_train

#Finding Confidence interval
def confidence_interval(mean, sd, max_expruns, confidence_level):
    mean = mean
    sd = sd
    N = max_expruns
    dof = N-1
    confidence_level = confidence_level
    t_crit = np.abs(t.ppf((1-confidence_level)/2,dof))
    CIa = mean-sd*t_crit/np.sqrt(N) 
    CIb = mean+sd*t_crit/np.sqrt(N)
    print('The Confidence Interval is:')
    print(CIa, CIb)


#PART 1: Testing Number of Neurons
max_expruns = 10
max_hidden = 21 #21 causewe want neuron 5,10,15,20 
learn_rate = 0.01

SGD_all = np.zeros(max_expruns) 
Adam_all = np.zeros(max_expruns) 
SGD2_all = np.zeros(max_expruns)  



#for learn_rate in range(0.1,1, 0.2):
    
for hidden_neurons in range(5,max_hidden, 5):
 
    for run_num in range(0,max_expruns): 
    
        x_train, x_test, y_train, y_test = read_data(0)   
            
        acc_sgd = keras_nn(x_train, x_test, y_train, y_test, 0, hidden_neurons, learn_rate, run_num) #SGD
           
        SGD_all[run_num] = acc_sgd
        
  
    
    mean = np.mean(SGD_all)
    sd = np.std(SGD_all)
    confidence_level = 0.95
    
    print(SGD_all, hidden_neurons,' SGD_all')
    print(mean, hidden_neurons, ' mean SGD_all')
    print(sd, hidden_neurons, ' std SGD_all')
    confidence_interval(mean, sd, max_expruns, confidence_level)

#PART 2: Testing Learning Rate
max_expruns = 10

hidden_neurons = 15
SGD_all = np.zeros(max_expruns) 
Adam_all = np.zeros(max_expruns) 
SGD2_all = np.zeros(max_expruns)  

def range_with_floats(start, stop, step):
    while stop > start:
        yield start
        start += step
for learn_rate in range_with_floats(0.1, 1.0, 0.2):

    for run_num in range(0,max_expruns): 
    
        x_train, x_test, y_train, y_test = read_data(0)   
            
        acc_sgd = keras_nn(x_train, x_test, y_train, y_test, 0, hidden_neurons, learn_rate, run_num) #SGD

           
        SGD_all[run_num] = acc_sgd

  
    
    mean = np.mean(SGD_all)
    sd = np.std(SGD_all)
    confidence_level = 0.95
    
    print(SGD_all, learn_rate,' SGD_all')
    print(mean, learn_rate, ' mean SGD_all')
    print(sd, learn_rate, ' std SGD_all')
    confidence_interval(mean, sd, max_expruns, confidence_level)
#Testing Number of Layers
max_expruns = 10
learn_rate = 0.07
hidden_neurons = 15

SGD_all = np.zeros(max_expruns) 
Adam_all = np.zeros(max_expruns) 
SGD2_all = np.zeros(max_expruns)  



#for learn_rate in range(0.1,1, 0.2):
    
#for hidden_neurons in range(5,max_hidden, 5):
 
for run_num in range(0,max_expruns): 
    
    x_train, x_test, y_train, y_test = read_data(0)   
            

    acc_sgd2 = keras_nn(x_train, x_test, y_train, y_test, 2, hidden_neurons, learn_rate,  run_num) #SGD2           
    SGD2_all[run_num] = acc_sgd2 # two hidden layers
  
    
mean = np.mean(SGD2_all)
sd = np.std(SGD2_all)
confidence_level = 0.95
    
print(SGD2_all, hidden_neurons,' SGD_all')
print(mean, hidden_neurons, ' mean SGD_all')
print(sd, hidden_neurons, ' std SGD_all')
confidence_interval(mean, sd, max_expruns, confidence_level)

#Testing Algorithm, please turn on the confusion matrix and auroc section in keras_nn function
#QUESTION 4
max_expruns = 10
hidden_neurons = 15
learn_rate = 0.07

Adam_all = np.zeros(max_expruns) 

for run_num in range(0,max_expruns): 
    
    x_train, x_test, y_train, y_test = read_data(0)   
            

    acc_adam = keras_nn(x_train, x_test, y_train, y_test, 1, hidden_neurons, learn_rate, run_num) #Adam            
    Adam_all[run_num] = acc_adam # two hidden layers
  
    
mean = np.mean(Adam_all)
sd = np.std(Adam_all)
confidence_level = 0.95
    
print(Adam_all, hidden_neurons,' Adam_all')
print(mean, hidden_neurons, ' mean Adam_all')
print(sd, hidden_neurons, ' std Adam_all')
confidence_interval(mean, sd, max_expruns, confidence_level)

