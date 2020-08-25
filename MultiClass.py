

Skip to content
Using Gmail with screen readers
in:sent logistic 
Meet
Start a meeting
Join a meeting
Hangouts

Machine Learning Logistic Regression

Mark Wang <usawangyu@gmail.com>
Attachments
Tue, Dec 10, 2019, 2:45 PM
to rohitgupta74


Attachments area

 
import sys
import numpy as np
import math
import random
from scipy.optimize import fmin_bfgs
from random import shuffle
from sklearn import datasets
from scipy.optimize import check_grad
from scipy.special import expit
from sklearn import model_selection
from sklearn import decomposition

def softmax(x, i):
    e = np.exp(-x)
    return ((e[i]).T/(1+np.array([np.sum(e,axis =0)]))).T
def compute_cost(currentw, xdata, targety, distincy):
    currentw = currentw.reshape((len(distincy), len(xdata[0])))
    resultgradient = []
    errorsum = float(0.0)
    alpha = np.dot(currentw, xdata.T)
    for i in range(len(distincy)):
        probability = softmax(alpha, i)
        logprobability = np.log(probability)
        targety_at_wi = np.zeros(len(targety))
        targety_at_wi [targety == distincy[i]] = 1
        error_at_i = np.sum((logprobability.T * targety_at_wi).T, axis =0)
        errorsum = error_at_i + errorsum
    cost = -errorsum       
    return float(cost)

def compute_gradient(currentw, xdata, targety, distincy):
    currentw = currentw.reshape((len(distincy), len(xdata[0])))
    resultgradient = []
     
    alpha = np.dot(currentw, xdata.T)
    for i in range(len(distincy)):
        probability = (1 - softmax(alpha, i)).flatten()
        targety_at_wi = np.zeros(len(targety))
        targety_at_wi [targety == distincy[i]] = 1
        noswitchbit = (xdata.T * probability).T
        gradientmatrix = (noswitchbit.T * targety_at_wi).T
        gradient_i = (float(1)/len(xdata)) * np.sum(gradientmatrix, axis=0)
        resultgradient.append(gradient_i)
    return np.array(resultgradient).flatten()

def generateBeginw(targety, xdata, distincy):
    beginw = np.zeros((len(distincy), (len(xdata[0]) -1)))
    beginw = np.insert(beginw,len(beginw[0]), 1, axis=1)
    return beginw.flatten()

def trainData(xdata, targety, distincy,lameda, stepsize):
    beginw = generateBeginw(targety, xdata, distincy)
    preivesw = beginw
    currentw = beginw
    nextw = beginw
    lasterror = sys.float_info.max
    currenterror = 9999999999
    gradientnorm = 100
    count = 0
    draw = []
    while(currenterror < lasterror):
        lasterror = currenterror 
        gradient = compute_gradient(currentw, xdata, targety, distincy)
        currenterror = compute_cost(currentw, xdata, targety, distincy)
        nextw = currentw * lameda- np.multiply(stepsize, gradient)
        preivesw = currentw
        currentw = nextw
 
        count += 1.0
        draw.append(currenterror)
        #print("Iteration " + str(count) + "-------error: " + str(currenterror))
    print("Find solution w : ")
    print(preivesw)
    print("------------------------------------------------")
    print("By using optimization Scipy Code")
    try:
        print(fmin_bfgs(compute_cost, x0=beginw, fprime=compute_gradient, args=(xdata, targety, distincy)))   
    except:
        print("unexpected error:" + sys.exc_info())
    
    print("------------------------------------------------")
    print("Check Gradient")
    try:
        print(check_grad(compute_cost, compute_gradient, preivesw, xdata, targety,distincy))
    except:
        print ("unexpected error:", sys.exc_info())
    #import matplotlib.pyplot as plt
    #plt.plot(draw)
    #plt.ylabel('Error')
    #plt.xlabel('Iterations')
    #plt.show()
    
 
    return preivesw

def testData(solutionw, testx, testy,distincy):
    print("------------------------------------------------")
    print("Test Data")
    solutionw = solutionw.reshape((len(distincy), len(testx[0])))
    resultlist= []
    alpha = np.dot(solutionw, testx.T)

    success = 0
    for i in range(len(testx)):
        probability_list = alpha/np.sum(alpha,axis = 0)
        if(np.argmax(probability_list.T[i]) == testy[i]):
            success += 1
    return success

def doLogisticRegressionMultiClass(xdata, targety, testx, testy, distincy,lameda, stepsize):
    solutionw = trainData(xdata, targety, distincy,lameda, stepsize)
    successCount = testData(solutionw, testx, testy, distincy) 
    print(str(successCount) + " out of " + str(len(testx)) + " are correct.") 
    return successCount
        
    
def doNormalization(originalx, testx):
    a = []
    a.append(np.amin(originalx))
    a.append(np.amax(originalx))
    a.append(np.amin(testx))
    a.append(np.amax(testx))
    originalx = (originalx-np.amin(a))/(np.amax(a) - np.amin(a))
    testx = (testx-np.amin(a))/(np.amax(a) - np.amin(a))
    return originalx, testx
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

print("Please select a data set to train:")
print("1. IRIS Data")
print("2. Handwritten digit")
print("3. CIFA-10")
selection = input("Enter a number: ")
trainx = []
trainy = []
testx = []
testy = []

totalsuccess = 0
#train IRIS and Handwritten 
if(selection == '1' or selection == '2'):
    if(selection == '1'):
        inputdata = datasets.load_iris()
    if(selection == '2'):
        inputdata = datasets.load_digits()
    for i in range(5):
        print(str(i) + "th loop:")
        randomstate = random.randint(0,100)
        trainx, testx, trainy, testy = model_selection.train_test_split(inputdata.data, inputdata.target, test_size=0.20, random_state=randomstate)
        trainbiasvector = np.ones((len(trainx), 1))
        testbiasvector = np.ones((len(testx), 1))
        trainx = np.c_[trainx, trainbiasvector]
        testx = np.c_[testx, testbiasvector]
        distincy = np.unique(inputdata.target)
         
        if(selection == '1'):
            totalsuccess += doLogisticRegressionMultiClass(trainx, trainy, testx, testy, distincy, 0.985, 0.1) #20
        if(selection == '2'):
            totalsuccess += doLogisticRegressionMultiClass(trainx, trainy, testx, testy, distincy, 0.80, 0.004) #20
    print("-----------------------------")
    print("Total " + str(totalsuccess) +" out of " + str(len(inputdata.target)) + " are Correct")
    print("Accuracy: " + str(totalsuccess*100.00/len(inputdata.target)) + "%")
# train CIFAR with nomalization
elif(selection == '3'):
    for i in range(5):
        inputdata = unpickle('cifar-10-batches-py\data_batch_'+ str(i+1))
        testdata = unpickle('cifar-10-batches-py\\test_batch')
        originalx = inputdata[b'data']
        testx = testdata[b'data']
        originalx, testx = doNormalization(originalx, testx)
        pca = decomposition.PCA(n_components=100)
        pca.fit(originalx)
        trainx = pca.transform(originalx)
        testx = pca.transform(testx)
        trainy = (inputdata[b'labels'])
        testy = testdata[b'labels']
        trainbiasvector = np.ones((len(trainx), 1))
        testbiasvector = np.ones((len(testx), 1))
        trainx = np.c_[trainx, trainbiasvector]
        testx = np.c_[testx, testbiasvector]
        distincy = np.unique(trainy)
        totalsuccess += doLogisticRegressionMultiClass(trainx, trainy, testx, testy, distincy, 0.99, 0.0005) #20
    print("-----------------------------")
    print("Total " + str(totalsuccess) +" out of " + str(5*len(inputdata[b'labels'])) + " are Correct")
    print("Accuracy: " + str(totalsuccess * 100.00 / (5*len(inputdata[b'labels']))) + "%")
 
 






LogisticRegressionMultiClass.py
Displaying LogisticRegressionMultiClass.py.
