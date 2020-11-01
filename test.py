from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np

#Preprocessor
preprocessor = MyPreProcessor()


#Linear Regression
print('Linear Regression')
X, Y = preprocessor.pre_process(0)
linear = MyLinearRegression()
linear.fit(X, Y, showGraph = True)
ypred = linear.predict(X)
print('Predicted Values:', ypred)
print('True Values:', Y)



# Test loss vs iterations for i+1 fold
# numRows = X.shape[0]
# rowsPerFold = int(numRows/10)
# i = 1
# testX = X[rowsPerFold*i:rowsPerFold*(i+1),:]
# testY = Y[rowsPerFold*i:rowsPerFold*(i+1),:]
# trainX_1 = X[0:rowsPerFold*i , :]
# trainY_1 = Y[0:rowsPerFold*i , :]
# trainX_2 = X[rowsPerFold*(i+1): , :]
# trainY_2 = Y[rowsPerFold*(i+1): , :]
# trainX = np.concatenate((trainX_1, trainX_2))
# trainY = np.concatenate((trainY_1, trainY_2))
# linear.fit(trainX, trainY, blockFunc = 0 , isCalcErrorPerEpoch = True, learningRate = 0.00001, numIterations = 200, testX = testX, testY = testY, showGraph = True)



#Normal Equation Code
# numRows = X.shape[0]
# rowsPerFold = int(numRows/10)

# testX = X[rowsPerFold*3:rowsPerFold*4,:]
# testY = Y[rowsPerFold*3:rowsPerFold*4,:]

# i = 3
# trainX_1 = X[0:rowsPerFold*i , :]
# trainY_1 = Y[0:rowsPerFold*i , :]
# trainX_2 = X[rowsPerFold*(i+1): , :]
# trainY_2 = Y[rowsPerFold*(i+1): , :]

# trainX = np.concatenate((trainX_1, trainX_2))
# trainY = np.concatenate((trainY_1, trainY_2))

# print("Parameters obtained through normal equation are", linear.normalEquation(trainX, trainY, testX, testY))



#Run Kfold
# linear.kFoldValidation(X, Y, 10, showGraph = True, learningRate = 0.00001, numIterations = 200, isCalcErrorPerEpoch = True)




###################################################################################################################################



#Logistic Regression
print('Logistic Regression')
X, Y = preprocessor.pre_process(2)
logistic = MyLogisticRegression()

logistic.fit(X, Y)
ypred = logistic.predict(X,isBatch = False)
print('Predicted Values:', ypred)
print('True Values:', Y)



#Validation loss vs iterations for 7:1:2 split
# numRows = X.shape[0]
# rowsPerFold = int(numRows/10)

# trainX = X[:7*rowsPerFold, :]
# trainY = Y[:7*rowsPerFold, :]

# valX = X[7*rowsPerFold: 8*rowsPerFold, :]
# valY = Y[7*rowsPerFold: 8*rowsPerFold, :]

# testX = X[8*rowsPerFold:, :]
# testY = Y[8*rowsPerFold:, :]

# logistic.fit(trainX, trainY, blockFunc = 0 , isCalcErrorPerEpoch = True, learningRate = 10, numIterations = 2000, testX = valX, testY = valY, showGraph = True)




#SkLearn regression
# logistic.skLearnAccuracy(X, Y)


#MLE
# print("Parameters obtained through MLE are", logistic.MLE(X,Y)[0])




#Split database
#logistic.getAccuracy(X, Y, numIterations = 2000, learningRate = 10, showGraph = True, isCalcErrorPerEpoch = True)