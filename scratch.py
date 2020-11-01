import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn import linear_model
import math
import random
import os 

class MyPreProcessor():
	"""
	My steps for pre-processing for the three datasets.
	"""

	def __init__(self):
		pass

	def pre_process(self, dataset):
		"""
		Reading the file and preprocessing the input and output.
		Note that you will encode any string value and/or remove empty entries in this function only.
		Further any pre processing steps have to be performed in this function too. 

		Parameters
		----------

		dataset : integer with acceptable values 0, 1, or 2
		0 -> Abalone Dataset
		1 -> VideoGame Dataset
		2 -> BankNote Authentication Dataset
		
		Returns
		-------
		X : 2-dimensional numpy array of shape (n_samples, n_features)
		y : 1-dimensional numpy array of shape (n_samples,)
		"""

		# np.empty creates an empty array only. You have to replace this with your code.
		X = []
		Y = []


		dir_path = os.path.dirname(os.path.realpath(__file__))

		#Implement Abalone Dataset
		if dataset == 0:
			#Getting dataset from file and preprocessing the dataframe
			reqDataframe = pd.read_table(dir_path + '/Dataset.data', sep = " ")
			reqDataframe.columns = ['sex', 'length', 'diameter', 'height', 'whole_weight',  'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
		
			#Make  extra dummy variables
			reqDataframe = pd.get_dummies(reqDataframe, columns = ['sex'])
		

			#Creating the numpy arrays
			X = reqDataframe[['sex_F' , 'sex_I', 'sex_M', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']].to_numpy()
			Y = reqDataframe[['rings']].to_numpy()


		# Implement for the video game dataset
		elif dataset == 1:
			#Getting dataset from file and preprocessing the dataframe
			originalDataframe = pd.read_csv(dir_path + '/VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv')
			reqDataframe = originalDataframe[['Critic_Score' , 'User_Score', 'Global_Sales']]
			reqDataframe = reqDataframe.replace('tbd', np.nan)
			reqDataframe = reqDataframe.dropna()
			reqDataframe['User_Score'] = reqDataframe['User_Score'].astype('float')

			# #Randomizing data
			# reqDataframe = reqDataframe.sample(frac=1, random_state=100)
		
			#Splitting input and output
			X = reqDataframe[['Critic_Score' , 'User_Score']].to_numpy()
			Y = reqDataframe[['Global_Sales']].to_numpy()


		# Implement for the banknote authentication dataset
		elif dataset == 2:
			
			#Getting dataset from file and preprocessing the dataframe
			reqDataframe = pd.read_table(dir_path + '/data_banknote_authentication.txt', sep = ",")
			reqDataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
			
			#Creating the numpy arrays
			X = reqDataframe[['variance', 'skewness', 'curtosis', 'entropy']].to_numpy()
			Y = reqDataframe[['class']].to_numpy()

		# Implement for the cancer data question
		elif dataset == 3:
			
			#Getting dataset from file and preprocessing the dataframe
			reqDataframe = pd.read_table(dir_path + '/Q4_Dataset.txt', sep = "\s+")
			reqDataframe.columns = ['class', 'spread', 'age']
			#print(reqDataframe)      
		
			#Creating the numpy arrays
			X = reqDataframe.iloc[:, 1:].to_numpy()
			Y = reqDataframe.iloc[:, 0].to_numpy()



		return X, Y

class MyLinearRegression():
	"""
	My implementation of Linear Regression.
	"""

	#Instance variables
	modelParametersRMS = None
	modelParametersMAE = None
	functionBlocked = -1;



	def __init__(self):
		pass


	#Method for gradient descent using Root Mean Square Error
	def gradientDescentRMS(self, modelParameters, inputArray, outputArray, learningRate, iterations, isCalcErrorPerEpoch, testX, testY):
		
		errHistory = []
		errPerEpoch = []
		numRows = inputArray.shape[0]
		numCols = inputArray.shape[1]

		for k in range(iterations):

			currModelParameters = copy.deepcopy(modelParameters)

			for i in range(numCols):
				additionalTerm = 0
				additionalTerm2 = 0
				
				for j in range(numRows):
					inputRow = inputArray[j];
					#print("currModelParameters" , currModelParameters)
					#print("inputRow", inputRow)
					hypothesis = np.dot(currModelParameters.T, inputRow)
					#print("hypothesis", hypothesis)
					additionalTerm2 = additionalTerm2 + ((hypothesis - outputArray[j])**2)
					additionalTerm = additionalTerm + ((hypothesis - outputArray[j]) * inputRow[i]) 
				
				#print("Additional term is", additionalTerm)
				additionalTerm2 = additionalTerm2/numRows
				additionalTerm2 = math.sqrt(additionalTerm2)
				#print("additional term2", additionalTerm2)
				modelParameters[i] = modelParameters[i] - (learningRate*(1/numRows)*(1/additionalTerm2)*additionalTerm)
						
			self.modelParametersRMS = modelParameters

			if(isCalcErrorPerEpoch==True):
				errPerEpoch.append(self.getErrorPerEpoch(modelParameters, testX, testY, isMAE = False))
			
			errHistory.append(additionalTerm2)
		return [modelParameters, errHistory, errPerEpoch]

	

	#Method for gradient descent using mean absolute error
	def gradientDescentMAE(self, modelParameters, inputArray, outputArray, learningRate, iterations, isCalcErrorPerEpoch, testX, testY):
		
		errHistory = []
		errPerEpoch = []
		numRows = inputArray.shape[0]
		numCols = inputArray.shape[1]

		for k in range(iterations):

			currModelParameters = copy.deepcopy(modelParameters)

			for i in range(numCols):
				additionalTerm = 0
				error = 0;
				for j in range(numRows):
					inputRow = inputArray[j];
					#print("currModelParameters" , currModelParameters)
					#print("inputRow", inputRow)
					hypothesis = np.dot(currModelParameters.T, inputRow)
					#print("hypothesis", hypothesis)
					tempError = hypothesis - outputArray[j]
					error = error + abs(tempError)
				  
					additionalTerm = additionalTerm + ((abs(tempError) /  (tempError))*inputRow[i])
				
				modelParameters[i] = modelParameters[i] - (learningRate*(1/numRows)*additionalTerm)

				#print("model parameters are", modelParameters)
			#print("error is", error/numRows)
			
			self.modelParametersMAE = modelParameters

			if(isCalcErrorPerEpoch==True):
				errPerEpoch.append(self.getErrorPerEpoch(modelParameters, testX, testY, isMAE = True))

			errHistory.append(error/numRows)

		return [modelParameters, errHistory, errPerEpoch]


	#Calculate Error per epoch
	def getErrorPerEpoch(self, modelParameters, testX, testY, isMAE):

		numRows = testX.shape[0]
		hypothesis = None
		err = 0



		for i in range(numRows):
			if(isMAE == True):
				hypothesis = np.dot(self.modelParametersMAE.T, testX[i])
				err = err + abs(hypothesis - testY[i])
			else:
				hypothesis = np.dot(self.modelParametersRMS.T, testX[i])
				err = err + (hypothesis - testY[i])**2

		if(isMAE==True):
			err = err/numRows
			return err

		else:
			err = err/numRows
			err = math.sqrt(err)
			return err



	''' X is the input training data
		Y is the Output training data
		learning rate and number of iterations are self explanatory
		showgraph decides if you want to construct a graph of training loss vs iterations
		blockFunc can block either RMS or MAE, if 0 blocks nothing, if 1 blocks MAE and if 2 blocks RMS
		isCalcErrorPerEpoch calculates the MAE or RMS for every parameter at each epoch
		testX and testY are provided if calcErrorPerEpoch is true'''
	def fit(self, X, Y, learningRate = 0.1, numIterations = 100, showGraph = False, blockFunc = 0, isCalcErrorPerEpoch = False, testX = None, testY = None):
		"""
		Fitting (training) the linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

		y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
		
		Returns
		-------
		self : an instance of self
		"""

		if(isCalcErrorPerEpoch == True and (testX is None or testY is None)):
			print("Please provide testX and testY to calculate the errors per epoch")
			return None

		elif(isCalcErrorPerEpoch==True):
			
			additionalColumn = np.ones((testX.shape[0], 1))
			testX = np.append(testX, additionalColumn, axis=1)
			
		additionalColumn = np.ones((X.shape[0], 1))
		X = np.append(X, additionalColumn, axis=1)
		
		#Get number of input rows
		numCols = X.shape[1]

		#Initialize the model paramters with 0s
		modelParametersRMS = np.zeros((numCols, 1))
		modelParametersMAE = np.zeros((numCols, 1))
		

		#MAE

		# #Problem 1
		# learningRate = 0.00001
		# numIterations = 100

		# #Problem 2
		# learningRate = 0.1
		# numIterations = 100

		self.functionBlocked = blockFunc

		if(blockFunc!=1):

			outputMAE = self.gradientDescentMAE(modelParametersMAE, X, Y, learningRate, numIterations, isCalcErrorPerEpoch, testX, testY)
			modelParametersMAE = outputMAE[0]
			errHistoryMAE = outputMAE[1]
			errPerEpochMAE = outputMAE[2] 
			#print("Model parameters MAE are", modelParametersMAE, modelParametersMAE.shape)
		   
			#assigning value to global variable for predict function
			self.modelParametersMAE = modelParametersMAE



		#RMS
		
		# #Problem 1
		# learningRate = 0.0001
		# numIterations = 100
	
		# #Problem 2
		# learningRate = 0.1
		# numIterations = 100

		if(blockFunc!=2):

			outputRMS = self.gradientDescentRMS(modelParametersRMS, X, Y, learningRate, numIterations, isCalcErrorPerEpoch, testX, testY)
			modelParametersRMS = outputRMS[0]
			errHistoryRMS = outputRMS[1]
			errPerEpochRMS = outputRMS[2]
			#print("Model parameters RMS are", modelParametersRMS, modelParametersRMS.shape)
			
			#assigning value to global variable for predict function
			self.modelParametersRMS = modelParametersRMS
	


		#Plotting graph       
		if(showGraph==True):
			if(blockFunc!=1):
				plt.plot(errHistoryMAE, label = "MAE")
			if(blockFunc!=2):
				plt.plot(errHistoryRMS, label = "RMS")
			
			plt.legend()
			plt.show()

		if(isCalcErrorPerEpoch==True):
			if(blockFunc!=1):
				plt.plot(errPerEpochMAE, label = "MAE")
			if(blockFunc!=2):
				plt.plot(errPerEpochRMS, label = "RMS")
			
			plt.legend()
			plt.show()			


		# #Sk Learn implementation
		# reg = linear_model.LinearRegression()
		# reg.fit(X, Y)
		# print("SK learn coeffs are ", reg.coef_)

		# fit function has to return an instance of itself or else it won't work with test.py
		return self



	def predict(self, X, isMAE = True):
		"""
		Predicting values using the trained linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

		Returns
		-------
		y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
		"""

		# return the numpy array y which contains the predicted values
		additionalColumn = np.ones((X.shape[0], 1))
		X = np.append(X, additionalColumn, axis=1)

		numRows = X.shape[0]
		numCols = X.shape[1]

		arrRMS = []
		arrMAE = []

		if(isMAE==True and self.functionBlocked!=1):
			for i in range(numRows):
				hypothesisMAE = np.dot(self.modelParametersMAE.T, X[i])
				arrMAE.append(hypothesisMAE[0])
		
		elif(isMAE==False and self.functionBlocked!=2):
			for i in range(numRows):
				hypothesisRMS = np.dot(self.modelParametersRMS.T, X[i])
				arrRMS.append(hypothesisRMS[0])
				
		else:
			print("Fit function was not run for the requested predicted values")
			return None
			
		if(isMAE==True):
			return arrMAE

		else:
			return arrRMS


	#Method for finding parameters by normal equation
	def normalEquation(self, trainX, trainY, testX, testY):
		
		firstTerm = np.matmul(trainX.T, trainX)
		inverseFirst = np.linalg.inv(firstTerm)

		secondTerm = np.matmul(trainX.T, trainY)

		modelParameters = np.matmul(inverseFirst, secondTerm)

		#Calculate train loss
		MAETrain = 0
		numRows = trainX.shape[0]
		for i in range(numRows):
			hypothesis = np.dot(modelParameters.T, trainX[i])
			MAETrain = MAETrain + abs(hypothesis - trainY[i])
		
		MAETrain = MAETrain/numRows
		print("MAE for training data is ", MAETrain)

		#Calculate train loss
		MAETest = 0
		numRows = testX.shape[0]
		for i in range(numRows):
			hypothesis = np.dot(modelParameters.T, testX[i])
			MAETest = MAETest + abs(hypothesis - testY[i])
		
		MAETest = MAETest/numRows
		print("MAE for testing data is ", MAETest)


		return modelParameters

		



	'''Method for kFoldValidation 
		X, Y are input and output
		K is the number of folds
		numIterations is the number of iterations each fold is to be run for
		set show graph to true to display training loss vs epochs
		set isCalcErrorPerEpoch to true to display validation loss after each epoch'''
	def kFoldValidation(self, X, Y, K, learningRate = 0.1, numIterations = 100, showGraph = False, isCalcErrorPerEpoch = False):

		numRows = X.shape[0]
		rowsPerSlice = int(numRows/K)

		# for first fold

		testX = X[0:rowsPerSlice, :]
		testY = Y[0:rowsPerSlice, :]

		trainX = X[rowsPerSlice: , :]
		trainY = Y[rowsPerSlice: , :]

		arrErrRMS = []
		arrErrMAE = []

		arrRMSParam = []
		arrMAEParam = []
		

		self.fit(trainX, trainY, learningRate = learningRate, numIterations = numIterations, showGraph = showGraph, isCalcErrorPerEpoch = isCalcErrorPerEpoch, testX = testX, testY = testY)
		arrRMSParam.append(self.modelParametersRMS)
		arrMAEParam.append(self.modelParametersMAE)

		predRMS = self.predict(testX, isMAE = False)
		predMAE = self.predict(testX, isMAE = True)


		numTestRows = testX.shape[0]
		avgRMS = 0
		avgMAE = 0

		for i in range(numTestRows):
			avgRMS = avgRMS + (predRMS[i] - testY[i])**2
			avgMAE = avgMAE + abs(predMAE[i] - testY[i])

		avgRMS = avgRMS/numTestRows
		avgRMS = math.sqrt(avgRMS)

		avgMAE = avgMAE/numTestRows

		arrErrRMS.append(avgRMS)
		arrErrMAE.append(avgMAE)


		print("RMS and MAE for fold " + str(1) + " " + "are " + str(avgRMS) + " " +  str(avgMAE))

		#For fold 2 to second last fold
		for i in range(1, K-1):
			testX = X[rowsPerSlice*i:rowsPerSlice*(i+1), :]
			testY = Y[rowsPerSlice*i:rowsPerSlice*(i+1), :]

			trainX_1 = X[0:rowsPerSlice*i , :]
			trainY_1 = Y[0:rowsPerSlice*i , :]

			trainX_2 = X[rowsPerSlice*(i+1): , :]
			trainY_2 = Y[rowsPerSlice*(i+1): , :]

			trainX = np.concatenate((trainX_1, trainX_2))
			trainY = np.concatenate((trainY_1, trainY_2))

			self.fit(trainX, trainY, learningRate = learningRate, numIterations = numIterations, showGraph = showGraph, isCalcErrorPerEpoch = isCalcErrorPerEpoch, testX = testX, testY = testY)
			arrRMSParam.append(self.modelParametersRMS)
			arrMAEParam.append(self.modelParametersMAE)
			
			predRMS = self.predict(testX, isMAE = False)
			predMAE = self.predict(testX, isMAE = True)

			numTestRows = testX.shape[0]
			avgRMS = 0
			avgMAE = 0

			for j in range(numTestRows):
				avgRMS = avgRMS + (predRMS[j] - testY[j])**2
				avgMAE = avgMAE + abs(predMAE[j] - testY[j])

			avgRMS = avgRMS/numTestRows
			avgRMS = math.sqrt(avgRMS)

			avgMAE = avgMAE/numTestRows

			arrErrRMS.append(avgRMS)
			arrErrMAE.append(avgMAE)

			print("RMS and MAE for fold " + str(i+1) + " " + "are " + str(avgRMS) + " " +  str(avgMAE))

		# for last fold

		testX = X[rowsPerSlice*(K-1):, :]
		testY = Y[rowsPerSlice*(K-1):, :]

		trainX = X[0:rowsPerSlice*(K-1): , :]
		trainY = Y[0:rowsPerSlice*(K-1): , :]
		
		self.fit(trainX, trainY, learningRate = learningRate, numIterations = numIterations, showGraph = showGraph, isCalcErrorPerEpoch = isCalcErrorPerEpoch, testX = testX, testY = testY)
		arrRMSParam.append(self.modelParametersRMS)
		arrMAEParam.append(self.modelParametersMAE)
		
		predRMS = self.predict(testX, isMAE = False)
		predMAE = self.predict(testX, isMAE = True)

		numTestRows = testX.shape[0]
		avgRMS = 0
		avgMAE = 0

		for i in range(numTestRows):
			avgRMS = avgRMS + (predRMS[i] - testY[i])**2
			avgMAE = avgMAE + abs(predMAE[i] - testY[i])

		avgRMS = avgRMS/numTestRows
		avgRMS = math.sqrt(avgRMS)

		avgMAE = avgMAE/numTestRows

		arrErrRMS.append(avgRMS)
		arrErrMAE.append(avgMAE)

		print("RMS and MAE for fold " + str(K) + " " + "are " + str(avgRMS) + " " +  str(avgMAE))

		reqIndex1 = -1
	
		minMAE = float("inf")
		for i in range(K):
			if(arrErrMAE[i]<minMAE):
				minMAE = arrErrMAE[i]
				reqIndex1 = i
		print("Minimum MAE " + str(minMAE) + " was achieved at fold " + str(reqIndex1+1))
		print("Parameters for this fold are ", arrMAEParam[reqIndex1])
			
		reqIndex2 = -1
	
		minRMS = float("inf")
		for i in range(K):
			if(arrErrRMS[i]<minRMS):
				minRMS = arrErrRMS[i]
				reqIndex2 = i

		print("Minimum RMS " + str(minRMS) + " was achieved at fold " + str(reqIndex2+1))
		print("Parameters for this fold are ", arrRMSParam[reqIndex2])
			

		return [arrMAEParam[reqIndex1], arrRMSParam[reqIndex2]]




class MyLogisticRegression():
	"""
	My implementation of Logistic Regression.
	"""
	modelParametersStoch = None
	modelParametersBatch = None
	functionBlocked = -1;

	def __init__(self):
		pass

	#Method for stochastic gradient descent
	def stochasticGradientDescent(self, modelParameters, inputArray, outputArray, learningRate, iterations, isCalcErrorPerEpoch, testX, testY):
		
		errHistory = []
		numRows = inputArray.shape[0]
		numCols = inputArray.shape[1]
		errPerEpoch = []

		for k in range(iterations):

			currModelParameters = copy.deepcopy(modelParameters)
			rowNum = random.randint(0, numRows-1)

			for i in range(numCols):
				additionalTerm = 0
				
				#Get a random row
				inputRow = inputArray[rowNum]

				#Calculating hypothesis
				thetaX = np.dot(currModelParameters.T, inputRow)
				try:
					hypothesis = 1/(1 + math.exp(-thetaX))
				except OverflowError:
					hypothesis = 0

				#print("hypothesis", hypothesis)
				additionalTerm = (hypothesis - outputArray[rowNum])*inputRow[i]
				
				modelParameters[i] = modelParameters[i] - (learningRate*additionalTerm)


			#print("model parameters are", modelParameters)
			#print("error is", error/numRows)

			self.modelParametersStoch = modelParameters

			if(isCalcErrorPerEpoch==True):
				errPerEpoch.append(self.getErrorPerEpoch(modelParameters, testX, testY, isBatch = False))
			
			#Calculating error
			error = 0;
			for j in range(numRows):
				inputRow = inputArray[j]

				thetaX = np.dot(currModelParameters.T, inputRow)
				#print(thetaX)
				try:
					hypothesis = 1/(1 + math.exp(-thetaX))
				except OverflowError:
					hypothesis = 0
				#print(hypothesis)

				tempError = (outputArray[j]*math.log(hypothesis + 0.00001) + ((1-outputArray[j])*math.log(1-hypothesis + 0.00001)))
				error = tempError + error
				
			errHistory.append(-(error/numRows))
			
		return [modelParameters, errHistory, errHistory]

		
	#Method for batch gradient descent
	def batchGradientDescent(self, modelParameters, inputArray, outputArray, learningRate, iterations, isCalcErrorPerEpoch, testX, testY):
		
		errHistory = []
		numRows = inputArray.shape[0]
		numCols = inputArray.shape[1]
		errPerEpoch = []

		for k in range(iterations):

			currModelParameters = copy.deepcopy(modelParameters)

			for i in range(numCols):
				additionalTerm = 0
				
				for j in range(numRows):
					inputRow = inputArray[j];
					#print("currModelParameters" , currModelParameters)
					#print("inputRow", inputRow)
					thetaX = np.dot(currModelParameters.T, inputRow)
					try:
						hypothesis = 1/(1 + math.exp(-thetaX))
					except OverflowError:
						hypothesis = 0
					#print("hypothesis", hypothesis)
					additionalTerm = additionalTerm + ((hypothesis - outputArray[j]) * inputRow[i]) 
				
				modelParameters[i] = modelParameters[i] - (learningRate*(1/numRows)*additionalTerm)

			self.modelParametersBatch = modelParameters

			if(isCalcErrorPerEpoch==True):
				errPerEpoch.append(self.getErrorPerEpoch(modelParameters, testX, testY, isBatch = True))

			#Calculating error
			error = 0;
			for j in range(numRows):
				inputRow = inputArray[j]
				thetaX = np.dot(currModelParameters.T, inputRow)
				#print(thetaX)
				try:
					hypothesis = 1/(1 + math.exp(-thetaX))
				except OverflowError:
					hypothesis = 0
				#print(hypothesis)

				tempError = (outputArray[j]*math.log(hypothesis + 0.00001) + ((1-outputArray[j])*math.log(1-hypothesis + 0.00001)))
				error = tempError + error

			errHistory.append(-(error/numRows))

		return [modelParameters, errHistory, errPerEpoch]



	#get loss after each epoch for newly obtained parameters
	def getErrorPerEpoch(self, modelParameters, testX, testY, isBatch):

		numRows = testX.shape[0]
		error = 0


		for i in range(numRows):
			if(isBatch == True):
				thetaXBatch = np.dot(self.modelParametersBatch.T, testX[i])
				try:
					hypothesisBatch = 1/(1 + math.exp(-thetaXBatch))
				except OverflowError:
					hypothesisBatch = 0
				tempError = (testY[i]*math.log(hypothesisBatch + 0.00001) + ((1-testY[i])*math.log(1-hypothesisBatch + 0.00001)))
				error = error + tempError

			else:
				thetaXStoch = np.dot(self.modelParametersStoch.T, testX[i])
				try:
					hypothesisStoch = 1/(1 + math.exp(-thetaXStoch))
				except OverflowError:
					hypothesisStoch = 0
				tempError = (testY[i]*math.log(hypothesisStoch + 0.00001) + ((1-testY[i])*math.log(1-hypothesisStoch + 0.00001)))
				error = error + tempError

		return (-error/numRows)


	''' X is the input training data
		Y is the Output training data
		learning rate and number of iterations are self explanatory
		showgraph decides if you want to construct a graph of training loss vs iterations
		blockFunc can block either RMS or MAE, if 0 blocks nothing, if 1 blocks MAE and if 2 blocks RMS
		isCalcErrorPerEpoch calculates the MAE or RMS for every parameter at each epoch
		testX and testY are provided if calcErrorPerEpoch is true'''

	def fit(self, X, Y, learningRate = 0.1, numIterations = 100, showGraph = False, blockFunc = 0, isCalcErrorPerEpoch = False, testX = None, testY = None):
		"""
		Fitting (training) the logistic model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

		y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
		
		Returns
		-------
		self : an instance of self
		"""

		if(isCalcErrorPerEpoch == True and (testX is None or testY is None)):
			print("Please provide testX and testY to calculate the errors per epoch")
			return None

		elif(isCalcErrorPerEpoch==True):
			
			additionalColumn = np.ones((testX.shape[0], 1))
			testX = np.append(testX, additionalColumn, axis=1)

		# fit function has to return an instance of itself or else it won't work with test.py
		additionalColumn = np.ones((X.shape[0], 1))
		X = np.append(X, additionalColumn, axis=1)

		#Get number of input rows
		numCols = X.shape[1]

		#Initialize the model paramters with 0s
		modelParametersStoch = np.zeros((numCols, 1))
		modelParametersBatch = np.zeros((numCols, 1))


		# # #Batch
		# numIterations = 100
		# learningRate = 0.1

		self.functionBlocked = blockFunc

		if(blockFunc!=1):

			outputBatch = self.batchGradientDescent(modelParametersBatch, X, Y, learningRate, numIterations, isCalcErrorPerEpoch, testX, testY)
			#print("Model parameters batch are", modelParametersBatch, modelParametersBatch.shape)
			modelParametersBatch = outputBatch[0]
			errHistoryBatch = outputBatch[1]
			errPerEpochBatch = outputBatch[2]

			self.modelParametersBatch = modelParametersBatch



		# # # #Stochastic
		# numIterations = 2000
		# learningRate = 10

		if(blockFunc!=2):

			outputStoch = self.stochasticGradientDescent(modelParametersStoch, X, Y, learningRate, numIterations, isCalcErrorPerEpoch, testX, testY)
			#print("Model parameters stoch are", modelParametersStoch, modelParametersStoch.shape)
			modelParametersStoch = outputStoch[0]
			errHistoryStoch = outputStoch[1]
			errPerEpochStoch = outputStoch[2]

			self.modelParametersStoch = modelParametersStoch

		
		#Plot graph
		if(showGraph==True):
			if(blockFunc!=1):
				plt.plot(errHistoryBatch, label = "Batch")
			
			if(blockFunc!=2):
				plt.plot(errHistoryStoch, label = "Stochastic")
			
			plt.legend()
			plt.show()

		if(isCalcErrorPerEpoch==True):
			if(blockFunc!=1):
				plt.plot(errPerEpochBatch, label = "Batch")
			if(blockFunc!=2):
				plt.plot(errPerEpochStoch, label = "Stoch")
			
			plt.legend()
			plt.show()			


		

		# fit function has to return an instance of itself or else it won't work with test.py

		return self

	def predict(self, X, isBatch = True):
		"""
		Predicting values using the trained logistic model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

		Returns
		-------
		y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
		"""

		# return the numpy array y which contains the predicted values

		additionalColumn = np.ones((X.shape[0], 1))
		X = np.append(X, additionalColumn, axis=1)

		numRows = X.shape[0]
		numCols = X.shape[1]

		arrStoch = []
		arrBatch = []

		if(isBatch==True and self.functionBlocked!=1):
			for i in range(numRows):
				thetaXBatch = np.dot(self.modelParametersBatch.T, X[i])
				hypothesisBatch = 1/(1 + math.exp(-thetaXBatch))
				arrBatch.append(hypothesisBatch)

		elif(isBatch==False and self.functionBlocked!=2):
			for i in range(numRows):

				thetaXStoch = np.dot(self.modelParametersStoch.T, X[i])
				try:
					hypothesisStoch = 1/(1 + math.exp(-thetaXStoch))
				except OverflowError:
					hypothesisStoch = 0
				arrStoch.append(hypothesisStoch)
				
		else:
			print("Fit function was not run for the requested predicted values")
			return None

		if(isBatch==True):
			return arrBatch

		else:
			return arrStoch


	''' Divide X and Y into defined ratios
		This functinos returns accuracy for the defined ratios of X and Y using both Stochastic and Batch Gradient Descent
		To see graphs of training loss vs epochs and testing loss vs epochs set showGraph ans isCalcErrorPerEpoch to true'''
	def getAccuracy(self, X, Y, learningRate = 0.1, numIterations = 100, showGraph = False, isCalcErrorPerEpoch = False):

		numRows = X.shape[0]
		rowsPerGroup = int(numRows/10)

		trainX = X[:(7*rowsPerGroup), :]
		trainY = Y[:(7*rowsPerGroup), :]

		valX = X[(7*rowsPerGroup):(8*rowsPerGroup), :]
		valY = Y[(7*rowsPerGroup):(8*rowsPerGroup), :]

		testX = X[(8*rowsPerGroup):, :]
		testY = Y[(8*rowsPerGroup):, :]

		self.fit(trainX, trainY, learningRate = learningRate, numIterations = numIterations, showGraph = showGraph, isCalcErrorPerEpoch = isCalcErrorPerEpoch, testX = testX, testY = testY)

		resultBatch = self.predict(testX, isBatch = True)
		resultStoch = self.predict(testX, isBatch = False)


		#For Testing Data
		numTestRows = testX.shape[0]
		
		correctStoch = 0
		correctBatch = 0

		for i in range(numTestRows):
			currStoch = 0
			if(resultStoch[i]>=0.5):
				currStoch = 1
			if(currStoch==testY[i]):
				correctStoch = correctStoch + 1;
			
			currBatch = 0
			if(resultBatch[i]>=0.5):
				currBatch = 1
			if(currBatch==testY[i]):
				correctBatch = correctBatch + 1;

		accuracyStoch = correctStoch/numTestRows
		accuracyBatch = correctBatch/numTestRows

		#For Training Data
		resultBatch = self.predict(trainX, isBatch = True)
		resultStoch = self.predict(trainX, isBatch = False)

		numTestRows = trainX.shape[0]
		
		correctStoch = 0
		correctBatch = 0

		for i in range(numTestRows):
			currStoch = 0
			if(resultStoch[i]>=0.5):
				currStoch = 1
			if(currStoch==trainY[i]):
				correctStoch = correctStoch + 1;
			
			currBatch = 0
			if(resultBatch[i]>=0.5):
				currBatch = 1
			if(currBatch==trainY[i]):
				correctBatch = correctBatch + 1;

		accuracyStochTrain = correctStoch/numTestRows
		accuracyBatchTrain = correctBatch/numTestRows

		
		
		print("Accuracy of Stoch for training data is", accuracyStochTrain)
		print("Accuracy of Stoch for testing data is", accuracyStoch)
		print("Parameters for stoch are ", self.modelParametersStoch)

		print("Accuracy of Batch for training data is", accuracyBatchTrain)
		print("Accuracy of Batch for testing data is", accuracyBatch)
		print("Parameters for Batch are ", self.modelParametersBatch)

		return [self.modelParametersStoch, self.modelParametersBatch]


	#SK Learn implementation of sklearn to find and compare accuracy with our implemented method
	def skLearnAccuracy(self, X, Y):


		logisticRegr = linear_model.LogisticRegression(max_iter = 2000)
		numRows = X.shape[0]
		rowsPerGroup = int(numRows/10)

		trainX = X[:(7*rowsPerGroup), :]
		trainY = Y[:(7*rowsPerGroup), :]

		valX = X[(7*rowsPerGroup):(8*rowsPerGroup), :]
		valY = Y[(7*rowsPerGroup):(8*rowsPerGroup), :]

		testX = X[(8*rowsPerGroup):, :]
		testY = Y[(8*rowsPerGroup):, :]

		y_arr = []
		for temp_arr in trainY:
			y_arr.append(temp_arr[0])

		logisticRegr.fit(trainX, y_arr)

		y_arr2 = []
		for temp_arr in testY:
			y_arr2.append(temp_arr[0])

			

		print("Accuracy on testing set was", logisticRegr.score(testX, y_arr2))
		print("Accuracy on training set was", logisticRegr.score(trainX, y_arr))
		


		#Maximum Likelihood Estimation for Question 4 implemented via gradient ascent to maximize the likelihood function
	def MLE(self, inputArray, outputArray, learningRate = 0.005, iterations = 60000):
		
		#Adjusting input
		additionalColumn = np.ones((inputArray.shape[0], 1))
		inputArray = np.append(inputArray, additionalColumn, axis=1)

		errHistory = []
		numRows = inputArray.shape[0]
		numCols = inputArray.shape[1]


		#Initialize the model paramters with 0s
		modelParameters = np.zeros((numCols, 1))


		for k in range(iterations):

			currModelParameters = copy.deepcopy(modelParameters)

			for i in range(numCols):
				additionalTerm = 0
				
				for j in range(numRows):
					inputRow = inputArray[j];
					#print("currModelParameters" , currModelParameters)
					#print("inputRow", inputRow)
					thetaX = np.dot(currModelParameters.T, inputRow)
					hypothesis = 1/(1 + math.exp(-thetaX))
					#print("hypothesis", hypothesis)
					additionalTerm = additionalTerm + ((outputArray[j] - hypothesis) * inputRow[i]) 
				
				#print("Additional term is", additionalTerm)
				#print("additional term2", additionalTerm2)

				#As we need to maximize the value we add
				modelParameters[i] = modelParameters[i] + (learningRate*(1/numRows)*additionalTerm)
			
			#print("Model Parameters are", modelParameters)

			#Calculating likelihood
			likelihood = 0;
			for j in range(numRows):
				inputRow = inputArray[j]
				thetaX = np.dot(currModelParameters.T, inputRow)
				#print(thetaX)
				hypothesis = 1/(1 + math.exp(-thetaX))
				#print(hypothesis)

				tempError = (outputArray[j]*math.log(hypothesis + 0.00001) + ((1-outputArray[j])*math.log(1-hypothesis + 0.00001)))
				likelihood = tempError + likelihood

			errHistory.append(likelihood)

		print("Graph of log of likelihood vs iterations is")
		plt.plot(errHistory)
		plt.show()

		return [modelParameters, errHistory]
