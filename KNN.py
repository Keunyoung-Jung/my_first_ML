import matplotlib.pyplot as plt
import numpy as np
import operator
from os import listdir

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32) :
        lineStr = fr.readline()
        for j in range(32) :
            returnVect[0,32*i+j] = int(lineStr[j])
            
    return returnVect

def classify(test,dataSet, label,k=3):
    distance = np.sqrt(((dataSet - test)**2).sum(1))
    
    Klabels = np.array(label)[distance.argsort()][:k]
    
    d = {}
    for l in Klabels :
        d[l] = d.get(l,0) + 1
        
    sortedList = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    
    return sortedList[0][0]

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m) :
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
        
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest) :
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest,trainingMat,hwLabels,3)
        print("th classifier came back with : %d , the real answer is : %d" % (classifierResult,classNumStr))
        
        if (classifierResult != classNumStr) : errorCount += 1.0
        
    print("\nthe total number of errors is : %d" % errorCount)
    print("\nthe total error rate is : %f "% (errorCount/float(mTest)))


