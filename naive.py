from numpy import *
from sklearn.naive_bayes import GaussianNB

def loadDataSet():
    postingList = [['my','dog','has','flea','problem','help','please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet :
        vocabSet = vocabSet | set(document)
        
    return list(vocabSet)

def setOfWordsVec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet :
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else : print('the word: %s is not in my Vacabulary!' %word)
    return returnVec
def trainNB0(trainMatrix , trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denum = 0.0
    p1Denum = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denum += sum(trainMatrix[i])
        else :
            p0Num += trainMatrix[i]
            p0Denum += sum(trainMatrix[i])
            
    p1Vect = p1Num/p1Denum
    p0Vect = p0Num/p0Denum
    return p0Vect,p1Vect,pAbusive
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + log(1.0 - pClass1)
    if p1>p0 :
        return 1
    else:
        return 0
    
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWordsVec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWordsVec(myVocabList, testEntry))
    print(testEntry,'classified as : ', classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWordsVec(myVocabList, testEntry))
    print(testEntry,'classified as : ', classifyNB(thisDoc,p0V,p1V,pAb))
listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print(myVocabList)    
print(setOfWordsVec(myVocabList, listOPosts[0]))
print(setOfWordsVec(myVocabList, listOPosts[3]))
#vec2Classify = array(setOfWordsVec(myVocabList, inputSet=['love','dog','cute']))
vec2Classify = array(setOfWordsVec(myVocabList, inputSet=['stop','garbage','stupid']))
trainMat = []
for postinDoc in listOPosts :
    trainMat.append(setOfWordsVec(myVocabList, postinDoc))
p0V, p1V, pAb = trainNB0(trainMat, listClasses)
print('-----------pAbusive----------')
print(pAb)
print('-----------p0Vector----------\n')
print(p0V)
print('-----------p1Vector----------\n')
print(p1V)
print(classifyNB(vec2Classify, p0V, p1V, pAb))
testingNB()

gnb = GaussianNB()
gnb.fit(trainMat,listClasses)
print(gnb.predict(trainMat[:2]))
