import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
import os

bar = '□□□□□□□□□□'
sw = 1
def percent_bar(array,count):   #퍼센트를 표시해주는 함수
    global bar
    global sw
    length = len(array)
    percent = (count/length)*100
    if count == 1 :
        print('preprocessing...txt -> png ')
    print('\r'+bar+'%3s'%str(int(percent))+'%',end='')
    if sw == 1 :
        if int(percent) % 10 == 0 :
            bar = bar.replace('□','■',1)
            sw = 0
    elif sw == 0 :
        if int(percent) % 10 != 0 :
            sw = 1
            
def preprocessing_img(filename) :
    #이미지 로드,리사이즈,이진화,열기연산,외곽선검출
    src = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    src = cv2.resize(src , (int(src.shape[1]/5),int(src.shape[0]/5)))
    src = src[0:src.shape[0]-10, 15:src.shape[1]-25]
    ret , bin = cv2.threshold(src,170,255,cv2.THRESH_BINARY_INV)
    bin = cv2.morphologyEx(bin , cv2.MORPH_OPEN , cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations = 2)
    im2 , contours , hierarchy = cv2.findContours(bin , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    
    #확인용 컬러영상
    color = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color , contours , -1 , (0,255,0),3)
    
    #리스트연산을 위해 초기변수 선언
    bR_arr = []
    digit_arr = []
    digit_arr2 = []
    count = 0
    
    #검출한 외곽선에 사각형을 그려서 배열에 추가
    for i in range(len(contours)) :
        bin_tmp = bin.copy()
        x,y,w,h = cv2.boundingRect(contours[i])
        bR_arr.append([x,y,w,h])
        
    #x값을 기준으로 배열을 정렬
    bR_arr = sorted(bR_arr, key=lambda num : num[0], reverse = False)

    #작은 노이즈데이터 버림,사각형그리기,12개씩 리스트로 다시 묶어서 저장
    for x,y,w,h in bR_arr :
        tmp_y = bin_tmp[y-2:y+h+2,x-2:x+w+2].shape[0]
        tmp_x = bin_tmp[y-2:y+h+2,x-2:x+w+2].shape[1]
        if  tmp_x and tmp_y > 10 :
            count += 1 
            cv2.rectangle(color,(x-2,y-2),(x+w+2,y+h+2),(0,0,255),1)
            digit_arr.append(bin_tmp[y-2:y+h+2,x-2:x+w+2])
            if count == 12 :
                digit_arr2.append(digit_arr)
                digit_arr = []
                count = 0
    
    #리스트에 저장된 이미지를 32x32의 크기로 리사이즈해서 순서대로 저장
    for i in range(0,len(digit_arr2)) :
        for j in range(len(digit_arr2[i])) :
            count += 1 
            if i == 0 :         #1일 경우 비율 유지를 위해 마스크를 만들어 그위에 얹어줌
                width = digit_arr2[i][j].shape[1]
                height = digit_arr2[i][j].shape[0]
                tmp = (height - width)/2
                mask = np.zeros((height,height))
                mask[0:height,int(tmp):int(tmp)+width] = digit_arr2[i][j]
                digit_arr2[i][j] = cv2.resize(mask,(32,32))
            else:
                digit_arr2[i][j] = cv2.resize(digit_arr2[i][j],(32,32))
            if i == 9 : i = -1
            cv2.imwrite('./kNN/testPNGs/'+str(i+1)+'_'+str(j)+'.png',digit_arr2[i][j])
            
    print(str(count)+'files are saved(preprocessing_img2png)')
        
    cv2.imshow('contours',color)
    cv2.imshow('bin',bin)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def preprocessing_txt(path):
    txtPaths = [os.path.join(path,f) for f in os.listdir(path)]
    count = 0
    #파일읽기
    for txtPath in txtPaths :
        count += 1
        filename = os.path.basename(txtPath)
        percent_bar(txtPaths,count)
        f = open(txtPath)
        img = []
        while True :
            tmp=[]
            text = f.readline()
            if not text :
                break
            for i in range(0,len(text)-1) : 
                #라인을 일어올때 text가 1일경우 255로 변경
                if int(text[i]) == 1 :
                    tmp.append(np.uint8(255))
                else :
                    tmp.append(np.uint8(0))
            img.append(tmp)
        img = np.array(img)
        cv2.imwrite('./kNN/trainingPNGs/'+filename.split('.')[0]+'.png',img)
    print('\n'+str(count)+'files are saved(preprocessing_txt2png)')
    
def KNN(train_x,train_y,test_x,test_y):     #knn알고리즘 결과출력
    print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(train_x,train_y)
    pre_arr = clf.predict(test_x)
    pre_arr = pre_arr.reshape(10,12)
    
    print('kNN의 테스트 세트 예측 :\n{}'.format(pre_arr))
    print('kNN의 테스트 세트 정확도 : {0:0.2f}%'.format(clf.score(test_x,test_y)*100))
    print('------------------------------------------------------')
    
def GNB(train_x,train_y,test_x,test_y):     #GaussianNB알고리즘 결과출력
    gnb = GaussianNB()
    gnb.fit(train_x,train_y)
    pre_arr = gnb.predict(test_x)
    pre_arr = pre_arr.reshape(10,12)
    
    print('GaussianNB의 테스트 세트 예측 :\n{}'.format(pre_arr))
    print('GaussianNB의 테스트 세트 정확도 : {0:0.2f}%'.format(gnb.score(test_x,test_y)*100))
    print('------------------------------------------------------')
    
def MNB(train_x,train_y,test_x,test_y):     #MultinomialNB알고리즘 결과출력
    mnb = MultinomialNB()
    mnb.fit(train_x,train_y)
    pre_arr = mnb.predict(test_x)
    pre_arr = pre_arr.reshape(10,12)
    
    print('MultinomialNB의 테스트 세트 예측 :\n{}'.format(pre_arr))
    print('MultinomialNB의 테스트 세트 정확도 : {0:0.2f}%'.format(mnb.score(test_x,test_y)*100))
    print('------------------------------------------------------')
    
def CNB(train_x,train_y,test_x,test_y):     #ComplementNB알고리즘 결과출력
    cnb = ComplementNB()
    cnb.fit(train_x,train_y)
    pre_arr = cnb.predict(test_x)
    pre_arr = pre_arr.reshape(10,12)
    
    print('ComplementNB의 테스트 세트 예측 :\n{}'.format(pre_arr))
    print('ComplementNB의 테스트 세트 정확도 : {0:0.2f}%'.format(cnb.score(test_x,test_y)*100))
    print('------------------------------------------------------')
    
def BNB(train_x,train_y,test_x,test_y):     #BernoulliNB알고리즘 결과출력
    bnb = BernoulliNB()
    bnb.fit(train_x,train_y)
    pre_arr = bnb.predict(test_x)
    pre_arr = pre_arr.reshape(10,12)
    
    print('BernoulliNB의 테스트 세트 예측 :\n{}'.format(pre_arr))
    print('BernoulliNB의 테스트 세트 정확도 : {0:0.2f}%'.format(bnb.score(test_x,test_y)*100))
    print('------------------------------------------------------')
    
def createdataset(directory):       #sklearn사용을 위해 데이터세트를 생성
    files = os.listdir(directory)
    x = []
    y = []
    for file in files:
        attr_x = cv2.imread(directory+file, cv2.IMREAD_GRAYSCALE)
        attr_x = attr_x.flatten()
        attr_y = file[0]
        x.append(attr_x)
        y.append(attr_y)
        
    x = np.array(x)
    y = np.array(y)
    return x , y
    
preprocessing_img('./digits.jpg')
preprocessing_txt('./kNN/trainingDigits')

print('Machine learning Calculating Result..')

train_dir = './kNN/trainingPNGs/'
train_x ,train_y = createdataset(train_dir)

test_dir = './kNN/testPNGs/'
test_x , test_y = createdataset(test_dir)

KNN(train_x, train_y, test_x, test_y)
GNB(train_x, train_y, test_x, test_y)
MNB(train_x, train_y, test_x, test_y)
CNB(train_x, train_y, test_x, test_y)
BNB(train_x, train_y, test_x, test_y)