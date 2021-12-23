# -*- coding: utf-8 -*-

#Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Veri Ön İşleme

#Veri Yükleme 

veriler = pd.read_csv('transfusion.csv')
#print(veriler)
x = veriler.iloc[:,0:4].values #bağımsız değişkenler (İlk 4 kolon)
y = veriler.iloc[:,4:].values #bağımlı değişken (Class kolonu)
#print(y)

#Verilerin Eğitim ve Test Kümesi Olarak Bölünmesi

from sklearn.model_selection  import train_test_split 
#Bağımsız ve bağımlı değişken ayrı ayrı test ve eğitim kümesine bölünür.
#Eğitim kümesi %67, test kümesi %33 olarak, rastgele bir şekilde belirlenir.
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#Verilerin Ölçeklenmesi

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() #StandardScaler sınıfından bir obje üretildi.
X_train = sc.fit_transform(x_train) #x_train kümesinden öğrenir ve değişkene uygular.
X_test = sc.transform(x_test) #x_test'i yeniden öğrenmez, sadece değişkene uygular.


#1. Lojistik Regresyon

#scikit learn kütüphanesinden LogisticRegression sınıfı kullanıldı.
from sklearn.linear_model import LogisticRegression 
logr = LogisticRegression(random_state=0) #LR sınıfından bir nesne üretildi. random_state default değeri 0'dır.
#logr objesi LogisticRegressiondan bilgiyi alır ve bu objeyi X_train ve y_train bilgileriyle eğitiyor.
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test) #Tahmin işlemi yapılır, y_pred değişkenine atanır.


'''
print("Tahmin edilen değerler:")
print(y_pred) #Yapılan tahmin sonuçları.
print("Gerçek test verileri:")
print(y_test) #Gerçek test verisi.
'''

#Karmaşıklık matrisi oluşturulur.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) #Sonuç verilerinin, gerçek ve tahmin değerleri arasında oluşturuldu.
print(" Lojistik Regresyon Karmaşıklık matrisi:")
print(cm)


#2. Naive Bayes
#GausianNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
#print(y_pred)

#Karmaşıklık matrisi oluşturulur.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) #Sonuç verilerinin, gerçek ve tahmin değerleri arasında oluşturuldu.
print("Gaussian Naive Bayes Karmaşıklık matrisi:")
print(cm)



#BernoulliNB
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)
#print(y_pred)

#Karmaşıklık matrisi oluşturulur.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) #Sonuç verilerinin, gerçek ve tahmin değerleri arasında oluşturuldu.
print("Bernoulli Naive Bayes Karmaşıklık matrisi:")
print(cm)


'''
#Accuracy score hesabı
from sklearn.metrics import accuracy_score
accuracyBernoulli = accuracy_score(y_test, y_pred)
print("Bernoulli Accuracy:")
print(accuracyBernoulli)
# Recall
from sklearn.metrics import recall_score
recallBernoulli = recall_score(y_test, y_pred, average=None)
print("Bernoulli Naive Bayes Recall:")
print(recallBernoulli)
# Precision
from sklearn.metrics import precision_score
precisionBernoulli= precision_score(y_test, y_pred, average=None)
print("Bernoulli Naive Bayes Precision:")
print(precisionBernoulli)
'''

# 2. KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("KNN Algoritmasının Karmaşıklık matrisi")
print(cm)

# 3. SVC (SVM classifier)
from sklearn.svm import SVC
svc = SVC(kernel='poly',probability=True)
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC Algoritmasının Karmaşıklık matrisi:')
print(cm)

# 5. Decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('Desicion Tree Karmaşıklık matrisi:')
print(cm)

# 6. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('Random Forest Karmaşıklık matrisi:')
print(cm)



'''
#k-katlamali capraz dogrulama 
from sklearn.model_selection import cross_val_score 

1. estimator : classifier (bizim durum)
2. X
3. Y
4. cv : kaç katlamalı

basari = cross_val_score(estimator = logr, X=X_train, y=y_train , cv = 4)
print(basari.mean())
print(basari.std())
'''


#burada bütün modelleri ımport ediyoruz.
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#burada modelleri bir listenin içerisine alıp parametreleri ile beraber tanımlıyoruz.
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Naive Bayes Bernoulli', BernoulliNB()))
models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
models.append(('K-NN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('RandomForestClassifier', RandomForestClassifier()))

#burada bir döngü vasıtasıyla tek tek bütün modelleri deneyerek sonuçları karşılaştırıyoruz. 
for name, model in models:
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn import metrics
    from sklearn.metrics import classification_report
    print(name,classification_report(y_test, y_pred))

    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(y_test, y_pred)*100))
    print("Recall : ", name,metrics.recall_score(y_test, y_pred,average=None))
    print("%s -> Precision: %%%.2f" % (name,metrics.precision_score(y_test, y_pred)*100))
    
    
#sınıflamanın tüm eşikleri için FPR ve TPR’yi hesaplar!
probs = bnb.predict_proba(X_test)
probs = probs[:, 1]
bnb_fpr, bnb_tpr, thresholds = metrics.roc_curve(y_test, probs)
bnb_auc = metrics.roc_auc_score(y_test, probs)

probs = logr.predict_proba(X_test)
probs = probs[:, 1]
logr_fpr, logr_tpr, thresholds = metrics.roc_curve(y_test, probs)
logr_auc = metrics.roc_auc_score(y_test, probs)


probs = knn.predict_proba(X_test)
probs = probs[:, 1]
knn_fpr, knn_tpr, thresholds = metrics.roc_curve(y_test, probs)
knn_auc = metrics.roc_auc_score(y_test, probs)

probs = dtc.predict_proba(X_test)
probs = probs[:, 1]
dtc_fpr, dtc_tpr, thresholds = metrics.roc_curve(y_test, probs)
dtc_auc = metrics.roc_auc_score(y_test, probs)

probs = rfc.predict_proba(X_test)
probs = probs[:, 1]
rfc_fpr, rfc_tpr, thresholds = metrics.roc_curve(y_test, probs)
rfc_auc = metrics.roc_auc_score(y_test, probs)

probs = svc.predict_proba(X_test)
probs = probs[:, 1]
svc_fpr, svc_tpr, thresholds = metrics.roc_curve(y_test, probs)
svc_auc = metrics.roc_auc_score(y_test, probs)

plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(bnb_fpr, bnb_tpr, 'b', marker='.', label = 'BernoilliNB = %0.8f' % bnb_auc,color="r" )
plt.plot(logr_fpr, logr_tpr, 'b', marker='.', label = 'Logistic = %0.8f' % logr_auc)
plt.plot(knn_fpr, knn_tpr, 'b', marker='.', label = 'KNN = %0.8f' % knn_auc,color="m")
plt.plot(dtc_fpr, dtc_tpr, 'b', marker='.', label = 'DesicionTree = %0.8f' % dtc_auc,color="y")
plt.plot(rfc_fpr, rfc_tpr, 'b', marker='.', label = 'rfc = %0.8f' % rfc_auc, color="c")
plt.plot(svc_fpr, svc_tpr, 'b', marker='.', label = 'svc = %0.3f' % svc_auc,color="g" )

plt.legend(loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Cross Validation

from sklearn.model_selection import cross_val_score
print('***********Çapraz Doğrulama Değerleri***********')
lojistikbasari = cross_val_score(estimator=logr, X= X_train, y= y_train, cv=10)
print("lojistik mean: " , lojistikbasari.mean())
print("lojistik standart sapma: " ,lojistikbasari.std())


gnbasari = cross_val_score(estimator=gnb, X= X_train, y= y_train, cv=10)
print("gausian naive bayes mean: " , gnbasari.mean())
print("gausian naive bayes standart sapma: " ,gnbasari.std())

bnbasari = cross_val_score(estimator=bnb, X= X_train, y= y_train, cv=10)
print("bernoulli naive bayes mean: " , bnbasari.mean())
print("bernoulli naive bayes standart sapma: " ,bnbasari.std())

knnbasari = cross_val_score(estimator=knn, X= X_train, y= y_train, cv=10)
print("KNN mean: " , knnbasari.mean())
print("KNN standart sapma: " ,knnbasari.std())

dtbasari = cross_val_score(estimator=dtc, X= X_train, y= y_train, cv=10)
print("Decision Tree mean: " , dtbasari.mean())
print("Decision Tree standart sapma: " ,dtbasari.std())

rfbasari = cross_val_score(estimator=rfc, X= X_train, y= y_train, cv=10)
print("Random Forest mean: " , rfbasari.mean());
print("Random Forest standart sapma: " ,rfbasari.std())

svcbasari = cross_val_score(estimator=svc, X= X_train, y= y_train, cv=10)
print("SVC mean: " , svcbasari.mean())
print("SVC standart sapma: " ,svcbasari.std())
    
