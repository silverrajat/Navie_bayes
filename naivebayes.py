

import pandas as pd
import sklearn.naive_bayes as nb
import helper as hp
@hp.timeit
def fitNaiveBayes(data):
    naivebayes_classifier=nb.GaussianNB()
    return naivebayes_classifier.fit(data[0],data[1])
csv_data=pd.read_csv("C:/Users/DELL/Desktop/niit/3rd semester/bank_contacts.csv")
train_x,train_y,test_x,test_y,labels=hp.split_data(csv_data,y='credit_application')
classifier=fitNaiveBayes((train_x,train_y))
predicted=classifier.predict(test_x)
hp.printModelSummary(test_y,predicted)

print('Naive Bayes model fitted successfully')





