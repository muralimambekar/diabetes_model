#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 04:33:44 2020

@author: murali
"""
class mlmodels:
    def mlmodels(X_train, X_test, y_train, y_test):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.metrics import confusion_matrix
        from sklearn import metrics
        
        acc=pd.DataFrame(columns=['Algorithm', 'Mean Accuracy', 'Std of Accuracy'])
        from sklearn.ensemble import RandomForestClassifier
        classifier=RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        from sklearn.model_selection import cross_val_score
        a = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        acc=acc.append({'Algorithm': 'Random Forest', 'Mean Accuracy': a.mean()*100, 'Std of Accuracy':a.std()*100},ignore_index=True)
        
        
        # Fitting Logistic Regression to the Training set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        a = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        acc=acc.append({'Algorithm': 'Logist Regression', 'Mean Accuracy': a.mean()*100, 'Std of Accuracy':a.std()*100},ignore_index=True)
       
        #KNN
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        a = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        acc=acc.append({'Algorithm': 'KNeighbors classifier', 'Mean Accuracy': a.mean()*100, 'Std of Accuracy':a.std()*100},ignore_index=True)
        
        
        
        # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        a = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        acc=acc.append({'Algorithm': 'SVM classifier', 'Mean Accuracy': a.mean()*100, 'Std of Accuracy':a.std()*100},ignore_index=True)
        
        
        
        # Fitting Kernel SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        a = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        acc=acc.append({'Algorithm': 'Kernel SVM', 'Mean Accuracy': a.mean()*100, 'Std of Accuracy':a.std()*100},ignore_index=True)
        
        
        
        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        a = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        acc=acc.append({'Algorithm': 'Naive Bayes', 'Mean Accuracy': a.mean()*100, 'Std of Accuracy':a.std()*100},ignore_index=True)
       
        
        
        # Fitting Decision Tree Classification to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        a = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        acc=acc.append({'Algorithm': 'Desicion Tree classifier', 'Mean Accuracy': a.mean()*100, 'Std of Accuracy':a.std()*100},ignore_index=True)
        
        # Fitting XGBoost to the Training set
        from xgboost import XGBClassifier
        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        a = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        acc=acc.append({'Algorithm': 'XGB classifier', 'Mean Accuracy': a.mean()*100, 'Std of Accuracy':a.std()*100},ignore_index=True)
            
        return acc
        

