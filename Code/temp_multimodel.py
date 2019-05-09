#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:19:21 2019

@author: sklarjg
"""

class modelResults:
    def __init__(self):
        self.tprs = []                    
        self.aucs = []                         
        self.importances = []
        self.accuracy = []                         
        self.precision = []                         
        self.recall = []                         
        self.f1 = []             
        self.shuffled_aucs = [] 
        self.shuffled_f1 = [] 
        self.shuffled_tprs = []   
        self.mean_fpr = np.linspace(0, 1, 101)  
    
    def to_df(self):
        return 1

class AGPCohortClassification:

    def __init__(self, feature_name, cohort, repeats, plot, save, title):
        self.feature_name = feature_name
        self.cohort = cohort
        self.rep = repeats
        self.plot = plot
        self.save = save
        self.title = title

    def classifyFeature(self):
        X, y = self.buildDataSubset()
        self.RepeatedKFoldCV(X, y)

    def buildDataSubset(self):
        X = otu_df.loc[self.cohort.index,:].astype(float).values
        y = self.cohort["target"]
        print y.value_counts()
        le = LabelEncoder()
        le.fit(y)
        X = np.log(X + 1.0)
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        X, y = shuffle(X, y)
        if len(X) > 2000:
            X = X[:2000,:]
            y = y[:2000]
        return X, y        

    def RepeatedKFoldCV(self, X, y):
        self.xgb = modelResults()
        self.rf = modelResults()
        self.lasso = modelResults()

        
        ##Repeat Cross Validation
        for n in range(1, self.rep + 1):
            print(str(n) + ","),
            y_shuffled = shuffle(y)
            cv = KFold(n_splits = 5, shuffle = True)
            for fold_num, (train, test) in enumerate(cv.split(X, y)):           
                
                #XGBoost
                self.trainXGB(X[train], X[test], y[train], y[test], False)
                self.trainXGB(X[train], X[test], y_shuffled[train], y_shuffled[test], True)
                
                self.trainRF(X[train], X[test], y[train], y[test], False)
                self.trainRF(X[train], X[test], y_shuffled[train], y_shuffled[test], True)
                
                self.trainLasso(X[train], X[test], y[train], y[test], False)
                self.trainLasso(X[train], X[test], y_shuffled[train], y_shuffled[test], True)
                
                
                
                
        ###average importance values over all repeats/folds
        self.importances = np.stack(self.importances)
        self.importances = pd.DataFrame(self.importances, columns = otu_df.columns).mean(axis = 0).sort_values(ascending = False)
        
        ##save importanes/ plot rocs for binary/diseas features (frequency cohorts are saved seperately)
        if self.save :
            self.importances.to_csv(save_path + "Importances/" + self.feature_name + ".csv")
        if self.plot:
            self.plotROC()
    
    def plotROC(self):
        plt.figure(1, figsize=(6, 6))
        plt.plot([-0.05, 1.05], [-0.05, 1.05], linestyle=':', lw=2, color='k', alpha=.8)
        
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        plt.plot(self.mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=0.9)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='b', alpha=.3, label=r'$\pm$ 1 std. dev.')
        
        mean_tpr = np.mean(self.shuffled_tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(self.shuffled_tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.shuffled_aucs)
        plt.plot(self.mean_fpr, mean_tpr, color='r',label=r'Mean Shuffled-ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=0.9)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='r', alpha=.3, label=r'$\pm$ 1 std. dev.')
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.title)
        plt.legend(loc="lower right")
        
        if self.save :
            plt.savefig(save_path + "ROCs/" + self.feature_name + ".png", dpi = 300)
        plt.show()
    
    
    def trainXGB(self, X_train, X_test, y_train, y_test, shuffle):
        
        alg = XGBClassifier(n_estimators=256, max_depth=5, learning_rate=0.1, colsample_bytree = 0.7, 
                            nthread=4, scoring='roc_auc', reg_alpha = 5, gamma = 0.1,
                            objective='binary:logistic', seed= 1235) 
        ##training and probability prediction:
        alg.fit(X_train, y_train, verbose=False, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=50) 
        y_pred = alg.predict_proba(X_test)[:,1]
        y_pred_class = alg.predict(X_test)
        y_true = y_test
        
        ##Evaluation Metrics:
        imp = alg.feature_importances_        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(y_true, y_pred_class)
        acc = accuracy_score(y_true, y_pred_class)
        prec = precision_score(y_true, y_pred_class)
        rec = recall_score(y_true, y_pred_class)
        
        if shuffle:
            self.shuffled_tprs.append(interp(self.mean_fpr, fpr, tpr))  ##For plotting roc curves
            self.shuffled_tprs[-1][0] = 0.0
            self.shuffled_aucs.append(roc_auc)
            self.shuffled_f1.append(f1)  
        else:
            self.importances.append(imp)
            self.tprs.append(interp(self.mean_fpr, fpr, tpr))  ##For plotting roc curves
            self.tprs[-1][0] = 0.0
            self.aucs.append(roc_auc)
            self.accuracy.append(acc)                        
            self.precision.append(prec)                          
            self.recall.append(rec)                         
            self.f1.append(f1)  
    
    def trainRF(self):
        
    def trainLasso(self):