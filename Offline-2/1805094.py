# %%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
np.random.seed(94)

# %%
def calc_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    return entropy(counts, base=2)

def calc_remainder(df, feature, target):
    feature_values = df[feature].unique()
    remainder = 0
    for value in feature_values:
        target_given_feature = df[df[feature] == value][target]
        weight = len(target_given_feature) / len(df)
        remainder += weight * calc_entropy(target_given_feature)
    return remainder

def calc_information_gain(df, feature, target):
    return calc_entropy(df[target]) - calc_remainder(df, feature, target)

def information_gain(df, target):
    features = df.columns.drop(target)
    return {feature: calc_information_gain(df, feature, target) for feature in features}

# %%
def telcoCustomerDataPreprocessing(K):
    dataFrame = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # print(dataFrame.shape)
    dataFrame = dataFrame.drop(['customerID'], axis=1)
    # print(dataFrame.shape)
    # dataFrame['MultipleLines'] = dataFrame['MultipleLines'].replace('No phone service', 'No')
    # for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
    #     dataFrame[col] = dataFrame[col].replace('No internet service', 'No')
    dataFrame['TotalCharges'] = dataFrame['TotalCharges'].replace(' ', np.nan)
    dataFrame['tenure'] = dataFrame['tenure'].astype(float)
    dataFrame['MonthlyCharges'] = dataFrame['MonthlyCharges'].astype(float)
    dataFrame['TotalCharges'] = dataFrame['TotalCharges'].astype(float)
    dataFrame['TotalCharges'] = dataFrame['TotalCharges'].fillna(dataFrame['TotalCharges'].mean())
    dataFrame['Churn']=dataFrame['Churn'].replace({'Yes':1,'No':0})
    # print(dataFrame['gender'].unique())
    dataFrame['gender']=dataFrame['gender'].replace({'Male':1,'Female':0})
    dataFrame['Partner']=dataFrame['Partner'].replace({'Yes':1,'No':0})
    dataFrame['Dependents']=dataFrame['Dependents'].replace({'Yes':1,'No':0})
    dataFrame['PhoneService']=dataFrame['PhoneService'].replace({'Yes':1,'No':0})
    dataFrame['PaperlessBilling']=dataFrame['PaperlessBilling'].replace({'Yes':1,'No':0})
    # print(dataFrame['PaperlessBilling'].unique())
    dataFrame=pd.get_dummies(
        dataFrame,
        columns=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                 'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']
    )
    # assign 0 to false and 1 to true for DataFrame
    dataFrame = dataFrame.replace({False: 0, True: 1})
    trainDataFrame, testDataFrame = train_test_split(dataFrame, test_size=0.2, random_state=94)
    trainDataFrame[['tenure','MonthlyCharges','TotalCharges']]=StandardScaler().fit_transform(trainDataFrame[['tenure','MonthlyCharges','TotalCharges']])
    testDataFrame[['tenure','MonthlyCharges','TotalCharges']]=StandardScaler().fit_transform(testDataFrame[['tenure','MonthlyCharges','TotalCharges']])

    #calculate information gain for each parameters and get get the top 10 parameters
    topTen=information_gain(trainDataFrame, 'Churn')
    # get the top 10 parameters
    topTen=sorted(topTen.items(), key=lambda x: x[1], reverse=True)
    topTen=topTen[:K]
    # print(topTen)

    trainDataFrame_target = trainDataFrame['Churn']
    trainDataFrame = trainDataFrame.drop(['Churn'], axis=1)
    testDataFrame_target = testDataFrame['Churn']
    testDataFrame = testDataFrame.drop(['Churn'], axis=1)

    #remove columns not in top 10
    for col in trainDataFrame.columns:
        if col not in [x[0] for x in topTen]:
            trainDataFrame=trainDataFrame.drop([col], axis=1)
            testDataFrame=testDataFrame.drop([col], axis=1)
    


    return trainDataFrame.to_numpy(), trainDataFrame_target.to_numpy(), testDataFrame.to_numpy(), testDataFrame_target.to_numpy()

    # dataFrame[['tenure','MonthlyCharges','TotalCharges']]=StandardScaler().fit_transform(dataFrame[['tenure','MonthlyCharges','TotalCharges']])

    # dataFrame_target = dataFrame['Churn']
    # dataFrame = dataFrame.drop(['Churn'], axis=1)

    #output dataFrame to file
    # dataFrame.to_csv('telcoCustomerDataPreprocessing.csv', index=False)
    # return dataFrame.to_numpy(), dataFrame_target.to_numpy()

# trainDataFrame, trainDataFrame_target, testDataFrame, testDataFrame_target = telcoCustomerDataPreprocessing(0)
# print(dataFrame)

# %%
def adultSalaryPreprocessing(K):
    trainDataFrame = pd.read_csv('adult.data')
    # testDataFrame = pd.read_csv('adult_test.csv')

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
    trainDataFrame.columns = columns
    # print(trainDataFrame['income'].unique())
    testDataFrame = pd.read_csv('adult.test',  names=columns)
    # check if first row, first column contains 'Cross validator'
    # print(testDataFrame.iloc[0,0])
    # testDataFrame=testDataFrame.drop([0],axis=0)
    if testDataFrame.iloc[0,0]=='|1x3 Cross validator':
        #remove first row
        testDataFrame=testDataFrame.drop([0],axis=0)
        # print(len(testDataFrame.columns))
    # print(testDataFrame.iloc[0,0])
    testDataFrame.columns = columns

    trainDataFrame = trainDataFrame.map(lambda x: x.strip() if isinstance(x, str) else x)
    testDataFrame = testDataFrame.map(lambda x: x.strip() if isinstance(x, str) else x)
    trainDataFrame['income'] =trainDataFrame['income'].replace({'<=50K':0,'>50K':1})
    testDataFrame['income'] =testDataFrame['income'].replace({'<=50K.':0,'>50K.':1})
    # print(trainDataFrame['income'].unique())
    imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    trainDataFrame[['workclass','occupation','native-country']]=trainDataFrame[['workclass','occupation','native-country']].replace('?',np.nan)
    testDataFrame[['workclass','occupation','native-country']]=testDataFrame[['workclass','occupation','native-country']].replace('?',np.nan)
    trainDataFrame[['workclass','occupation','native-country']]=imputer.fit_transform(trainDataFrame[['workclass','occupation','native-country']])
    testDataFrame[['workclass','occupation','native-country']]=imputer.fit_transform(testDataFrame[['workclass','occupation','native-country']])
    # trainDataFrame = trainDataFrame[trainDataFrame['native-country'] != 'Holand-Netherlands']
    # testDataFrame = testDataFrame[testDataFrame['native-country'] != 'Holand-Netherlands']
    trainDataFrame=pd.get_dummies(
        trainDataFrame,
        columns=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    )
    testDataFrame=pd.get_dummies(
        testDataFrame,
        columns=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    )
    trainDataFrame=trainDataFrame.replace({False:0,True:1})
    testDataFrame=testDataFrame.replace({False:0,True:1})

    for col in ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']:
        trainDataFrame[col]=trainDataFrame[col].astype(float)
        testDataFrame[col]=testDataFrame[col].astype(float)
        trainDataFrame[col]=StandardScaler().fit_transform(trainDataFrame[[col]])
        testDataFrame[col]=StandardScaler().fit_transform(testDataFrame[[col]])

    topTen=information_gain(trainDataFrame, 'income')
    # get the top 10 parameters
    topTen=sorted(topTen.items(), key=lambda x: x[1], reverse=True)
    topTen=topTen[:K]
    # print(topTen)

    trainDataFrame_target = trainDataFrame['income']
    trainDataFrame = trainDataFrame.drop(['income'], axis=1)
    testDataFrame_target = testDataFrame['income']
    testDataFrame = testDataFrame.drop(['income'], axis=1)

    for col in trainDataFrame.columns:
        if col not in [x[0] for x in topTen]:
            trainDataFrame=trainDataFrame.drop([col], axis=1)
            if col not in testDataFrame.columns:
                testDataFrame[col]=0
            testDataFrame=testDataFrame.drop([col], axis=1)

    # trainDataFrame.to_csv('adult_train.csv',index=False)

    return trainDataFrame.to_numpy(), trainDataFrame_target.to_numpy(), testDataFrame.to_numpy(), testDataFrame_target.to_numpy()

    # trainDataFrame=trainData
    
    



# adultSalaryPreprocessing()

# %%
def creditCardDataPreprocessing(K,partial=True):
    dataFrame=pd.read_csv('creditcard.csv')
    dataFrameNegatives=dataFrame[dataFrame['Class']==0]
    dataFramePositives=dataFrame[dataFrame['Class']==1]
    trainDataFrameNegatives, testDataFrameNegatives=train_test_split(dataFrameNegatives, test_size=0.2, random_state=94)
    trainDataFramePositives, testDataFramePositives=train_test_split(dataFramePositives, test_size=0.2, random_state=94)
    if(partial):
        trainDataFrameNegatives=trainDataFrameNegatives.sample(n=16000, random_state=94)
        testDataFrameNegatives=testDataFrameNegatives.sample(n=4000, random_state=94)
    trainDataFrame=pd.concat([trainDataFrameNegatives, trainDataFramePositives],ignore_index=True).sample(frac=1, random_state=94)
    testDataFrame=pd.concat([testDataFrameNegatives, testDataFramePositives],ignore_index=True).sample(frac=1, random_state=94)
    
    for col in trainDataFrame.columns:
        if col!='Class':
            trainDataFrame[col]=StandardScaler().fit_transform(trainDataFrame[[col]])
            testDataFrame[col]=StandardScaler().fit_transform(testDataFrame[[col]])
    p=K
    # topTen=information_gain(trainDataFrame, 'Class')
    # # get the top 10 parameters
    # # topTen=sorted(topTen.items(), key=lambda x: x[1], reverse=True)
    # topTen=topTen[:K]
    # print(topTen)
    
    trainDataFrame_target=trainDataFrame['Class']
    testDataFrame_target=testDataFrame['Class']
    trainDataFrame=trainDataFrame.drop(['Class'], axis=1)
    testDataFrame=testDataFrame.drop(['Class'], axis=1)

    # for col in trainDataFrame.columns:
    #     if col not in [x[0] for x in topTen]:
    #         trainDataFrame=trainDataFrame.drop([col], axis=1)
    #         testDataFrame=testDataFrame.drop([col], axis=1)
    #conduct loop for all columns except Class

    

    return trainDataFrame.to_numpy(), trainDataFrame_target.to_numpy(), testDataFrame.to_numpy(), testDataFrame_target.to_numpy()

# trainDataFrame, trainDataFrame_target, testDataFrame, testDataFrame_target = creditCardDataPreprocessing(3)
# print(trainDataFrame.shape)

# %%
def gradientDescent(X,y,y_hat):
    n=X.shape[0]
    dw=np.dot(X.T,(y-y_hat)*y_hat*(1-y_hat))/n
    return dw

def train(X,y):
    y=y.reshape(y.shape[0],1)
    X=(X-X.mean(axis=0))/X.std(axis=0) 
    samples=X.shape[0]
    features=X.shape[1]
    epochs=1000
    lr=0.01
    w=np.zeros((features+1,1))
    X=np.concatenate((np.ones((samples,1)),X),axis=1)
    y=y.reshape(samples,1)
    y_hat=1/(1+np.exp(-np.dot(X,w)))
    # print(y_hat.shape)
    for i in range(epochs):
        y_hat=1/(1+np.exp(-np.dot(X,w)))
        dw=gradientDescent(X,y,y_hat)
        w=w+lr*dw
        loss=np.mean((y-(1/(1+np.exp(-np.dot(X,w)))))**2)
        earlyTerminateThreshold=0.5
        if(loss<earlyTerminateThreshold):
            break
    return w




# %%
def predict(X,w):
    
    X=(X-X.mean(axis=0))/X.std(axis=0)
    samples=X.shape[0]
    X=np.concatenate((np.ones((samples,1)),X),axis=1)
    y_hat=1/(1+np.exp(-np.dot(X,w)))
    calculatedTargets=[]
    for i in y_hat:
        if(i>0.5):
            calculatedTargets.append(1)
        else:
            calculatedTargets.append(0)
    return np.array(calculatedTargets)

# %%
def performance(y,y_hat):
    y=y.reshape(y.shape[0],1)
    y_hat=y_hat.reshape(y_hat.shape[0],1)
    accuracy = accuracy_score(y, y_hat)
    sensitivity = recall_score(y,y_hat) 
    precision = precision_score(y,y_hat)

    f1Score = f1_score(y,y_hat)
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()

    specificity = tn / (tn+fp)
    falseDiscoveryRate = fp / (tp+fp)

    return accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score

# %% [markdown]
# Churn Logistic regression

def churnLogisticRegression(K):

    trainChurnFeatures, trainChurnTarget, testChurnFeatures, testChurnTarget = telcoCustomerDataPreprocessing(K)


    w=train(trainChurnFeatures,trainChurnTarget)
    accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(trainChurnTarget,predict(trainChurnFeatures,w))
    print('Logistic Regression: ============')
    print('Churn train: ============')
    print('accuracy: ',accuracy)
    print('sensitivity: ',sensitivity)
    print('specificity: ',specificity)
    print('precision: ',precision)
    print('falseDiscoveryRate: ',falseDiscoveryRate)
    print('f1Score: ',f1Score)

    accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(testChurnTarget,predict(testChurnFeatures,w))
    print('Churn test: ============')
    print('accuracy: ',accuracy)
    print('sensitivity: ',sensitivity)
    print('specificity: ',specificity)
    print('precision: ',precision)
    print('falseDiscoveryRate: ',falseDiscoveryRate)
    print('f1Score: ',f1Score)


# %% [markdown]
# Adult Logistic Regression
def adultLogisticRegression(K):
    trainAdultFeatures, trainAdultTarget, testAdultFeatures, testAdultTarget = adultSalaryPreprocessing(K)

    w=train(trainAdultFeatures,trainAdultTarget)
    accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(trainAdultTarget,predict(trainAdultFeatures,w))
    print('Logistic Regression: ============')
    print('Adult train: ============')
    print('accuracy: ',accuracy)
    print('sensitivity: ',sensitivity)
    print('specificity: ',specificity)
    print('precision: ',precision)
    print('falseDiscoveryRate: ',falseDiscoveryRate)
    print('f1Score: ',f1Score)

    accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(testAdultTarget,predict(testAdultFeatures,w))
    print('Adult test: ============')
    print('accuracy: ',accuracy)
    print('sensitivity: ',sensitivity)
    print('specificity: ',specificity)
    print('precision: ',precision)
    print('falseDiscoveryRate: ',falseDiscoveryRate)
    print('f1Score: ',f1Score)

# %% [markdown]
# Credit Card Logistic Regression
def creditCardLogisticRegression(K,partial=True):
    trainCreditFeatures, trainCreditTarget, testCreditFeatures, testCreditTarget=creditCardDataPreprocessing(K,partial)

    w=train(trainCreditFeatures,trainCreditTarget)
    accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(trainCreditTarget,predict(trainCreditFeatures,w))
    print('Logistic Regression: ============')
    print('Creditcard train: ============')
    print('accuracy: ',accuracy)
    print('sensitivity: ',sensitivity)
    print('specificity: ',specificity)
    print('precision: ',precision)
    print('falseDiscoveryRate: ',falseDiscoveryRate)
    print('f1Score: ',f1Score)

    accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(testCreditTarget,predict(testCreditFeatures,w))
    print('Creditcard test: ============')
    print('accuracy: ',accuracy)
    print('sensitivity: ',sensitivity)
    print('specificity: ',specificity)
    print('precision: ',precision)
    print('falseDiscoveryRate: ',falseDiscoveryRate)
    print('f1Score: ',f1Score)



# %% [markdown]
# AdaBoost

# %%
def adaboost(X,y,K):
    samples=X.shape[0]
    y=y.reshape(y.shape[0],1)
    w=np.full((samples),1/samples)
    hypotheses=[]
    np.random.seed(94)
    z=[]
    example=np.concatenate((X,y),axis=1)
    features=example.shape[1]-y.shape[1]
    targetCols=y.shape[1]
    for i in range(K):
        data=example[np.random.choice(samples,size=samples,replace=True,p=w)]
        data_X=data[:,:features]
        data_y=data[:,-targetCols:]
        h=train(data_X,data_y)
        y_hat=predict(X,h)
        error=0
        for j in range(samples):
            if(y[j]!=y_hat[j]):
                error+=w[j]
            else:
                error+=0
        if(error>0.5):
            continue
        hypotheses.append(h)
        for j in range(samples):
            if(y[j]==y_hat[j]):
                w[j]=w[j]*error/(1-error)
            else:
                w[j]=w[j]*1
        #normalize w
        w=w/np.sum(w)
        # print(w)
        z.append(np.log2((1-error)/error))
    z=np.array(z)
    # print('z shape: ',z.shape)
    # print('hypotheses shape: ',len(hypotheses) )
    return hypotheses,z

    

# %%
def weightedMajorityPredict(X,hypotheses,z):
    z=z.reshape(z.shape[0],1)
    samples=X.shape[0]
    nHypos=len(hypotheses)
    # print('nHypos: ',nHypos)
    X=(X-X.mean(axis=0))/X.std(axis=0)
    X=np.concatenate((np.ones((samples,1)),X),axis=1)
    y_hat=[]
    for i in range(nHypos):
        temp=1/(1+np.exp(-np.dot(X,hypotheses[i])))
        arr=[]
        for j in temp:
            if(j>=0.5):
                arr.append(1)
            else:
                arr.append(-1)

        y_hat.append(arr)
        
    y_hat=np.array(y_hat)
    # print(y_hat.shape)
    # print(z.shape)
    weight=np.dot(y_hat.T,z)
    calculatedTargets=[]
    for i in weight:
        if(i>=0):
            calculatedTargets.append(1)
        else:
            calculatedTargets.append(0)

    return np.array(calculatedTargets)


# %% [markdown]
# Churn Adaboost
def churnAdaboost(K):
    trainChurnFeatures, trainChurnTarget, testChurnFeatures, testChurnTarget = telcoCustomerDataPreprocessing(K)

    trainChurnTarget=trainChurnTarget.reshape(trainChurnTarget.shape[0],1)
    testChurnTarget=testChurnTarget.reshape(testChurnTarget.shape[0],1)
    print('Adaboost: ============')
    for k in range(5,21,5):
        hypothesis,z=adaboost(trainChurnFeatures,trainChurnTarget,k)
        accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(trainChurnTarget,weightedMajorityPredict(trainChurnFeatures,hypothesis,z))
        print(f'Train Churn k={k}: ============ Accuracy: {accuracy}')

        accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(testChurnTarget,weightedMajorityPredict(testChurnFeatures,hypothesis,z))
        print(f'Test Churn k={k}: ============ Accuracy: {accuracy}')

# %% [markdown]
# Adult AdaBoost
def adultAdaboost(K):
    trainAdultFeatures, trainAdultTarget, testAdultFeatures, testAdultTarget = adultSalaryPreprocessing(K)

    print('Adaboost: ============')
    for k in range(5,21,5):
        hypothesis,z=adaboost(trainAdultFeatures,trainAdultTarget,k)
        accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(trainAdultTarget,weightedMajorityPredict(trainAdultFeatures,hypothesis,z))
        print(f'Train Adult k={k}: ============ Accuracy: {accuracy}')

        accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(testAdultTarget,weightedMajorityPredict(testAdultFeatures,hypothesis,z))
        print(f'Test Adult k={k}: ============ Accuracy: {accuracy}')

# %% [markdown]
# Creditcard AdaBoost
def creditCardAdaboost(K,partial=True):
    trainCreditFeatures, trainCreditTarget, testCreditFeatures, testCreditTarget=creditCardDataPreprocessing(K,partial)

    print('Adaboost: ============')
    for k in range(5,21,5):
        hypothesis,z=adaboost(trainCreditFeatures,trainCreditTarget,k)
        accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(trainCreditTarget,weightedMajorityPredict(trainCreditFeatures,hypothesis,z))
        print(f'Train Credit k={k}: ============ Accuracy: {accuracy}')

        accuracy, sensitivity, specificity, precision, falseDiscoveryRate, f1Score=performance(testCreditTarget,weightedMajorityPredict(testCreditFeatures,hypothesis,z))
        print(f'Test Credit k={k}: ============ Accuracy: {accuracy}')


# %% [markdown]
# run specific dataset
datasets=['churn','adult','creditcard']

# take input from user
index=int(input('Enter the index of the dataset you want to run: \n1. Churn\n2. Adult\n3. Creditcard\n'))
if(index==1):
    K=int(input('Enter the number of features you want to use: '))
    churnLogisticRegression(K)
    churnAdaboost(K)
elif(index==2):
    K=int(input('Enter the number of features you want to use: '))
    adultLogisticRegression(K)
    adultAdaboost(K)
elif(index==3):
    K=int(input('Enter the number of features you want to use: '))
    partial=int(input('Enter Choice for dataset size\n1.partial dataset \n0.full dataset '))
    if(partial==1):
        partial=True
    elif(partial==0):
        partial=False
    else:
        print('Invalid input. Please try again.')
    creditCardLogisticRegression(K,partial)
    creditCardAdaboost(K,partial)
else:
    print('Invalid input. Please try again.')
