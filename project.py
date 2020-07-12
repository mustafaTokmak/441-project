import pandas as pd 
import time 
import random 
import numpy as np
import json
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt



"""data = pd.read_csv("cmu_data.csv") 
print(data.head())
start = time.time()
data_dict = {}
for index, rows in data.iterrows(): 
    #print(data.iloc[index,3:].values)
    #print(data.iloc[index,0])
    print(index)
    if not data.iloc[index,0] in data_dict:
        data_dict[data.iloc[index,0]] = []
    data_dict[data.iloc[index,0]].append(data.iloc[index,3:].tolist())    
end = time.time()
duration = end-start
print(duration)

with open('cmu_dict_data','w') as f:
    f.write(json.dumps(data_dict,indent=True))"""


with open('cmu_dict_data','r') as f:
    data = f.read()

data_dict = json.loads(data)

print(len(data_dict))


train_and_test_data = {}
print(len(data_dict))
train_size = 20
test_size = 200
test_subject_size = test_size / 50
def get_random_test_and_train_sample(data_dict,subject,train_size,test_size):
    train = []
    test = []
    temp = data_dict[subject]
    random.shuffle(temp)
    for i in range(train_size):
        t = temp[i]
        train.append(t)

    for i in range(test_size):
        t = temp[train_size+i]
        test.append(t)
    
    for key in data_dict.keys():
        if key == subject:
            continue
        test = test + data_dict[key][:test_subject_size]
        
    """for i in range(int(test_size/test_subject_size)):
        while True:
            r = random.randint(0,len(data_dict.keys())-1)
            key = (list(data_dict.keys()))[r]
            if key == subject:
                continue
            else:
                break
        temp = data_dict[key]
        random.shuffle(temp)
        temp = temp[:test_subject_size]
        test = test + temp"""
    return test,train

test,train = get_random_test_and_train_sample(data_dict,"s002",train_size,test_size)
print(len(test))
print(len(train))


start = time.time()
for key in data_dict.keys():
    train_and_test_data[key] = {"train":[],"test":[]} 
    train = []
    test = []
    test,train = get_random_test_and_train_sample(data_dict,key,train_size,test_size)
    train_and_test_data[key]["train"] = train
    train_and_test_data[key]["test"] = test

end = time.time()
duration = end-start
print(duration)


"""    
params = {'max_features': 0.1, 'max_samples': 5,
            'n_estimators': 300, 'n_jobs': -1, 'random_state': 11,'behaviour':'new'}
            """
classifier = IsolationForest(contamination=0, max_features=1,
            max_samples=500, n_estimators=3000, n_jobs=-1, random_state=11)

#classifier = IsolationForest(contamination=0)
from sklearn import  metrics
import matplotlib.pyplot as plt

def get_eer(tpr,fpr):
    eer = 100
    for i in range(len(tpr)):
        if 1-tpr[i] < fpr[i]:
            eer = ( (1-tpr[i]) + (1-tpr[i-1]) + fpr[i] + fpr[i-1])/4
            break
    #print(eer)
    return eer 
    
#classifier.set_params(**params)
start = time.time()
scores = {}
result_scores = {}
aucs = []
eers = []

weight = []
for i in range(test_size*2):
    weight.append(-0.5 + i*(1/(test_size*2)))
tpr_list = []
fpr_list = []
for key in data_dict.keys():
    print(key)
    predictors_data = train_and_test_data[key]["train"]
    target_data = [1]*train_size
    classifier.fit(predictors_data, target_data)
    score = classifier.decision_function(train_and_test_data[key]["test"])
    #print(score)
    scores[key] = score.tolist()
    y = np.array(([1]*test_size)+([-1]*test_size))
    #print(y)
    fpr, tpr, thresholds = metrics.roc_curve(y, score,drop_intermediate=False)
    tpr_list.append(tpr[:350])
    fpr_list.append(fpr[:350])

    print(len(thresholds))
    eer = get_eer(tpr,fpr)
    eers.append(eer)
    auc = metrics.roc_auc_score(y,score)
    aucs.append(auc)
    result_scores[key] = {'tpr':[],'fpr':[]}
    result_scores[key]['tpr'] = tpr.tolist()
    result_scores[key]['fpr'] = fpr.tolist()
    end = time.time()
    duration = end-start
    print(duration)

    #plt.plot(fpr,tpr)
end = time.time()
duration = end-start
print(duration)
from statistics import mean 
print(mean(aucs))
print(mean(eers))



tpr_list = np.array(tpr_list)
fpr_list = np.array(fpr_list)

tpr_avg = np.mean(tpr_list, axis=0)
fpr_avg = np.mean(fpr_list, axis=0)
plt.plot(fpr_avg,tpr_avg)
plt.show()

"""
min_score = 10
max_score = -10

for key in scores.keys():
    if(min(scores[key]) < min_score):
        min_score = min(scores[key])
    if(max(scores[key]) > max_score):
        max_score = max(scores[key])



print(min_score)
print(max_score)


def get_roc_curve(y,score,min_score,max_score):
    fpr_list = [] 
    tpr_list = [] 
    thresholds = []

    step = (max_score - min_score) / 100
    for i in range(100):
        threshold = min_score + i*step
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        thresholds.append(threshold)
        for i in range(len(y)):
            if y[i] == 1 and threshold <= score[i]:
                tp = tp +1
            if y[i] == 1 and threshold > score[i]:
                fn = fn +1
            if y[i] == 0 and threshold <= score[i]:
                fp = fp +1
            if y[i] == 0 and threshold > score[i]:
                tn = tn +1
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return fpr_list, tpr_list, thresholds



new_result_scores = {}
new_eers = []
tpr_list = []
fpr_list = []
for key in scores.keys():
    score = scores[key]
    y = np.array(([1]*test_size)+([0]*test_size))
    fpr, tpr, thresholds = get_roc_curve(y, score,min_score,max_score)
    #print(thresholds)
    eer = get_eer(tpr,fpr)
    new_eers.append(eer)
    new_result_scores[key] = {'tpr':[],'fpr':[]}
    new_result_scores[key]['tpr'] = tpr
    tpr_list.append(tpr)
    new_result_scores[key]['fpr'] = fpr
    fpr_list.append(fpr)

    

print(mean(new_eers))
"""