import pandas as pd 
import time 
import random 
import numpy as np
import json
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn import  metrics
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
import manhattan_distance as md


with open('cmu_dict_data','r') as f:
    data = f.read()

data_dict = json.loads(data)

print(len(data_dict))


## train size 200, test samples chosen from each subject first 5 sample(CMU paper method)
train_and_test_data = {}
print(len(data_dict))
train_size = 200
test_size = 200
def get_test_and_train_sample_with_first_data(data_dict,subject,train_size,test_size):
    first_data_size  = 5
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
        test = test + data_dict[key][:4]
    return test,train




start = time.time()
for key in data_dict.keys():
    train_and_test_data[key] = {"train":[],"test":[]} 
    train = []
    test = []
    test,train = get_test_and_train_sample_with_first_data(data_dict,key,train_size,test_size)
    train_and_test_data[key]["train"] = train
    train_and_test_data[key]["test"] = test

end = time.time()
duration = end-start
print(duration)


classifier = IsolationForest(contamination=0, max_features=1,
            max_samples=500, n_estimators=3000, n_jobs=-1, random_state=11)



def get_eer(tpr,fpr,thresholds):
    eer = 100
    for i in range(len(tpr)):
        if 1-tpr[i] < fpr[i]:
            eer = ( (1-tpr[i]) + (1-tpr[i-1]) + fpr[i] + fpr[i-1])/4
            eer_threshold = thresholds[i-1]
            return eer,eer_threshold
    #print(eer)
    return 0,0
    
start = time.time()
scores = {}
isof_result_scores = {}
isof_aucs = []
isof_eers = []
isof_eer_thresholds = []
md_aucs = []
md_eers = []
md_eer_thresholds = []

isof_tpr_list = []
isof_fpr_list = []
md_tpr_list = []
md_fpr_list = []

for key in data_dict.keys():
    print(key)
    predictors_data = train_and_test_data[key]["train"]
    target_data = [1]*train_size
    
    score = md.scaled_manhattan_distance(train_and_test_data[key]["train"],train_and_test_data[key]["test"])
    y = np.array(([1]*test_size)+([-1]*test_size))
    
    #print(y)
    fpr, tpr, thresholds = metrics.roc_curve(y, score,drop_intermediate=False)
    md_tpr_list.append(tpr[:int(test_size*1.75)])
    md_fpr_list.append(fpr[:int(test_size*1.75)])

    eer,eer_threshold = get_eer(tpr,fpr,thresholds)
    print(eer,eer_threshold)
    md_eers.append(eer)
    md_eer_thresholds.append(eer_threshold)
    auc = metrics.roc_auc_score(y,score)
    md_aucs.append(auc)



    classifier.fit(predictors_data, target_data)
    score = classifier.decision_function(train_and_test_data[key]["test"])
    print(score)
    #print(score)
    scores[key] = score.tolist()
    y = np.array(([1]*test_size)+([-1]*test_size))
    #print(y)
    fpr, tpr, thresholds = metrics.roc_curve(y, score,drop_intermediate=False)
    plt.plot(fpr,tpr)
    isof_tpr_list.append(tpr[:int(test_size*1.75)])
    isof_fpr_list.append(fpr[:int(test_size*1.75)])

    eer,eer_threshold = get_eer(tpr,fpr,thresholds)
    print(eer,eer_threshold)
    isof_eers.append(eer)
    isof_eer_thresholds.append(eer_threshold)
    auc = metrics.roc_auc_score(y,score)
    isof_aucs.append(auc)

    end = time.time()
    duration = end-start
    print(duration)

    #plt.plot(fpr,tpr)
end = time.time()
duration = end-start
print(duration)
from statistics import mean 

print("isof_aucs")
print(mean(isof_aucs))
print("isof_eers")
print(mean(isof_eers))
print("isof_eer_thresholds")
print(mean(isof_eer_thresholds))

print("md_aucs")
print(mean(md_aucs))
print("md_eers")
print(mean(md_eers))
print("md_eer_thresholds")
print(mean(np.array(md_eer_thresholds)))




md_tpr_list = np.array(md_tpr_list)
md_fpr_list = np.array(md_fpr_list)
md_tpr_avg = np.mean(md_tpr_list, axis=0)
md_fpr_avg = np.mean(md_fpr_list, axis=0)
plt.plot(md_fpr_avg,md_tpr_avg,label="scaled_manhattan_distance")

isof_tpr_list = np.array(isof_tpr_list)
isof_fpr_list = np.array(isof_fpr_list)
isof_tpr_avg = np.mean(isof_tpr_list, axis=0)
isof_fpr_avg = np.mean(isof_fpr_list, axis=0)
plt.figure()
plt.plot(isof_fpr_avg,isof_tpr_avg,label="isolation_forest")
plt.show()

