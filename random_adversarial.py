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

def get_random_test_and_train_sample(data_dict,subject,train_size,test_size):
    test_subject_size = 10
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
    
    ### for random choose
    for i in range(int(test_size/test_subject_size)):
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
        test = test + temp
    return test,train

def get_test_sample_with_k_means(data_dict,subject,train_size,test_size,k):
    test = []
    all_data = []
    for key in data_dict.keys():
        if key == subject:
            continue
        all_data = all_data + data_dict[key]
        
    times = int(test_size / k )
    random.shuffle(all_data)
    fold_size = int(len(all_data) / k)
    for i in range(k):
        fold = all_data[i*fold_size:(i+1)*fold_size]
        fold = np.array(fold)
        fold = np.mean(fold, axis=0)
        test.append(fold)
    return test




start = time.time()


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


train_sizes = [4,10,20,50,100,200]
result = {'4':{},'10':{},'20':{},'50':{},'100':{},'200':{}}
for train_size in train_sizes:
    k_means = [1,4,10,20,50,100,200]
    isof_k_means_result={}
    md_k_means_result={}

    for key in data_dict.keys():
        train_and_test_data[key] = {"train":[],"test":[],} 
        train = []
        test = []
        test,train = get_random_test_and_train_sample(data_dict,key,train_size,test_size)
        train_and_test_data[key]["train"] = train
        train_and_test_data[key]["test"] = test


        print(key)
        predictors_data = train
        target_data = [1]*train_size
        
        score = md.scaled_manhattan_distance(train,test)
        y = np.array(([1]*test_size)+([-1]*test_size))
        
        #print(y)
        fpr, tpr, thresholds = metrics.roc_curve(y, score,drop_intermediate=False)
        md_tpr_list.append(tpr[:int(test_size*1.75)])
        md_fpr_list.append(fpr[:int(test_size*1.75)])

        eer,eer_threshold = get_eer(tpr,fpr,thresholds)

        #KMEANS
        for k in k_means:
            k_means_test = get_test_sample_with_k_means(data_dict,key,train_size,test_size,k)
            k_means_score = md.scaled_manhattan_distance(train,k_means_test)
            success = 0
            fail = 0
            for s in (k_means_score):
                if s > eer_threshold:
                    success += 1
                else:
                    fail += 1
            if not k in md_k_means_result:
                md_k_means_result[k] = []    
            md_k_means_result[k].append(success/k)
                    
        
        md_eers.append(eer)
        md_eer_thresholds.append(eer_threshold)

        auc = metrics.roc_auc_score(y,score)
        md_aucs.append(auc)



        classifier.fit(predictors_data, target_data)
        score = classifier.decision_function(test)
        
        #print(score)
        scores[key] = score.tolist()
        y = np.array(([1]*test_size)+([-1]*test_size))
        #print(y)
        fpr, tpr, thresholds = metrics.roc_curve(y, score,drop_intermediate=False)
        plt.plot(fpr,tpr)
        isof_tpr_list.append(tpr[:int(test_size*1.75)])
        isof_fpr_list.append(fpr[:int(test_size*1.75)])

        eer,eer_threshold = get_eer(tpr,fpr,thresholds)

        for k in k_means:
            k_means_test = get_test_sample_with_k_means(data_dict,key,train_size,test_size,k)
            k_means_score = classifier.decision_function(k_means_test)
            success = 0
            fail = 0
            for s in (k_means_score):
                if s > eer_threshold:
                    success += 1
                else:
                    fail += 1
            if not k in isof_k_means_result:
                isof_k_means_result[k] = []    
            isof_k_means_result[k].append(success/k)


       
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
    result[str(train_size)]["mean(isof_aucs)"] = mean(isof_aucs)
    print("isof_eers")
    print(mean(isof_eers))
    result[str(train_size)]["mean(isof_eers)"] = mean(isof_eers)
    print("isof_eer_thresholds")
    print(mean(isof_eer_thresholds))
    result[str(train_size)]["mean(isof_eer_thresholds)"] = mean(isof_eer_thresholds)



    print("md_aucs")
    print(mean(md_aucs))
    result[str(train_size)]["mean(md_aucs)"] = mean(md_aucs)
    print("md_eers")
    print(mean(md_eers))
    result[str(train_size)]["mean(md_eers)"] = mean(md_eers)
    print("md_eer_thresholds")
    print(mean(np.array(md_eer_thresholds)))
    result[str(train_size)]["md_eer_thresholds"] = mean(np.array(md_eer_thresholds))



    print("isof_k_means_result")
    print(isof_k_means_result)
    result[str(train_size)]["isof_k_means_result"] = isof_k_means_result
    print("md_k_means_result")
    print(md_k_means_result)
    result[str(train_size)]["md_k_means_result"] = md_k_means_result
    

with open("result.json",'w') as f:
    try:
        f.write(json.dumps(result))
    except:
        a = 1


with open("result",'w') as f:
    try:
        f.write(str(result))
    except:
        a = 1
    
    
"""

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

"""