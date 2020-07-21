import json
from statistics import mean 

with open("result.json",'r') as f:
    data = f.read()


data = json.loads(data)
for key in data.keys():
    for d in data[key]["isof_k_means_result"].keys():
        print(d)
        for i in data[key]["isof_k_means_result"][d].keys():
            print(i)
                
                

#train size -> {,...... isof_k_means_result -> {used_data_rate  --> {k_means -> [s1,s2,s3] }}}
