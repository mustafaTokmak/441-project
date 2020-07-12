import random

count = 0
for i in range(10):
    r = random.randint(0,2)
    #print(r)
    print(i)
    if r == 0:
        print("r")
        i = i-1
        continue
