#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[5]:


answers = {}


# In[6]:


allRatings = []
for l in readCSV("assignment1/train_Interactions.csv.gz"):
    allRatings.append(l)
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerItem = defaultdict(set)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    usersPerItem[b].add(u)


# In[7]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("assignment1/train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# In[8]:


### Q1
len(return1)


# In[9]:


books = set()
users = set()
userBookRead = defaultdict(set)

for user, book, _ in allRatings:
    books.add(book)
    users.add(user)
    userBookRead[user].add(book)


# In[10]:


negativeRatingsValid = []
for user, book, _ in ratingsValid:
    notRead = list(books.difference(userBookRead[user]))
    negativeSample = random.choice(notRead)
    negativeRatingsValid.append((user, negativeSample, False))


# In[11]:


for i in range(len(ratingsValid)):
    u,b,_ = ratingsValid[i]
    ratingsValid[i] = (u,b, True)


# In[12]:


validation = ratingsValid+negativeRatingsValid


# In[13]:


predictions = []
for _,book,_ in validation:
    if book in return1:
        predictions.append(True)
    else:
        predictions.append(False)


# In[14]:


y = [t[2] for t in validation]
print(y[0])


# In[15]:


def accuracy(predictions, y):
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    return (TP+TN)/len(predictions)


# In[16]:


answers['Q1'] = accuracy(predictions, y)
assertFloat(answers['Q1'])


# In[17]:


### Q2


# In[18]:


return2 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return2.add(i)
    if count > totalRead/1.4: break # 1.4 means threashold of 71.43


# In[19]:


predictions2 = []
for _,book,_ in validation:
    if book in return2:
        predictions2.append(True)
    else:
        predictions2.append(False)


# In[20]:


answers['Q2'] = [71.43, accuracy(predictions2, y)]


# In[21]:


answers['Q2']


# In[22]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[23]:


### Q3


# In[24]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


# In[25]:


predictions3 = []
threshold = .003
for user,book,_ in validation:
    maxJaccard = 0
    for bPrime,_ in ratingsPerUser[user]:
        maxJaccard = max(maxJaccard, Jaccard(usersPerItem[book], usersPerItem[bPrime]))
    if threshold < maxJaccard:
        predictions3.append(True)
    else:
        predictions3.append(False)


# In[26]:


accuracy(predictions3, y)


# In[27]:


answers['Q3'] = accuracy(predictions3, y)
assertFloat(answers['Q3'])


# In[28]:


### Q4


# In[29]:


predictions4 = []
threshold = .045
for user,book,_ in validation:
    maxJaccard = 0
    for bPrime,_ in ratingsPerUser[user]:
        maxJaccard = max(maxJaccard, Jaccard(usersPerItem[book], usersPerItem[bPrime]))
    if threshold < maxJaccard or book in return2:
        predictions4.append(True)
    else:
        predictions4.append(False)


# In[30]:


answers['Q4'] = accuracy(predictions4, y)
assertFloat(answers['Q4'])


# In[31]:


### Q5


# In[32]:


def predict(user, book):
    maxJaccard = 0
    for bPrime,_ in ratingsPerUser[user]:
        maxJaccard = max(maxJaccard, Jaccard(usersPerItem[book], usersPerItem[bPrime]))
    if threshold < maxJaccard or book in return2:
        return True
    else:
        return False


# In[33]:


import csv
with open("predictions_Read.csv", "w", newline='') as file:
    writer = csv.writer(file)
    for l in open("assignment1/pairs_Read.csv"):
        if l.startswith("userID"):
            writer.writerow(["userID", "bookID", "prediction"])
            continue
        u,b = l.strip().split(',')
        row = [u, b, predict(u,b)]
        writer.writerow(row)


# In[34]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"
assert type(answers['Q5']) == str


# In[35]:


### Q9


# In[36]:


allRatings = []
for l in readCSV("assignment1/train_Interactions.csv.gz"):
    allRatings.append(l)
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerItem = defaultdict(set)
userBiases = defaultdict(float)
itemBiases = defaultdict(float)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    usersPerItem[b].add(u)


# In[37]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[38]:


mean = sum([r for _,_,r in ratingsTrain])/len(ratingsTrain)
alwaysPredictMean = [mean for d in ratingsValid]
y = [d[2] for d in ratingsValid]
MSE(alwaysPredictMean, y)


# In[39]:


N = len(ratingsTrain)
nUsers = len(ratingsPerUser)
nItems = len(ratingsPerItem)
users = list(ratingsPerUser.keys())
items = list(ratingsPerItem.keys())


# In[40]:


alpha = mean


# In[41]:


userBiases = defaultdict(float)
itemBiases = defaultdict(float)


# In[42]:


def prediction(user, item):
    if user in userBiases and item in itemBiases:
        return alpha + userBiases[user] + itemBiases[item]
    elif user in userBiases:
        return alpha + userBiases[user]
    elif item in itemBiases:
        return alpha + itemBiases[item]
    else:
        return alpha 


# In[43]:


labels = [d[2] for d in ratingsTrain]


# In[68]:


for u in users:
    userBiases[u] = 0.0


# In[69]:


for i in items:
    itemBiases[i] = 0.0


# In[70]:


lamb = 1
for i in range(50):
    dalpha = 0
    for u,b,r in ratingsTrain:
        dalpha += r - (userBiases[u]+itemBiases[b])
    alpha = dalpha/N
    
    dUserBiases = defaultdict(float)
    for u in users:
        dUserBiases[u] = 0.0
    dItemBiases = defaultdict(float)
    for i in items:
        dItemBiases[u] = 0.0
        
    for user in userBiases:
        for book,rating in ratingsPerUser[user]:
            dUserBiases[user] += rating - (alpha+itemBiases[book])
        dUserBiases[user] = dUserBiases[user]/(lamb+len(ratingsPerUser[user]))
    userBiases = dUserBiases
    for item in itemBiases:
        for u, r in ratingsPerItem[item]:
            dItemBiases[item] += r - (alpha+userBiases[u])
        dItemBiases[item] = dItemBiases[item]/(lamb+len(ratingsPerItem[item]))
    itemBiases = dItemBiases
    


# In[71]:


pred = [prediction(u,b) for u,b,_ in ratingsValid]
y = [r for _,_,r in ratingsValid]


# In[72]:


mse = MSE(pred,y)


# In[73]:


mse


# In[74]:


answers['Q9'] = mse
assertFloat(answers['Q9'])


# In[75]:


### Q10


# In[76]:


sortusers = sorted(userBiases.items(), key=lambda kv: kv[1], reverse=True)


# In[79]:


sortusers[0]


# In[80]:


answers['Q10'] = [sortusers[0][0], sortusers[-1][0], float(sortusers[0][1]), float(sortusers[-1][1])]
assert [type(x) for x in answers['Q10']] == [str, str, float, float]


# In[81]:


### Q11


# In[82]:


for u in users:
    userBiases[u] = 0.0
for i in items:
    itemBiases[i] = 0.0
lamb = 5
for i in range(50):
    dalpha = 0
    for u,b,r in ratingsTrain:
        dalpha += r - (userBiases[u]+itemBiases[b])
    alpha = dalpha/N
    
    dUserBiases = defaultdict(float)
    for u in users:
        dUserBiases[u] = 0.0
    dItemBiases = defaultdict(float)
    for i in items:
        dItemBiases[u] = 0.0
        
    for user in userBiases:
        for book,rating in ratingsPerUser[user]:
            dUserBiases[user] += rating - (alpha+itemBiases[book])
        dUserBiases[user] = dUserBiases[user]/(lamb+len(ratingsPerUser[user]))
    userBiases = dUserBiases
    for item in itemBiases:
        for u, r in ratingsPerItem[item]:
            dItemBiases[item] += r - (alpha+userBiases[u])
        dItemBiases[item] = dItemBiases[item]/(lamb+len(ratingsPerItem[item]))
    itemBiases = dItemBiases


# In[83]:


pred2 = [prediction(u,b) for u,b,_ in ratingsValid]
y = [r for _,_,r in ratingsValid]
validMSE = MSE(pred2,y)
validMSE


# In[84]:


answers['Q11'] = (5, validMSE)
assertFloat(answers['Q11'][0])
assertFloat(answers['Q11'][1])


# In[85]:


import csv
with open("predictions_Rating.csv", "w", newline='') as file:
    writer = csv.writer(file)
    for l in open("assignment1/pairs_Rating.csv"):
        if l.startswith("userID"):
            writer.writerow(["userID", "bookID", "rating"])
            continue
        u,b = l.strip().split(',')
        row = [u, b, prediction(u,b)]
        writer.writerow(row)


# In[ ]:





# In[86]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




