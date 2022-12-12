#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy
import urllib
import scipy.optimize
import random
import sklearn
from sklearn import linear_model
import gzip
from collections import defaultdict


# In[88]:


import warnings
warnings.filterwarnings("ignore")


# In[89]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[90]:


f = open("5year.arff", 'r')


# In[91]:


# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)


# In[92]:


answers = {}


# In[93]:


def accuracy(predictions, y):
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    return (TP+TN)/len(predictions)


# In[94]:


def BER(predictions, y):
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    BER = 1 - 1/2 * (TPR + TNR)
    return BER


# In[95]:


## Q1


# In[96]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[97]:


mod = sklearn.linear_model.LogisticRegression(C=1.0)
mod.fit(X,y)
predictions = mod.predict(X)


# In[98]:


answers['Q1'] = [accuracy(predictions, y), BER(predictions,y)] # Accuracy and balanced error rate


# In[99]:


assertFloatList(answers['Q1'], 2)


# In[100]:


### Q2


# In[101]:


mod = sklearn.linear_model.LogisticRegression(C=1.0, class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X)


# In[102]:


answers['Q2'] = [accuracy(predictions, y), BER(predictions,y)]
assertFloatList(answers['Q2'], 2)


# In[103]:


### Q3


# In[104]:


random.seed(3)
random.shuffle(dataset)


# In[105]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[106]:


Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[107]:


len(Xtrain), len(Xvalid), len(Xtest)


# In[108]:


mod = sklearn.linear_model.LogisticRegression(C=1.0, class_weight='balanced')
mod.fit(Xtrain,ytrain)
pred_train = mod.predict(Xtrain)
pred_valid = mod.predict(Xvalid)
pred_test = mod.predict(Xtest)


# In[109]:


answers['Q3'] = [BER(pred_train, ytrain), BER(pred_valid, yvalid), BER(pred_test, ytest)]
assertFloatList(answers['Q3'], 3)


# In[110]:


### Q4


# In[111]:


berList = []


# In[112]:


for n in [10**-4, 10**-3, 10**-2, 10**-1,10**-0, 10**-1,10**2, 10**3, 10**4]:
    mod = sklearn.linear_model.LogisticRegression(C=n, class_weight='balanced')
    mod.fit(Xtrain,ytrain)
    pred_valid = mod.predict(Xvalid)
    berList.append(BER(pred_valid, yvalid))


# In[113]:


answers['Q4'] = berList
assertFloatList(answers['Q4'], 9)


# In[114]:


### Q5


# In[115]:


print(berList)


# In[116]:


bestC = 10**2
mod = sklearn.linear_model.LogisticRegression(C=bestC, class_weight='balanced')
mod.fit(Xtrain,ytrain)
pred_test = mod.predict(Xtest)


# In[117]:


answers['Q5'] = [bestC, BER(pred_test, ytest)]
assertFloatList(answers['Q5'], 2)


# In[82]:


### Q6


# In[83]:


f = open("young_adult_10000.json")
dataset = []
for l in f:
    dataset.append(eval(l))


# In[84]:


dataTrain = dataset[:9000]
dataTest = dataset[9000:]


# In[130]:


dataTrain[0]


# In[120]:


usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair


# In[144]:


for d in dataTrain:
    user,item = d['user_id'], d['book_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerItem[item].append(d)
    reviewsPerUser[user].append(d)
    ratingDict[(user, item)] = d['rating'] #are we guaranteed no duplicates?


# In[122]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


# In[123]:


def mostSimilar(i):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem: # For all items
        if i == i2: continue # other than the query
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:10]


# In[126]:


answers['Q6'] = mostSimilar('2767052')


# In[127]:


assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)


# In[128]:


### Q7


# In[146]:


ratingMean = sum([d['rating'] for d in dataTrain]) / len(dataTrain)


# In[156]:


def predictRating(user, item):
    try:
        itemRatingMean = sum([d['rating'] for d in reviewsPerItem[item]]) / len(reviewsPerItem[item])
    except:
        itemRatingMean = ratingMean
        
    ratings_similarity_sum = 0
    similarity_sum = 0
    for d in reviewsPerUser[user]:
        j = d['book_id']
        if j == item: 
            continue
        rating = d['rating'] #user rating for other items
        try:
            avg_rating = sum([d['rating'] for d in reviewsPerItem[j]]) / len(reviewsPerItem[j])
        except:
            avg_rating = ratingMean
        similarity = Jaccard(usersPerItem[item],usersPerItem[j])
        ratings_similarity_sum += (rating-avg_rating)*similarity
        similarity_sum += similarity
    if similarity_sum == 0:
        return itemRatingMean
    else:
        return itemRatingMean + (ratings_similarity_sum/similarity_sum)


# In[132]:


dataTrain[0]


# In[133]:


u, i = dataTrain[0]['user_id'], dataTrain[0]['book_id']


# In[139]:


predictRating(u, i)


# In[140]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[148]:


Predictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]


# In[149]:


Labels = [d['rating'] for d in dataTest]


# In[151]:


answers['Q7'] = MSE(Predictions, Labels)


# In[153]:


### Q8


# In[154]:


ratingMean = sum([d['rating'] for d in dataTrain]) / len(dataTrain)


# In[157]:


def predictRating2(user, item):
    try:
        userRatingMean = sum([d['rating'] for d in reviewsPerUser[user]]) / len(reviewsPerUser[user])
    except:
        userRatingMean = ratingMean
        
    ratings_similarity_sum = 0
    similarity_sum = 0
    for d in reviewsPerItem[item]:
        j = d['user_id']
        if j == user: 
            continue
        rating = d['rating'] #the item's rating for other users
        try:
            avg_rating = sum([d['rating'] for d in reviewsPerUser[j]]) / len(reviewsPerUser[j])
        except:
            avg_rating = ratingMean
        similarity = Jaccard(itemsPerUser[user],itemsPerUser[j])
        ratings_similarity_sum += (rating-avg_rating)*similarity
        similarity_sum += similarity
    if similarity_sum == 0:
        return userRatingMean
    else:
        return userRatingMean + (ratings_similarity_sum/similarity_sum)


# In[158]:


Predictions2 = [predictRating2(d['user_id'], d['book_id']) for d in dataTest]
answers['Q8'] = MSE(Predictions2, Labels)


# In[159]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




