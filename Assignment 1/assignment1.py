#!/usr/bin/env python
# coding: utf-8

# In[144]:


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
import csv


# In[145]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[146]:


allRatings = []
for l in readCSV("assignment1/train_Interactions.csv.gz"):
    allRatings.append(l)
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)
items = set()
ratingDict = {}
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    ratingDict[(u,b)] = r
    usersPerItem[b].add(u)
    itemsPerUser[u].add(b)
    items.add(b)


# In[147]:


itemAverages = {}
    
for i, ls in ratingsPerItem.items():
    itemAverages[i] = sum([t[1] for t in ls]) / len(ls)


# In[148]:


books = set()
users = set()
userBookRead = defaultdict(set)

for user, book, _ in allRatings:
    books.add(book)
    users.add(user)
    userBookRead[user].add(book)


# In[149]:


negativeRatingsValid = []
for user, book, _ in ratingsValid:
    notRead = list(books.difference(userBookRead[user]))
    negativeSample = random.choice(notRead)
    negativeRatingsValid.append((user, negativeSample, 0))


# In[150]:


for i in range(len(ratingsValid)):
    u,b,_ = ratingsValid[i]
    ratingsValid[i] = (u,b, 1)
validation = ratingsValid+negativeRatingsValid



# In[151]:


def Cosine(i1, i2):
    # Between two items
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer = 0
    denom1 = 0
    denom2 = 0
    for u in inter:
        numer += ratingDict[(u,i1)]*ratingDict[(u,i2)]
    for u in usersPerItem[i1]:
        denom1 += ratingDict[(u,i1)]**2
    for u in usersPerItem[i2]:
        denom2 += ratingDict[(u,i2)]**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom



# In[152]:


bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("assignment1/train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return2 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return2.add(i)
    if count > totalRead/1.4: break


# In[153]:


def predict(user, book, trues):
    maxSim = 0
    for bPrime,_ in ratingsPerUser[user]:
        maxSim = max(maxSim, Cosine(book, bPrime))
    if trues > 10000:
        return 0
    if threshold < maxSim or book in return2:
        trues += 1
        return 1
    else:
        return 0



# In[154]:


threshold = .06
with open("predictions_Read.csv", "w", newline='') as file:
    writer = csv.writer(file)
    for l in open("assignment1/pairs_Read.csv"):
        if l.startswith("userID"):
            writer.writerow(["userID", "bookID", "prediction"])
            continue
        u,b = l.strip().split(',')
        read = predict(u,b, trues)
        if read:
            trues += 1
        row = [u, b, read]
        writer.writerow(row)



# Q3


# In[187]:


allRatings = []
for l in readCSV("assignment1/train_Interactions.csv.gz"):
    allRatings.append(l)

ratingsTrain = allRatings
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerItem = defaultdict(set)
userBiases = defaultdict(float)
itemBiases = defaultdict(float)
userIDs,itemIDs = {},{}
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    usersPerItem[b].add(u)
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not b in itemIDs: itemIDs[b] = len(itemIDs)

nUsers,nItems = len(userIDs),len(itemIDs)


# In[188]:


mean = sum([r for _,_,r in ratingsTrain])/len(ratingsTrain)
alpha = mean
N = len(ratingsTrain)
nUsers = len(ratingsPerUser)
nItems = len(ratingsPerItem)
users = list(ratingsPerUser.keys())
items = list(ratingsPerItem.keys())
userBiases = defaultdict(float)
itemBiases = defaultdict(float)


# In[189]:


def prediction(user, item):
    if user in userBiases and item in itemBiases:
        num = alpha + userBiases[user] + itemBiases[item]
        if num > 5:
            return 5
        else: 
            return num
    elif user in userBiases:
        num = alpha + userBiases[user]
        if num > 5:
            return 5
        else: 
            return num
    elif item in itemBiases:
        num = alpha + itemBiases[item]
        if num > 5:
            return 5
        else: 
            return num
    else:
        return alpha 


# In[190]:


for u in users:
    userBiases[u] = 0.0
for i in items:
    itemBiases[i] = 0.0
lamb = 4
for i in range(70):
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


# In[191]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)



# In[192]:


with open("predictions_Rating.csv", "w", newline='') as file:
    writer = csv.writer(file)
    for l in open("assignment1/pairs_Rating.csv"):
        if l.startswith("userID"):
            writer.writerow(["userID", "bookID", "prediction"])
            continue
        u,b = l.strip().split(',')
        row = [u, b, prediction(u,b)]
        writer.writerow(row)




