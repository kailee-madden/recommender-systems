import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import numpy
import random
import gzip
import math
import sklearn
import statistics

### Each question can be run as a standalone and by default when running homework1.py gets called ###

def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

def Q1(dataset):

    def feature(datum):
        feat = datum['review_text'].count("!")
        return [1] + [feat]

    X = [feature(d) for d in dataset]
    Y = [d['rating'] for d in dataset]
    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, Y)
    Theta = model.coef_
    y_pred = model.predict(X)
    sse = sum([x**2 for x in (Y - y_pred)])
    mse = sse / len(Y)
    answers['Q1'] = [Theta[0], Theta[1], mse]
    assertFloatList(answers['Q1'], 3)

def Q2(dataset):

    def feature(datum):
        feat = [1] 
        feat.append(len(datum['review_text'])) 
        feat.append(datum['review_text'].count("!")) 
        return feat
    
    X = [feature(d) for d in dataset]
    Y = [d['rating'] for d in dataset]
    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, Y)
    Theta = model.coef_
    y_pred = model.predict(X)
    sse = sum([x**2 for x in (Y - y_pred)])
    mse = sse / len(Y)
    answers['Q2'] = [Theta[0], Theta[1], Theta[2], mse]
    assertFloatList(answers['Q2'], 4)

def Q3(dataset):

    def feature(datum, deg):
        # feature for a specific polynomial degree
        feat = [1]
        feat.append(datum['review_text'].count("!"))
        if deg >= 2:
            feat.append((datum['review_text'].count("!"))**2)
        if deg >= 3:
            feat.append((datum['review_text'].count("!"))**3)
        if deg >= 4:
            feat.append((datum['review_text'].count("!"))**4)
        if deg >= 5:
            feat.append((datum['review_text'].count("!"))**5)
        return feat
    
    MSEs = []
    for i in range(1,6):
        X = numpy.array([feature(d, i) for d in dataset])
        Y = [d['rating'] for d in dataset]
        model = sklearn.linear_model.LinearRegression(fit_intercept=False)
        model.fit(X, Y)
        y_pred = model.predict(X)
        sse = sum([x**2 for x in (Y - y_pred)])
        mse = sse / len(Y)
        MSEs.append(mse)
    answers['Q3'] = MSEs
    assertFloatList(answers['Q3'], 5)

def Q4(dataset):
    train = dataset[:5000]
    test = dataset[5000:]

    def feature(datum, deg):
        # feature for a specific polynomial degree
        feat = [1]
        feat.append(datum['review_text'].count("!"))
        if deg >= 2:
            feat.append((datum['review_text'].count("!"))**2)
        if deg >= 3:
            feat.append((datum['review_text'].count("!"))**3)
        if deg >= 4:
            feat.append((datum['review_text'].count("!"))**4)
        if deg >= 5:
            feat.append((datum['review_text'].count("!"))**5)
        return feat
    
    MSEs = []
    for i in range(1,6):
        X = numpy.array([feature(d, i) for d in train])
        Y = [d['rating'] for d in train]
        model = sklearn.linear_model.LinearRegression(fit_intercept=False)
        model.fit(X, Y)
        X_test = numpy.array([feature(d, i) for d in test])
        Y_test = [d['rating'] for d in test]
        y_pred = model.predict(X_test)
        sse = sum([x**2 for x in (Y_test - y_pred)])
        mse = sse / len(Y)
        MSEs.append(mse)
    answers['Q4'] = MSEs
    assertFloatList(answers['Q4'], 5)

def Q5(dataset):
    train = dataset[:5000]
    test = dataset[5000:]
    Y = [d['rating'] for d in test]
    med = statistics.median(Y)
    y_pred = [med for d in test]
    summation = 0
    for i in range(len(Y)):
        summation += abs(Y[i] - y_pred[i])
    mae = summation/len(Y)
    answers['Q5'] = mae
    assertFloat(answers['Q5'])

def Q6(dataset):
    X = [[1, d['review/text'].count("!")] for d in dataset]
    y = [d['user/gender'] == 'Female' for d in dataset]
    mod = sklearn.linear_model.LogisticRegression()
    mod.fit(X,y)
    predictions = mod.predict(X)
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    BER = 1 - 1/2 * (TPR + TNR)
    answers['Q6'] = [TP, TN, FP, FN, BER]
    assertFloatList(answers['Q6'], 5)

def Q7(dataset):
    mod = sklearn.linear_model.LogisticRegression(class_weight='balanced')
    X = [[1, d['review/text'].count("!")] for d in dataset]
    y = [d['user/gender'] == 'Female' for d in dataset]
    mod.fit(X,y)
    predictions = mod.predict(X)
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    BER = 1 - 1/2 * (TPR + TNR)
    answers['Q7'] = [TP, TN, FP, FN, BER]
    assertFloatList(answers['Q7'], 5)

    def Q8():
        probs = mod.predict_proba(X)
        probY = list(zip([p[1] for p in probs], [p[1] > 0.5 for p in probs], y))
        probY.sort(reverse=True)
        precisionList = []
        prev = 0
        for i in [1, 10, 100, 1000, 10000]:
            labs = [x[2] for x in probY[:i]]
            prec = sum(labs) / len(labs)
            precisionList.append(prec)
        answers['Q8'] = precisionList
        assertFloatList(answers['Q8'], 5)
    Q8() 

if __name__ == "__main__":
    global answers
    answers = {}

    f = open("young_adult_10000.json")
    books = []
    for l in f:
        books.append(json.loads(l))
    
    f = open("beer_50000.json")
    beers = []
    for l in f:
        if 'user/gender' in l:
            beers.append(eval(l))
    
    # answer all the hw questions
    Q1(books)
    Q2(books)
    Q3(books)
    Q4(books)
    Q5(books)

    Q6(beers)
    Q7(beers) # Q8 is called within Q7

    f = open("answers_hw1.txt", 'w') # Write your answers to a file
    f.write(str(answers) + '\n')
    f.close()
