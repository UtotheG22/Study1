# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:15:16 2020

@author: oops_
"""
import xlrd
import jieba
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
data = xlrd.open_workbook('CarpList1.xlsx')
table = data.sheet_by_name('Sheet1')
Title_1 = str(table.col_values(0))
nrows = table.nrows
result = []

nameList = []
topWords = []
counts = {}

def get_data():
    result=[]
    for i in range(nrows):
        col=table.row_values(i)
        #print(col)
        result.append(col)
        #print(result)
    return result
    
if __name__ == '__main__':
    Title_3 = get_data()
    #print(Title_3)
    

def wordSegment():

    rowNumber = table.nrows
    for i in range(0, rowNumber):
        cell = str(table.cell(i, 0).value)
        words = jieba.lcut(cell)
        for word in words:
            nameList.append(word)
        i += 1

    for word in nameList:
        # if len(word) == 1:
        #     continue
        # else:
        counts[word] = counts.get(word, 0) + 1

    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)

    for i in range(40):
        word, count = items[i]
        topWords.append(items[i][0])
        print(word, count)
#wordSegment()

Final_data = []
for i in range(nrows):
    Final_data.append([])
    for j in range(6):
        Final_data[i].append(0)
#print(len(Title_3[4][0]))

for i in range(nrows):
    Final_data[i][0] = len(Title_3[i][0])
#print(Final_data[i][1])
#print(Title_3[1][0][2])
#for i in range(nrows):
#    for k in range(len(Title_3[i][0])):
for i in range(nrows):
    for k in range(len(Title_3[i][0])):
        if Title_3[i][0][k] == "！":
            Final_data[i][1] = Final_data[i][1] + 4
        if Title_3[i][0][k] == "？":
            Final_data[i][1] = Final_data[i][1] + 4
            
for i in range(nrows):
    Search_word = jieba.lcut(Title_3[i][0], cut_all = True)
    for l in range(len(Search_word)):
        if Search_word[l] == "震惊":
            Final_data[i][2] = Final_data[i][2] + 4
        if Search_word[l] == "转发":
            Final_data[i][2] = Final_data[i][3] + 4
        if Search_word[l] == "食品":
            Final_data[i][2] = Final_data[i][4] + 4
print(Final_data)

X = Final_data
pca = PCA(n_components=2) 
pca.fit(X)
K_data=pca.fit_transform(X)
#print(pca.explained_variance_ratio_)
print(K_data)
x = []
y = []
colors = np.random.rand(len(K_data))
for i in range (len(K_data)):
    x.append(0)
    x[i] = K_data[i][0]
    y.append(0)
    y[i] = K_data[i][1]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x[:170],y[:170],c = "#6699FF", marker = 'o')
ax1.scatter(x[136:137], y[136:137], c = "#FF9933", marker = 'o')
plt.show()

x = K_data
kmeans = KMeans(n_clusters = 4)
kmeans.fit(x)
plt.plot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],'ro')

K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    meandistortions.append(sum(np.min(cdist(x, kmeans.cluster_centers_,'euclidean'), axis=1)) / x.shape[0])
plt.figure()
plt.grid(True)
ax1 = plt.subplot(2,1,1)
ax1.plot(x[:,0], x[:,1], 'k.')

