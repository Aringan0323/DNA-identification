#!/usr/bin/env python
# coding: utf-8

# First, we must import the training data csv file into a 2-D python array. Because this data seems to be DNA codon sequences, we should split the right column of the data up into groups of 3 codons.

# In[2]:


t_data = open("C:/Users/Adam/Desktop/training.data")
train = t_data.readlines()
for i,line in enumerate(train):
    train[i] = line.strip().split(",")
    for j in range(0,len(train[i][1]),3):
        train[i].append(train[i][1][j:j+3])
    train[i].pop(1)
print(train[0])


# Now, we should split the data into a data vector and a label vector.

# In[3]:


train_data = []
train_labels = []
for row in train:
    train_labels.append(row[0])
    train_data.append(row[1:])
print(train_labels[0])
print(train_data[0])


# In biology, the purpose of codons is to code for the production of 20 different amino acids. We can reduce the dimensionality of the data vector by converting the vector of lists of 3 letters into lists of the amino acids that those groups of codons code for. We need to create a dictionary mapping each group of three letters to the amino acid that it codes for. Then we can represent each of those amino acids as a number ranging from 0 to 19. If there are any typos in the data that include a letter that is not A, T, C, or G, the group of codons that contains that letter will just be assigned to an imaginary amino acid 21 for the sake of keeping that data entry while minimizing the affect of the typo.

# In[4]:


amino_acids_to_codons ={
    "Phe":["TTT","TTC"],
    "Leu":["TTA","TTG","CTT","CTC","CTA","CTG"],
    "Ile":["ATT","ATC","ATA"],
    "Met":["ATG"],
    "Val":["GTT","GTC","GTA","GTG"],
    "Ser":["TCT","TCC","TCA","TCG","AGT","AGC"],
    "Pro":["CCC","CCT","CCA","CCG"],
    "Thr":["ACT","ACC","ACA","ACG"],
    "Ala":["GCT","GCC","GCA","GCG"],
    "Tyr":["TAT","TAC"],
    "Stop":["TAA","TAG","TGA"],
    "His":["CAT","CAC"],
    "Gln":["CAA","CAG"],
    "Asn":["AAT","AAC"],
    "Lys":["AAA","AAG"],
    "Asp":["GAT","GAC"],
    "Glu":["GAA","GAG"],
    "Cys":["TGT","TGC"],
    "Trp":["TGG"],
    "Arg":["CGT","CGC","CGA","CGG","AGA","AGG"],
    "Gly":["GGT","GGC","GGA","GGG"]
}

codons_to_amino_acids = {}
for i, amino in enumerate(amino_acids_to_codons):
    for codon in amino_acids_to_codons[amino]:
        codons_to_amino_acids[codon] = i
print(amino_acids_to_codons)
print(codons_to_amino_acids)


# In[5]:


def numerical(data, key):
    for i,row in enumerate(data):
        for j,codon in enumerate(row):
            if codon not in key: #Inevitably there will be typos in the data, so we assign any nonsensical codons to imaginary amino acid "21".
                data[i][j] = 21
            else:
                data[i][j] = key[codon]

numerical(train_data, codons_to_amino_acids)
print(train_data[0])


# Now, we can put the data into a pandas dataframe and shuffle it before we split it into a validation and training set.

# In[6]:


import pandas as pd
import random

train_data_df = pd.DataFrame(train_data)
train_data_df["Labels"] = train_labels
train_data_df


# In[7]:


import numpy as np

def val_split(df, split):
    
    X = df.to_numpy()
    np.random.shuffle(X)
    y = X.T[-1]
    X = X.T[:-1]
    X = X.T
    
    split = int(split*len(train_data))

    X_train = X[:split]
    y_train = y[:split]
    
    X_val = X[split:]
    y_val = y[split:]
    
    return X_train,y_train,X_val,y_val


# In[8]:


Xt, yt, Xv, yv = val_split(train_data_df, 0.85)

ttd_df = pd.DataFrame(Xt)
ttd_df["Labels"] = yt

tvd_df = pd.DataFrame(Xv)
tvd_df["Labels"] = yv

ttd_df


# In[8]:


tvd_df


# Before we start testing models, we should find a way to measure the accuracy of the model. I implemented this with the accuracy_score function in sklearn.

# In[9]:


from sklearn.metrics import accuracy_score

def print_acc(true, pred):
    print(accuracy_score(true, pred))


# First, we test how well a basic linear support vector machine can predict on new data.

# In[10]:


from sklearn import svm

clf_svm = svm.LinearSVC()
clf_svm.fit(Xt, yt)

pred_svm = clf_svm.predict(Xv)
print_acc(yv, pred_svm)


# With an accuracy of just above 50%, the SVM will not be the model that we use. Next we can try a decision tree.

# In[11]:


from sklearn import tree

clf_dt = tree.DecisionTreeClassifier()
clf_dt = clf_dt.fit(Xt, yt)
tree.plot_tree(clf_dt)

pred = clf_dt.predict(Xv)

print_acc(yv, pred)


# With an accuracy of roughly 87%, the classification tree model works much better than the SVM, but I am still not satisfied with an accuracy of 87%. next we can try a Random Forest Classifier to use bagging to try to get a better accuracy score.

# In[12]:


from sklearn.ensemble import RandomForestClassifier

clf_rfc = RandomForestClassifier(n_estimators = 20)
clf_rfc = clf_rfc.fit(Xt, yt)

pred_rfc = clf_rfc.predict(Xv)
print_acc(yv, pred_rfc)


# At 89% accuracy, the Random Forest model is slightly better that the classification tree model. Lets try the AdaBoost classifier to see if boosting can give us a better accuracy.

# In[13]:


from sklearn.ensemble import AdaBoostClassifier

clf_ada = AdaBoostClassifier(n_estimators = 30)
clf_ada = clf_ada.fit(Xt,yt)

pred_ada = clf_ada.predict(Xv)
print_acc(yv, pred_ada)


# The AdaBoost classifier has essentially the same accuracy as the Random Forest method, so this method is no better. Next, lets try to use a Neural Network to get a better accuracy.

# In[14]:


from sklearn.neural_network import MLPClassifier

clf_nn = MLPClassifier(activation='relu', solver='adam', max_iter=500)
clf_nn.fit(Xt,yt)

pred_nn = clf_nn.predict(Xv)
print_acc(yv, pred_nn)


# The neural network performed worse than all of the other methods other than the Linear SVM. Maybe we can alter our data in such a way that makes it easier for the neural network to read. To do this, we can use one hot encoding to transform each data entry into a matrix where the rows represent the amino acids and the columns represent the columns in the entry. This transforms our data into a grid of only 1s and 0s, where there is a 1 at position (row, column) if there is the amino acid represented by that row in that column of the data, and a 0 otherwise.

# In[24]:


from sklearn.preprocessing import OneHotEncoder

X = np.concatenate((Xt, Xv), axis=0)

enc_oh = OneHotEncoder().fit(X)
Xt_oh = enc_oh.transform(Xt).toarray()
Xv_oh = enc_oh.transform(Xv).toarray()

clf_nn = MLPClassifier(activation='relu', solver='adam', max_iter=200)
clf_nn.fit(Xt_oh,yt)

pred_nn = clf_nn.predict(Xv_oh)
print_acc(yv, pred_nn)


# In[31]:


[coef.shape for coef in clf_nn.coefs_]


# This method works much better than any other method with an accuracy of 93%. Now, we can train the model on the entire training dataset and use that model to predict on the test dataset.

# In[17]:


test = open("C:/Users/Adam/Desktop/test.data")
test_dt = test.readlines()
test_ds = []
for i,line in enumerate(test_dt):
    test_ds.append([line.strip()])
    for j in range(0,len(test_ds[i][0]),3):
        test_ds[i].append(test_ds[i][0][j:j+3])
    test_ds[i].pop(0)

    

test_df = pd.DataFrame(test_ds)

test_df

    


# In[18]:


y = np.concatenate((yt, yv), axis=0)

X_oh = enc_oh.transform(X).toarray()

clf_nn_fin = MLPClassifier(activation='relu', solver='adam', max_iter=200).fit(X_oh, y)


# In[19]:


numerical(test_ds, codons_to_amino_acids)

test_arr = np.array(test_ds)

test_oh = enc_oh.transform(test_arr)

pred_fin = clf_nn_fin.predict(test_oh)

pred_nn_df = pd.DataFrame(pred_fin,columns=["Multi-Layer Perceptron Prediction"])

pred_nn_df


# In[24]:


pred_nn_df.to_csv("C:/Users/Adam/Desktop/HW5_results.csv")

