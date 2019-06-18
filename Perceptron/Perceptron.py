#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np


# In[50]:


df = pd.read_csv('dataset_3.csv',header=None)
df.columns=['Index','PHIX1','PHIX2','Y']
df = df.set_index('Index')
df['Y']=df['Y'].replace(0,-1)
print("Preprocessing done")


# In[51]:


print("Preparing Scatterplot")
colors = np.where(df.Y == -1 , 'r', 'k') #Colors to classes, Red to 1 and Black to -1
df.plot(kind='scatter', x='PHIX1', y='PHIX2', s=2, c=colors)
plt.show()


# In[52]:


#A helper function to create a line
#Input : Weights W
#Output : A plot with the line corresponding to W
def plot_disc(W,epoch):
    #Draw the line
    #W[0] + W[1]*x + W[2]*y=0
    #y = mx + b
    m=-1*W[1]/W[2]
    b=-1*W[0]/W[2]
    phix1 = np.linspace(-5,5,100)
    phix2 = m*phix1 + b
    plt.plot(phix1, phix2, '-r', label='Discriminator')
    plt.title('Graph of Perceptron Discriminator')
    plt.xlabel('PHIX1', color='#1C2833')
    plt.ylabel('PHIX2', color='#1C2833')
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig('img'+str(epoch)+'.png')
    plt.show()
    


# In[40]:


#BATCH GRADIENT DESCENT
count=10
x=[]
sum_e=0
while(count!=0):
    epochs=0
    learning_rate = 0.0001
    np.random.seed(count)
    W=np.random.randn(3,1)

    #print(W)
    #print("EPOCH NUMBER : ", epochs)

    #df.plot(kind='scatter', x='PHIX1', y='PHIX2', s=2, c=colors)
    #plot_disc(W,epochs)
    x_epochs=[]
    y_errors=[]
    x_epochs.append(epochs)
    while(epochs!=1000):
        df = df.sample(frac=1)
        #Take the previous line and do the substitution
        results = W[0] + W[1]*df['PHIX1'] + W[2]*df['PHIX2']
        activated = W[0] + W[1]*df['PHIX1'] + W[2]*df['PHIX2']
        activated[activated<0]=-1
        activated[activated>=0]=1
        #Take the missclassified points only
        #print("Number of misclassified points : ", len(results[activated!=df['Y']]))
        #Find total error
        new_error=-1*sum((results * df['Y'])[activated!=df['Y']])
        #print("Total Error : ", new_error)
        y_errors.append(new_error)
        if(new_error==0):
            break
        old_error=-1*sum((results * df['Y'])[activated!=df['Y']])

        #Do the gradient descent
        W[0] = W[0] + learning_rate * sum(df['Y'][activated!=df['Y']])
        W[1] = W[1] + learning_rate * sum((df['Y']*df['PHIX1'])[activated!=df['Y']])
        W[2] = W[2] + learning_rate * sum((df['Y']*df['PHIX2'])[activated!=df['Y']])

        #print("NEW WEIGHTS FOUND")
        #New weights found!!
        #Scatter the points
        epochs+=1
        #print("")
        #print("EPOCH NUMBER : ", epochs)
        x_epochs.append(epochs)
        #df.plot(kind='scatter', x='PHIX1', y='PHIX2', s=2, c=colors)
        #plot_disc(W,epochs)
    #print(epochs,",")
    sum_e+=epochs
    count-=1
print(sum_e/10)


# In[55]:


#STOCHASTIC GRADIENT DESCENT
from sklearn.utils import shuffle
from time import time
count=10
sum_e=0
x=[]
while(count!=0):
    
    epochs=0
    learning_rate = 0.001
    W=[1,1,1]
    df = df.sample(frac=1)
    shuffled_index = df.index
    #print(W)
    #print("EPOCH NUMBER : ", epochs)

    #df.plot(kind='scatter', x='PHIX1', y='PHIX2', s=2, c=colors)
    #plot_disc(W,epochs)
    x_epochs=[]
    y_errors=[]
    x_epochs.append(epochs)
    last_checked=0

    while(epochs<2*10e5):
        #Take the previous line and do the substitution
        idx=epochs%1000 #Take a single data point's index
        idx=shuffled_index[idx]
        epochs+=1
        result = W[0] + W[1]*df['PHIX1'][idx] + W[2]*df['PHIX2'][idx]
        activated = W[0] + W[1]*df['PHIX1'][idx] + W[2]*df['PHIX2'][idx]
        if(activated<0):
            activated=-1
        else:
            activated=1
        '''
        Interesting note : In BGD, we were taking error on the entire dataset. Hence 
        the number of misclassified points was available to us implicitly as a byproduct.
        Here, however, that is not the case. We will need to terminate when we have correctly 
        classified all points - and not drag on till 10e6 epochs. Hence we count the number
        of misclassified points ALSO on each epoch - just to check if we may terminate now.
        Error and GD are still on a single point
        '''

        if(activated!=df['Y'][idx]):
            #print("Misclassfied")
            #print(epochs-1,(epochs-1)%1000)
            last_checked=0 #Tell the next iteration to check termination in ELIF
            #Won't do the check here cause update is required OBVIOUSLY

        elif(last_checked==0):
            #May do a check here if last_checked==0
            #print("Correctly Classified and check needed")
            #print(epochs-1,(epochs-1)%1000)
            x_epochs.append(epochs)
            y_errors.append(0)
            check_result = W[0] + W[1]*df['PHIX1'] + W[2]*df['PHIX2']
            check_activated = W[0] + W[1]*df['PHIX1'] + W[2]*df['PHIX2']
            check_activated[check_activated<0]=-1
            check_activated[check_activated>=0]=1
            num_misclassified= len(check_result[check_activated!=df['Y']])

            if(num_misclassified==0):
                #print("CONVERGED")
                break
            else:
                #print("Total misclassified", num_misclassified)
                last_checked=1 #Tell the next iteration that a check was done here and no update was done,
                               #so simply go on to ELSE without an expensive check again
                continue
        else:
            #print("Correctly Classified and check not needed")
            #print(epochs-1,(epochs-1)%1000)
            x_epochs.append(epochs)
            y_errors.append(0)
            continue



        #Find total error
        new_error = -1*result*df['Y'][idx]
        y_errors.append(new_error)
        #print("Total Error : ", new_error)
        if(new_error==0):
            break        
        #Do the gradient descent
        W[0] = W[0] + learning_rate * df['Y'][idx]
        W[1] = W[1] + learning_rate * (df['Y'][idx]*df['PHIX1'][idx])
        W[2] = W[2] + learning_rate * (df['Y'][idx]*df['PHIX2'][idx])

        #print("NEW WEIGHTS FOUND")
        #New weights found!!
        #Scatter the points
        #print("")
        #print("EPOCH NUMBER : ", epochs)
        x_epochs.append(epochs)
        #df.plot(kind='scatter', x='PHIX1', y='PHIX2', s=2, c=colors)
        #plot_disc(W,epochs)
    #print(epochs,",")
    sum_e+=epochs
    x.append(epochs)
    count-=1
print(sum_e/10)


# In[ ]:




