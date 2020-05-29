#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


# In[2]:


dataset = mnist.load_data('mymnist.db')


# In[3]:


train , test = dataset
X_train , y_train = train
X_test , y_test = test


# In[4]:


X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)


# In[5]:


X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[6]:


y_train_cat = to_categorical(y_train)


# In[7]:


model = Sequential()
model.add(Dense(units=300, input_dim=28*28, activation='relu'))
model.add(Dense(units=150, activation='relu'))
model.add(Dense(units=75, activation='relu'))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=10, activation='softmax'))


# In[8]:


model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[9]:


y_pred = model.fit(X_train, y_train_cat, epochs=5)

