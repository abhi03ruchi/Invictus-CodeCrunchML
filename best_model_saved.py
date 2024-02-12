#!/usr/bin/env python
# coding: utf-8

# In[4]:


data = pd.read_csv(r"C:\Users\dushy\Desktop\My Expression.csv")
data.shape


# In[5]:


get_ipython().run_cell_magic('capture', '', '!pip install scikit-plot\nimport pandas as pd\nimport numpy as np\nimport scikitplot\nimport random\nimport seaborn as sns\nimport keras\nimport os\nfrom matplotlib import pyplot\nimport matplotlib.pyplot as plt\nimport tensorflow as tf\nfrom tensorflow.keras.utils import to_categorical\nimport warnings\nfrom tensorflow.keras.models import Sequential\nfrom keras.callbacks import EarlyStopping\nfrom keras import regularizers\nfrom keras.callbacks import ModelCheckpoint,EarlyStopping\nfrom tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax\nfrom keras.preprocessing import image\nfrom keras.utils.vis_utils import plot_model\nfrom keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization,MaxPooling2D,Activation,Input\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nwarnings.simplefilter("ignore")\nfrom keras.models import Model\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\nfrom keras.regularizers import l1, l2\nimport plotly.express as px\nfrom matplotlib import pyplot as plt\nfrom sklearn.metrics import confusion_matrix\nfrom sklearn.metrics import classification_report')


# In[6]:


data.isnull().sum()


# In[7]:


data.head()


# In[8]:


CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
fig = px.bar(x = CLASS_LABELS,
             y = [list(data['emotion']).count(i) for i in np.unique(data['emotion'])] , 
             color = np.unique(data['emotion']) ,
             color_continuous_scale="Emrld") 
fig.update_xaxes(title="Emotions")
fig.update_yaxes(title = "Number of Images")
fig.update_layout(showlegend = True,
    title = {
        'text': 'Train Data Distribution ',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()


# In[9]:


data = data.sample(frac=1)


# In[10]:


labels = to_categorical(data[['emotion']], num_classes=7)


# In[11]:


train_pixels = data["pixels"].astype(str).str.split(" ").tolist()
train_pixels = np.uint8(train_pixels)


# In[12]:


pixels = train_pixels.reshape((35887*2304,1))


# In[13]:


scaler = StandardScaler()
pixels = scaler.fit_transform(pixels)


# In[14]:


pixels = train_pixels.reshape((35887, 48, 48,1))


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(pixels, labels, test_size=0.1, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)


# In[16]:


print(X_train.shape)
print(X_test.shape)
print(X_val.shape)


# In[17]:


plt.figure(figsize=(15,23))
label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}
i = 1
for i in range (7):
    img = np.squeeze(X_train[i])
    plt.subplot(1,7,i+1)
    plt.imshow(img)
    index = np.argmax(y_train[i])
    plt.title(label_dict[index])
    plt.axis('off')
    i += 1
plt.show()


# In[18]:


datagen = ImageDataGenerator(  width_shift_range = 0.1,
                               height_shift_range = 0.1,
                               horizontal_flip = True,
                               zoom_range = 0.2)
valgen = ImageDataGenerator(   width_shift_range = 0.1,
                               height_shift_range = 0.1,
                               horizontal_flip = True,
                               zoom_range = 0.2) 


# In[26]:


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


# In[27]:


datagen.fit(X_train)
valgen.fit(X_val)


# In[28]:


train_generator = datagen.flow(X_train, y_train, batch_size=64)
val_generator = datagen.flow(X_val, y_val, batch_size=64)


# In[29]:


def cnn_model():

  model= tf.keras.models.Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48,1)))
  model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
  model.add(BatchNormalization())
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
      
  model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
  model.add(BatchNormalization())
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
  model.add(BatchNormalization())
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
  model.add(BatchNormalization())
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten()) 
  model.add(Dense(256,activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.25))
      
  model.add(Dense(512,activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.25))

  model.add(Dense(7, activation='softmax'))
  model.compile(
    optimizer = Adam(lr=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])
  return model


# In[30]:


model = cnn_model()


# In[33]:


model.compile(
    optimizer = Adam(lr=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])
model.summary()


# In[35]:


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1, 
                              restore_best_weights=True,mode="max",patience = 5),
                ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                                save_best_only=True,mode="max")]
history = model.fit(train_generator,
                    epochs=15,9
                    batch_size=32,   
                    verbose=1,
                    callbacks=[checkpointer],
                    validation_data=val_generator)


# In[36]:


plt.plot(history.history["loss"],'r', label="Training Loss")
plt.plot(history.history["val_loss"],'b', label="Validation Loss")
plt.legend()


# In[37]:


plt.plot(history.history["accuracy"],'r',label="Training Accuracy")
plt.plot(history.history["val_accuracy"],'b',label="Validation Accuracy")
plt.legend()


# In[38]:


loss = model.evaluate(X_test,y_test) 
print("Test Acc: " + str(loss[1]))


# In[39]:


preds = model.predict(X_test)
y_pred = np.argmax(preds , axis = 1 )


# In[40]:


label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}

figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(X_test.shape[0], size=24, replace=False)):
    ax = figure.add_subplot(4, 6, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(X_test[index]))
    predict_index = label_dict[(y_pred[index])]
    true_index = label_dict[np.argmax(y_test,axis=1)[index]]
    
    ax.set_title("{} ({})".format((predict_index), 
                                  (true_index)),
                                  color=("green" if predict_index == true_index else "red"))


# In[41]:


CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]

cm_data = confusion_matrix(np.argmax(y_test, axis = 1 ), y_pred)
cm = pd.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (15,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


# In[42]:


from sklearn.metrics import classification_report
print(classification_report(np.argmax(y_test, axis = 1 ),y_pred,digits=3))


# In[43]:


model = cnn_model()


# In[48]:


model.compile(optimizer=tf.keras.optimizers.SGD(0.0001),
                loss='categorical_crossentropy',
                metrics = ['accuracy'])
history = model.fit(train_generator,
                    epochs=20,
                    batch_size=32,   
                    verbose=1,
                    callbacks=[checkpointer],
                    validation_data=val_generator)


# In[49]:


loss = model.evaluate(X_test,y_test) 
print("Test Acc: " + str(loss[1]))


# In[50]:


plt.plot(history.history["loss"],'r', label="Training Loss")
plt.plot(history.history["val_loss"],'b', label="Validation Loss")
plt.legend()


# In[51]:


plt.plot(history.history["accuracy"],'r',label="Training Accuracy")
plt.plot(history.history["val_accuracy"],'b',label="Validation Accuracy")
plt.legend()


# In[52]:


model = cnn_model()
model.compile(
    optimizer = Adam(lr=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])
checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1, 
                              restore_best_weights=True,mode="max",patience = 100),
                              ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                              save_best_only=True,mode="max")]
history = model.fit(train_generator,
                    epochs=50,
                    batch_size=64,   
                    verbose=1,
                    callbacks=[checkpointer],
                    validation_data=val_generator)


# In[53]:


loss = model.evaluate(X_test,y_test) 
print("Test Acc: " + str(loss[1]))


# In[54]:


preds = model.predict(X_test)
y_pred = np.argmax(preds , axis = 1 )


# In[55]:


CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]

cm_data = confusion_matrix(np.argmax(y_test, axis = 1 ), y_pred)
cm = pd.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


# In[56]:


print(classification_report(np.argmax(y_test, axis = 1 ),y_pred,digits=3))


# In[57]:


model.save('best_model.h5')


# In[59]:


from tensorflow.keras.models import load_model

model = load_model('best_model.h5')


# In[ ]:




