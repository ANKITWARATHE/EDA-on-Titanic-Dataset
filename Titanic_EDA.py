#!/usr/bin/env python
# coding: utf-8

# # --------------------------------------------Titanic EDA---------------------------------------------

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


pip install ydata-profiling


# In[2]:


from ydata_profiling import ProfileReport


# In[3]:


titanic_train=pd.read_csv("https://raw.githubusercontent.com/ANKITWARATHE/Titanic-Project/main/titanic_train.csv")
titanic_test=pd.read_csv("https://raw.githubusercontent.com/ANKITWARATHE/Titanic-Project/main/test.csv")

combine = [titanic_train, titanic_test]


# In[4]:


combine


# In[5]:


titanic_train.columns


# In[6]:


titanic_test.columns


# In[7]:


# Top 5 record of titanic train data
titanic_train.head(10)


# In[8]:


titanic_train.tail()


# # Examining Data

# In[9]:


titanic_train.shape #shows total number of rows and columns in data set


# In[10]:


titanic_train.describe()


# ## Insights:
# 
# 1.Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224)
# 
# 2.Survived is a categorical feature with 0 or 1 values
# 
# 3.Around 38% samples survived representative of the actual survival rate at 32%
# 
# 4.Fares varied significantly with few passengers (<1%) paying as high as $512.
# 
# 5.Few elderly passengers (<1%) within age range 65-80.

# # Data Profiling
# By pandas profiling, an interactive HTML report gets generated which contains all the information about the columns of the dataset, like the counts and type of each column.
# 
# 1.Detailed information about each column, coorelation between different columns and a sample of dataset
# 
# 2.It gives us visual interpretation of each column in the data
# 
# 3.Spread of the data can be better understood by the distribution plot
# 
# 4.Grannular level analysis of each column.

# In[11]:


titanic_profile = ProfileReport(titanic_train, title= "Pandas Profiling Report")


# In[12]:


titanic_profile


# In[13]:


titanic_profile.to_file(output_file="Pandas Profiling Report.html")


# ## Data Preprocessing
# - Check for Errors and Null Values
# 
# - Replace Null Values with appropriate values
# 
# - Drop down features that are incomplete and are not too relevant for analysis
# 
# - Create new features that can would help to improve prediction

# In[14]:


miss1=titanic_train.isnull().sum()
miss= (titanic_train.isnull().sum()/len(titanic_train))*100
miss_data=pd.concat([miss1,miss],axis=1,keys=['Total','%'])
print(miss_data)


# The Age, Cabin and Embarked have null values.Lets fix them

# **Filling missing age by median**

# In[15]:


new_age = titanic_train.Age.median()


# In[16]:


new_age


# In[17]:


titanic_train.Age.fillna(new_age, inplace = True)


# **Filling missing Embarked by mode**

# In[18]:


titanic_train.Embarked = titanic_train.Embarked.fillna(titanic_train['Embarked'].mode()[0])


# In[19]:


titanic_train.isnull().sum()


# **Cabin feature may be dropped as it is highly incomplete or contains many null values**

# In[20]:


titanic_train.drop('Cabin', axis = 1,inplace = True)


# In[21]:


titanic_train.isnull().sum()


# **PassengerId Feature may be dropped from training dataset as it does not contribute to survival**

# In[22]:


titanic_train.drop('PassengerId', axis = 1,inplace = True)


# **Ticket feature may be dropped down**

# In[23]:


titanic_train.drop('Ticket', axis = 1,inplace = True)


# ## Creating New Fields

# 1.Create New Age Bands to improve prediction Insights
# 
# 2.Create a new feature called Family based on Parch and SibSp to get total count of family members on board
# 
# 3.Create a Fare range feature if it helps our analysis

# # AGE-BAND

# In[24]:


titanic_train['Age_band']=0

titanic_train.loc[titanic_train['Age']<=1,'Age_band']="Infant"

titanic_train.loc[(titanic_train['Age']>1)&(titanic_train['Age']<=12),'Age_band']="Children"

titanic_train.loc[titanic_train['Age']>12,'Age_band']="Adults"

titanic_train.head(10)


# # Fare-Band

# In[25]:


titanic_train['FareBand']=0
titanic_train.loc[(titanic_train['Fare']>=0)&(titanic_train['Fare']<=10),'FareBand']=1
titanic_train.loc[(titanic_train['Fare']>10)&(titanic_train['Fare']<=15),'FareBand']=2
titanic_train.loc[(titanic_train['Fare']>15)&(titanic_train['Fare']<=35),'FareBand']=3
titanic_train.loc[titanic_train['Fare']>35,'FareBand']=4
titanic_train.head(10)


# We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
# 
# - In the following code we extract Title feature using regular expressions. The RegEx pattern (\w+.) matches the first word which ends with a dot character within Name feature. The expand=False flag returns a DataFrame.

# In[26]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(titanic_train['Title'], titanic_train['Sex'])


# We can replace many titles with a more common name or classify them as Rare.

# In[27]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
titanic_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# We can convert the categorical titles to ordinal.

# In[28]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

titanic_train.head()


# ## Insights
# 
# - Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
# - Survival among Title Age bands varies slightly.
# - Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).
# 
# **Decision**
# 
# We decide to retain the new Title feature for model training

# ### Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
# Converting Sex feature to a new feature called Gender where female=1 and male=0.

# In[29]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

titanic_train.head()


# **Extracting Titles Now we can drop down Name feature**

# In[30]:


titanic_train.drop('Name', axis = 1,inplace = True)


# In[31]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

titanic_train.head()


# We can also create an artificial feature combining Pclass and Age.

# In[32]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

titanic_train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# # Post Pandas Profiling : Checking Data after data preparation

# In[34]:


profile = ProfileReport(titanic_train)
profile.to_file(output_file="Titanic_after_preprocessing.html")


# # Data Visualization

# ### 4.1 What is Total Count of Survivals and Victims?

# In[33]:


titanic_train.groupby(['Survived'])['Survived'].count()# similar functions unique(),sum(),mean() etc


# In[34]:


plt = titanic_train.Survived.value_counts().plot(kind ='bar')
plt.set_xlabel('DIED OR SURVIVED')
plt.set_ylabel('Passenger Count')


# ## Insights
# 
# - Only 342 Passengers Survived out of 891
# - Majority Died which conveys there were less chances of Survival

# ### 4.2 Which gender has more survival rate?

# In[35]:


titanic_train.groupby(['Survived', 'Sex']).count()["Age"]


# In[36]:


sns.countplot(data=titanic_train,x ='Survived',hue='Sex')


# In[37]:


titanic_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()


# ### Insights
# 
# - Female has better chances of Survival "LADIES FIRST"
# - There were more males as compared to females ,but most of them died.

# ### 4.3 What is Survival rate based on Person type?

# In[38]:


titanic_train.groupby(['Survived', 'Age_band']).count()['Sex']


# In[39]:


titanic_train[titanic_train['Age_band'] == 'Adults'].Survived.groupby(titanic_train.Survived).count().plot(kind='pie', figsize=(6, 6),explode=[0,0.05],autopct='%1.1f%%')
plt.axis('equal')
plt.legend(["Died","Survived"])
plt.set_title("Adult survival rate")
#plt.show()


# ------------------------------------------**ADULT-SURVIVAL RATE**--------------------------------------------------------------

# In[40]:


titanic_train[titanic_train['Age_band'] == 'Children'].Survived.groupby(titanic_train.Survived).count().plot(kind='pie', figsize=(6, 6),explode=[0,0.05],autopct='%1.1f%%')
plt.axis('equal')
#plt.legend(["Died","Survived"])
plt.set_title("Child survival rate")
#plt.show()


# ------------------------------------------**CHILD-SURVIVAL RATE**--------------------------------------------------------------

# In[41]:


titanic_train[titanic_train['Age_band'] == 'Infant'].Survived.groupby(titanic_train.Survived).count().plot(kind='pie', figsize=(6, 6),explode=[0,0.05],autopct='%1.1f%%')
plt.axis('equal')
#plt.legend(["Died","Survived"])
plt.set_title("Infant survival rate")
#plt.show()


# ------------------------------------------**INFANT-SURVIVAL RATE**--------------------------------------------------------------

# ### Insights
# 
# - Majority Passengers were Adults
# 
# - Almost half of the total number of children survived.
# 
# - Most of the Adults failed to Survive
# 
# - More than 85percent of Infant Survived

# ### 4.4 Did Economy Class had an impact on survival rate?

# In[42]:


titanic_train.groupby(['Pclass', 'Survived'])['Survived'].count()


# In[44]:


sns.barplot(x= 'Pclass',y ='Survived', data = titanic_train)


# In[46]:


sns.barplot(x = 'Pclass',y = 'Survived',hue='Sex', data=titanic_train)


# ### Insights
# 
# - Most of the passengers travelled in Third class but only 24per of them survived
# 
# - If we talk about survival ,more passengers in First class survived and again female given more priority
# 
# - Economic Class affected Survival rate and Passengers travelling with First Class had higher ratio of survival as compared to Class 2 and 3.

# ### 4.5 **What is Survival Propability based on Embarkment of passengers?**
# 

# Titanic’s first voyage was to New York before sailing to the Atlantic Ocean it picked passengers from three ports Cherbourg(C), Queenstown(Q), Southampton(S). Most of the Passengers in Titanicic embarked from the port of Southampton.Lets see how embarkemt affected survival probability.

# In[48]:


sns.countplot(x = 'Embarked',data=titanic_train)


# In[50]:


plt = titanic_train[['Embarked', 'Survived']].groupby('Embarked').mean().Survived.plot(kind = 'bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Survival Probability')


# ### Gender Survival based on Embarkment and Pclass

# In[51]:


pd.crosstab([titanic_train.Sex, titanic_train.Survived,titanic_train.Pclass],[titanic_train.Embarked], margins=True)


# In[52]:


sns.violinplot(x='Embarked',y='Pclass',hue='Survived',data=titanic_train,split=True)


# In[53]:


sns.catplot(x="Embarked", y="Survived", hue="Sex",
            col="Pclass", aspect=.8,kind='bar',
             data=titanic_train);


# ### Insights:
# 
# - Most Passengers from port C Survived.
# 
# - Most Passengers were from Southampton(S).
# 
# - Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
# 
# - Males had better survival rate in Port C when compared for S and Q ports.
# 
# - Females had least Survival rate in Pclass 3

# ### 4.6 How is Fare distributed for Passesngers?
# 
# 

# In[54]:


titanic_train['Fare'].min()


# In[55]:


titanic_train['Fare'].max()


# In[56]:


titanic_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[57]:


titanic_train.groupby(['FareBand', 'Survived'])['Survived'].count()


# In[61]:


sns.swarmplot(x='Survived', y='Fare', data=titanic_train)


# ### Insights
# 
# - Majority Passenger's fare lies in 0-100 dollars range
# - Passengers who paid more Fares had more chances of Survival
# - Fare as high as 514 dollars was purcharsed by very few.(Outlier)

# ### 4.7 What was Average fare by Pclass & Embark location?

# In[65]:


sns.boxplot(x="Pclass", y="Fare", data=titanic_train,hue="Embarked",fliersize=5)


# In[66]:


sns.boxplot(x="Embarked", y="Fare", data=titanic_train)


# ### Insights
# 
# - First Class Passengers paid major part of total Fare.
# - Passengers who Embarked from Port C paid Highest Fare

# ### 4.8 Segment Age in bins with size of 10

# In[67]:


plt=titanic_train['Age'].hist(bins=20)
plt.set_ylabel('Passengers')
plt.set_xlabel('Age of Passengers')
plt.set_title('Age Distribution of Titanic Passengers',size=17, y=1.08)


# ### Insights:
# 
# - The youngest passenger on the Titanic were toddlers under 6 months
# - The oldest were of 80 years of age.
# - The mean for passengers was a bit over 29 years i.e there were more young passengers in the ship.

# ### Lets see how Age has correlation with Survival

# In[70]:


sns.distplot(titanic_train[titanic_train['Survived']==1]['Age'])


# In[71]:


sns.distplot(titanic_train[titanic_train['Survived']==0]['Age'])


# In[72]:


sns.violinplot(x='Sex',y='Age',hue='Survived',data=titanic_train,split=True)


# ### Insights
# 
# - Most of the passengers died.
# - Majority of passengers were between 25-40,most of them died
# - Female are more likely to survival

# ### 4.9 Did Solo Passenger has less chances of Survival ?

# In[73]:


titanic_train['FamilySize']=0
titanic_train['FamilySize']=titanic_train['Parch']+titanic_train['SibSp']
titanic_train['SoloPassenger']=0
titanic_train.loc[titanic_train.FamilySize==0,'SoloPassenger']=1


# In[79]:


sns.catplot(y= 'SoloPassenger',x = 'Survived',data=titanic_train,kind = 'point')


# In[80]:


sns.violinplot(y='SoloPassenger',x='Sex',hue='Survived',data=titanic_train,split=True)


# In[82]:


sns.catplot(x ='SoloPassenger',y = 'Survived',hue='Pclass',col="Embarked",data=titanic_train,kind = 'point')


# ### Insights
# 
# - Most of the Passengers were travelling Solo and most of them died
# - Solo Females were more likely to Survive as compared to males
# - Passengers Class have a positive correlation with Solo Passenger Survival
# - Passengers Embarked from Port Q had Fifty -Fifty Chances of Survival

# ### 4.10 How did total family size affected Survival Count?

# In[83]:


for i in titanic_train:
    titanic_train['FamilySize'] = titanic_train['SibSp'] + titanic_train['Parch'] + 1

titanic_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[84]:


sns.barplot(x='FamilySize', y='Survived', hue='Sex', data=titanic_train)


# ### Insights
# 
# - Both men and women had a massive drop of survival with a FamilySize over 4.
# - The chance to survive as a man increased with FamilySize until a size of 4
# - Men are not likely to Survive with FamilySize 5 and 6
# - Big Size Family less likihood of Survival

# ### 4.11 How can you correlate Pclass/Age/Fare with Survival rate?

# In[85]:


sns.pairplot(titanic_train[["FareBand","Age","Pclass","Survived"]],vars= ["FareBand","Age","Pclass"],hue="Survived", dropna=True,markers=["o", "s"])


# ### Insights:
# - Fare and Survival has positive correlation
# 
# - We cannt relate age and Survival as majority of travellers were of mid age
# 
# - Higher Class Passengers had more likeihood of Survival

# ### 4.12 Which features had most impact on Survival rate?

# In[86]:


sns.heatmap(titanic_train.corr(),annot=True)


# ### Insights:
# 
# - Older women have higher rate of survival than older men . Also, older women has higher rate of survival than younger women; an opposite trend to the one for the male passengers.
# - All the features are not necessary to predict Survival
# - More Features creates Complexitity
# - Fare has positive Correlation
# - For Females major Survival Chances , only for port C males had more likeihood of Survival.

# # Conclusion : "If you were young female travelling in First Class and embarked from port -C then you have best chances of Survival in Titanic"
# 
# -  Most of the Passengers Died
# - "Ladies & Children First" i.e **76% of Females and 16% of Children** Survived
# -  Gender , Passenger type & Classs are mostly realted to Survival.
# -  Survival rate diminishes significantly for Solo  Passengers
# -  Majority of Male Died
# -  Males with Family had better Survival rate as compared to Solo Males

# In[ ]:




