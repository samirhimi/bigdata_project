#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[5]:


df = pd.read_csv('openb_pod_list_defaultv3.csv', sep="\t")


# In[6]:


df.describe()


# In[7]:


df.head(30)


# In[ ]:





# In[8]:


pod_phase = df['pod_phase']

for value in pod_phase.unique():
    print(value)


# In[9]:


print(df.groupby('pod_phase').size())


# In[10]:


qos = df['qos']

for value in qos.unique():
    print(value)

print(df.groupby('qos').size())


# In[11]:


# histograms
df.hist()
plt.show()


# In[12]:


# box and whisker plots
df.plot(kind='box', subplots=True, layout=(8,8), sharex=False, sharey=False)
plt.show()


# In[13]:


gpu_spec = df['gpu_spec']

for value in gpu_spec.unique():
    print(value)


# In[14]:


df.head(100)
#df2 = df.drop('deletion_time', axis=1) 


# In[16]:


df3 = df.drop('gpu_spec', axis=1) 


# In[18]:


data = df3[['cpu_milli', 'memory_mib', 'num_gpu', 'gpu_milli', 'creation_time','deletion_time', 'scheduled_time','pod_phase','qos' ]]
data = df3[['cpu_milli', 'memory_mib', 'num_gpu', 'gpu_milli', 'creation_time','qos','pod_phase']]
print(data.head(10))
null_count = data.isnull().sum().sum()
print('Number of null values:', null_count)
data['pod_phase'].replace(['RUN_SUCC', 'FAIL_PEND'],
                        [0, 1], inplace=True)

data['qos'].replace(['BE','LS','Burstable', 'Guaranteed'],
                        [0, 1, 2, 3], inplace=True)


# In[19]:


import networkx as nx
import matplotlib.pyplot as plt









print(data.head(10))
data.dropna(inplace=True)

G = nx.DiGraph()

G.add_nodes_from(data.columns)

for col1 in data.columns:
    for col2 in data.columns:
        if col1 != col2:
            correlation = data[col1].corr(data[col2], 'pearson')
            G.add_edge(col1, col2, weight=correlation)
most_connected_node = max(G, key=G.degree)

pos = nx.spring_layout(G, k=0.9, seed=42)

node_colors = ['lightblue' if col != most_connected_node else 'red' for col in G.nodes]
node_sizes = [500 if col != most_connected_node else 800 for col in G.nodes]

edge_colors = [d['weight'] for _, _, d in G.edges(data=True)]
edge_widths = [2 * abs(weight) for weight in edge_colors]

plt.figure(figsize=(12, 8))
nx.draw_networkx(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes,
                 edge_color=edge_colors, width=edge_widths, edge_cmap=plt.cm.coolwarm,
                 alpha=0.7, font_size=10, arrows=True)

plt.annotate(f'Most Connected: {most_connected_node}', xy=pos[most_connected_node], xytext=(0, 20),
             textcoords='offset points', ha='center', va='center', color='red',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='red', facecolor='white'))

clusters = [cluster for cluster in nx.weakly_connected_components(G) if len(cluster) > 1]
for cluster in clusters:
    cluster_nodes = list(cluster)
    nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, node_color='yellow', node_size=600)

edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
print (edge_labels)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm)
sm.set_array(edge_colors)
#plt.colorbar(sm)

plt.title('Correlation of Pod deployment in Alababa Dataset')
plt.tight_layout()
plt.show()


# In[20]:


# Split-out validation dataset
array = data.values
X = array[:,0:6]
y = array[:,6]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
print (y)


# In[21]:


# In[ ]:





# In[22]:


df['cpu_milli'].plot()


# In[23]:


df['creation_time'].plot()


# In[24]:


df['scheduled_time'].plot()


# In[ ]:





# In[25]:


df['deletion_time'].plot()


# In[26]:


df.plot.scatter(x='creation_time', y='cpu_milli')


# In[27]:


data.head()


# In[28]:


import seaborn as sns
import sklearn
import warnings
 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score




# In[29]:


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('LDA', +()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[30]:


# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()


# In[31]:


...
# Make predictions on validation dataset
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:




