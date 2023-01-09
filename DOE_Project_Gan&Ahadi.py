"""
ISyE 6413: Design and Analysis of Experiments
Fall 2022
Team members: Yishin Gan - Pouya Ahadi
"""

### Loading required packages
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.datasets import make_classification
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_gaussian_quantiles
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
import random

#%%
"""
Part I: comparing entropy sampling and random sapmling:
    -In this part, we use sequantial design to build a classifier in in iterative process.
    -We compare random sampling with entropy sampling
"""

#################### reading data set: you can comment the code for any data set that is not used

#### Spmbase UCI
data = pd.read_csv('spambase.data',header =None)
data = np.asarray(data)
X = data[:,:57]
y = data[:,57]

# #### Statlog (heart) UCI
# data = pd.read_csv('heart.dat', header =None, delimiter=' ')
# data = np.asarray(data)
# X = data[:,:13]
# y = data[:,13]
# y = y - 1


#%%
############################################# Train/ Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    shuffle = True, random_state=42)
c = len(set(y_train))
if c == 2:
    avg_method = 'binary'
else:
    avg_method = 'macro'

############################################# Initial Parameters and Inputs
classifier = RandomForestClassifier(random_state=1) # we can change to any other desired classifier
n_cycles = 15# number of AL cycles we want to try
budget = 5 # sampling budget per cycle

############################################# AL Initialization
n_train = X_train.shape[0] # size of train data
p_init = 0.01 # proportion of train data that we want to use for initialization of AL

I = list(np.arange(n_train))
L_init = random.sample(I, int(n_train*p_init) ) # set of initially labeled data
U_init = [each for each in I if each not in L_init] # set of initially unlabeled data

# trining initial classifier and finding initial accuracy and F1 score
clf = classifier
clf.fit(X_train[L_init], y_train[L_init])
y_pred = clf.predict(X_test)
initial_accuracy = accuracy_score(y_test, y_pred)
initial_f1 = f1_score(y_test, y_pred, average=avg_method)

############################################# Defining AL function                                 
def AL1():
    ############################# Entropy Sampling (ES) ############################# 
    learner = ActiveLearner(estimator=classifier,
                            query_strategy=entropy_sampling,
                            X_training=X_train[L_init],
                            y_training=y_train[L_init])
    
    L = L_init # L represent the list of labeled data at each cycle
    U = U_init # U represent the list of unlabeled data at each cycle
    
    accuracy_ES = [] # accuracy lits for entropy sampling: it saves accuracy of all iterations
    F1_score_ES = [] # F1 lits for entropy sampling: it saves accuracy of all iterations
    
    accuracy_ES.append(initial_accuracy)
    F1_score_ES.append(initial_f1)
    
    for i in range(n_cycles): # AL loop
        index_pool = U
        # following commands finds the query points for from unlabeled data based on budget
        query_index, query_instance = learner.query(X_train[index_pool], n_instances = budget )
        X_AL_split, Y_AL_split = X_train[index_pool][query_index], y_train[index_pool][query_index]
        
        # following commands retraint the classifier based on new labeled data and evaluate model
        learner.teach(X=X_AL_split, y=Y_AL_split)
        pred_test = learner.predict(X_test)
        F1_score_ES.append(f1_score(y_test, pred_test, average=avg_method))
        accuracy_ES.append(accuracy_score(y_test, pred_test))

        # following lines updates the unlabeled and labeled data sets for the next cycle
        L = [*L, *np.array(index_pool)[query_index]]
        U = [each for each in I if each not in L]

    ############################# Random Sampling (RS) #############################
    learner = ActiveLearner(estimator=classifier,
                            X_training=X_train[L_init],
                            y_training=y_train[L_init])
    
    L = L_init # L represent the list of labeled data at each cycle
    U = U_init # U represent the list of unlabeled data at each cycle
    
    accuracy_RS = [] # accuracy lits for entropy sampling: it saves accuracy of all iterations
    F1_score_RS = [] # F1 lits for entropy sampling: it saves accuracy of all iterations
    
    accuracy_RS.append(initial_accuracy)
    F1_score_RS.append(initial_f1 )
    
    for i in range(n_cycles): # AL loop
        index_pool = U
        # following commands finds the query points for from unlabeled data based on budget
        query_index = np.random.choice(U, budget, replace=False)
        X_AL_split, Y_AL_split = X_train[query_index], y_train[query_index]
        
        # following commands retraint the classifier based on new labeled data and evaluate model
        learner.teach(X=X_AL_split, y=Y_AL_split)
        pred_test = learner.predict(X_test)
        F1_score_RS.append(f1_score(y_test, pred_test, average=avg_method))
        accuracy_RS.append(accuracy_score(y_test, pred_test))

        # following lines updates the unlabeled and labeled data sets for the next cycle
        L = [*L, *query_index]
        U = [each for each in I if each not in L]
   
    return accuracy_ES, F1_score_ES, accuracy_RS, F1_score_RS

#%%
############################################# Running AL and showing results
round_num = 10 # number of replications for AL
results = []
for r in range(round_num):
    results.append(AL1())
    
##plotting
import matplotlib.ticker as ticker

AL1_acc_res = np.array([each[1] for each in results])
AL2_acc_res = np.array([each[3] for each in results])

plt.rcParams["figure.figsize"] = [8, 5.5]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5

num_all = round_num * (n_cycles+1)

AL1_repeat_pd = pd.DataFrame.from_dict({"acc":AL1_acc_res.reshape(-1), "iteration":list(range(0,n_cycles+1))*round_num}, orient="columns")
AL2_repeat_pd = pd.DataFrame.from_dict({"acc":AL2_acc_res.reshape(-1), "iteration":list(range(0,n_cycles+1))*round_num}, orient="columns")

repeat_pd = pd.concat([AL1_repeat_pd,AL2_repeat_pd])

repeat_pd["method"] = ["Entropy Sampling"]*num_all + ["Random Sampling"]*num_all

plt.rcParams['figure.figsize'] = [15,10]
repeat_pd_median = repeat_pd.groupby(["iteration","method"]).median().reset_index()
ax = sns.lineplot(x='iteration',y='acc',hue='method',data=repeat_pd_median, palette={"Random Sampling":'blue',"Entropy Sampling":'red'})

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(17)

legend = ax.legend(loc='lower right' , borderpad=1.5,
                    labelspacing=0.5,ncol=1,
                    handlelength=2,fontsize = 25) #prop={'size':15}

legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_facecolor('lightgray')
legend.legendPatch.set_edgecolor("black")

plt.xlabel('cycle', fontsize=30)
plt.ylabel('F1 Score', fontsize=30)
plt.title('Heart Statlog Data Set', fontsize=30)

# plt.savefig('DOE_ESvsRS_1rep_median_heartstatlog', dpi = 300)
plt.show()


#%%
"""
Part II: adding exploration to experiments:
    -In this part, we divide our sampling budget into two parts.
    -First part is for exploitation using entropy sampling.
    - Second part is for exploration. We use Max-min distance design for exploration.
    - We compare the case when we don't add exploration vs adding exploration
"""

########################### Simulate data and adding gaussian clusters
plt.rcParams["figure.figsize"] = [8, 5.5]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5

## 3 class case - without noise
X1,y1= make_classification(n_samples=19900, n_features=2, n_informative=2,
                          n_redundant=0, n_repeated=0, n_classes=2,
                          n_clusters_per_class=1,class_sep=1.8,flip_y=0,
                          weights=[0.99,0.01], random_state=5)

X2, y2 = make_gaussian_quantiles(mean=(-2, 3), cov=0.1, n_samples=50,
                                 n_features=2,n_classes=1, random_state=2)

X3, y3 = make_gaussian_quantiles(mean=(-5, -6), cov=0.1, n_samples=50,
                                 n_features=2,n_classes=1, random_state=3)

y2 = np.ones(len(y2))
y3 = np.ones(len(y3))

X = np.concatenate([X1,X2,X3])
y = np.concatenate([y1,y2,y3])

## plot data
ax= plt.subplot()
sns.scatterplot(X[:,0],X[:,1], palette=['blue','red'],
                hue=y,ax=ax, s = 3, edgecolor ="black");
legend = ax.legend(loc='upper left' , borderpad=1.5,
                    labelspacing=0.5,ncol=1,
                    handlelength=3,fontsize = 12) #prop={'size':15}
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_facecolor('lightgray')
legend.legendPatch.set_edgecolor("black")

plt.xlabel('X1', fontsize=15)
plt.ylabel('X2', fontsize=15)

plt.title("3 Clusters - skewness = 1.5 %", fontsize=20)

# plt.savefig("Data - 3 clusters.jpg", dpi=300)
plt.show();

########################### Simulate data and adding gaussian clusters
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    shuffle = True, random_state=42)
c = len(set(y_train))
if c == 2:
    avg_method = 'binary'
else:
    avg_method = 'macro'

############################################# Initial Parameters and Inputs
classifier = RandomForestClassifier(random_state=1) # we can change to any other desired classifier
n_cycles = 15# number of AL cycles we want to try
budget = 5 # sampling budget per cycle

############################################# AL Initialization
n_train = X_train.shape[0] # size of train data
p_init = 0.001# proportion of train data that we want to use for initialization of AL

I = list(np.arange(n_train))
L_init = random.sample(I, int(n_train*p_init) ) # set of initially labeled data
U_init = [each for each in I if each not in L_init] # set of initially unlabeled data

# trining initial classifier and finding initial accuracy and F1 score
clf = classifier
clf.fit(X_train[L_init], y_train[L_init])
y_pred = clf.predict(X_test)
initial_accuracy = accuracy_score(y_test, y_pred)
initial_f1 = f1_score(y_test, y_pred, average=avg_method)

############################################# Defining AL function 
from scipy.spatial.distance import cdist                                
def AL2():
    ############################# Entropy Sampling (exploitation only) ############################# 
    learner = ActiveLearner(estimator=classifier,
                            query_strategy=entropy_sampling,
                            X_training=X_train[L_init],
                            y_training=y_train[L_init])
    
    L = L_init # L represent the list of labeled data at each cycle
    U = U_init # U represent the list of unlabeled data at each cycle
    
    accuracy_ES = [] # accuracy lits for entropy sampling: it saves accuracy of all iterations
    F1_score_ES = [] # F1 lits for entropy sampling: it saves accuracy of all iterations
    
    accuracy_ES.append(initial_accuracy)
    F1_score_ES.append(initial_f1)
    
    for i in range(n_cycles): # AL loop
        index_pool = U
        # following commands finds the query points for from unlabeled data based on budget
        query_index, query_instance = learner.query(X_train[index_pool], n_instances = budget )
        X_AL_split, Y_AL_split = X_train[index_pool][query_index], y_train[index_pool][query_index]
        
        # following commands retraint the classifier based on new labeled data and evaluate model
        learner.teach(X=X_AL_split, y=Y_AL_split)
        pred_test = learner.predict(X_test)
        F1_score_ES.append(f1_score(y_test, pred_test, average=avg_method))
        accuracy_ES.append(accuracy_score(y_test, pred_test))

        # following lines updates the unlabeled and labeled data sets for the next cycle
        L = [*L, *np.array(index_pool)[query_index]]
        U = [each for each in I if each not in L]

    ############################# Entropy Sampling with Exploration (ESE) #############################
    learner = ActiveLearner(estimator=classifier,
                            X_training=X_train[L_init],
                            y_training=y_train[L_init])
    
    L = L_init # L represent the list of labeled data at each cycle
    U = U_init # U represent the list of unlabeled data at each cycle
    
    accuracy_ESE = [] # accuracy lits for entropy sampling: it saves accuracy of all iterations
    F1_score_ESE = [] # F1 lits for entropy sampling: it saves accuracy of all iterations
    
    accuracy_ESE.append(initial_accuracy)
    F1_score_ESE.append(initial_f1 )
    
    for i in range(n_cycles): # AL loop
        index_pool = U
        ## query for exploitation
        query_index1, query_instance = learner.query(X_train[index_pool], n_instances = int(budget *exploitation_ratio) )
        X_exploitation, Y_exploitation = X_train[index_pool][query_index1], y_train[index_pool][query_index1]
        
        ## query for exploration
        query_index2 = [] 
        L_Q = [*L, *np.array(index_pool)[query_index1]] # labeled and queried union
        idx = [each for each in index_pool if each not in L_Q] # list of neither labeled nor querried in pool
        
        ## following loop is performing max-min distance design in a sequential way (as mentioned in our reports and slides)
        for j in range(budget - int(budget *exploitation_ratio)): 
            dist = cdist(X_train[idx], X_train[L_Q])
            query = idx[np.argmax(np.min(dist, axis=1))]
            query_index2.append(query)
            idx.remove(query)
            L_Q.append(query)
            
        X_exploration, Y_exploration = X_train[query_index2], y_train[query_index2]
        
        # following commands retraint the classifier based on new labeled data and evaluate model
        X_AL_split = np.concatenate([X_exploitation, X_exploration]) # combining exploration and exploitation querries
        Y_AL_split = np.concatenate([Y_exploitation, Y_exploration])
        learner.teach(X=X_AL_split, y=Y_AL_split)
        pred_test = learner.predict(X_test)
        F1_score_ESE.append(f1_score(y_test, pred_test, average=avg_method))
        accuracy_ESE.append(accuracy_score(y_test, pred_test))

        # following lines updates the unlabeled and labeled data sets for the next cycle
        L = [*L, *np.array(index_pool)[query_index1]]
        L = [*L, *query_index2]
        U = [each for each in I if each not in L]

    return accuracy_ES, F1_score_ES, accuracy_ESE, F1_score_ESE

#%%
############################################# Initial Parameters and Inputs
classifier = RandomForestClassifier(random_state=1) 
n_cycles = 15
round_num = 1
exploitation_ratio = 0.9 # ratio of budget we want to use for exploitation
budget = 100

############################################# Running AL and showing results
results = []
for r in range(round_num):
    results.append(AL2())
 
#%%
##plotting
import matplotlib.ticker as ticker

AL1_acc_res = np.array([each[1] for each in results])
AL2_acc_res = np.array([each[3] for each in results])

plt.rcParams["figure.figsize"] = [8, 5.5]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5

num_all = round_num * (n_cycles+1)

AL1_repeat_pd = pd.DataFrame.from_dict({"acc":AL1_acc_res.reshape(-1), "iteration":list(range(0,n_cycles+1))*round_num}, orient="columns")
AL2_repeat_pd = pd.DataFrame.from_dict({"acc":AL2_acc_res.reshape(-1), "iteration":list(range(0,n_cycles+1))*round_num}, orient="columns")

repeat_pd = pd.concat([AL1_repeat_pd,AL2_repeat_pd])

repeat_pd["method"] = ["Only Exploitation (Uncertainty Sampling)"]*num_all + ["With Exploration"]*num_all

plt.rcParams['figure.figsize'] = [15,10]
repeat_pd_median = repeat_pd.groupby(["iteration","method"]).median().reset_index()
ax = sns.lineplot(x='iteration',y='acc',hue='method',data=repeat_pd_median, palette={"Only Exploitation (Uncertainty Sampling)":'blue',"With Exploration":'red'})

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(17)

legend = ax.legend(loc='lower right' , borderpad=1.5,
                    labelspacing=0.5,ncol=1,
                    handlelength=2,fontsize = 25) #prop={'size':15}

legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_facecolor('lightgray')
legend.legendPatch.set_edgecolor("black")

plt.xlabel('cycle', fontsize=30)
plt.ylabel('F1 Score', fontsize=30)
plt.title('Data with 3 clusters', fontsize=30)
# plt.savefig('DOE_ESvsESS_1rep_3cluster', dpi = 300)
plt.show()