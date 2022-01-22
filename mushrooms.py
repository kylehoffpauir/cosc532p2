import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



"""     
    1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
                                  knobbed=k,sunken=s
     2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
     3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
                                  pink=p,purple=u,red=e,white=w,yellow=y
     4. bruises?:                 bruises=t,no=f
     5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
                                  musty=m,none=n,pungent=p,spicy=s
     6. gill-attachment:          attached=a,descending=d,free=f,notched=n
     7. gill-spacing:             close=c,crowded=w,distant=d
     8. gill-size:                broad=b,narrow=n
     9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
                                  green=r,orange=o,pink=p,purple=u,red=e,
                                  white=w,yellow=y
    10. stalk-shape:              enlarging=e,tapering=t
    11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
                                  rhizomorphs=z,rooted=r,missing=?
    12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    16. veil-type:                partial=p,universal=u
    17. veil-color:               brown=n,orange=o,white=w,yellow=y
    18. ring-number:              none=n,one=o,two=t
    19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
                                  none=n,pendant=p,sheathing=s,zone=z
    20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
                                  orange=o,purple=u,white=w,yellow=y
    21. population:               abundant=a,clustered=c,numerous=n,
                                  scattered=s,several=v,solitary=y
    22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
                                  urban=u,waste=w,woods=d
"""
# names of all the columns
colNames = ["edibility","cap-shape", "cap-surface", "cap-color", "bruises", "odor",
            "gill-attachment", "gill-spacing", "gill-size", "gill-color",
            "stalk-shape", "stalk-root", "stalk-surface-above-ring",
            "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring",
            "veil-type", "veil-color", "ring-number","ring-type", "spore-print-color",
            "population", "habitat"]

# read in the data (nominal converted to interval)
data = pd.read_csv('shrooms_modded.csv', sep=",", header=None)
data.columns = colNames
#print(data)

data_crosstab = pd.crosstab(data['edibility'],
                            data["ring-number"],
                            margins = False)
print(data_crosstab)

y=['edibility']


# y holds whether or not the mushroom is poisonous
y = data.iloc[:, 0].values
# x holds everything else
x = data.iloc[:, 1:].values

rfe = RFE(LogisticRegression(solver="liblinear"))
rfe = rfe.fit(x, y)
print(rfe.support_)
print(rfe.ranking_)
# columns rfe finds useful
useful = ['cap-surface', 'bruises',
          'gill-attachment', 'gill-spacing', 'gill-size',
          'stalk-root', 'stalk-surface-above-ring',
          'veil-color', 'ring-type', 'spore-print-color', 'population']
x = data.loc[:, useful]


# use the statsmodel to identify statistically insignificant features ( P < 0.05 ) and delete them
# logit_model=sm.Logit(y,x)
# result=logit_model.fit()
# print(result.summary2())


# removing statistically insignificant features (veil-color and gill-attachment)
useful = ['bruises', 'gill-spacing', 'gill-size',
          'stalk-root', 'stalk-surface-above-ring',
          'ring-type', 'spore-print-color', 'population']
x = data.loc[:, useful]
# by deleting these two features that are statistically insignificant, we actually do not change accuracy at all.


logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())

#logreg on the useful data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
train_score_list = []
test_score_list = []

# tested using every solver available and got the same
# weighted this way bc id rather mistakenly id a poison than an an edible
# more cautious
logreg = LogisticRegression(solver="liblinear", class_weight={0:.7, 1:.3})
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)


y_pred_train = logreg.predict(x_train)
y_pred_test = logreg.predict(x_test)
y_pred_val = logreg.predict(x_val)
from sklearn.metrics import accuracy_score
print("training accuracy: " + str(accuracy_score(y_train, y_pred_train)))
print("testing accuracy: " + str(accuracy_score(y_test, y_pred_test)))
print("validation accuracy: " + str(accuracy_score(y_val, y_pred_val)))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))
print(logreg.get_params())


# visualize conf matrix:

# class_names=[0,1] # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# # create heatmap
# sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')

"""
using test size = .2 and random state = 0 
gives us confusion matrix of
  True+   False+
  [1154     12]
  [ 68    1204]
 False-    True-
 so we have 2358 correct predictions and 90 incorrect
 BUT! only 12 FP
 
 and an accuracy report of:
              precision   recall  f1-score   support

        P 0     0.94      0.99      0.97      1166
        E 1     0.99      0.95      0.97      1272

    accuracy                        0.97      2438
   macro avg    0.97      0.97      0.97      2438
weighted avg    0.97      0.97      0.97      2438


precision: tp / (tp+fp) -- the ratio of not creating a false positive
recall: tp / (tp+fn) -- how well the classifier finds all the positives
f1-score: "weighted harmonic mean of precision and recall" -- 0 bad, 1 good.
support: number of occurance of a p or e in y_test
 
 We have a 99% precision for identifying if a mushroom is edible!
"""

logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

importances = pd.DataFrame(data={
    'Attribute': x_train.columns,
    'Importance': logreg.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()
