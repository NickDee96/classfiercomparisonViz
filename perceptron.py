import pandas as pd
from sklearn import linear_model

df=pd.read_csv("sonar.all-data.csv",header=None)

from sklearn.model_selection import train_test_split

y=df[60]
x=df.drop([60],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.25, random_state=12)



##plotting the df from PCA

from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding

pca_3d = PCA(n_components=3)
PCs_3d = pd.DataFrame(pca_3d.fit_transform(x_test.reset_index(drop=True)))
PCs_3d["label"]=y_test.reset_index(drop=True)

def color_gen(ar):
    col=[]
    for i in ar:
        if i=="R":
            col.append("#5edce6")
        else:
            col.append("#f03a98")
    return col


import plotly.graph_objs as go
fig=go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=PCs_3d[0],
        y=PCs_3d[1],
        z=PCs_3d[2],
        mode="markers",
        marker=dict(
            color=color_gen(PCs_3d["label"])
        )
    )
)
import plotly.express as px
fig = px.scatter_matrix(
    df[[x for x in range(10)].append(60)],
    color=60)
fig.show()




##Logistic regression
lgRg=linear_model.LogisticRegression()
lgRg.fit(x_train,y_train)
lgRg.score(x_test,y_test) 

## Least squares support vector machines
from sklearn import svm
clf=svm.SVC()
clf.fit(x_tclf = neighbors.KNeighborsClassifier(15, weights=weights)
clf.fit(x_train, y_train)
clf.score(x_test,y_test)

#Quadratic Classifiers (Quadratic Discriminant Analysis)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf=QuadraticDiscriminantAnalysis()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)


## K-Nearest Neighbour
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(15, weights="")
clf.fit(x_train, y_train)
clf.score(x_test,y_test)

## Random Forest
## Neural Networks
## Learning Vector Quantization