import streamlit as st
from sklearn import datasets
import numpy as np


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split #split our data into training and test data
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA #principale component algorithm
import matplotlib.pyplot as plt

st.title("Machine Learning Interactive Web App")

st.write("""
# Explore different classifers
## Which one is the best?
""")
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://st4.depositphotos.com/4376739/20147/v/1600/depositphotos_201476170-stock-illustration-big-data-visualization-artificial-intelligence.jpg")
    }
   
    </style>
    """,
    unsafe_allow_html=True
)

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris","Breast Cancer", "Wine dataset"))


classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN","SVM", "Random Forest"))

def get_dataset(dataset_name):
    if( dataset_name=="Iris"):
        data = datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data = datasets.load_breast_cancer()
    else :
        data = datasets.load_wine()
    X = data.data
    Y = data.target
    return X,Y

X,Y = get_dataset(dataset_name)



st.write("Shape of dataset ", X.shape) #for iris for example : les infos sur sepals nepals en cm etc ..
st.write("number of classes",len(np.unique(Y))) #les différents types


def add_parameter_ui(clf_name):
    params = dict()
    if(clf_name == "KNN"):
        K = st.sidebar.slider("K",1,15) #créer un nouveau slider spécifique à KNN
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"]=C
    else : 
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100) #number of trees
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if(clf_name == "KNN"):
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else : 
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)
    return clf

clf = get_classifier(classifier_name,params)

#Classification

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234) #20% of data is used for testing
clf.fit(X_train,Y_train)  #Fit API is the same for all the classifiers in sklearn
y_predict = clf.predict(X_test)

acc = accuracy_score(Y_test,y_predict)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")


#PLOT
pca = PCA(2) #to keep only 2D data
X_projected = pca.fit_transform(X) #unsupervised technic we don't need Y =label(target)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

figure = plt.figure()
plt.scatter(x1,x2,c=Y, alpha = 0.8,cmap="plasma")   #alpha makeit a little bit transaprent
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.colorbar()

st.pyplot(figure)
#plt.show()



#TODO : add more parameters and add other classifiers then add feature scaling