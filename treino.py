import sys
import scipy
import numpy as np
import matplotlib
import pandas
import sklearn
import joblib
from pathlib import Path 
import seaborn as sns
import pandas as pd #trabalhar com dados tabulares -> trabahar com dados em formato de tabela
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt #fazer figuras e plotagens para melhor visualização dos dados
from sklearn import model_selection  #
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score # metrica para definir em a % de acerto do modelo
from sklearn.linear_model import LogisticRegression # modelo de classificação
from sklearn.tree import DecisionTreeClassifier #modelo que trata os dados por arvore de desisão
from sklearn.neighbors import KNeighborsClassifier #
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # modelo que trabalha com analize de probabilidade
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH1= MODEL_DIR / "predict_iris_model1.joblib"
MODEL_PATH2 = MODEL_DIR / "predict_iris_model2.joblib"
MODEL_PATH3 = MODEL_DIR / "predict_iris_model3.joblib"
MODEL_PATH4= MODEL_DIR / "predict_iris_model4.joblib"
MODEL_PATH5= MODEL_DIR / "predict_iris_model5.joblib"
MODEL_PATH6 = MODEL_DIR / "predict_iris_model6.joblib"

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(url, names = attributes)
#print(df.head(3))
#print(df.shape)
#print(df.dtypes)
# print(df.info())

# df.plot(kind='box', subplots=True, layout=(2,2)) #grafico de caixa e bigode para analize da distribuição dos dados
# plt.show()

numeric_df = df.select_dtypes(include="number")
# correlations = numeric_df.corr()
# sns.heatmap(correlations, cmap="BrBG", annot=True, fmt=".3f")
# plt.show()

x = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["class"]

n_seed_test = 2
teste_porcentagem = 0.20

x_train,x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=teste_porcentagem, random_state=n_seed_test, shuffle=True)

pipeline1 = Pipeline(steps=[("logreg", LogisticRegression(max_iter=100, random_state=n_seed_test))])
pipeline2 = Pipeline(steps=[("tree", DecisionTreeClassifier())])
pipeline3 = Pipeline(steps=[("kN", KNeighborsClassifier())])
pipeline4 = Pipeline(steps=[("linear", LinearDiscriminantAnalysis())])
pipeline5 = Pipeline(steps=[("svc", SVC())])
pipeline6 = Pipeline(steps=[("gau", GaussianNB())])


pipeline1.fit(x_train, y_train)
predicao = pipeline1.predict(x_test)

pipeline2.fit(x_train, y_train)
predicao = pipeline2.predict(x_test)

pipeline3.fit(x_train, y_train)
predicao = pipeline3.predict(x_test)

pipeline4.fit(x_train, y_train)
predicao = pipeline4.predict(x_test)

pipeline5.fit(x_train, y_train)
predicao = pipeline5.predict(x_test)

pipeline6.fit(x_train, y_train)
predicao = pipeline6.predict(x_test)

print("acuracia: ", accuracy_score(y_test, predicao))
# print("precisao: ", precision_score(y_test, predicao))
# print("recall: ", recall_score(y_test, predicao))
print("report: ", classification_report(y_test, predicao))

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(pipeline1, MODEL_PATH1)
joblib.dump(pipeline2, MODEL_PATH2)
joblib.dump(pipeline3, MODEL_PATH3)
joblib.dump(pipeline4, MODEL_PATH4)
joblib.dump(pipeline5, MODEL_PATH5)
joblib.dump(pipeline6, MODEL_PATH6)
print(f"\nModelo salvo em: {MODEL_PATH1}")
print(f"\nModelo salvo em: {MODEL_PATH2}")
print(f"\nModelo salvo em: {MODEL_PATH3}")
print(f"\nModelo salvo em: {MODEL_PATH4}")
print(f"\nModelo salvo em: {MODEL_PATH5}")
print(f"\nModelo salvo em: {MODEL_PATH6}")

