import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



df = pd.read_csv('data/hotel.csv')
#se débarrasser des valeurs nulles
df = df.dropna()

# garder toujours le meme echantillon code "34"
np.random.seed(34)
# Prélèvement d'un échantillon représentatif de 30%
df1 = df.sample(frac = 0.3)

# Ajout de la colonne des sentiments
df1['sentiments'] = df1.Rating.apply(lambda x: 0 if x in [1, 2] else 1)


X = df1['Review']
y = df1['sentiments']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=24)

# Vectoriser les données textuelles - avec nombre d occurrences (entiers)
cv = CountVectorizer()

# # Vectoriser les données textuelles - avec tfidf (reels)
# cv = TfidfVectorizer()

X_entrainement = cv.fit_transform(X_train)
X_evaluation = cv.transform(X_test)
# print(X_entrainement.toarray())

# Afficher les mots dans le corpus
# np.set_printoptions(threshold=sys.maxsize)
# print(cv.get_feature_names_out())


# # Entraîner le modèle
# lr = LogisticRegression()
# lr.fit(X_entrainement, y_train)

# # Score
# lr_score = lr.score(X_evaluation, y_test)
# print(lr_score)
# print("\n")

# # Prédire les étiquettes pour les données de test
# prediction = lr.predict(X_evaluation)
# print(prediction)
# print("\n")

## OU

# # arbre de decision
# arbre = tree.DecisionTreeRegressor()
# arbre = arbre.fit(X_entrainement, y_train)
# prediction = arbre.predict(X_evaluation)
# tree.plot_tree(arbre)
# plt.show()

## OU

reseau = MLPClassifier(hidden_layer_sizes=(8,8,8))
reseau = reseau.fit(X_entrainement, y_train)
prediction = reseau.predict(X_evaluation)


# Matrice de Confusion
cm_lr = confusion_matrix(y_test, prediction)
tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
print(cm_lr)
print("\n")
print(tn, fp, fn, tp)
print("\n")

# Taux vrais positifs et vrais négatifs
tpr_lr = round(tp/(tp + fn), 4)
tnr_lr = round(tn/(tn + fp), 4)
tpp_lr = round(tp/(tp + fp), 4)
print(tpr_lr, tnr_lr, tpp_lr)
print("\n")

print(accuracy_score(y_test, prediction)) # performence
print(precision_score(y_test, prediction))
print(recall_score(y_test, prediction))
print(f1_score(y_test, prediction))
