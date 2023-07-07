import pandas as pd
from textblob import TextBlob


#charger le dataset
df = pd.read_csv('data/hotel.csv')
print(df.head())

#supprimer les doublons
df.drop_duplicates(subset ='Review', keep = 'first', inplace = True)

#caster la colonne en tring
df['Review'] = df['Review'].astype('str')

#methode pour retouner la polarite
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

#methode pour retouner la polarite
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#Ajouter la colonne Polarite/subjectivite en appliquant la methode get_poraty a chaque ligne
df['polarity'] = df['Review'].apply(get_polarity)
df['subjectivity'] = df['Review'].apply(get_subjectivity)

print(df.head())
print("\n")
print(df.sample(10))
