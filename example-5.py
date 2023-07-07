import pandas as pd
from textblob import TextBlob


#charger le dataset
df = pd.read_csv('data/trumptweets.csv')

#supprimer les doublons
df.drop_duplicates(subset ='content', keep = 'first', inplace = True)

#caster la colonne en string
df['content'] = df['content'].astype('str')

#methode pour retouner la polarite
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

#methode pour retouner la polarite
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


#Ajouter la colonne Polarite/subjectivite en appliquant la methode get_poraty a chaque ligne
df['polarity'] = df['content'].apply(get_polarity)
df['subjectivity'] = df['content'].apply(get_subjectivity)

print(f"negative:\t{len(df[df['polarity'] < 0])}")
print(f"positive:\t{len(df[df['polarity'] > 0])}")
print(f"neutre:\t\t{len(df[df['polarity'] == 0])}")
