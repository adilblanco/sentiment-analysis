# # # # # # # #
# Basé régles #
# # # # # # # #

import nltk
from textblob import TextBlob

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


print("\n* * * * * * * * * * * * * * * * * *\n")
# text = 'I had an awesome day'
# text = 'I had an bad day'
# text = 'I had an horrible day'
text = 'I had not a good day'

# créer un objet textblob
blob_text = TextBlob(text)
print(f"text:\t\t{blob_text}")

# afficher les mots avec les tags
tags = blob_text.tags
print(f"tags:\t\t{tags}")

# afficher la subjectivite
sentiment = blob_text.sentiment
print(f"subjectivite:\t{sentiment}")

# afficher la polaite
# Polarité : à quel point un mot est positif ou négatif. -1 est très négatif. +1 est très positif.
polarite = sentiment.polarity
print(f"polarite:\t{polarite}")

# afficher la Subjectivité
# Subjectivité : À quel point un mot est subjectif ou opiniâtre. 0 est un fait. +1 est beaucoup d'opinion.
subjectivite = sentiment.subjectivity
print(f"subjectivite:\t{subjectivite}")
