from textblob import TextBlob

text = ("Sentiment analysis (also known as opinion mining or emotion AI) is the use of natural language processing, "
        "text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, "
        "and study affective states and subjective information. Sentiment analysis is widely applied "
        "to voice of the customer materials such as reviews and survey responses, online and social "
        "media, and healthcare materials for applications that range from marketing "
        "to customer service to clinical medicine. ")

# créer un objet textblob
blob_text = TextBlob(text)
print(f"text:\t{blob_text}")

# afficher les mots avec les tags
tags = blob_text.tags
print(f"tags:\t{tags}")

# afficher la subjectivite
sentiment = blob_text.sentiment
print(f"sentiment:\t{sentiment}")

# afficher la polaite
# Polarité : à quel point un mot est positif ou négatif. -1 est très négatif. +1 est très positif.
polarite = sentiment.polarity
print(f"polarite:\t{polarite}")

# afficher la Subjectivité
# Subjectivité : À quel point un mot est subjectif ou opiniâtre. 0 est un fait. +1 est beaucoup d'opinion.
subjectivite = sentiment.subjectivity
print(f"subjectivite:\t{subjectivite}")