from textblob import TextBlob

feedbacks = ['I love the app is amazing ',
             "The experience was bad as hell",
             "This app is really helpful",
             "Damn the app tastes like shit ",
             'Please don\'t download the app you will regret it ']

feedbacks_positifs = []
feedbacks_negatifs = []

for feedback in feedbacks:
    polarite_feedback = TextBlob(feedback).sentiment.polarity
    if polarite_feedback > 0:
        feedbacks_positifs.append(feedback)
        continue
    feedbacks_negatifs.append(feedback)

print('Nombre de feebacks positifs : {}'.format(len(feedbacks_positifs)))
print(feedbacks_positifs)
print('Nombre de feebacks negatifs : {}'.format(len(feedbacks_negatifs)))
print(feedbacks_negatifs)
