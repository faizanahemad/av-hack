# Innoplexus-Online-Hiring-Hackathon-Sentiment-Analysis

![image](https://user-images.githubusercontent.com/37707687/62003677-4119b700-b138-11e9-89ad-60725dc3f6f8.png)

## About the Solution
- Used 1D CNN
- Multiple values of Thresholds for improving F1, (no improvement).

### Areas of Improvement
- Sentence Shuffle
- Hand held preprocessing like [this](https://github.com/pawangeek/Ccmps/blob/master/innoplexus/preprocessing(part1).ipynb)
- Use only words which are in both train and test set. Works in a competition but not in real-world
- Use Fast.ai's pretrained Bert like [here](https://github.com/pawangeek/Ccmps/blob/master/innoplexus/bert%2Bfastai(model).ipynb)
- Word Contractions like 're->are, I'm -> I am et
- Mis-Spelling correction using edit distance and pre-existing dictionary of words

## About the competition

### Problem Statement
Sentiment Analysis for drugs/medicines Nowadays the narrative of a brand is not only built and controlled by the company that owns the brand. For this reason, companies are constantly looking out across Blogs, Forums, and other social media platforms, etc for checking the sentiment for their various products and also competitor products to learn how their brand resonates in the market. This kind of analysis helps them as part of their post-launch market research. This is relevant for a lot of industries including pharma and their drugs.

The challenge is that the language used in this type of content is not strictly grammatically correct. Some use sarcasm. Others cover several topics with different sentiments in one post. Other users post comments and reply and thereby indicating his/her sentiment around the topic.

Sentiment can be clubbed into 3 major buckets - Positive, Negative and Neutral Sentiments.

You are provided with data containing samples of text. This text can contain one or more drug mentions. Each row contains a unique combination of the text and the drug mention. Note that the same text can also have different sentiment for a different drug.

Given the text and drug name, the task is to predict the sentiment for texts contained in the test dataset. Given below is an example of text from the dataset:

#### Example:
Stelara is still fairly new to Crohn's treatment. This is why you might not get a lot of replies. I've done some research, but most of the "time to work" answers are from Psoriasis boards. For Psoriasis, it seems to be about 4-12 weeks to reach a strong therapeutic level. The good news is, Stelara seems to be getting rave reviews from Crohn's patients. It seems to be the best med to come along since Remicade. I hope you have good success with it. My daughter was diagnosed Feb. 19/07, (13 yrs. old at the time of diagnosis), with Crohn's of the Terminal Illium. Has used Prednisone and Pentasa. Started Imuran (02/09), had an abdominal abscess (12/08). 2cm of Stricture. Started ​Remicade in Feb. 2014, along with 100mgs. of Imuran.

For Stelara the above text is ​positive​ while for Remicade the above text is ​negative​.

### Data Description
train.csv
Contains the labelled texts with sentiment values for a given drug

test.csv
test.csv contains texts with drug names for which the participants are expected to predict the correct sentiment

sample_submission.csv
sample_submission.csv contains the submission format for the predictions against the test set. NA single csv needs to be submitted as a solution. The submission file must contain only 2 columns unique_hash, sentiment

### Evaluation Metric
The metric used for evaluating the performance of the classification model would be macro F1-Score.

### Public and Private Split
The texts in the test data are further randomly divided into Public (40%) and Private (60%) data. Your initial responses will be checked and scored on the Public data. The final rankings would be based on your private score which will be published once the competition is over.

Private Leaderboard: 70 (Score: 0.5017273731)

## References
- [Model Stacking](https://www.kaggle.com/general/18793)
- [Rank 14 Solution with Fast.ai Pretrained Bert and data specific text processing](https://github.com/pawangeek/Ccmps/tree/master/innoplexus)
- [Better Solution with Text Augment](https://github.com/rajat5ranjan/AV-Innoplexus-Online-Hiring-Hackathon-Sentiment-Analysis)
- [Rank-27: Word Contractions](https://github.com/Laxminarayen/Innoplex_Hackathon/blob/master/Sentiment%20Classification%20ML%20%2B%20Keras%20Functional.ipynb)
- [Another User Solution](https://github.com/chetanambi/Innoplexus-Online-Hiring-Hackathon-Sentiment-Analysis/blob/master/Sentiment%20Analysis_Final%20Solution_0.5230949840.ipynb)
- [Misspellings and Contractions](https://github.com/nursnaaz/AV-Innoplexus)
- [Deep and Wide LSTM+GRU with Spell correct](https://github.com/anandthirwani/Innoplexus-Online-Hiring-Hackathon-Sentiment-Analysis)
- [Soln 2: DNN](https://github.com/shravankoninti/AV/blob/master/Innoplexus_25_July_2019/AV_Innoplex_25072019ipynb.ipynb)
- [Spacy and word Cloud Soln](https://github.com/saroj1994/Innoplexus-Online-Hiring-Hackathon-Sentiment-Analysis/blob/master/innoplexus_hackathon_submission_code.ipynb)


