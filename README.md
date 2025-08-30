# TikTokTechJam2025
---
# Preprocess and Clean Data
---
# Feature Engineering
---

The feature engineering part made use of HuggingFace's library for sentimental analysis. 
Upon evaluation, the program appends the following columns to the dataframe:
- sentiment_label: POSITIVE, NEUTRAL or NEGATIVE based on the sentimental analysis result using "cardiffnlp/twitter-roberta-base-sentiment"
- sentiment_score: The score associated with the label
- expected_sentiment: Expected reviewer's sentiment based off their rating (1-2: negative, 3: neutral, 4-5: positive)
- sentiment_match: Checks if there is a match between sentimental_label and expected_sentiment
- detailed_match: Checks the confidence level on the match/mismatch based off sentiment_score

By evaluating the review's sentiments and cross-referencing it to the ratings, it helps to even out the subjective nature of rating (e.g. some reviewers might rate more harshly despite good reviews).

It also helps to detects ratings are that are intentionally misleading (e.g. providing a good review but giving a bad rating).

---
# LLM Prompt Engineering 
---
# Rule-based Policy Checks
---
# ML Model Training
---
