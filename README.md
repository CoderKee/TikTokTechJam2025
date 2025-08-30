# TikTokTechJam2025
---
# Preprocess and Clean Data

The preprocessing step ingests a concatenation of JSON objects, keeps only review-like objects (must include user_id & gmap_id and at least text or rating), and standardizes fields for modeling. Picture-only and response-only fragments are excluded.

The pipeline appends the following columns to the dataframe:

- text_raw – Original review text (kept for audit/demos).

- text_clean – Lightly normalized text with PII tokens redacted to <URL>, <EMAIL>, <PHONE>, <HANDLE> (regex used only for redaction, not policy classification).

- time_dt_utc – Review timestamp converted from ms to UTC datetime.

- review_year / review_month / review_dow / review_hour – Calendar features derived from time_dt_utc.

- resp_text – Business response text (if any).

- resp_time_ms / resp_time_dt_utc – Response time in ms and UTC datetime.

- resp_delay_hours – Hours between review and business response (-1 if no response).

- n_pics / has_images – Count of photo URLs and a presence flag.

- len_chars / len_words – Basic length signals.

- digit_ratio / punct_ratio / exclam_count – Cheap quality/spam cues.

- has_resp – Whether a business response exists (0/1).

Additional hygiene:

- Drop rows with empty text_clean.

- De-duplicate exact repeated texts per place (gmap_id) via a normalized key.

- Cast core types (user_id/name/gmap_id→string, rating→nullable int).

This helps to produce a consistent, privacy-aware table that’s easy to explore, label with an LLM, and train models on—while removing obvious noise and preserving useful metadata (timing, images, responses).
---
# Rule-based Policy Checks
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

Review Classification with DeepSeek API
This part of the project focuses on automatically labeling Google Reviews using the DeepSeek API and Python. Reviews are classified into four categories:
Advertisement – Promotes another business (not the reviewed one).
Irrelevant Content – Off-topic, such as politics or personal stories.
Rant without visiting – Angry criticism without actually visiting the place.
None – Legitimate review based on an actual experience.
Overview
Reviews are processed in batches for efficiency.
The system uses custom prompts to ensure consistent classification according to defined rules.
Parallel processing allows multiple batches to be classified simultaneously.
Results are stored in the df_clean DataFrame under the column GPT-label.
Key Points
Uses Hugging Face / DeepSeek models for NLP-based classification.
Combines rule-based guidance with AI-powered predictions.
Designed to filter low-quality or irrelevant reviews automatically, improving downstream analysis or moderation.
---
# ML Model Training
---
