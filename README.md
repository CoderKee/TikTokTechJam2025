# TikTokTechJam2025

---

## Preprocess and Clean Data

The preprocessing step ingests a concatenation of JSON objects, keeps only review-like objects (must include `user_id` & `gmap_id` and at least `text` or `rating`), and standardizes fields for modeling. Picture-only and response-only fragments are excluded.  

The pipeline appends the following columns to the dataframe:

- `text_raw` – Original review text (kept for audit/demos).  
- `text_clean` – Lightly normalized text with PII tokens redacted to `<URL>`, `<EMAIL>`, `<PHONE>`, `<HANDLE>` (regex used only for redaction).  
- `time_dt_utc` – Review timestamp converted from ms to UTC datetime.  
- `review_year` / `review_month` / `review_dow` / `review_hour` – Calendar features derived from `time_dt_utc`.  
- `resp_text` – Business response text (if any).  
- `resp_time_ms` / `resp_time_dt_utc` – Response time in ms and UTC datetime.  
- `resp_delay_hours` – Hours between review and business response (-1 if no response).  
- `n_pics` / `has_images` – Count of photo URLs and a presence flag.  
- `len_chars` / `len_words` – Basic length signals.  
- `digit_ratio` / `punct_ratio` / `exclam_count` – Indicators of low-quality or spammy content.  
- `has_resp` – Whether a business response exists (0/1).  

Additional hygiene steps:

- Drop rows with empty `text_clean`.  
- De-duplicate exact repeated texts per place (`gmap_id`) via a normalized key.  
- Cast core types (`user_id`/`name`/`gmap_id` → string, `rating` → nullable int).  

This produces a **consistent, privacy-aware table** that’s easy to explore, label with an LLM, and train models on—while removing obvious noise and preserving useful metadata (timing, images, responses).

---

## Rule-based Policy Checks

*(Add your description of the rule-based policy checks here. You can mention keyword filtering, preliminary categorization, etc.)*

---

## Feature Engineering

We used **Hugging Face Transformers** for sentiment analysis. The program appends the following columns to the dataframe:

- `sentiment_label` – POSITIVE, NEUTRAL, or NEGATIVE based on the "cardiffnlp/twitter-roberta-base-sentiment" model.  
- `sentiment_score` – Confidence score associated with the label.  
- `expected_sentiment` – Expected reviewer sentiment based on rating (1-2: negative, 3: neutral, 4-5: positive).  
- `sentiment_match` – Checks if `sentiment_label` matches `expected_sentiment`.  
- `detailed_match` – Checks confidence level of the match/mismatch using `sentiment_score`.  

**Purpose:**  
- Helps account for the subjective nature of ratings (e.g., harsh ratings despite positive text).  
- Detects intentionally misleading reviews (e.g., positive text with a negative rating).

---

## LLM Prompt Engineering

**Review Classification with DeepSeek API**  

This component automatically labels Google Reviews using DeepSeek API:

**Categories:**

- **Advertisement** – Promotes another business (not the reviewed one).  
- **Irrelevant Content** – Off-topic (politics, personal stories).  
- **Rant without visiting** – Angry criticism without actually visiting the place.  
- **None** – Legitimate review based on actual experience.  

**Overview:**

- Reviews are processed in **batches** for efficiency.  
- Custom prompts ensure consistent classification according to defined rules.  
- **Parallel processing** allows multiple batches to be classified simultaneously.  
- Results are stored in the `df_clean` DataFrame under the column `GPT-label`.

**Key Points:**

- Uses **Hugging Face / DeepSeek models** for NLP-based classification.  
- Combines **rule-based guidance** with AI-powered predictions.  
- Designed to **filter low-quality or irrelevant reviews** automatically, improving downstream analysis and moderation.

---

## ML Model Training

*(Add your description of the model training process here. Include info about model architecture, dataset, evaluation metrics like precision/recall/F1, and fine-tuning.)*

