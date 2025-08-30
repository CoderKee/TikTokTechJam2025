# TikTokTechJam2025

---

## Preprocess and Clean Data

The preprocessing step ingests a concatenation of JSON objects, keeps only review-like objects (must include `user_id` & `gmap_id` and at least `text` or `rating`), and standardizes fields for modeling. Picture-only and response-only fragments are excluded.  

The pipeline appends the following columns to the dataframe:

- `text_raw` – Original review text (kept for audit/demos).  
- `text_clean` – Lightly normalized text with PII tokens redacted to `<URL>`, `<EMAIL>`, `<PHONE>`, `<HANDLE>` (regex used only for redaction). Preprocessing pipeline applied, as discussed below.
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

## Rule-Based Policy Checks

Before training our ML model, we implemented a **rule-based system** to preliminarily classify reviews into categories. This system serves as both a filtering mechanism and a way to generate labeled samples for training.

### Purpose
- Quickly identify and label reviews containing obvious **spam or off-topic content**.  
- Create a **balanced dataset** for model training by sampling equal numbers from each category.  
- Provide **mutually-exclusive labels** to help guide the ML model.

### Categories and Rules

1. **Advertisement**  
   - Detects promotional content such as URLs, competitor mentions, sales, discounts, subscriptions, or calls-to-action (e.g., “buy now”, “visit our page”).  

2. **Irrelevant Content**  
   - Detects off-topic text related to politics, jobs, education, personal life, entertainment, or unrelated products.  

3. **Rant Without Visit**  
   - Detects reviews from people who have **not actually visited** the place, including phrases like “never been”, “heard about it”, “just passing by”, or “planning to visit”.  

4. **None**  
   - Reviews that **do not match any of the above rules** and are likely genuine.

### Methodology
- Each review is checked against all rules.  
- **Priority assignment**: RantWithoutVisit > Advertisement > Irrelevant > None.  
- Samples are drawn to create a **balanced dataset** with roughly equal numbers of reviews from each category (up to 3,500 per class).  

### Outcome
- Produces a **high-quality labeled dataset** for model training.  
- Helps the ML model focus on subtle patterns beyond obvious rule-based signals.  
- Ensures the final training set is **balanced**, reducing bias toward the majority “normal” reviews.

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

## Data Preprocessing

- Text converted to lowercase
- Lemmatization (grouping together the inflected forms of a word)
- Punctuation removal
- Stop word removal (little semantic value)

---

## ML Model Training

After generating labeled data through both rule-based checks and LLM-assisted labeling, we trained a **DeBERTa-based sequence classification model** to automatically classify reviews into four categories: Advertisement, Irrelevant Content, Rant without visiting, and None.

### Dataset Preparation
- Used the labeled dataset (`GPT-labels`) with **text_clean** as input.  
- Converted labels into integers for modeling.  
- Split data into **train (90%)** and **test (10%)** sets, maintaining class balance.  
- Applied **oversampling** to underrepresented classes to handle data imbalance and improve model learning. This was done with inverse class frequency.

### Tokenization
- Tokenized review text using the **DeBERTa tokenizer**, truncating and padding sequences to a fixed length of 256 tokens.  

### Model Architecture
- Fine-tuned **`microsoft/deberta-v3-base`** for sequence classification with 4 output labels.  
- Implemented and compared **custom loss functions**:
  - **Focal Loss** to handle class imbalance and emphasize difficult-to-classify examples.
  - **Class-weighted Cross-Entropy** for class balancing with CE loss.
- Used `attention_probs_dropout_prob = 0.2` for regularisation.

### Training Setup
- **Batch size:** 32 
- **Learning rate:** 2e-5  
- **Epochs:** 5
  - Used early stopping at 350 steps based on assessment of macro F1-score.
  - Note that "epochs" is a misnomer after minority class oversampling.
- **Weight decay:** 0.01
- **Warmup_ratio:** 0.01
   - 52 steps in our implementation, to give adaptive optimizer better statistics.
- **Evaluation:** Monitored metrics and saved every 25 steps, including accuracy, macro F1-score, and per-class precision/recall/F1.  

### Metrics
- Trained model was evaluated using:
  - **Accuracy** and **macro F1-score** for overall performance.  
  - **Per-class precision, recall, and F1-score** to ensure all categories were learned effectively.  

### Outcome
- The model learns LLM-provided (GPT-3.5-turbo) labels from subtle textual cues to classify reviews automatically.
- Balancing techniques and custom loss functions help mitigate bias toward the majority class.  
- Final model serves as the core of **Toktok**, enabling automatic detection and filtering of unhelpful or irrelevant reviews across large datasets.

