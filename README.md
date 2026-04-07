# Sentiment Intelligence Engine
### Real-Time AI/ML Mini Project — Python + Flask

---

## Overview

A fully functional, real-time sentiment analysis web application built
entirely in Python from scratch — no external ML libraries (PyTorch,
TensorFlow, or scikit-learn) are required.

The project demonstrates core concepts in Natural Language Processing (NLP)
and Machine Learning through a hand-crafted pipeline that includes:

- Lexicon-based sentiment scoring
- Negation handling and context windows
- Intensifier boosting (very, extremely, incredibly, etc.)
- Multi-class emotion profiling (joy, trust, fear, anger, sadness, etc.)
- Subjectivity vs. objectivity estimation
- Sentence-level breakdown and per-sentence scoring
- Flesch-Kincaid reading grade estimation
- Session-level analytics and trend visualization

---

## Tech Stack

| Layer       | Technology               |
|-------------|--------------------------|
| Backend     | Python 3.12, Flask       |
| NLP Engine  | Custom (built from scratch) |
| Frontend    | HTML, CSS, Vanilla JS    |
| Deployment  | Any machine with Python  |

---

## How to Run

### Step 1 — Install dependency

```bash
pip install flask
```

### Step 2 — Start the server

```bash
python app.py
```

### Step 3 — Open in browser

```
http://localhost:5000
```

---

## API Reference

### POST /api/analyze

Analyse any text and receive a full sentiment report.

**Request body:**
```json
{ "text": "This product is absolutely fantastic!" }
```

**Response:**
```json
{
  "label": "Positive",
  "compound": 0.8571,
  "confidence": 92.8,
  "pos_score": 1.8,
  "neg_score": 0.0,
  "subjectivity": 74.3,
  "objectivity": 25.7,
  "emotions": { "joy": 0.63, "anticipation": 0.37 },
  "key_pos_words": ["fantastic"],
  "key_neg_words": [],
  "sentence_breakdown": [...],
  "word_count": 6,
  "sentence_count": 1,
  "reading_grade": 7.2,
  "processing_ms": 0.42
}
```

### GET /api/demo

Returns a random pre-written sample text with its analysis.

### GET /api/history

Returns session-level analytics — counts of positive, negative, and
neutral analyses made so far, plus a trend array.

---

## Project Concepts Demonstrated

1. **NLP Pipeline Design** — tokenization, negation windows, intensifier detection
2. **Lexicon-based ML** — weighted word scoring, polarity scoring
3. **Feature Engineering** — compound score, subjectivity ratio, emotion vectors
4. **Multi-class Classification** — Positive / Negative / Neutral with confidence
5. **Data Aggregation** — real-time session statistics and trend tracking
6. **REST API Design** — clean JSON endpoints with Flask
7. **Single-page Application** — dynamic frontend without any framework

---

## File Structure

```
sentiment_engine/
    app.py           Main application (server + ML engine + frontend)
    requirements.txt Python dependencies
    README.md        This file
```

---

## Author Note

This project was built to demonstrate a complete AI/ML pipeline using only
the Python standard library and Flask. The NLP engine inside app.py contains
over 300 curated lexicon entries, a negation handling system, and an emotion
profiling layer — all implemented as clean, readable Python classes.
