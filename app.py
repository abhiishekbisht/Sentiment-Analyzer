"""
Sentiment Intelligence Engine
------------------------------
A real-time AI/ML web application that performs multi-dimensional
sentiment analysis using a hand-crafted NLP pipeline built from scratch.

Author  : College AI/ML Mini Project
Stack   : Python 3, Flask, Custom NLP Engine (no external ML libs needed)
Run     : python app.py  ->  open http://localhost:5000
"""

import math
import re
import json
import time
import random
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ---------------------------------------------------------------------------
# LEXICON  —  hand-crafted, research-inspired sentiment word database
# ---------------------------------------------------------------------------

POSITIVE_LEXICON = {
    # Strong positives (weight 1.0)
    "excellent": 1.0, "outstanding": 1.0, "brilliant": 1.0, "superb": 1.0,
    "fantastic": 1.0, "magnificent": 1.0, "phenomenal": 1.0, "extraordinary": 1.0,
    "exceptional": 1.0, "wonderful": 1.0, "amazing": 1.0, "incredible": 1.0,
    "perfect": 1.0, "flawless": 1.0, "spectacular": 1.0, "splendid": 1.0,
    # Medium positives (weight 0.7)
    "good": 0.7, "great": 0.7, "nice": 0.7, "positive": 0.7, "happy": 0.7,
    "pleased": 0.7, "satisfied": 0.7, "impressed": 0.7, "delighted": 0.7,
    "love": 0.7, "loved": 0.7, "enjoy": 0.7, "enjoyed": 0.7, "like": 0.7,
    "liked": 0.7, "appreciate": 0.7, "appreciated": 0.7, "glad": 0.7,
    "helpful": 0.7, "useful": 0.7, "efficient": 0.7, "effective": 0.7,
    "reliable": 0.7, "trustworthy": 0.7, "honest": 0.7, "genuine": 0.7,
    "clean": 0.7, "smooth": 0.7, "fast": 0.7, "quick": 0.7, "responsive": 0.7,
    # Light positives (weight 0.4)
    "okay": 0.4, "ok": 0.4, "fine": 0.4, "decent": 0.4, "alright": 0.4,
    "acceptable": 0.4, "reasonable": 0.4, "adequate": 0.4, "fair": 0.4,
    "interesting": 0.4, "informative": 0.4, "clear": 0.4, "simple": 0.4,
    "fun": 0.4, "cute": 0.4, "cool": 0.4, "solid": 0.4, "standard": 0.4,
    # Emotions
    "joy": 0.8, "joyful": 0.8, "excited": 0.8, "exciting": 0.8, "thrilled": 0.9,
    "grateful": 0.8, "thankful": 0.8, "inspired": 0.8, "motivated": 0.7,
    "confident": 0.7, "proud": 0.8, "peaceful": 0.6, "calm": 0.5, "safe": 0.6,
    "hopeful": 0.7, "optimistic": 0.7, "cheerful": 0.8, "warm": 0.6,
}

NEGATIVE_LEXICON = {
    # Strong negatives (weight 1.0)
    "terrible": 1.0, "awful": 1.0, "horrible": 1.0, "dreadful": 1.0,
    "atrocious": 1.0, "abysmal": 1.0, "disastrous": 1.0, "catastrophic": 1.0,
    "pathetic": 1.0, "disgusting": 1.0, "revolting": 1.0, "appalling": 1.0,
    "worst": 1.0, "unacceptable": 1.0, "inexcusable": 1.0, "outrageous": 1.0,
    # Medium negatives (weight 0.7)
    "bad": 0.7, "poor": 0.7, "wrong": 0.7, "negative": 0.7, "unhappy": 0.7,
    "disappointed": 0.7, "disappointing": 0.7, "frustrated": 0.7, "frustrating": 0.7,
    "hate": 0.7, "hated": 0.7, "dislike": 0.7, "disliked": 0.7, "annoyed": 0.7,
    "annoying": 0.7, "upset": 0.7, "angry": 0.7, "broken": 0.7, "failed": 0.7,
    "failure": 0.7, "useless": 0.7, "worthless": 0.7, "waste": 0.7, "slow": 0.7,
    "buggy": 0.7, "crash": 0.7, "crashed": 0.7, "error": 0.6, "issue": 0.5,
    "problem": 0.5, "unreliable": 0.7, "misleading": 0.7, "confusing": 0.6,
    # Light negatives (weight 0.4)
    "mediocre": 0.4, "average": 0.3, "bland": 0.4, "boring": 0.4, "dull": 0.4,
    "lacking": 0.4, "missing": 0.4, "limited": 0.3, "basic": 0.3, "plain": 0.3,
    # Emotions
    "sad": 0.7, "sadness": 0.7, "fear": 0.8, "scared": 0.8, "anxious": 0.7,
    "worried": 0.6, "stressed": 0.7, "tired": 0.5, "exhausted": 0.7,
    "lonely": 0.7, "depressed": 0.9, "hopeless": 0.9, "miserable": 0.9,
    "jealous": 0.6, "envious": 0.6, "regret": 0.7, "regretful": 0.7,
}

INTENSIFIERS = {
    "very": 1.3, "extremely": 1.5, "incredibly": 1.4, "absolutely": 1.4,
    "totally": 1.2, "completely": 1.3, "utterly": 1.4, "deeply": 1.3,
    "highly": 1.2, "really": 1.2, "so": 1.1, "quite": 1.1, "rather": 1.1,
    "pretty": 1.1, "fairly": 1.0, "genuinely": 1.2, "truly": 1.2,
    "seriously": 1.3, "especially": 1.2, "particularly": 1.2,
}

NEGATORS = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "hardly", "barely", "scarcely", "without", "lack", "lacking",
    "cannot", "cant", "wont", "dont", "doesnt", "isnt", "wasnt",
    "arent", "werent", "hasnt", "havent", "shouldnt", "wouldnt", "couldnt",
}

EMOTION_CATEGORIES = {
    "joy":        ["happy", "joy", "joyful", "excited", "delight", "delighted", "wonderful",
                   "amazing", "fantastic", "love", "loved", "cheerful", "thrilled", "glad"],
    "trust":      ["reliable", "trustworthy", "honest", "genuine", "safe", "confident",
                   "secure", "stable", "consistent", "dependable", "accurate"],
    "fear":       ["scared", "fear", "afraid", "anxious", "worried", "nervous", "panic",
                   "terror", "horror", "dread", "frightened", "uneasy", "stress", "stressed"],
    "surprise":   ["surprised", "unexpected", "unbelievable", "shocking", "astonishing",
                   "remarkable", "incredible", "sudden", "extraordinary"],
    "sadness":    ["sad", "sadness", "unhappy", "depressed", "miserable", "lonely",
                   "disappointed", "heartbroken", "grief", "sorrow", "regret", "hopeless"],
    "disgust":    ["disgusting", "revolting", "horrible", "awful", "terrible", "gross",
                   "appalling", "atrocious", "unacceptable", "pathetic", "vile"],
    "anger":      ["angry", "frustrated", "annoyed", "outraged", "furious", "mad",
                   "irritated", "infuriated", "enraged", "hate", "despise"],
    "anticipation":["hopeful", "optimistic", "excited", "looking forward", "expecting",
                    "eager", "motivated", "inspired", "curious", "interested"],
}


# ---------------------------------------------------------------------------
# NLP ENGINE
# ---------------------------------------------------------------------------

class SentimentEngine:
    """
    A multi-dimensional NLP sentiment analysis engine built from scratch.
    Implements: lexicon scoring, negation handling, intensifier boosting,
    emotion profiling, subjectivity estimation, and confidence scoring.
    """

    def __init__(self):
        self.pos_lexicon = POSITIVE_LEXICON
        self.neg_lexicon = NEGATIVE_LEXICON
        self.intensifiers = INTENSIFIERS
        self.negators = NEGATORS
        self.emotion_map = EMOTION_CATEGORIES
        self.history = []          # stores past analyses for session analytics

    # --- tokenizer -----------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        return [t for t in text.split() if t]

    # --- sentence splitter ---------------------------------------------------

    def split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    # --- core lexicon scorer with negation window ----------------------------

    def score_tokens(self, tokens: list[str]) -> dict:
        pos_score = 0.0
        neg_score = 0.0
        pos_words = []
        neg_words = []
        negation_window = 0
        intensifier_boost = 1.0

        for i, token in enumerate(tokens):
            # track negation (flips polarity for next 3 words)
            if token in self.negators:
                negation_window = 3
                continue

            # track intensifiers
            if token in self.intensifiers:
                intensifier_boost = self.intensifiers[token]
                continue

            # score word
            if token in self.pos_lexicon:
                weight = self.pos_lexicon[token] * intensifier_boost
                if negation_window > 0:
                    neg_score += weight * 0.8
                    neg_words.append(token)
                else:
                    pos_score += weight
                    pos_words.append(token)

            elif token in self.neg_lexicon:
                weight = self.neg_lexicon[token] * intensifier_boost
                if negation_window > 0:
                    pos_score += weight * 0.5
                    pos_words.append(token)
                else:
                    neg_score += weight
                    neg_words.append(token)

            # decay windows
            if negation_window > 0:
                negation_window -= 1
            intensifier_boost = 1.0

        return {
            "pos_score": pos_score,
            "neg_score": neg_score,
            "pos_words": pos_words,
            "neg_words": neg_words,
        }

    # --- subjectivity estimator ----------------------------------------------

    def estimate_subjectivity(self, tokens: list[str]) -> float:
        """Returns 0.0 (objective) to 1.0 (highly subjective)"""
        opinion_words = set(self.pos_lexicon) | set(self.neg_lexicon)
        opinion_count = sum(1 for t in tokens if t in opinion_words)
        if not tokens:
            return 0.0
        raw = opinion_count / len(tokens)
        return round(min(raw * 3.5, 1.0), 3)   # scaled to feel natural

    # --- emotion profiler ----------------------------------------------------

    def profile_emotions(self, tokens: list[str]) -> dict:
        token_set = set(tokens)
        scores = {}
        for emotion, words in self.emotion_map.items():
            hits = sum(1 for w in words if w in token_set)
            if hits:
                scores[emotion] = round(hits / len(words), 4)
        # normalize
        total = sum(scores.values())
        if total:
            scores = {k: round(v / total, 4) for k, v in scores.items()}
        return scores

    # --- sentence-level analysis for multi-sentence texts --------------------

    def analyze_sentences(self, text: str) -> list[dict]:
        sentences = self.split_sentences(text)
        results = []
        for s in sentences:
            tokens = self.tokenize(s)
            scored = self.score_tokens(tokens)
            p, n = scored["pos_score"], scored["neg_score"]
            total = p + n
            if total == 0:
                label = "neutral"
                compound = 0.0
            elif p > n:
                label = "positive"
                compound = round((p - n) / (total + 1e-6), 3)
            else:
                label = "negative"
                compound = round((p - n) / (total + 1e-6), 3)
            results.append({"sentence": s, "label": label, "compound": compound})
        return results

    # --- main analyze method -------------------------------------------------

    def analyze(self, text: str) -> dict:
        if not text or not text.strip():
            return {"error": "Empty input"}

        start = time.time()
        tokens = self.tokenize(text)

        if len(tokens) < 2:
            return {"error": "Please enter at least a few words for meaningful analysis."}

        scored = self.score_tokens(tokens)
        pos_score = scored["pos_score"]
        neg_score = scored["neg_score"]
        total_score = pos_score + neg_score

        # compound score  [-1, +1]
        if total_score == 0:
            compound = 0.0
        else:
            compound = (pos_score - neg_score) / (total_score + 0.001)
            compound = max(-1.0, min(1.0, compound))

        # label + confidence
        if compound >= 0.05:
            label = "Positive"
            confidence = round(0.5 + compound * 0.5, 4)
        elif compound <= -0.05:
            label = "Negative"
            confidence = round(0.5 + abs(compound) * 0.5, 4)
        else:
            label = "Neutral"
            confidence = round(1.0 - abs(compound) * 2, 4)

        subjectivity = self.estimate_subjectivity(tokens)
        emotions = self.profile_emotions(tokens)
        sentence_analysis = self.analyze_sentences(text)

        # reading level estimate (Flesch–Kincaid grade approximation)
        word_count = len(tokens)
        sentence_count = max(len(self.split_sentences(text)), 1)
        avg_word_len = sum(len(t) for t in tokens) / max(word_count, 1)
        fk_grade = round(0.39 * (word_count / sentence_count) + 11.8 * (avg_word_len / 3) - 15.59, 1)
        fk_grade = max(1, min(fk_grade, 18))

        # token stats
        word_freq = Counter(tokens)
        top_words = [{"word": w, "count": c} for w, c in word_freq.most_common(8)
                     if w not in {"the", "a", "an", "is", "it", "in", "of", "to", "and",
                                  "that", "this", "was", "for", "on", "are", "with", "as"}][:6]

        elapsed = round((time.time() - start) * 1000, 2)

        result = {
            "label": label,
            "compound": round(compound, 4),
            "confidence": round(confidence * 100, 1),
            "pos_score": round(pos_score, 3),
            "neg_score": round(neg_score, 3),
            "subjectivity": round(subjectivity * 100, 1),
            "objectivity": round((1 - subjectivity) * 100, 1),
            "emotions": emotions,
            "key_pos_words": list(set(scored["pos_words"]))[:6],
            "key_neg_words": list(set(scored["neg_words"]))[:6],
            "sentence_breakdown": sentence_analysis,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "top_words": top_words,
            "reading_grade": fk_grade,
            "processing_ms": elapsed,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

        # store in history
        self.history.append({
            "text_preview": text[:60] + ("..." if len(text) > 60 else ""),
            "label": label,
            "compound": result["compound"],
            "timestamp": result["timestamp"],
        })
        if len(self.history) > 20:
            self.history.pop(0)

        return result

    def get_history_stats(self) -> dict:
        if not self.history:
            return {"total": 0, "positive": 0, "negative": 0, "neutral": 0, "trend": []}
        counts = Counter(h["label"] for h in self.history)
        trend = [{"label": h["label"], "compound": h["compound"], "time": h["timestamp"]}
                 for h in self.history[-10:]]
        return {
            "total": len(self.history),
            "positive": counts.get("Positive", 0),
            "negative": counts.get("Negative", 0),
            "neutral": counts.get("Neutral", 0),
            "trend": trend,
        }


engine = SentimentEngine()


# ---------------------------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    result = engine.analyze(text)
    return jsonify(result)


@app.route("/api/history")
def history():
    return jsonify(engine.get_history_stats())


@app.route("/api/demo")
def demo():
    samples = [
        "This product is absolutely fantastic! I love every single feature, it works perfectly.",
        "The service was terrible. I waited for hours and nothing worked. Completely unacceptable.",
        "The package arrived on time. The product is what was described in the listing.",
        "I am incredibly disappointed. The quality is poor and the support team was no help at all.",
        "Really enjoying this! The interface is clean, fast, and the experience is genuinely delightful.",
        "Not the worst thing I have used, but it is certainly lacking in several key areas.",
    ]
    text = random.choice(samples)
    result = engine.analyze(text)
    result["demo_text"] = text
    return jsonify(result)


# ---------------------------------------------------------------------------
# HTML TEMPLATE  —  full single-page app served by Flask
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentiment Intelligence Engine</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Manrope:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d0f14;
    --surface: #13161d;
    --surface2: #1a1e28;
    --border: rgba(255,255,255,0.07);
    --accent: #7cffa0;
    --accent2: #ff6b6b;
    --accent3: #ffd97d;
    --accent4: #7eb8ff;
    --text: #e8eaf0;
    --muted: #6b7280;
    --radius: 12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Manrope', sans-serif;
    font-size: 15px;
    line-height: 1.6;
    min-height: 100vh;
  }
  /* --- layout --- */
  .shell { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
  header {
    display: flex; align-items: flex-end; justify-content: space-between;
    padding-bottom: 2rem; border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
  }
  .logo-block .eyebrow {
    font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--accent); font-family: 'DM Mono', monospace; font-weight: 500;
  }
  .logo-block h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.6rem, 3.5vw, 2.4rem);
    font-weight: 400; color: var(--text); line-height: 1.1;
  }
  .logo-block h1 em { font-style: italic; color: var(--accent); }
  .version-tag {
    font-size: 11px; font-family: 'DM Mono', monospace;
    color: var(--muted); background: var(--surface2);
    padding: 4px 10px; border-radius: 20px; border: 1px solid var(--border);
  }
  /* --- grid --- */
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .col-full { grid-column: 1 / -1; }
  /* --- cards --- */
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.5rem;
  }
  .card-label {
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted); font-family: 'DM Mono', monospace; margin-bottom: 1rem;
  }
  /* --- input area --- */
  .input-wrap { position: relative; }
  textarea {
    width: 100%; background: var(--surface2); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem; font-family: 'Manrope', sans-serif; font-size: 15px;
    resize: vertical; min-height: 130px; outline: none;
    transition: border-color 0.2s;
  }
  textarea:focus { border-color: rgba(124,255,160,0.3); }
  textarea::placeholder { color: var(--muted); }
  .btn-row { display: flex; gap: 0.75rem; margin-top: 0.85rem; flex-wrap: wrap; }
  button {
    cursor: pointer; font-family: 'Manrope', sans-serif;
    font-size: 13px; font-weight: 600; border: none;
    border-radius: 6px; padding: 9px 20px; transition: opacity 0.15s, transform 0.1s;
  }
  button:active { transform: scale(0.97); }
  .btn-primary { background: var(--accent); color: #0d0f14; }
  .btn-secondary { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
  .btn-demo { background: transparent; color: var(--accent4); border: 1px solid rgba(126,184,255,0.3); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  /* --- result badge --- */
  .verdict {
    display: flex; align-items: center; gap: 1rem;
    padding: 1.2rem 1.4rem; border-radius: 10px;
    background: var(--surface2); margin-bottom: 1.2rem;
    border: 1px solid var(--border);
  }
  .verdict-dot {
    width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0;
    box-shadow: 0 0 12px currentColor;
  }
  .dot-positive { background: var(--accent); color: var(--accent); }
  .dot-negative { background: var(--accent2); color: var(--accent2); }
  .dot-neutral  { background: var(--accent3); color: var(--accent3); }
  .verdict-label {
    font-family: 'DM Serif Display', serif; font-size: 1.5rem; line-height: 1;
  }
  .verdict-conf {
    margin-left: auto; font-family: 'DM Mono', monospace;
    font-size: 13px; color: var(--muted);
  }
  /* --- compound bar --- */
  .bar-track {
    height: 8px; background: var(--surface2);
    border-radius: 4px; overflow: hidden; margin: 0.5rem 0;
  }
  .bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.5s cubic-bezier(.22,1,.36,1), background 0.5s;
  }
  /* --- stat grid --- */
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.5rem; }
  .stat-box {
    background: var(--surface2); border-radius: 8px;
    padding: 0.75rem 1rem; border: 1px solid var(--border);
  }
  .stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
  .stat-val { font-family: 'DM Mono', monospace; font-size: 1.2rem; font-weight: 500; margin-top: 2px; }
  /* --- emotion bars --- */
  .emo-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.45rem 0; }
  .emo-name { width: 80px; font-size: 12px; color: var(--muted); flex-shrink: 0; text-transform: capitalize; }
  .emo-bar { flex: 1; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; }
  .emo-fill { height: 100%; border-radius: 3px; transition: width 0.6s cubic-bezier(.22,1,.36,1); }
  .emo-pct { width: 38px; text-align: right; font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); }
  /* --- keyword tags --- */
  .tag-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
  .tag {
    font-size: 12px; padding: 3px 10px; border-radius: 20px;
    font-family: 'DM Mono', monospace;
  }
  .tag-pos { background: rgba(124,255,160,0.1); color: var(--accent); border: 1px solid rgba(124,255,160,0.2); }
  .tag-neg { background: rgba(255,107,107,0.1); color: var(--accent2); border: 1px solid rgba(255,107,107,0.2); }
  /* --- sentence breakdown --- */
  .sent-item {
    padding: 0.6rem 0.9rem; border-radius: 6px; margin-bottom: 0.5rem;
    background: var(--surface2); border-left: 3px solid var(--border);
    font-size: 13px; line-height: 1.5;
  }
  .sent-item.pos { border-left-color: var(--accent); }
  .sent-item.neg { border-left-color: var(--accent2); }
  .sent-item.neu { border-left-color: var(--accent3); }
  .sent-meta { font-size: 11px; color: var(--muted); font-family: 'DM Mono', monospace; margin-top: 3px; }
  /* --- history trend --- */
  .trend-dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 2px;
    cursor: default;
  }
  /* --- history stats row --- */
  .hist-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem; }
  .hist-stat { font-family: 'DM Mono', monospace; font-size: 12px; }
  /* --- loading --- */
  .loading { display: none; align-items: center; gap: 0.5rem; color: var(--muted); font-size: 13px; }
  .loading.active { display: flex; }
  .spinner {
    width: 16px; height: 16px; border: 2px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  /* --- empty state --- */
  .empty { color: var(--muted); font-size: 13px; text-align: center; padding: 2rem 0; }
  /* --- word cloud grid --- */
  .wc-grid { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
  .wc-word {
    font-family: 'DM Mono', monospace; border-radius: 4px;
    background: var(--surface2); padding: 3px 8px;
    color: var(--muted); font-size: 12px;
  }
  /* --- meta footer --- */
  .meta-row {
    display: flex; gap: 1.5rem; flex-wrap: wrap;
    font-size: 11px; color: var(--muted); font-family: 'DM Mono', monospace;
    margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border);
  }
  /* --- responsive --- */
  @media (max-width: 680px) {
    .grid { grid-template-columns: 1fr; }
    .stat-grid { grid-template-columns: 1fr 1fr; }
  }
</style>
</head>
<body>
<div class="shell">

  <header>
    <div class="logo-block">
      <p class="eyebrow">AI / ML Project &mdash; Python NLP</p>
      <h1>Sentiment <em>Intelligence</em> Engine</h1>
    </div>
    <span class="version-tag">v1.0 &mdash; real-time</span>
  </header>

  <div class="grid">

    <!-- INPUT PANEL -->
    <div class="card col-full">
      <p class="card-label">Input &mdash; text to analyse</p>
      <div class="input-wrap">
        <textarea id="textInput" placeholder="Type or paste any text here. Try a product review, a tweet, a news excerpt, or your own sentence..."></textarea>
      </div>
      <div class="btn-row">
        <button class="btn-primary" onclick="runAnalysis()">Analyse</button>
        <button class="btn-demo" onclick="loadDemo()">Load demo</button>
        <button class="btn-secondary" onclick="clearAll()">Clear</button>
        <div class="loading" id="loadingIndicator">
          <div class="spinner"></div> Processing...
        </div>
      </div>
    </div>

    <!-- VERDICT + SCORES -->
    <div class="card" id="verdictCard">
      <p class="card-label">Verdict</p>
      <div class="empty" id="verdictEmpty">Run an analysis to see results.</div>
      <div id="verdictContent" style="display:none">
        <div class="verdict" id="verdictBadge">
          <span class="verdict-dot" id="verdictDot"></span>
          <span class="verdict-label" id="verdictLabel"></span>
          <span class="verdict-conf" id="verdictConf"></span>
        </div>
        <p style="font-size:12px;color:var(--muted);margin-bottom:6px;">Compound score</p>
        <div class="bar-track"><div class="bar-fill" id="compoundBar"></div></div>
        <div class="stat-grid">
          <div class="stat-box">
            <div class="stat-label">Positive weight</div>
            <div class="stat-val" id="posScore" style="color:var(--accent)"></div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Negative weight</div>
            <div class="stat-val" id="negScore" style="color:var(--accent2)"></div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Subjectivity</div>
            <div class="stat-val" id="subjectivity"></div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Objectivity</div>
            <div class="stat-val" id="objectivity"></div>
          </div>
        </div>
        <div class="meta-row">
          <span id="wordCount"></span>
          <span id="sentCount"></span>
          <span id="gradeLevel"></span>
          <span id="procTime"></span>
        </div>
      </div>
    </div>

    <!-- EMOTION PROFILE -->
    <div class="card" id="emotionCard">
      <p class="card-label">Emotion profile</p>
      <div class="empty" id="emotionEmpty">Emotion breakdown will appear here.</div>
      <div id="emotionContent" style="display:none"></div>
    </div>

    <!-- KEYWORD SIGNALS -->
    <div class="card col-full" id="keywordCard">
      <p class="card-label">Detected signal words</p>
      <div class="empty" id="keywordEmpty">Key sentiment words will be highlighted here.</div>
      <div id="keywordContent" style="display:none">
        <p style="font-size:12px;color:var(--muted);margin-bottom:6px;">Positive signals</p>
        <div class="tag-row" id="posWords"></div>
        <p style="font-size:12px;color:var(--muted);margin:0.75rem 0 6px;">Negative signals</p>
        <div class="tag-row" id="negWords"></div>
        <p style="font-size:12px;color:var(--muted);margin:0.75rem 0 6px;">Top words in text</p>
        <div class="wc-grid" id="topWords"></div>
      </div>
    </div>

    <!-- SENTENCE BREAKDOWN -->
    <div class="card col-full" id="sentenceCard">
      <p class="card-label">Sentence-level breakdown</p>
      <div class="empty" id="sentenceEmpty">Individual sentence scores will appear here.</div>
      <div id="sentenceContent" style="display:none"></div>
    </div>

    <!-- SESSION HISTORY -->
    <div class="card col-full" id="historyCard">
      <p class="card-label">Session analytics</p>
      <div class="empty" id="historyEmpty">Your analysis history will appear here.</div>
      <div id="historyContent" style="display:none">
        <div class="hist-row" id="histStats"></div>
        <div style="margin-top:0.75rem;display:flex;flex-wrap:wrap;gap:4px;" id="trendDots"></div>
      </div>
    </div>

  </div>

</div>

<script>
const EMOTION_COLORS = {
  joy: "#7cffa0", trust: "#7eb8ff", fear: "#c084fc",
  surprise: "#ffd97d", sadness: "#6b7280", disgust: "#ff6b6b",
  anger: "#ff6b6b", anticipation: "#fb923c"
};

async function runAnalysis() {
  const text = document.getElementById("textInput").value.trim();
  if (!text) return;
  setLoading(true);
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    renderResults(data);
    loadHistory();
  } catch(e) {
    alert("Network error. Please ensure the server is running.");
  } finally {
    setLoading(false);
  }
}

async function loadDemo() {
  setLoading(true);
  try {
    const res = await fetch("/api/demo");
    const data = await res.json();
    document.getElementById("textInput").value = data.demo_text;
    renderResults(data);
    loadHistory();
  } finally {
    setLoading(false);
  }
}

function clearAll() {
  document.getElementById("textInput").value = "";
  ["verdictContent","emotionContent","keywordContent","sentenceContent","historyContent"].forEach(id => {
    document.getElementById(id).style.display = "none";
  });
  ["verdictEmpty","emotionEmpty","keywordEmpty","sentenceEmpty","historyEmpty"].forEach(id => {
    document.getElementById(id).style.display = "block";
  });
}

function renderResults(d) {
  // --- verdict ---
  const dot = document.getElementById("verdictDot");
  const label = document.getElementById("verdictLabel");
  const conf = document.getElementById("verdictConf");
  dot.className = "verdict-dot dot-" + d.label.toLowerCase();
  label.textContent = d.label;
  label.style.color = d.label === "Positive" ? "var(--accent)" :
                      d.label === "Negative" ? "var(--accent2)" : "var(--accent3)";
  conf.textContent = "Confidence: " + d.confidence + "%";

  const pct = ((d.compound + 1) / 2 * 100).toFixed(1);
  const bar = document.getElementById("compoundBar");
  bar.style.width = pct + "%";
  bar.style.background = d.compound > 0 ? "var(--accent)" :
                         d.compound < 0 ? "var(--accent2)" : "var(--accent3)";

  document.getElementById("posScore").textContent = d.pos_score.toFixed(3);
  document.getElementById("negScore").textContent = d.neg_score.toFixed(3);
  document.getElementById("subjectivity").textContent = d.subjectivity + "%";
  document.getElementById("objectivity").textContent = d.objectivity + "%";
  document.getElementById("wordCount").textContent = "Words: " + d.word_count;
  document.getElementById("sentCount").textContent = "Sentences: " + d.sentence_count;
  document.getElementById("gradeLevel").textContent = "Grade level: " + d.reading_grade;
  document.getElementById("procTime").textContent = "Processed in " + d.processing_ms + " ms";
  show("verdictContent", "verdictEmpty");

  // --- emotions ---
  const emoDiv = document.getElementById("emotionContent");
  if (Object.keys(d.emotions).length === 0) {
    emoDiv.innerHTML = "<p style='color:var(--muted);font-size:13px;'>No strong emotion signals detected.</p>";
  } else {
    const sorted = Object.entries(d.emotions).sort((a,b) => b[1]-a[1]);
    emoDiv.innerHTML = sorted.map(([emo, score]) => {
      const pct = (score * 100).toFixed(1);
      const color = EMOTION_COLORS[emo] || "#7cffa0";
      return `<div class="emo-row">
        <span class="emo-name">${emo}</span>
        <div class="emo-bar"><div class="emo-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="emo-pct">${pct}%</span>
      </div>`;
    }).join("");
  }
  show("emotionContent", "emotionEmpty");

  // --- keywords ---
  document.getElementById("posWords").innerHTML =
    d.key_pos_words.length ? d.key_pos_words.map(w => `<span class="tag tag-pos">${w}</span>`).join("") :
    "<span style='color:var(--muted);font-size:12px;'>None detected</span>";
  document.getElementById("negWords").innerHTML =
    d.key_neg_words.length ? d.key_neg_words.map(w => `<span class="tag tag-neg">${w}</span>`).join("") :
    "<span style='color:var(--muted);font-size:12px;'>None detected</span>";
  document.getElementById("topWords").innerHTML =
    d.top_words.map(w => `<span class="wc-word">${w.word} <strong style="color:var(--text)">${w.count}</strong></span>`).join("");
  show("keywordContent", "keywordEmpty");

  // --- sentences ---
  const sentDiv = document.getElementById("sentenceContent");
  if (d.sentence_breakdown.length <= 1) {
    sentDiv.innerHTML = "<p style='color:var(--muted);font-size:13px;'>Enter multiple sentences for breakdown.</p>";
  } else {
    sentDiv.innerHTML = d.sentence_breakdown.map(s => {
      const cls = s.label === "positive" ? "pos" : s.label === "negative" ? "neg" : "neu";
      return `<div class="sent-item ${cls}">
        ${s.sentence}
        <div class="sent-meta">${s.label} &mdash; score: ${s.compound}</div>
      </div>`;
    }).join("");
  }
  show("sentenceContent", "sentenceEmpty");
}

async function loadHistory() {
  const res = await fetch("/api/history");
  const h = await res.json();
  if (h.total === 0) return;
  document.getElementById("histStats").innerHTML = `
    <span class="hist-stat" style="color:var(--accent)">Positive: ${h.positive}</span>
    <span class="hist-stat" style="color:var(--accent2)">Negative: ${h.negative}</span>
    <span class="hist-stat" style="color:var(--accent3)">Neutral: ${h.neutral}</span>
    <span class="hist-stat" style="color:var(--muted)">Total: ${h.total}</span>
  `;
  document.getElementById("trendDots").innerHTML = h.trend.map(t => {
    const col = t.label === "Positive" ? "var(--accent)" :
                t.label === "Negative" ? "var(--accent2)" : "var(--accent3)";
    return `<span class="trend-dot" style="background:${col}" title="${t.label} @ ${t.time}"></span>`;
  }).join("");
  show("historyContent", "historyEmpty");
}

function show(contentId, emptyId) {
  document.getElementById(contentId).style.display = "block";
  document.getElementById(emptyId).style.display = "none";
}

function setLoading(on) {
  document.getElementById("loadingIndicator").classList.toggle("active", on);
  document.querySelectorAll("button").forEach(b => b.disabled = on);
}

document.getElementById("textInput").addEventListener("keydown", e => {
  if (e.key === "Enter" && e.ctrlKey) runAnalysis();
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("  Sentiment Intelligence Engine")
    print("  --------------------------------")
    print("  Server  : http://localhost:5000")
    print("  API     : POST /api/analyze")
    print("  Demo    : GET  /api/demo")
    print("  History : GET  /api/history")
    print()
    app.run(debug=True, port=5000)
