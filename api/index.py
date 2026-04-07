"""
Sentiment Intelligence Engine
------------------------------
A real-time AI/ML web application that performs multi-dimensional
sentiment analysis using a hand-crafted NLP pipeline built from scratch.

Author  : College AI/ML Mini Project
Stack   : Python 3, Flask, Custom NLP Engine (no external ML libs needed)
Run     : python api/index.py  ->  open http://localhost:5000
"""

import math
import re
import json
import time
import random
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, 
            template_folder="../frontend", 
            static_folder="../frontend",
            static_url_path="/frontend")

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
    return render_template("index.html")


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
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
