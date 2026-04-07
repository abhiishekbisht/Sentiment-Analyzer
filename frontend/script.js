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
