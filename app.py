import os, json, logging, random, re
from flask import Flask, render_template, request, jsonify, send_from_directory
from deep_translator import GoogleTranslator as Translator
from gtts import gTTS
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# -------------------- FLASK SETUP --------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- DATA FILES --------------------
INTENTS_FILE = "intents.json"
STUDENTS_FILE = "students.json"

with open(INTENTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
intents = data.get("intents", [])

# Load student data at startup to avoid File I/O on every request
with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
    STUDENT_DATA = json.load(f)

# -------------------- PATTERN PREPROCESSING --------------------
pattern_map, intent_by_tag, all_patterns, pattern_to_tag = {}, {}, [], []
for it in intents:
    tag = it.get("tag", "")
    intent_by_tag[tag] = it
    for p in it.get("patterns", []):
        kp = p.strip().lower()
        pattern_map.setdefault(kp, []).append(tag)
        all_patterns.append(kp)
        pattern_to_tag.append(tag)

# -------------------- BERT SETUP --------------------
EMBED, PATTERN_EMB = None, None
if SentenceTransformer and all_patterns:
    try:
        EMBED = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        PATTERN_EMB = EMBED.encode(all_patterns, convert_to_numpy=True)
        logger.info("✅ SentenceTransformer embeddings initialized.")
    except Exception as e:
        logger.warning(f"⚠️ SentenceTransformer init failed: {e}")

# -------------------- HELPER DATA --------------------
SEARCH_KEYWORDS = [
    "MCA syllabus", "admission process", "MCA fees", "college timing",
    "scholarship options", "hostel information", "faculty details",
    "CU infrastructure", "library info", "college events",
    "canteen facility", "placement records", "academic calendar",
    "college contact info", "MCA specialization", "list of MCA courses",
    "CU official website", "HOD name", "scholarship details"
]

def generate_search_suggestions(user_message: str, n: int = 5):
    words = user_message.lower().split()
    filtered = [k for k in SEARCH_KEYWORDS if any(w in k.lower() for w in words)]
    if not filtered:
        filtered = random.sample(SEARCH_KEYWORDS, k=min(n, len(SEARCH_KEYWORDS)))
    return [f"Search {s}" for s in random.sample(filtered, k=min(n, len(filtered)))]

def suggest_questions(uid=None, n=6):
    pool = []
    for it in intents:
        if "suggestions" in it:
            pool.extend(it["suggestions"])
    random.shuffle(pool)
    return pool[:n] if pool else ["college timing", "MCA syllabus", "fees", "hostel", "admission process"]

def translate_and_speak(text, lang_code="en"):
    try:
        translated = Translator(source="auto", target=lang_code).translate(text)
    except Exception:
        translated = text
    audio_url = None
    try:
        tts = gTTS(translated, lang=lang_code)
        out_path = os.path.join("static", "output.mp3")
        tts.save(out_path)
        audio_url = "/static/output.mp3"
    except Exception:
        pass
    return translated, audio_url

# -------------------- CORE INTENT LOGIC --------------------
def resolve_intent(text):
    txt = text.strip().lower()
    if not txt:
        return "unknown", "Please rephrase your question."
    if txt in pattern_map:
        tag = pattern_map[txt][0]
        resp = random.choice(intent_by_tag[tag].get("responses", ["Sorry, I didn’t understand."]))
        return tag, resp
    if EMBED and PATTERN_EMB is not None:
        q_emb = EMBED.encode([txt], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, PATTERN_EMB)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        tag = pattern_to_tag[best_idx]
        if best_score >= 0.6:
            resp = random.choice(intent_by_tag[tag].get("responses", ["Sorry, I didn’t understand."]))
            return tag, resp
    return "unknown", random.choice(["Sorry, I didn’t understand.", "Could you rephrase that?"])

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/set_uid", methods=["POST"])
def set_uid():
    try:
        data = request.get_json(force=True)
        raw_uid = (data.get("uid") or "").strip().upper()
        uid_norm = re.sub(r"[^A-Z0-9]", "", raw_uid)

        logger.info(f"🔍 Checking UID: {uid_norm}")

        intents_list = STUDENT_DATA.get("intents", [])
        match = next((entry for entry in intents_list if entry.get("tag", "").upper() == uid_norm), None)

        if not match:
            return jsonify({"ok": False, "message": "UID not found"}), 404

        responses = match.get("responses", [])
        response = random.choice(responses) if responses else "Record found but no response data."
        suggestions = suggest_questions(uid_norm, n=6)

        return jsonify({
            "ok": True,
            "response": response,
            "suggestions": suggestions
        })

    except Exception as e:
        logger.exception("set_uid error")
        return jsonify({"ok": False, "message": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    d = request.get_json(force=True)
    msg = d.get("message", "")
    uid = d.get("uid", "").upper().strip() or None
    lang = d.get("language", "en")

    tag, resp = resolve_intent(msg)
    translated, audio = translate_and_speak(resp, lang)
    intent_sug = suggest_questions(uid)
    search_sug = generate_search_suggestions(msg)
    suggestions = list(dict.fromkeys(intent_sug + search_sug))[:6]

    return jsonify({"response": translated, "audio": audio, "suggestions": suggestions})

@app.route("/reset", methods=["POST"])
def reset():
    return jsonify({"ok": True})

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
