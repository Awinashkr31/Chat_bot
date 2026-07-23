import os, json, logging, random, re
import hashlib, uuid, time, concurrent.futures
from threading import Thread
from logging.handlers import RotatingFileHandler
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from deep_translator import GoogleTranslator as Translator
from gtts import gTTS
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# -------------------- FLASK SETUP --------------------
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__, static_folder="static", template_folder="templates")

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

os.makedirs("logs", exist_ok=True)
log_handler = RotatingFileHandler("logs/app.log", maxBytes=5000000, backupCount=3)
log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# -------------------- DATA FILES --------------------
INTENTS_FILE = "intents.json"
STUDENTS_FILE = os.environ.get("STUDENTS_DATA_PATH", "students.json")

with open(INTENTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
intents = data.get("intents", [])

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

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

if SentenceTransformer and all_patterns:
    try:
        os.makedirs("models", exist_ok=True)
        CACHE_FILE = "models/pattern_embeddings.npy"
        HASH_FILE = "models/intents_hash.txt"
        current_hash = get_file_hash(INTENTS_FILE)
        
        EMBED = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        
        if os.path.exists(CACHE_FILE) and os.path.exists(HASH_FILE):
            with open(HASH_FILE, "r") as f:
                saved_hash = f.read().strip()
            if saved_hash == current_hash:
                PATTERN_EMB = np.load(CACHE_FILE)
                logger.info("✅ Loaded cached embeddings.")
        
        if PATTERN_EMB is None:
            PATTERN_EMB = EMBED.encode(all_patterns, convert_to_numpy=True)
            np.save(CACHE_FILE, PATTERN_EMB)
            with open(HASH_FILE, "w") as f:
                f.write(current_hash)
            logger.info("✅ SentenceTransformer embeddings initialized and cached.")
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

def cleanup_old_audio():
    now = time.time()
    static_dir = "static"
    if not os.path.exists(static_dir): return
    for f in os.listdir(static_dir):
        if f.startswith("output_") and f.endswith(".mp3"):
            path = os.path.join(static_dir, f)
            if now - os.path.getmtime(path) > 300: # 5 minutes
                try: os.remove(path)
                except: pass

def _do_translate_and_tts(text, lang_code, out_path):
    try:
        translated = Translator(source="auto", target=lang_code).translate(text)
    except Exception:
        translated = text
    try:
        tts = gTTS(translated, lang=lang_code)
        tts.save(out_path)
        return translated, True
    except Exception:
        return translated, False

def translate_and_speak(text, lang_code="en"):
    Thread(target=cleanup_old_audio).start()
    
    unique_id = uuid.uuid4().hex
    out_path = os.path.join("static", f"output_{unique_id}.mp3")
    audio_url = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_translate_and_tts, text, lang_code, out_path)
        try:
            translated, audio_ok = future.result(timeout=5)
            if audio_ok:
                audio_url = f"/static/output_{unique_id}.mp3"
        except concurrent.futures.TimeoutError:
            translated = text # Fallback to original text on timeout

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
@limiter.limit("60 per minute")
def set_uid():
    try:
        try:
            data = request.get_json(force=True)
        except Exception:
            return jsonify({"ok": False, "message": "Invalid JSON"}), 400
            
        raw_uid = data.get("uid", "")
        if not isinstance(raw_uid, str):
            return jsonify({"ok": False, "message": "Invalid UID format"}), 400
            
        raw_uid = raw_uid.strip().upper()
        uid_norm = re.sub(r"[^A-Z0-9]", "", raw_uid)
        
        if not re.match(r"^\d{2}[A-Z]{3}\d{5}$", uid_norm):
            return jsonify({"ok": False, "message": "Invalid UID pattern. Expected format like 24MCA20002"}), 400

        logger.info(f"🔍 Checking UID: {uid_norm}")

        with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
            student_data = json.load(f)

        intents_list = student_data.get("intents", [])
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
@limiter.limit("60 per minute")
def chat():
    try:
        d = request.get_json(force=True)
    except Exception:
        return jsonify({"response": "Invalid JSON", "audio": None, "suggestions": []}), 400
        
    msg = d.get("message", "")
    if not isinstance(msg, str):
        return jsonify({"response": "Invalid message format", "audio": None, "suggestions": []}), 400
        
    if len(msg) > 500:
        return jsonify({"response": "Message too long (max 500 characters)", "audio": None, "suggestions": []}), 400
        
    uid_raw = d.get("uid", "")
    uid = uid_raw.upper().strip() if isinstance(uid_raw, str) else None
    
    lang_raw = d.get("language", "en")
    lang = lang_raw if isinstance(lang_raw, str) else "en"

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
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
