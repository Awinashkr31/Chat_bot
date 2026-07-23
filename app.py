import os, json, logging, random, re
import hashlib, uuid, time, concurrent.futures
from threading import Thread
from logging.handlers import RotatingFileHandler
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from deep_translator import GoogleTranslator as Translator
from gtts import gTTS
from sklearn.metrics.pairwise import cosine_similarity
from config import Config
from retrieval import RetrievalSystem
import rapidfuzz
import re

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
INTENTS_FILE = Config.INTENTS_FILE
STUDENTS_FILE = Config.STUDENTS_FILE
CACHE_FILE = Config.CACHE_FILE
HASH_FILE = Config.HASH_FILE

# -------------------- CUSTOM LOGGERS --------------------
def setup_file_logger(name, log_file):
    l = logging.getLogger(name)
    l.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(Config.LOGS_DIR, log_file), encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(message)s"))
    l.addHandler(fh)
    return l

intent_logger = setup_file_logger("intent_logger", "intent_queries.log")
unmatched_logger = setup_file_logger("unmatched_logger", "unmatched_queries.log")
corrections_logger = setup_file_logger("corrections_logger", "corrections.log")

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
EMBED, PATTERN_EMB, retriever = None, None, None

VOCAB = set()
for intent in intents:
    for pattern in intent.get("patterns", []):
        words = re.findall(r'\b\w+\b', pattern.lower())
        VOCAB.update(words)
VOCAB = sorted(list(VOCAB))

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
        
        EMBED = SentenceTransformer(Config.SENTENCE_MODEL)
        
        if os.path.exists(CACHE_FILE) and os.path.exists(HASH_FILE):
            with open(HASH_FILE, "r") as f:
                saved_hash = f.read().strip()
            if saved_hash == current_hash:
                PATTERN_EMB = np.load(CACHE_FILE)
                logger.info("Loaded cached embeddings.")
        
        if PATTERN_EMB is None:
            PATTERN_EMB = EMBED.encode(all_patterns, convert_to_numpy=True)
            np.save(CACHE_FILE, PATTERN_EMB)
            with open(HASH_FILE, "w") as f:
                f.write(current_hash)
            logger.info("SentenceTransformer embeddings initialized and cached.")
            
        retriever = RetrievalSystem(EMBED)
        retriever.load_documents("docs")
    except Exception as e:
        logger.warning(f"SentenceTransformer init failed: {e}")

# -------------------- HELPER DATA --------------------
SESSION_STORE = {}

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
def resolve_intent(text, session_id=None):
    txt = text.strip().lower()
    if not txt:
        return "unknown", "Please rephrase your question.", [], False
        
    recent_tags = SESSION_STORE.get(session_id, {}).get("history", []) if session_id else []
    
    def score_text(t):
        if t in pattern_map:
            return pattern_map[t][0], 1.0, None
        if EMBED and PATTERN_EMB is not None:
            q_emb = EMBED.encode([t], convert_to_numpy=True)
            sims = cosine_similarity(q_emb, PATTERN_EMB)[0]
            for i, p_tag in enumerate(pattern_to_tag):
                if p_tag in recent_tags:
                    sims[i] += 0.05
            best_idx = int(sims.argmax())
            return pattern_to_tag[best_idx], float(sims[best_idx]), sims
        return "unknown", 0.0, None

    tag, best_score, sims = score_text(txt)
    corrected_txt = txt
    
    if best_score < 0.75:
        words = re.findall(r'\b\w+\b', txt)
        corrected_words = []
        changed = False
        for w in words:
            if w not in VOCAB:
                match = rapidfuzz.process.extractOne(w, VOCAB, scorer=rapidfuzz.distance.Levenshtein.distance)
                if match and match[1] <= 2:
                    corrected_words.append(match[0])
                    changed = True
                else:
                    corrected_words.append(w)
            else:
                corrected_words.append(w)
                
        if changed:
            candidate_txt = " ".join(corrected_words)
            c_tag, c_best_score, c_sims = score_text(candidate_txt)
            if c_best_score > best_score:
                corrected_txt = candidate_txt
                tag, best_score, sims = c_tag, c_best_score, c_sims
                corrections_logger.info(f"{time.time()}|{txt}|{corrected_txt}")
    
    # Log query
    intent_logger.info(f"{time.time()}|{corrected_txt}|{tag}|{best_score:.4f}")
            
    if best_score >= 0.75:
        resp = random.choice(intent_by_tag[tag].get("responses", ["Sorry, I didn’t understand."]))
        return tag, resp, [], False
            
    elif best_score >= 0.55:
        resp = random.choice(intent_by_tag[tag].get("responses", ["Sorry, I didn’t understand."]))
        
        unmatched_logger.info(f"{time.time()}|{corrected_txt}|{tag}|{best_score:.4f}")
            
        alt_tags = []
        if sims is not None:
            sorted_indices = sims.argsort()[::-1]
            for idx in sorted_indices[1:]:
                alt_tag = pattern_to_tag[idx]
                if alt_tag != tag and alt_tag not in alt_tags:
                    alt_tags.append(alt_tag)
                if len(alt_tags) >= 2: break
                
        did_you_mean = [intent_by_tag[t]["patterns"][0] for t in alt_tags if intent_by_tag[t].get("patterns")]
        resp += "\n\n*(Confidence is slightly low. Did you mean something else?)*"
        return tag, resp, did_you_mean, True
            
    unmatched_logger.info(f"{time.time()}|{corrected_txt}|none|{best_score:.4f}")
        
    # Fallback to local FAISS retrieval
    if retriever:
        rag_response = retriever.search(corrected_txt)
        if rag_response:
            return "document_retrieval", rag_response, [], False
                
    return "unknown", random.choice(["Sorry, I didn’t understand.", "Could you rephrase that?"]), [], False

# -------------------- ROUTES --------------------
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    # Consider adjusting CSP based on exact app needs (inline scripts, fonts, etc.)
    # response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

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

        logger.info(f"Checking UID: {uid_norm}")

        with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
            student_data = json.load(f)

        intents_list = student_data.get("intents", [])
        match = next((entry for entry in intents_list if entry.get("tag", "").upper() == uid_norm), None)

        if not match:
            return jsonify({"ok": False, "message": "UID not found"}), 404

        responses = match.get("responses", [])
        raw_response = random.choice(responses) if responses else ""
        
        # Extract PII
        patterns = match.get("patterns", [])
        first_name = patterns[1] if len(patterns) > 1 else "Student"
        
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', raw_response)
        email = email_match.group(0) if email_match else ""
        
        clean_response = f"Hi {first_name} — how can I help with MCA at CU today?"
        suggestions = suggest_questions(uid_norm, n=6)

        return jsonify({
            "ok": True,
            "response": clean_response,
            "suggestions": suggestions,
            "profile": {
                "name": first_name,
                "roll": uid_norm,
                "email": email
            }
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
    
    session_id = d.get("session_id", "default")
    
    lang_raw = d.get("language", "en")
    lang = lang_raw if isinstance(lang_raw, str) else "en"

    tag, resp, alt_suggestions, is_clarify = resolve_intent(msg, session_id=session_id)
    
    # Update context
    if session_id not in SESSION_STORE: SESSION_STORE[session_id] = {"history": []}
    if tag != "unknown" and tag != "document_retrieval":
        SESSION_STORE[session_id]["history"].append(tag)
        SESSION_STORE[session_id]["history"] = SESSION_STORE[session_id]["history"][-3:]
    
    translated, audio = translate_and_speak(resp, lang)
    intent_sug = suggest_questions(uid)
    search_sug = generate_search_suggestions(msg)
    
    suggestions = list(dict.fromkeys(alt_suggestions + intent_sug + search_sug))[:6]

    return jsonify({"response": translated, "audio": audio, "suggestions": suggestions, "is_clarify": is_clarify})

@app.route("/reset", methods=["POST"])
def reset():
    return jsonify({"ok": True})

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=Config.FLASK_DEBUG)
