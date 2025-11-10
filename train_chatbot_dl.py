
import os, json, logging, random, pickle
from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

# ----- CONFIG -----
DATA_FILE = "intents.json"   # dataset file
SENTENCE_MODEL = "distiluse-base-multilingual-cased-v2"
MIN_SAMPLES_PER_CLASS = 2
TEST_SIZE = 0.15
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 4
DROPOUT = 0.3
MODEL_DIR = "models"
MODEL_OUT = os.path.join(MODEL_DIR, "deep_intent_model.h5")
BEST_MODEL_OUT = os.path.join(MODEL_DIR, "best_deep_intent_model.h5")
LE_OUT = os.path.join(MODEL_DIR, "label_encoder.pkl")
META_OUT = os.path.join(MODEL_DIR, "model_metadata.json")
AUGMENT_SMALL_CLASSES = True
# ------------------

os.makedirs("logs", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(filename="logs/deep_model_builder.log", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deep_builder")
logger.addHandler(logging.StreamHandler())

def load_texts_labels(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    intents = payload.get("intents", [])
    texts, labels = [], []
    for intent in intents:
        tag = intent.get("tag", "")
        for p in intent.get("patterns", []) or []:
            if isinstance(p, str) and p.strip():
                texts.append(p.strip())
                labels.append(tag)
    return texts, labels

def simple_augment(text):
    opts = [
        lambda s: s,
        lambda s: s + ".",
        lambda s: s + " please",
        lambda s: s + " ?",
        lambda s: s.replace("?", "")
    ]
    return random.choice(opts)(text)

def augment_balance(texts, labels, min_samples=MIN_SAMPLES_PER_CLASS):
    counts = Counter(labels)
    new_texts = list(texts)
    new_labels = list(labels)
    if AUGMENT_SMALL_CLASSES:
        for tag, cnt in counts.items():
            if cnt < min_samples:
                need = min_samples - cnt
                samples = [t for t, l in zip(texts, labels) if l == tag]
                if not samples:
                    continue
                for _ in range(need):
                    base = random.choice(samples)
                    new_texts.append(simple_augment(base))
                    new_labels.append(tag)
    else:
        kept = [(t, l) for t, l in zip(texts, labels) if Counter(labels)[l] >= min_samples]
        if not kept:
            raise ValueError("No classes left after filtering by min samples")
        new_texts, new_labels = zip(*kept)
    return new_texts, new_labels

def build_keras_model(input_dim, num_classes, dropout=0.3):
    model = Sequential([
        Dense(512, activation="relu", input_shape=(input_dim,)),
        Dropout(dropout),
        Dense(256, activation="relu"),
        Dropout(dropout),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    logger.info("Starting deep model builder...")
    texts, labels = load_texts_labels(DATA_FILE)
    logger.info("Loaded %d patterns across %d raw tags", len(texts), len(set(labels)))

    texts_aug, labels_aug = augment_balance(texts, labels, MIN_SAMPLES_PER_CLASS)
    logger.info("Using %d samples after augmentation", len(texts_aug))

    le = LabelEncoder()
    y = le.fit_transform(labels_aug)
    num_classes = len(le.classes_)
    logger.info("Number of classes: %d", num_classes)

    y_cat = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    logger.info("Loading SentenceTransformer: %s", SENTENCE_MODEL)
    embedder = SentenceTransformer(SENTENCE_MODEL)
    X = embedder.encode(texts_aug, convert_to_numpy=True, show_progress_bar=True)
    logger.info("Embeddings shape: %s", X.shape)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    except Exception as e:
        logger.warning("Stratified split failed (%s), using random split", e)
        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = build_keras_model(X.shape[1], num_classes, DROPOUT)
    model.summary(print_fn=logger.info)

    es = EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True, verbose=1)
    chk = ModelCheckpoint(BEST_MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, chk], verbose=1)

    if os.path.exists(BEST_MODEL_OUT):
        model = tf.keras.models.load_model(BEST_MODEL_OUT)
        logger.info("Loaded best model checkpoint")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Evaluation -> Loss: %.4f | Accuracy: %.4f", loss, acc)
    print(f" Deep model accuracy: {acc*100:.2f}%")

    y_test_int = np.argmax(y_test, axis=1)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_int = np.argmax(y_pred_probs, axis=1)
    unique_test_labels = np.unique(y_test_int)
    target_names = le.inverse_transform(unique_test_labels)
    report = classification_report(y_test_int, y_pred_int, labels=unique_test_labels, target_names=target_names, zero_division=0)
    print("\nClassification report:\n", report)
    logger.info("Classification report:\n%s", report)

    model.save(MODEL_OUT)
    with open(LE_OUT, "wb") as f:
        pickle.dump(le, f)

    meta = {"classes": list(le.classes_), "accuracy": float(acc)}
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("Saved model -> %s, label encoder -> %s, metadata -> %s", MODEL_OUT, LE_OUT, META_OUT)
    print("\n Saved deep model and label encoder successfully!")

if __name__ == "__main__":
    main()
