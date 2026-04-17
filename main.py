"""
main.py — IBN Chatbot RAG
Architecture :
  1. INDEXATION  : sentence-transformers convertit la dataset en vecteurs FAISS
  2. RETRIEVAL   : cherche les 3 passages les plus proches de la question
  3. GENERATION  : Ollama génère une réponse avec le contexte récupéré

Installer :
  pip install fastapi uvicorn sentence-transformers faiss-cpu ollama requests

Lancer :
  ollama serve          (terminal 1)
  python main.py        (terminal 2)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, re, os, requests, numpy as np
import uvicorn

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
DATASET_FILE   = "industry5_dataset.json"
INDEX_FILE     = "faiss_index.bin"      # index vectoriel sauvegardé
EMBED_FILE     = "embeddings.npy"       # vecteurs sauvegardés
EMBED_MODEL    = "all-MiniLM-L6-v2"    # modèle sentence-transformers
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "llama3.2:1b"
TOP_K          = 3                      # nombre de passages récupérés
MIN_SCORE      = 0.30                   # seuil similarité cosine (0 à 1)

# ═══════════════════════════════════════════════════
# CHARGEMENT DATASET
# ═══════════════════════════════════════════════════
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"✅ Dataset chargé : {len(dataset)} questions")

# ═══════════════════════════════════════════════════
# DÉTECTION LANGUE
# ═══════════════════════════════════════════════════
FR_WORDS = {
    "je","tu","il","nous","vous","la","le","les","un","une",
    "est","que","qui","comment","pourquoi","quand","quel",
    "faire","dois","puis","peut","faut","quoi","mon","ma",
    "des","sur","pas","dans","avec","pour","par"
}

def detect_lang(text: str) -> str:
    words = set(text.lower().split())
    return "fr" if len(words & FR_WORDS) >= 1 else "en"

# ═══════════════════════════════════════════════════
# INDEXATION RAG
# ═══════════════════════════════════════════════════
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️  RAG non disponible — installe : pip install sentence-transformers faiss-cpu")

# Chargement du modèle d'embedding
embedder = None
faiss_index = None
index_items = []   # liste des passages indexés (question + réponse)

def build_index():
    """
    Construit l'index FAISS à partir de la dataset.
    Chaque entrée est représentée par :
      - question_en + answer_en
      - question_fr + answer_fr
    """
    global embedder, faiss_index, index_items

    print("\n📐 Construction de l'index RAG...")
    print(f"   Modèle : {EMBED_MODEL}")

    embedder = SentenceTransformer(EMBED_MODEL)

    # Construire les passages à indexer
    passages = []
    for item in dataset:
        # Passage anglais
        passages.append({
            "text":     f"{item['question_en']} {item['answer_en']}",
            "question": item["question_en"],
            "answer_en":item["answer_en"],
            "answer_fr":item.get("answer_fr",""),
            "category": item["category"],
            "risk":     item["risk_level"],
            "follow_en":item.get("follow_up_en",""),
            "follow_fr":item.get("follow_up_fr",""),
            "lang":     "en",
            "id":       item["id"],
        })
        # Passage français
        if item.get("question_fr"):
            passages.append({
                "text":     f"{item['question_fr']} {item['answer_fr']}",
                "question": item["question_fr"],
                "answer_en":item["answer_en"],
                "answer_fr":item.get("answer_fr",""),
                "category": item["category"],
                "risk":     item["risk_level"],
                "follow_en":item.get("follow_up_en",""),
                "follow_fr":item.get("follow_up_fr",""),
                "lang":     "fr",
                "id":       item["id"],
            })

    index_items.extend(passages)

    # Générer les embeddings
    texts = [p["text"] for p in passages]
    print(f"   Indexation de {len(texts)} passages...")

    if os.path.exists(EMBED_FILE) and os.path.exists(INDEX_FILE):
        # Charger depuis le disque si déjà calculé
        print("   Chargement de l'index existant...")
        vectors = np.load(EMBED_FILE)
        faiss_index = faiss.read_index(INDEX_FILE)
    else:
        # Calculer et sauvegarder
        vectors = embedder.encode(texts, show_progress_bar=True,
                                   convert_to_numpy=True, normalize_embeddings=True)
        np.save(EMBED_FILE, vectors)

        # Créer index FAISS (similarité cosine via produit scalaire sur vecteurs normalisés)
        dim = vectors.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)   # Inner Product = cosine si normalisé
        faiss_index.add(vectors.astype(np.float32))
        faiss.write_index(faiss_index, INDEX_FILE)

    print(f"✅ Index RAG prêt — {faiss_index.ntotal} vecteurs\n")

# Construire l'index au démarrage
if RAG_AVAILABLE:
    build_index()

# ═══════════════════════════════════════════════════
# RETRIEVAL — chercher les passages proches
# ═══════════════════════════════════════════════════
def retrieve(query: str, lang: str, top_k: int = TOP_K):
    """
    Retourne les top_k passages les plus proches de la query.
    Score entre 0 et 1 (cosine similarity).
    """
    if not RAG_AVAILABLE or faiss_index is None:
        return []

    # Encoder la query
    q_vec = embedder.encode([query], convert_to_numpy=True,
                              normalize_embeddings=True).astype(np.float32)

    # Chercher dans FAISS
    scores, indices = faiss_index.search(q_vec, top_k * 3)

    results = []
    seen_ids = set()

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or score < MIN_SCORE:
            continue
        item = index_items[idx]

        # Éviter les doublons (même question en FR et EN)
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])

        results.append({
            "score":    float(score),
            "item":     item,
        })

        if len(results) >= top_k:
            break

    return results

# ═══════════════════════════════════════════════════
# GENERATION — Ollama avec contexte RAG
# ═══════════════════════════════════════════════════
def build_prompt(query: str, passages: list, lang: str) -> tuple[str, str]:
    """
    Construit le system prompt et le prompt utilisateur
    avec les passages récupérés comme contexte.
    """
    # System prompt
    if lang == "fr":
        system = (
            "Tu es un assistant technique expert en Industrie 5.0, "
            "maintenance industrielle et sécurité des machines. "
            "Réponds en français de façon claire et structurée. "
            "Utilise le contexte fourni pour répondre précisément. "
            "Si la réponse n'est pas dans le contexte, utilise tes connaissances générales. "
            "Sois concis et pratique."
        )
    else:
        system = (
            "You are a technical expert assistant in Industry 5.0, "
            "industrial maintenance, and machine safety. "
            "Reply in English clearly and in a structured way. "
            "Use the provided context to answer precisely. "
            "If the answer is not in the context, use your general knowledge. "
            "Be concise and practical."
        )

    # Construire le contexte avec les passages récupérés
    context_lines = []
    for i, r in enumerate(passages, 1):
        item = r["item"]
        answer = item["answer_fr"] if lang == "fr" and item["answer_fr"] else item["answer_en"]
        question = item["question"]
        context_lines.append(f"[Passage {i}]\nQ: {question}\nA: {answer}")

    context = "\n\n".join(context_lines)

    # Prompt final
    if lang == "fr":
        prompt = (
            f"Contexte technique pertinent :\n\n{context}\n\n"
            f"Question du technicien : {query}\n\n"
            f"Réponse détaillée :"
        )
    else:
        prompt = (
            f"Relevant technical context:\n\n{context}\n\n"
            f"Technician question: {query}\n\n"
            f"Detailed answer:"
        )

    return system, prompt


def ask_ollama(system: str, prompt: str) -> str | None:
    """Envoie le prompt à Ollama et retourne la réponse."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,    # réponses cohérentes et précises
                    "top_p":       0.9,
                    "num_predict": 400,    # longueur max de la réponse
                },
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        return None
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None

# ═══════════════════════════════════════════════════
# FALLBACK FUZZY (si RAG indisponible)
# ═══════════════════════════════════════════════════
def tokenize(text):
    return set(re.findall(r'\b\w{3,}\b', text.lower()))

for item in dataset:
    item["_tokens_en"] = tokenize(item["question_en"])
    item["_tokens_fr"] = tokenize(item.get("question_fr",""))

def fuzzy_fallback(query: str, lang: str):
    """Matching fuzzy simple si RAG non disponible."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None, 0

    q   = query.lower().strip()
    qt  = tokenize(q)
    best_score, best_item = 0, None

    for item in dataset:
        qk = "question_fr" if lang=="fr" else "question_en"
        tk = f"_tokens_{lang}"
        question = item[qk].lower()
        q_tokens = item.get(tk, set())

        s1 = fuzz.ratio(q, question)
        s2 = fuzz.token_set_ratio(q, question)
        s3 = (len(qt & q_tokens) / max(len(qt | q_tokens),1)) * 100 if q_tokens and qt else 0
        score = s1*0.3 + s2*0.5 + s3*0.2

        if score > best_score:
            best_score = score
            best_item  = item

    return best_item, round(best_score, 1)

# ═══════════════════════════════════════════════════
# API FASTAPI
# ═══════════════════════════════════════════════════
app = FastAPI(title="IBN RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

@app.post("/chat")
def chat(msg: Message):
    query = msg.message.strip()
    lang  = detect_lang(query)

    # ── Mode RAG ──
    if RAG_AVAILABLE and faiss_index is not None:

        # 1. RETRIEVAL
        passages = retrieve(query, lang, TOP_K)

        if not passages:
            # Aucun passage pertinent — Ollama seul
            system, prompt = build_prompt(query, [], lang)
            answer = ask_ollama(system, prompt)
            if not answer:
                no_match = {
                    "en": "❌ No relevant information found. Try rephrasing.",
                    "fr": "❌ Aucune information pertinente trouvée. Essayez de reformuler."
                }
                return {
                    "answer":     no_match[lang],
                    "confidence": 0.0,
                    "category":   None,
                    "risk":       None,
                    "follow_up":  None,
                    "source":     "none",
                    "lang":       lang,
                    "passages":   [],
                }
            return {
                "answer":     answer,
                "confidence": 0.0,
                "category":   "General",
                "risk":       None,
                "follow_up":  None,
                "source":     "ollama_only",
                "lang":       lang,
                "passages":   [],
            }

        # 2. Top passage pour les métadonnées
        top     = passages[0]
        top_item= top["item"]
        score   = top["score"]

        # 3. Si score très élevé (>0.90) → réponse directe dataset
        if score >= 0.90:
            answer   = top_item["answer_fr"] if lang=="fr" and top_item["answer_fr"] else top_item["answer_en"]
            follow   = top_item[f"follow_{lang}"] if score < 0.95 else ""
            return {
                "answer":     answer,
                "confidence": round(score, 2),
                "category":   top_item["category"],
                "risk":       top_item["risk"],
                "follow_up":  follow,
                "source":     "dataset",
                "lang":       lang,
                "matched_id": top_item["id"],
                "passages":   [{"score": r["score"], "question": r["item"]["question"]}
                               for r in passages],
            }

        # 4. GENERATION — Ollama avec contexte RAG
        system, prompt = build_prompt(query, passages, lang)
        answer = ask_ollama(system, prompt)

        if answer:
            return {
                "answer":     answer,
                "confidence": round(score, 2),
                "category":   top_item["category"],
                "risk":       top_item["risk"],
                "follow_up":  top_item.get(f"follow_{lang}", ""),
                "source":     "rag",
                "lang":       lang,
                "matched_id": top_item["id"],
                "passages":   [{"score": round(r["score"],2), "question": r["item"]["question"]}
                               for r in passages],
            }

        # Ollama indisponible → réponse directe dataset
        answer = top_item["answer_fr"] if lang=="fr" and top_item["answer_fr"] else top_item["answer_en"]
        return {
            "answer":     answer,
            "confidence": round(score, 2),
            "category":   top_item["category"],
            "risk":       top_item["risk"],
            "follow_up":  top_item.get(f"follow_{lang}", ""),
            "source":     "dataset_fallback",
            "lang":       lang,
            "matched_id": top_item["id"],
            "passages":   [],
        }

    # ── Mode Fallback Fuzzy (si RAG non disponible) ──
    item, score = fuzzy_fallback(query, lang)
    if item and score >= 40:
        answer = item["answer_fr"] if lang=="fr" and item.get("answer_fr") else item["answer_en"]
        return {
            "answer":     answer,
            "confidence": round(score/100, 2),
            "category":   item["category"],
            "risk":       item["risk_level"],
            "follow_up":  item.get(f"follow_up_{lang}", ""),
            "source":     "fuzzy_fallback",
            "lang":       lang,
        }

    no_match = {
        "en": "❌ I couldn't find a matching answer. Try rephrasing.",
        "fr": "❌ Aucune réponse trouvée. Essayez de reformuler."
    }
    return {
        "answer":     no_match[lang],
        "confidence": 0.0,
        "category":   None,
        "risk":       None,
        "follow_up":  None,
        "source":     "none",
        "lang":       lang,
    }


@app.get("/health")
def health():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
        models    = [m["name"] for m in r.json().get("models", [])]
    except:
        ollama_ok = False
        models    = []

    return {
        "status":        "ok",
        "questions":     len(dataset),
        "rag_available": RAG_AVAILABLE,
        "index_size":    faiss_index.ntotal if faiss_index else 0,
        "embed_model":   EMBED_MODEL,
        "ollama":        "online" if ollama_ok else "offline",
        "models":        models,
        "ollama_model":  OLLAMA_MODEL,
    }


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  IBN Chatbot — Architecture RAG")
    print(f"  Dataset    : {len(dataset)} questions")
    print(f"  Embedding  : {EMBED_MODEL}")
    print(f"  LLM        : {OLLAMA_MODEL}")
    print(f"  Top-K      : {TOP_K} passages récupérés")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)