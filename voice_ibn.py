"""
voice_ibn.py
═══════════════════════════════════════════════════════
Chatbot vocal IBN — Placement de services en temps réel

Architecture :
  1. Enregistrement audio (sounddevice)
  2. Transcription (Whisper)
  3. Détection intention IBN (dataset_4.json)
  4. Moteur de placement → nœud optimal
  5. Réponse vocale (pyttsx3)
  6. Mise à jour dashboard (WebSocket)

Installer :
  pip install openai-whisper sounddevice scipy pyttsx3 fastapi uvicorn websockets

Lancer :
  python voice_ibn.py
  Ouvrir : http://localhost:8081
"""

import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile, os, json, warnings, threading, time, asyncio
import requests as http_requests
import numpy as np

# ── Isolation du bruit ──
try:
    import noisereduce as nr
    NOISEREDUCE_OK = True
except ImportError:
    NOISEREDUCE_OK = False
    print("⚠️  noisereduce non installé — pip install noisereduce")

try:
    import webrtcvad
    WEBRTCVAD_OK = True
except ImportError:
    WEBRTCVAD_OK = False
    print("⚠️  webrtcvad non installé — pip install webrtcvad")

# ── Neo4j ──
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️  neo4j non installé — pip install neo4j")
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
DATASET_FILE  = "dataset_4.json"
WHISPER_MODEL = "base"
SAMPLE_RATE   = 16000
SERVER_PORT   = 8081

# ═══════════════════════════════════════════════════════
# CHARGEMENT DATASET
# ═══════════════════════════════════════════════════════
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

intentions  = data["intentions"]
services    = data["services"]
nodes       = data["nodes"]
latency_map = data["latency"]

print(f"✅ Dataset chargé : {len(nodes)} nœuds, {len(services)} services, {len(intentions)} intentions")

# ═══════════════════════════════════════════════════════
# MAPPING VOCAL — phrases naturelles → intentions
# ═══════════════════════════════════════════════════════
VOICE_MAPPING = {
    "i1":  ["replace power unit", "power unit", "remplacer unité"],
    "i2":  ["machine status", "operational status", "état machine", "statut", "operational stage", "machine stage", "operational state", "retrieve operational", "status of the machine"],
    "i3":  ["ar sequence", "ar assembly", "ar service", "deploy ar",
            "déployer ar", "service ar", "ar generator", "augmented reality",
            "show me the ar", "show ar", "ar for power", "ar for assembly"],
    "i4":  ["highlight errors", "errors assembly", "erreurs assemblage"],
    "i5":  ["motor temperature", "temperature anomaly", "température moteur",
            "motor overheat", "surchauffe moteur", "check temperature"],
    "i6":  ["lubrication", "lubrifier", "moving parts", "lubricate"],
    "i7":  ["ar belt", "belt replacement", "remplacement courroie", "belt ar"],
    "i8":  ["conveyor belt", "belt wear", "belt damage", "detect belt"],
    "i9":  ["ar gear", "gear alignment", "alignement engrenage"],
    "i10": ["vibration data", "vibration analysis", "analyse vibration",
            "abnormal patterns", "vibration abnormal"],
    "i11": ["electrical connections", "signal integrity", "connexions électriques"],
    "i12": ["ar safety overlay", "safety overlay", "ar safety"],
    "i13": ["temperature sensors", "current sensors", "monitor temperature"],
    "i14": ["detect errors", "error detection", "identify error",
            "error detector", "identify problem", "use error detector"],
    "i15": ["full diagnostics", "complete diagnostics", "diagnostic complet", "full dies", "diagnostics after maintenance", "perform full", "perform diagnostics"],
    "i16": ["predictive maintenance", "maintenance prédictive", "high risk"],
    "i17": ["ar troubleshoot", "ar fault guidance", "ar dépannage"],
    "i18": ["post maintenance efficiency", "operational efficiency", "post-maintenance", "post mantanons", "verify the post", "maintenance operational efficiency", "return system to production"],
    "i19": ["ar summary maintenance", "maintenance summary ar"],
    "i20": ["components attention", "maintenance alert"],
    "i21": ["belt tension", "conveyor tension", "belt alignment"],
    "i22": ["ar fan", "fan assembly", "ar ventilateur"],
    "i23": ["fan vibration", "vibration fan", "fan analysis"],
    "i24": ["ar hazard", "hazardous areas", "ar danger"],
    "i25": ["voice command", "voice conversion", "conversion vocale",
            "voice control", "commande vocale"],
    "i26": ["machine errors logs", "error logs", "retrieve logs"],
    "i27": ["ar valve", "valve replacement", "remplacement vanne"],
    "i28": ["hydraulic leak", "fuite hydraulique", "leak detection", "hydraulic"],
    "i29": ["pipeline pressure", "pressure levels", "pression pipeline"],
    "i30": ["ar gearbox", "gearbox inspection", "inspection gearbox"],
    "i31": ["gearbox teeth", "gear wear", "usure engrenage"],
    "i32": ["motor load", "motor vibration", "analyze motor", "charge moteur"],
    "i33": ["cable connections", "electrical cable", "câbles électriques"],
    "i34": ["ar reactivate", "ar safety reactivate"],
    "i35": ["sensor monitoring", "monitor sensors", "surveiller capteurs"],
    "i36": ["assembly inconsistency", "detect inconsistency"],
    "i37": ["system diagnostics", "full system diagnostics"],
    "i38": ["predictive alert", "high risk alert"],
    "i39": ["ar troubleshooting faults", "ar guidance troubleshoot"],
    "i40": ["post maintenance verify", "verify efficiency"],
    "i41": ["ar summary all steps", "ar all maintenance"],
    "i42": ["components alert", "alert components"],
    "i43": ["video capture", "quality review", "capture vidéo"],
    "i44": ["fault history", "historical fault data", "historique pannes"],
    "i45": ["ar motor replacement", "motor replacement ar"],
    "i46": ["motor anomaly", "motor current", "detect motor fault",
            "motor fault", "anomaly motor", "anomalie moteur",
            "activate anomaly", "anomaly analyzer", "activate anomaly analyzer"],
    "i47": ["alignment mechanical", "mechanical alignment", "verify alignment"],
    "i48": ["final check", "final system check", "vérification finale"],
    "i49": ["maintenance report", "generate report", "rapport maintenance"],
    "i50": ["maintenance cycle alert", "next maintenance cycle"],
}

# ═══════════════════════════════════════════════════════
# ÉTAT GLOBAL — partagé entre vocal et dashboard
# ═══════════════════════════════════════════════════════
state = {
    "nodes":      [],
    "placements": [],
    "listening":  False,
    "last_text":  "",
    "last_intent":"",
    "last_node":  "",
    "stats": {
        "total": 0,
        "success": 0,
        "fail": 0
    }
}

def init_nodes(reset_load=True):
    if reset_load or not state["nodes"]:
        state["nodes"] = []
        for nd in nodes:
            lats    = latency_map.get(nd["id"], [50])
            avg_lat = round(sum(lats) / len(lats), 1)
            lat_min = round(min(lats), 1)
            lat_max = round(max(lats), 1)
            state["nodes"].append({
                "id":        nd["id"],
                "type":      nd["type"],
                "cpu":       nd["capacity"]["CPU"],
                "mem":       nd["capacity"]["MEM"],
                "disk":      nd["capacity"]["DISK"],
                "bw":        nd["capacity"]["BW"],
                "cpu_used":  0,
                "mem_used":  0,
                "disk_used": 0,
                "bw_used":   0,
                "lat":       avg_lat,
                "lat_min":   lat_min,
                "lat_max":   lat_max,
                "intents":   [],
                "active":    False,
            })

init_nodes()

# ═══════════════════════════════════════════════════════
# NEO4J
# ═══════════════════════════════════════════════════════
NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "emnakhouloudazizyoussef"
NEO4J_DB       = "neo4j"

neo4j_driver = None

def neo4j_connect():
    global neo4j_driver
    if not NEO4J_AVAILABLE:
        return False
    try:
        neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        neo4j_driver.verify_connectivity()
        print("✅ Neo4j connecté")
        return True
    except Exception as e:
        print(f"⚠️  Neo4j non disponible : {e}")
        neo4j_driver = None
        return False

def neo4j_write_voice_placement(intent: dict, results: list, text: str):
    if not neo4j_driver:
        return
    try:
        with neo4j_driver.session(database=NEO4J_DB) as session:
            ts = time.strftime("%H:%M:%S")
            session.run("""
                MATCH (i:Intention {id:$iid})-[r:PLACED_ON]->()
                DELETE r
            """, iid=intent["id"])

            for r in results:
                if not r["node"]:
                    session.run("""
                        MERGE (i:Intention {id:$iid})
                        SET i.description = $desc,
                            i.failed      = true,
                            i.voice_text  = $text,
                            i.timestamp   = $ts,
                            i.success     = false
                    """, iid=intent["id"], desc=intent["description"],
                        text=text[:80], ts=ts)
                else:
                    node_id = r["node"]
                    lat     = r["lat"]
                    grouped = r.get("grouped", False)
                    session.run("""
                        MERGE (i:Intention {id:$iid})
                        SET i.description = $desc,
                            i.failed      = false,
                            i.success     = true,
                            i.voice_text  = $text,
                            i.timestamp   = $ts,
                            i.services    = $svcs
                        WITH i
                        MERGE (n:IbnNode {id:$nid})
                        MERGE (i)-[p:PLACED_ON]->(n)
                        SET p.latency   = $lat,
                            p.timestamp = $ts,
                            p.grouped   = $grouped,
                            p.voice     = true
                    """,
                        iid=intent["id"], desc=intent["description"],
                        text=text[:80], ts=ts,
                        svcs=", ".join(intent["services"]),
                        nid=node_id, lat=lat, grouped=grouped,
                    )
            print(f"   📡 Neo4j mis à jour : {intent['id']}")
    except Exception as e:
        print(f"   ⚠️  Neo4j error : {e}")

def neo4j_clear_voice():
    if not neo4j_driver:
        return
    try:
        with neo4j_driver.session(database=NEO4J_DB) as session:
            session.run("""
                MATCH (i:Intention)-[r:PLACED_ON]->()
                WHERE r.voice = true
                DELETE r
            """)
            session.run("""
                MATCH (i:Intention)
                WHERE i.voice_text IS NOT NULL
                REMOVE i.voice_text, i.timestamp
                SET i.failed = false, i.success = false
            """)
        print("🗑️  Neo4j — placements vocaux effacés")
    except Exception as e:
        print(f"⚠️  Neo4j clear error : {e}")

neo4j_connect()

# ═══════════════════════════════════════════════════════
# MOTEUR DE PLACEMENT
# ═══════════════════════════════════════════════════════
def total_resources(service_ids):
    total = {"CPU": 0, "MEM": 0, "DISK": 0, "BW": 0}
    for s in services:
        if s["id"] in service_ids:
            for r in total:
                total[r] += s["resources"].get(r, 0)
    return total

def get_available(node_id):
    nd_state = next((n for n in state["nodes"] if n["id"] == node_id), None)
    nd_cap   = next((n for n in nodes if n["id"] == node_id), None)
    if not nd_state or not nd_cap:
        return None
    return {
        "CPU":  nd_cap["capacity"]["CPU"]  - nd_state["cpu_used"],
        "MEM":  nd_cap["capacity"]["MEM"]  - nd_state["mem_used"],
        "DISK": nd_cap["capacity"]["DISK"] - nd_state["disk_used"],
        "BW":   nd_cap["capacity"]["BW"]   - nd_state["bw_used"],
    }

def best_node_for(required, qos_latency):
    best_node, best_lat = None, 9999
    for nd in nodes:
        avail = get_available(nd["id"])
        if not avail:
            continue
        if not all(avail.get(r, 0) >= required[r] for r in required):
            continue
        lats = latency_map.get(nd["id"], [])
        if not lats:
            continue
        avg_lat = sum(lats) / len(lats)
        if avg_lat <= qos_latency and avg_lat < best_lat:
            best_lat  = avg_lat
            best_node = nd["id"]
    return best_node, round(best_lat, 1) if best_node else None

def select_node(required, qos_latency, service_ids):
    # Étape 1 : placement groupé
    node_id, lat = best_node_for(required, qos_latency)
    if node_id:
        return [{"service": "ALL", "node": node_id, "lat": lat, "grouped": True}]

    # Étape 2 : placement distribué service par service
    results = []
    for svc_id in service_ids:
        svc = next((s for s in services if s["id"] == svc_id), None)
        if not svc:
            continue
        svc_req = {
            "CPU":  svc["resources"]["CPU"],
            "MEM":  svc["resources"]["MEM"],
            "DISK": svc["resources"]["DISK"],
            "BW":   svc["resources"]["BW"],
        }
        nd, lt = best_node_for(svc_req, qos_latency)
        if nd:
            results.append({"service": svc_id, "node": nd, "lat": lt, "grouped": False})
            apply_service_to_node(nd, svc_req)
        else:
            results.append({"service": svc_id, "node": None, "lat": None, "grouped": False})

    return results if results else None

def apply_service_to_node(node_id, req):
    import random
    for nd in state["nodes"]:
        if nd["id"] == node_id:
            nd["cpu_used"]  += req["CPU"]
            nd["mem_used"]  += req["MEM"]
            nd["disk_used"] += req["DISK"]
            nd["bw_used"]   += req["BW"]
            nd["active"]     = True
            lats = latency_map.get(node_id, [50])
            base = sum(lats) / len(lats)
            nd["lat"] = round(base + random.uniform(-2, 4), 1)
            break

def apply_placement(intent, results):
    for r in results:
        if not r["node"]:
            continue
        if r.get("grouped"):
            req = total_resources(intent["services"])
            apply_service_to_node(r["node"], req)
        for nd in state["nodes"]:
            if nd["id"] == r["node"]:
                if intent["id"] not in nd["intents"]:
                    nd["intents"].append(intent["id"])
                break

def detect_intention(text: str):
    results = detect_multiple_intentions(text)
    return results[0] if results else None

def prefilter_intentions(text: str, top_n: int = 10) -> list:
    text_lower = text.lower()
    text_words = set(text_lower.split())

    service_keywords = {
        "s1": ["voice", "vocal", "speak", "audio", "conversion", "voix", "record"],
        "s2": ["protocol", "convert", "status", "operational", "retrieve", "logs"],
        "s3": ["ar", "augmented", "reality", "visual", "inspect", "overlay",
               "sequence", "guidance", "lunettes", "glasses", "assembly"],
        "s4": ["content", "display", "show", "highlight", "summary", "overlay"],
        "s5": ["error", "detect", "fault", "wear", "damage", "check", "verify",
               "inconsisten", "alignment"],
        "s6": ["anomaly", "analyze", "analyzer", "vibration", "motor", "temperature",
               "pressure", "sensor", "load", "current"],
    }

    scored = []
    for intent in intentions:
        score = 0
        desc_words = set(intent["description"].lower().split())
        score += len(desc_words & text_words) * 2

        for svc_id in intent["services"]:
            for kw in service_keywords.get(svc_id, []):
                if kw in text_lower:
                    score += 3
                    break

        if score > 0:
            scored.append((score, intent))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:top_n]]

def detect_with_ollama(text: str) -> list:
    import re

    candidates = prefilter_intentions(text, top_n=10)
    if not candidates:
        return []

    # Vérifier que i3 (ou l'intention correcte) est bien dans les candidats
    candidate_ids = {i["id"] for i in candidates}
    print(f"   📋 Candidats pré-filtrés : {sorted(candidate_ids)}")

    word_count = len(text.split())
    max_n = 1 if word_count <= 8 else 3

    candidates_list = "\n".join(
        [f"{i['id']}: {i['description']}"
         for i in candidates]
    )

    print(f"   📋 Envoi de {len(candidates)} candidats à Ollama...")

    prompt = f"""You are a classifier. Pick the BEST matching intention ID for the technician request.
OUTPUT: Only the ID(s) separated by commas. Example: i3
No explanations, no service codes, no dashes, no ranges.
MAX {max_n} ID(s).

CANDIDATE INTENTIONS:
{candidates_list}

TECHNICIAN REQUEST: {text}

BEST ID(s):"""

    try:
        resp = http_requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":  "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 15,
                }
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return []

        raw    = resp.json().get("response", "")
        answer = raw.replace("\n", ",").replace(" ", "").strip().lower()
        print(f"   🤖 Ollama → {answer}")

        found_ids = re.findall(r'i\d+', answer)
        found_ids = list(dict.fromkeys(found_ids))[:max_n]

        # Garder seulement les IDs qui étaient dans les candidats
        detected = []
        for iid in found_ids:
            if iid not in candidate_ids:
                print(f"   ⚠️  Ollama a inventé {iid} (hors candidats) — ignoré")
                continue
            intent = next((i for i in intentions if i["id"] == iid), None)
            if intent:
                detected.append(intent)

        return detected

    except Exception as e:
        print(f"   ⚠️  Ollama error : {e}")
        return []

def keyword_fallback(text: str) -> list:
    text_lower = text.lower()
    text_words = set(text_lower.split())
    scored = {}

    # Score 1 — VOICE_MAPPING (phrases naturelles — priorité maximale)
    for intent_id, keywords in VOICE_MAPPING.items():
        for kw in keywords:
            if kw in text_lower:
                scored[intent_id] = scored.get(intent_id, 0) + 5
                break

    # Score 2 — mots communs avec description
    for intent in intentions:
        desc_w = set(intent["description"].lower().split())
        overlap = len(desc_w & text_words)
        if overlap > 0:
            scored[intent["id"]] = scored.get(intent["id"], 0) + overlap

    # Score 3 — service keywords
    svc_kw = {
        "s1": ["voice","vocal","conversion","speak","audio"],
        "s2": ["protocol","status","retrieve","logs","operational"],
        "s3": ["ar","augmented","visual","inspect","overlay","sequence","deploy","assembly"],
        "s4": ["display","content","show","highlight","summary"],
        "s5": ["error","detect","fault","wear","damage","verify","check"],
        "s6": ["anomaly","analyze","vibration","motor","temperature","sensor","load"],
    }
    for intent in intentions:
        for svc_id in intent["services"]:
            for kw in svc_kw.get(svc_id, []):
                if kw in text_lower:
                    scored[intent["id"]] = scored.get(intent["id"], 0) + 2
                    break

    # Pénalité intentions larges (i15, i37, i48)
    for intent in intentions:
        nb = len(intent["services"])
        if nb >= 5:
            scored[intent["id"]] = scored.get(intent["id"], 0) * 0.15
        elif nb >= 4:
            scored[intent["id"]] = scored.get(intent["id"], 0) * 0.5

    sorted_intents = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    max_intents    = 1 if len(text.split()) <= 8 else 3

    print(f"\n   📊 Top 5 keyword scores :")
    for iid, sc in sorted_intents[:5]:
        print(f"      {iid} → {round(sc,1)}")

    selected, used_svcs = [], set()
    for intent_id, score in sorted_intents:
        if score < 2:
            break
        intent = next((i for i in intentions if i["id"] == intent_id), None)
        if not intent:
            continue
        new_svcs = set(intent["services"]) - used_svcs
        if new_svcs:
            selected.append(intent)
            used_svcs.update(intent["services"])
        if len(selected) >= max_intents:
            break

    return selected

def detect_multiple_intentions(text: str) -> list:
    """
    Détection hybride intelligente :
    - Français → Keyword fallback direct
    - Anglais  → Ollama sur candidats pré-filtrés
                 Si Ollama invalide → keyword pur (Ollama ignoré complètement)
    """
    fr_words = {"je", "tu", "il", "nous", "vous", "la", "le", "les", "un", "une",
                "est", "que", "qui", "comment", "veux", "dois", "puis", "faut",
                "déployer", "activer", "analyser", "détect"}
    words = set(text.lower().split())
    is_french = len(words & fr_words) >= 1

    if is_french:
        print(f"\n   🇫🇷 Langue FR → Keyword fallback")
        detected = keyword_fallback(text)
        if detected:
            print(f"   ✅ Keyword : {[i['id'] for i in detected]}")
        return detected

    # ── Anglais → Ollama ──
    print(f"\n   🧠 Langue EN → Ollama (candidats pré-filtrés)...")
    ollama_detected = detect_with_ollama(text)

    text_lower = text.lower()

    if ollama_detected:
        # Validation par score keyword : l'intention Ollama doit scorer
        # mieux que le top-1 keyword OU être dans le top-3 keyword
        kw_scores = {}
        for intent_id, keywords in VOICE_MAPPING.items():
            for kw in keywords:
                if kw in text_lower:
                    kw_scores[intent_id] = kw_scores.get(intent_id, 0) + 5
                    break
        for intent in intentions:
            desc_w = set(intent["description"].lower().split())
            overlap = len(desc_w & set(text_lower.split()))
            if overlap:
                kw_scores[intent["id"]] = kw_scores.get(intent["id"], 0) + overlap

        # Top 3 keyword IDs
        top_kw = [iid for iid, _ in sorted(kw_scores.items(),
                   key=lambda x: x[1], reverse=True)[:3]]

        valid = []
        for intent in ollama_detected:
            ollama_score = kw_scores.get(intent["id"], 0)
            top1_score   = kw_scores.get(top_kw[0], 0) if top_kw else 0

            # Valide si : dans top3 keyword OU score >= 50% du top1
            if intent["id"] in top_kw or (top1_score > 0 and ollama_score >= top1_score * 0.5):
                valid.append(intent)
            else:
                print(f"   ⚠️  Ollama {intent['id']} rejeté (kw_score={ollama_score}, top1={top1_score}, top3={top_kw})")

        if valid:
            print(f"   ✅ Ollama validé par keyword-score : {[i['id'] for i in valid]}")
            return valid

        # Ollama invalide → keyword pur
        print(f"   ⚠️  Ollama invalide → Keyword fallback pur")
        detected = keyword_fallback(text)
        if detected:
            print(f"   ✅ Keyword : {[i['id'] for i in detected]}")
        return detected

    # Ollama retourne rien → keyword fallback
    print(f"   ⚠️  Ollama vide → Keyword fallback")
    detected = keyword_fallback(text)
    if detected:
        print(f"   ✅ Keyword : {[i['id'] for i in detected]}")
    return detected

# ═══════════════════════════════════════════════════════
# WHISPER
# ═══════════════════════════════════════════════════════
print(f"⏳ Chargement Whisper '{WHISPER_MODEL}'...")
whisper_model = whisper.load_model(WHISPER_MODEL)
print("✅ Whisper prêt")

def remove_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    if not NOISEREDUCE_OK:
        return audio
    try:
        audio_f = audio.flatten().astype(np.float32)
        noise_sample = audio_f[:int(sr * 0.5)]
        reduced = nr.reduce_noise(
            y=audio_f,
            sr=sr,
            y_noise=noise_sample,
            prop_decrease=0.85,
            stationary=False,
        )
        return reduced.reshape(audio.shape)
    except Exception as e:
        print(f"   ⚠️  noisereduce error : {e}")
        return audio

def detect_voice_activity(audio: np.ndarray, sr: int) -> np.ndarray:
    if not WEBRTCVAD_OK:
        return audio
    try:
        vad = webrtcvad.Vad(2)
        audio_int16 = (audio.flatten() * 32767).astype(np.int16)
        frame_ms   = 30
        frame_size = int(sr * frame_ms / 1000)
        frames     = []
        voiced     = []

        for i in range(0, len(audio_int16) - frame_size, frame_size):
            frame = audio_int16[i:i + frame_size]
            frame_bytes = frame.tobytes()
            try:
                is_speech = vad.is_speech(frame_bytes, sr)
            except:
                is_speech = True
            frames.append(frame)
            voiced.append(is_speech)

        context = 3
        keep = set()
        for i, v in enumerate(voiced):
            if v:
                for j in range(max(0, i-context), min(len(voiced), i+context+1)):
                    keep.add(j)

        if not keep:
            print("   ⚠️  VAD : aucune voix détectée, audio original gardé")
            return audio

        kept_frames = [frames[i] for i in sorted(keep)]
        audio_clean = np.concatenate(kept_frames).astype(np.float32) / 32767.0
        voice_ratio = len(keep) / max(len(frames), 1) * 100
        print(f"   🎙️  VAD : {voice_ratio:.0f}% de voix détectée")
        return audio_clean.reshape(-1, 1)

    except Exception as e:
        print(f"   ⚠️  webrtcvad error : {e}")
        return audio

def transcribe(audio_array) -> tuple[str, str]:
    print("   🔊 Traitement audio...")
    audio = audio_array.astype(np.float32)
    if audio.max() > 1.0:
        audio = audio / 32768.0

    if NOISEREDUCE_OK:
        audio = remove_noise(audio, SAMPLE_RATE)
        print("   ✅ noisereduce appliqué")

    if WEBRTCVAD_OK:
        audio = detect_voice_activity(audio, SAMPLE_RATE)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp.name, SAMPLE_RATE, audio)

    result = whisper_model.transcribe(
        temp.name,
        fp16=False,
        initial_prompt=(
            "IBN technician commands: deploy AR service, voice conversion, "
            "anomaly analyzer, error detector, motor fault, belt replacement, "
            "vibration analysis, hydraulic leak, sensor calibration, "
            "AR guidance, predictive maintenance, detect fault."
        ),
        language=None,
        temperature=0.0,
    )

    try:
        os.remove(temp.name)
    except:
        pass

    return result["text"].strip(), result.get("language", "en")

# ═══════════════════════════════════════════════════════
# TTS
# ═══════════════════════════════════════════════════════
def speak(text: str, lang: str = "en"):
    try:
        short = text[:300].replace('"', '').replace("'", '')
        import subprocess
        subprocess.Popen([
            'PowerShell', '-Command',
            f'Add-Type -AssemblyName System.Speech; '
            f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
            f'$s.Speak("{short}")'
        ], creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        print(f"   ⚠️  TTS error : {e}")

# ═══════════════════════════════════════════════════════
# FASTAPI + WEBSOCKET DASHBOARD
# ═══════════════════════════════════════════════════════
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

ws_clients: list[WebSocket] = []

async def broadcast(message: dict):
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(message)
        except:
            dead.append(ws)
    for d in dead:
        ws_clients.remove(d)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    await ws.send_json({"type": "init", "state": state})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_clients.remove(ws)

@app.get("/")
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

@app.get("/state")
async def get_state():
    return state

# ═══════════════════════════════════════════════════════
# GÉNÉRATION DE RÉPONSE NATURELLE VIA OLLAMA
# ═══════════════════════════════════════════════════════
def generate_response(text: str, detected: list, placements: list, lang: str) -> str:
    if not detected:
        return "Aucune intention détectée. Veuillez réessayer." if lang == "fr" else "No intention detected. Please try again."

    placement_summary = []
    for i, intent in enumerate(detected):
        p = placements[i] if i < len(placements) else None
        if p and p.get("success"):
            node = p.get("node") or (p.get("nodes", ["?"])[0])
            lat  = p.get("lat", "?")
            placement_summary.append(
                f"- {intent['description']} → deployed on node {node} ({lat}ms)"
            )
        else:
            placement_summary.append(
                f"- {intent['description']} → FAILED (insufficient resources)"
            )

    placements_text = "\n".join(placement_summary)
    response_lang   = "French" if lang == "fr" else "English"

    prompt = f"""You are an IBN assistant for industrial technicians wearing VR glasses.

The technician said: "{text}"

Based on their request, the following services were deployed:
{placements_text}

Write a SHORT response (2-3 sentences max) in {response_lang} that:
1. Acknowledges what the technician asked
2. Confirms which services were deployed and on which nodes
3. Is clear and professional

Response:"""

    try:
        resp = http_requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":  "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 80,
                }
            },
            timeout=20,
        )
        if resp.status_code == 200:
            answer = resp.json().get("response", "").strip()
            answer = answer.replace("\n", " ").strip()
            if answer:
                print(f"   💬 Ollama réponse : {answer[:100]}...")
                return answer
    except Exception as e:
        print(f"   ⚠️  Ollama response error : {e}")

    # Fallback
    if lang == "fr":
        parts = [f"{p.get('id','?')} sur {p.get('node') or p.get('nodes',['?'])[0]}"
                 for p in placements if p.get("success")]
        if parts:
            return f"{len(parts)} service(s) déployé(s) : {', '.join(parts)}."
        return "Impossible de placer les services. Ressources insuffisantes."
    else:
        parts = [f"{p.get('id','?')} on {p.get('node') or p.get('nodes',['?'])[0]}"
                 for p in placements if p.get("success")]
        if parts:
            return f"{len(parts)} service(s) deployed: {', '.join(parts)}."
        return "Cannot place services. Insufficient resources."


# ═══════════════════════════════════════════════════════
# BOUCLE VOCALE (thread séparé)
# ═══════════════════════════════════════════════════════
def voice_loop():
    print("\n" + "═"*55)
    print("  IBN Voice — Placement de services")
    print(f"  Dataset : {len(intentions)} intentions, {len(nodes)} nœuds")
    print(f"  Dashboard : http://localhost:{SERVER_PORT}")
    print("═"*55)
    print("\n💡 Exemples de commandes vocales :")
    print("   'Show me the AR sequence for power unit assembly'")
    print("   'Deploy AR service on the field'")
    print("   'I need voice conversion'")
    print("   'Activate anomaly analysis'")
    print("   'Je veux déployer le service AR'\n")

    loop = asyncio.new_event_loop()

    while True:
        try:
            input("\n⏎  Appuie sur ENTRÉE pour parler...")
            state["listening"] = True

            loop.run_until_complete(broadcast({
                "type": "listening",
                "listening": True
            }))

            print("🎤 Enregistrement... (ENTRÉE pour arrêter)")
            audio = sd.rec(int(60 * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE, channels=1)
            input()
            sd.stop()
            state["listening"] = False

            print("⏳ Transcription...")
            text, lang = transcribe(audio)

            if not text:
                print("❌ Aucun texte détecté")
                continue

            print(f"\n🗣️  [{lang.upper()}] : {text}")
            state["last_text"] = text

            loop.run_until_complete(broadcast({
                "type": "transcript",
                "text": text,
                "lang": lang
            }))

            detected = detect_multiple_intentions(text)

            if not detected:
                msg = "No IBN intention detected. Please try again."
                if lang == "fr":
                    msg = "Aucune intention IBN détectée. Réessayez."
                print(f"\nℹ️  {msg}")
                speak(msg, lang)
                loop.run_until_complete(broadcast({
                    "type": "no_intent", "text": text
                }))
                continue

            print(f"\n🎯 {len(detected)} intention(s) détectée(s) :")
            all_responses = []

            for intent in detected:
                print(f"\n{'─'*45}")
                print(f"   📌 {intent['id']} — {intent['description']}")
                print(f"   Services  : {', '.join(intent['services'])}")
                state["last_intent"] = intent["id"]

                req = total_resources(intent["services"])
                print(f"   Ressources: CPU={req['CPU']} MEM={req['MEM']} BW={req['BW']}Mbps")

                results   = select_node(req, intent["QoS"]["latency"], intent["services"])
                placed    = [r for r in results if r["node"]]
                failed    = [r for r in results if not r["node"]]
                is_grouped= len(results) == 1 and results[0].get("grouped")
                success   = len(placed) > 0
                state["stats"]["total"] += 1

                if success:
                    state["stats"]["success"] += 1
                    apply_placement(intent, results)

                    if is_grouped:
                        node_id = placed[0]["node"]
                        lat     = placed[0]["lat"]
                        state["last_node"] = node_id
                        print(f"   ✅ Groupé → {node_id.upper()} ({lat}ms)")
                        if lang == "fr":
                            r_msg = f"{intent['id']} sur {node_id} ({lat}ms)"
                        else:
                            r_msg = f"{intent['id']} on {node_id} ({lat}ms)"
                    else:
                        parts = [f"{r['service']}→{r['node'].upper()}({r['lat']}ms)"
                                 for r in placed]
                        summary = ", ".join(parts)
                        print(f"   ✅ Distribué : {summary}")
                        if failed:
                            print(f"   ⚠️  Échec partiel : {[r['service'] for r in failed]}")
                        r_msg = summary

                    all_responses.append(r_msg)

                    placement = {
                        "id":      intent["id"],
                        "desc":    intent["description"][:50],
                        "services":intent["services"],
                        "node":    placed[0]["node"] if is_grouped else None,
                        "nodes":   [r["node"] for r in placed],
                        "lat":     placed[0]["lat"],
                        "success": True,
                        "grouped": is_grouped,
                        "time":    time.strftime("%H:%M:%S"),
                        "text":    text,
                    }
                    state["placements"].insert(0, placement)
                    neo4j_write_voice_placement(intent, results, text)

                else:
                    state["stats"]["fail"] += 1
                    print(f"   ❌ Aucun nœud disponible")
                    state["placements"].insert(0, {
                        "id":      intent["id"],
                        "desc":    intent["description"][:50],
                        "services":intent["services"],
                        "node":    None,
                        "success": False,
                        "time":    time.strftime("%H:%M:%S"),
                        "text":    text,
                    })

            state["placements"] = state["placements"][:20]

            response = generate_response(text, detected, state["placements"][:len(detected)], lang)
            print(f"\n🔊 {response}")
            speak(response, lang)

            loop.run_until_complete(broadcast({
                "type":      "placement",
                "intents":   [i["id"] for i in detected],
                "intent":    detected[0]["id"],
                "node":      state["last_node"],
                "lat":       state["placements"][0].get("lat") if state["placements"] else 0,
                "services":  [s for i in detected for s in i["services"]],
                "nodes":     state["nodes"],
                "placements":state["placements"],
                "stats":     state["stats"],
            }))

        except KeyboardInterrupt:
            print("\n\n👋 Arrêt vocal")
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            continue

# ═══════════════════════════════════════════════════════
# DASHBOARD HTML (intégré)
# ═══════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="fr" data-theme="dark">
<head>
<meta charset="UTF-8">
<title>IBN Voice Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@400;500;600&display=swap" rel="stylesheet">
<style>
[data-theme="dark"]  { --bg:#070b12; --s:#111827; --s2:#1a2540; --b:rgba(56,189,248,0.12); --a:#38bdf8; --t:#e2e8f0; --t2:#94a3b8; --t3:#475569; }
[data-theme="light"] { --bg:#f0f4f8; --s:#fff; --s2:#f1f5fb; --b:rgba(14,165,233,0.18); --a:#0284c7; --t:#0f172a; --t2:#334155; --t3:#94a3b8; }
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--t);height:100vh;overflow:hidden;transition:background .3s}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(var(--b) 1px,transparent 1px),linear-gradient(90deg,var(--b) 1px,transparent 1px);background-size:40px 40px;opacity:.5;pointer-events:none}
.topbar{display:flex;align-items:center;gap:12px;padding:10px 24px;background:var(--s);border-bottom:1px solid var(--b);position:relative;z-index:10;flex-shrink:0}
.logo{display:flex;align-items:center;gap:10px}
.logo-icon{width:34px;height:34px;background:linear-gradient(135deg,#0ea5e9,#0369a1);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;box-shadow:0 0 16px rgba(14,165,233,.3)}
.logo-title{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:var(--a)}
.logo-sub{font-size:10px;color:var(--t3)}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:10px}
.mic-badge{display:flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;font-size:10px;padding:5px 12px;border-radius:3px;border:1px solid;transition:all .3s}
.mic-badge.idle{background:rgba(100,116,139,.1);color:var(--t3);border-color:rgba(100,116,139,.2)}
.mic-badge.listening{background:rgba(248,113,113,.12);color:#fca5a5;border-color:rgba(248,113,113,.3);animation:micPulse 1s ease-in-out infinite}
@keyframes micPulse{0%,100%{box-shadow:0 0 0 0 rgba(248,113,113,.3)}50%{box-shadow:0 0 0 6px rgba(248,113,113,0)}}
.theme-btn{background:var(--s2);border:1px solid var(--b);border-radius:6px;padding:6px 10px;cursor:pointer;font-size:14px;transition:all .15s}
.theme-btn:hover{border-color:var(--a)}
.main{display:flex;height:calc(100vh - 53px);gap:0;position:relative;z-index:1}
.left{flex:1;display:flex;flex-direction:column;gap:10px;padding:14px;overflow-y:auto}
.left::-webkit-scrollbar{width:3px}
.left::-webkit-scrollbar-thumb{background:var(--b);border-radius:2px}
.card{background:var(--s);border:1px solid var(--b);border-radius:6px;padding:14px;transition:background .3s,border .3s}
.card-title{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:var(--a);letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:7px}
.card-title::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--a);box-shadow:0 0 6px var(--a);flex-shrink:0}
.transcript-box{background:var(--s2);border:1px solid var(--b);border-radius:4px;padding:12px;font-size:13px;color:var(--t);min-height:48px;font-style:italic;line-height:1.6}
.transcript-box.listening{border-color:rgba(248,113,113,.4);animation:borderPulse 1.5s ease-in-out infinite}
@keyframes borderPulse{0%,100%{border-color:rgba(248,113,113,.2)}50%{border-color:rgba(248,113,113,.6)}}
.intent-node{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px}
.info-box{background:var(--s2);border:1px solid var(--b);border-radius:4px;padding:10px}
.info-label{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--t3);letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px}
.info-val{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:var(--a)}
.info-sub{font-size:10px;color:var(--t3);margin-top:2px}
.kpi-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.kpi{background:var(--s2);border:1px solid var(--b);border-radius:4px;padding:10px}
.kpi-val{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:var(--t)}
.kpi-lbl{font-size:9px;color:var(--t3);letter-spacing:.08em;text-transform:uppercase;margin-top:2px}
.right{width:360px;min-width:300px;display:flex;flex-direction:column;gap:10px;padding:14px;border-left:1px solid var(--b);overflow-y:auto}
.right::-webkit-scrollbar{width:3px}
.right::-webkit-scrollbar-thumb{background:var(--b);border-radius:2px}
.node-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:6px}
.node-card{background:var(--s2);border:1px solid var(--b);border-radius:4px;padding:9px;transition:all .3s;border-left:3px solid var(--b)}
.node-card.gw{border-left-color:#f59e0b}
.node-card.cp{border-left-color:var(--a)}
.node-card.active{border-color:rgba(34,211,238,.4);box-shadow:0 0 10px rgba(34,211,238,.1)}
.node-id{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;color:var(--a);margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}
.node-badge{font-size:8px;padding:1px 5px;border-radius:2px;font-family:'JetBrains Mono',monospace}
.node-badge.gw{background:rgba(245,158,11,.15);color:#f59e0b}
.node-badge.cp{background:rgba(56,189,248,.12);color:var(--a)}
.res-row{display:flex;align-items:center;gap:5px;margin-bottom:3px}
.res-lbl{font-family:'JetBrains Mono',monospace;font-size:8px;color:var(--t3);width:26px;flex-shrink:0}
.bar{flex:1;height:4px;background:var(--s);border-radius:2px;overflow:hidden}
.bar-fill{height:100%;border-radius:2px;transition:width .5s ease,background .3s}
.res-pct{font-family:'JetBrains Mono',monospace;font-size:8px;color:var(--t2);width:24px;text-align:right}
.log-list{display:flex;flex-direction:column;gap:4px;max-height:280px;overflow-y:auto}
.log-list::-webkit-scrollbar{width:3px}
.log-item{display:flex;align-items:flex-start;gap:7px;padding:7px 9px;border-radius:3px;border-left:2px solid;animation:logIn .25s ease both}
@keyframes logIn{from{opacity:0;transform:translateX(-6px)}to{opacity:1;transform:translateX(0)}}
.log-item.ok{background:rgba(34,211,238,.05);border-color:#22d3ee}
.log-item.fail{background:rgba(248,113,113,.05);border-color:#f87171}
.log-icon{font-size:11px;flex-shrink:0}
.log-content{flex:1;min-width:0}
.log-id{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;color:var(--a)}
.log-desc{font-size:11px;color:var(--t2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.log-detail{font-family:'JetBrains Mono',monospace;font-size:9px;color:#22d3ee;margin-top:1px}
.log-fail-txt{font-family:'JetBrains Mono',monospace;font-size:9px;color:#f87171;margin-top:1px}
.log-time{font-family:'JetBrains Mono',monospace;font-size:8px;color:var(--t3);flex-shrink:0}
.log-empty{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--t3);padding:12px;text-align:center}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">
    <div class="logo-icon">🎤</div>
    <div>
      <div class="logo-title">IBN · Voice Dashboard</div>
      <div class="logo-sub">Placement temps réel · Whisper · Dataset 4</div>
    </div>
  </div>
  <div class="topbar-right">
    <div class="mic-badge idle" id="micBadge">⬤ IDLE</div>
    <button class="theme-btn" onclick="toggleTheme()">🌙</button>
  </div>
</div>

<div class="main">

  <div class="left">

    <div class="card">
      <div class="card-title">Reconnaissance Vocale</div>
      <div class="transcript-box" id="transcriptBox">
        En attente de commande vocale...
      </div>
      <div class="intent-node" id="intentNode" style="display:none">
        <div class="info-box">
          <div class="info-label">Intention détectée</div>
          <div class="info-val" id="intentVal">—</div>
          <div class="info-sub" id="intentDesc">—</div>
        </div>
        <div class="info-box">
          <div class="info-label">Nœud sélectionné</div>
          <div class="info-val" id="nodeVal">—</div>
          <div class="info-sub" id="nodeLatency">—</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Statistiques</div>
      <div class="kpi-row">
        <div class="kpi">
          <div class="kpi-val" id="kpiTotal" style="color:#7dd3fc">0</div>
          <div class="kpi-lbl">Total</div>
        </div>
        <div class="kpi">
          <div class="kpi-val" id="kpiSuccess" style="color:#22d3ee">0</div>
          <div class="kpi-lbl">Succès</div>
        </div>
        <div class="kpi">
          <div class="kpi-val" id="kpiFail" style="color:#f87171">0</div>
          <div class="kpi-lbl">Échecs</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">État des Nœuds en Temps Réel</div>
      <div class="node-grid" id="nodeGrid"></div>
    </div>

  </div>

  <div class="right">

    <div class="card" style="flex:1">
      <div class="card-title">Journal de Placement</div>
      <div class="log-list" id="logList">
        <div class="log-empty">Aucun placement — parle pour commencer</div>
      </div>
    </div>

  </div>

</div>

<script>
const ws = new WebSocket('ws://localhost:""" + str(SERVER_PORT) + """/ws');

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'init') {
    renderNodes(msg.state.nodes);
    renderLog(msg.state.placements);
    updateStats(msg.state.stats);
  }
  if (msg.type === 'listening') {
    setListening(true);
  }
  if (msg.type === 'transcript') {
    setListening(false);
    document.getElementById('transcriptBox').textContent = msg.text;
    document.getElementById('transcriptBox').classList.remove('listening');
  }
  if (msg.type === 'placement') {
    document.getElementById('intentNode').style.display = 'grid';
    document.getElementById('intentVal').textContent    = msg.intent;
    document.getElementById('nodeVal').textContent      = msg.node.toUpperCase();
    document.getElementById('nodeLatency').textContent  = `${msg.lat} ms`;
    document.getElementById('intentDesc').textContent   = msg.services.join(', ');
    renderNodes(msg.nodes);
    renderLog(msg.placements);
    updateStats(msg.stats);
  }
  if (msg.type === 'placement_failed') {
    document.getElementById('nodeVal').textContent     = 'ÉCHEC';
    document.getElementById('nodeLatency').textContent = 'Aucun nœud disponible';
    renderNodes(msg.nodes);
    updateStats(msg.stats);
  }
  if (msg.type === 'no_intent') {
    document.getElementById('transcriptBox').textContent =
      `❓ "${msg.text}" — Aucune intention IBN détectée`;
  }
};

function setListening(v) {
  const badge = document.getElementById('micBadge');
  const box   = document.getElementById('transcriptBox');
  if (v) {
    badge.className = 'mic-badge listening';
    badge.textContent = '🎤 LISTENING';
    box.textContent = '🎤 Écoute en cours...';
    box.classList.add('listening');
  } else {
    badge.className = 'mic-badge idle';
    badge.textContent = '⬤ IDLE';
    box.classList.remove('listening');
  }
}

function barColor(pct) {
  if (pct >= 80) return '#f87171';
  if (pct >= 50) return '#f59e0b';
  return '#22d3ee';
}

function pct(used, cap) {
  return cap > 0 ? Math.min(100, Math.round(used / cap * 100)) : 0;
}

function renderNodes(nodes) {
  const grid = document.getElementById('nodeGrid');
  grid.innerHTML = nodes.map(n => {
    const cpu  = pct(n.cpu_used, n.cpu);
    const mem  = pct(n.mem_used, n.mem);
    const disk = pct(n.disk_used, n.disk);
    const bw   = pct(n.bw_used,  n.bw);
    const isGW = n.type === 'gateway';
    const latColor = n.lat > 70 ? '#f87171' : n.lat > 50 ? '#f59e0b' : '#22d3ee';
    return `
    <div class="node-card ${isGW?'gw':'cp'} ${n.active?'active':''}">
      <div class="node-id">
        ${n.id.toUpperCase()}
        <span class="node-badge ${isGW?'gw':'cp'}">${isGW?'GW':'CP'}</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-bottom:6px">
        <div style="font-family:'JetBrains Mono',monospace;font-size:8px;background:var(--s);border-radius:3px;padding:3px 5px">
          <div style="color:var(--t3);font-size:7px">CPU</div>
          <div style="color:#38bdf8;font-weight:700">${n.cpu_used}/${n.cpu}</div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:8px;background:var(--s);border-radius:3px;padding:3px 5px">
          <div style="color:var(--t3);font-size:7px">MEM</div>
          <div style="color:#818cf8;font-weight:700">${n.mem_used}/${n.mem}G</div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:8px;background:var(--s);border-radius:3px;padding:3px 5px">
          <div style="color:var(--t3);font-size:7px">DISK</div>
          <div style="color:#34d399;font-weight:700">${n.disk_used}/${n.disk}G</div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:8px;background:var(--s);border-radius:3px;padding:3px 5px">
          <div style="color:var(--t3);font-size:7px">BW</div>
          <div style="color:#f59e0b;font-weight:700">${n.bw_used}/${n.bw}M</div>
        </div>
      </div>
      ${[['CPU',cpu,'#38bdf8'],['MEM',mem,'#818cf8'],['BW',bw,'#f59e0b']].map(([l,v,c])=>`
      <div class="res-row">
        <span class="res-lbl">${l}</span>
        <div class="bar"><div class="bar-fill" style="width:${v}%;background:${barColor(v)}"></div></div>
        <span class="res-pct">${v}%</span>
      </div>`).join('')}
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;margin-top:5px;display:flex;justify-content:space-between;align-items:center">
        <span style="color:${latColor}">⏱ ${n.lat}ms</span>
        <span style="color:${n.active?'#22d3ee':'var(--t3)'}">
          ${n.intents.length} intent${n.intents.length!==1?'s':''}
        </span>
      </div>
    </div>`;
  }).join('');
}

function renderLog(placements) {
  const list = document.getElementById('logList');
  if (!placements || placements.length === 0) {
    list.innerHTML = '<div class="log-empty">Aucun placement — parle pour commencer</div>';
    return;
  }
  list.innerHTML = placements.map(p => `
    <div class="log-item ${p.success?'ok':'fail'}">
      <div class="log-icon">${p.success?'✅':'❌'}</div>
      <div class="log-content">
        <div class="log-id">${p.id}</div>
        <div class="log-desc">${p.desc}</div>
        ${p.success
          ? `<div class="log-detail">→ ${p.node ? p.node.toUpperCase() : (p.nodes&&p.nodes[0]?p.nodes[0].toUpperCase():'?')}  (${p.lat}ms)</div>`
          : `<div class="log-fail-txt">ÉCHEC — QoS non satisfait</div>`}
      </div>
      <div class="log-time">${p.time}</div>
    </div>`).join('');
}

function updateStats(stats) {
  document.getElementById('kpiTotal').textContent   = stats.total;
  document.getElementById('kpiSuccess').textContent = stats.success;
  document.getElementById('kpiFail').textContent    = stats.fail;
}

function toggleTheme() {
  const html = document.documentElement;
  const t    = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', t);
  document.querySelector('.theme-btn').textContent = t === 'dark' ? '🌙' : '☀️';
}
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    voice_thread = threading.Thread(target=voice_loop, daemon=True)
    voice_thread.start()

    print(f"\n🌐 Dashboard : http://localhost:{SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")