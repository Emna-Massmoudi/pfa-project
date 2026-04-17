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
import pyttsx3
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
# ÉTAT GLOBAL — partagé entre vocal et dashboard
# ═══════════════════════════════════════════════════════
state = {
    "nodes":      [],          # état actuel des nœuds
    "placements": [],          # historique des placements
    "listening":  False,       # micro actif ?
    "last_text":  "",          # dernière transcription
    "last_intent":"",          # dernière intention détectée
    "last_node":  "",          # dernier nœud sélectionné
    "stats": {
        "total": 0,
        "success": 0,
        "fail": 0
    }
}

# Initialiser l'état des nœuds
def init_nodes():
    state["nodes"] = []
    for nd in nodes:
        lats = latency_map.get(nd["id"], [50])
        avg_lat = round(sum(lats) / len(lats), 1)
        state["nodes"].append({
            "id":       nd["id"],
            "type":     nd["type"],
            "cpu":      nd["capacity"]["CPU"],
            "mem":      nd["capacity"]["MEM"],
            "disk":     nd["capacity"]["DISK"],
            "bw":       nd["capacity"]["BW"],
            "cpu_used": 0,
            "mem_used": 0,
            "disk_used":0,
            "bw_used":  0,
            "lat":      avg_lat,
            "intents":  [],
            "active":   False,
        })

init_nodes()

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
    """Retourne les ressources disponibles d'un nœud."""
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
    """Trouve le meilleur nœud pour des ressources données."""
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
    """
    Algorithme de placement hybride :
    Étape 1 : Essayer de placer TOUS les services sur UN seul nœud
    Étape 2 : Si impossible → distribuer chaque service sur le meilleur nœud
    Retourne : liste de {service, node, lat} ou None si échec total
    """
    # ── Étape 1 : placement groupé ──
    node_id, lat = best_node_for(required, qos_latency)
    if node_id:
        return [{"service": "ALL", "node": node_id, "lat": lat, "grouped": True}]

    # ── Étape 2 : placement distribué service par service ──
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
            # Réserver les ressources immédiatement pour les prochains services
            apply_service_to_node(nd, svc_req)
        else:
            results.append({"service": svc_id, "node": None, "lat": None, "grouped": False})

    return results if results else None

def apply_service_to_node(node_id, req):
    """Réserve les ressources d'un service sur un nœud."""
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
    """Met à jour l'état des nœuds après placement distribué."""
    import random
    for r in results:
        if not r["node"]:
            continue
        # Si groupé, appliquer toutes les ressources de l'intention
        if r.get("grouped"):
            req = total_resources(intent["services"])
            apply_service_to_node(r["node"], req)
        # Sinon déjà appliqué dans select_node
        for nd in state["nodes"]:
            if nd["id"] == r["node"]:
                if intent["id"] not in nd["intents"]:
                    nd["intents"].append(intent["id"])
                break

def detect_intention(text: str):
    """Détecte UNE intention IBN (compatibilité)."""
    results = detect_multiple_intentions(text)
    return results[0] if results else None

def detect_multiple_intentions(text: str):
    """
    Détecte PLUSIEURS intentions IBN dans un paragraphe.
    Retourne une liste triée par score décroissant.
    """
    text_lower = text.lower()
    text_words = set(text_lower.split())

    # Mots-clés des services
    service_keywords = {
        "s1": ["voice", "vocal", "speak", "audio", "parole", "voix", "conversion"],
        "s2": ["protocol", "convert", "protocole", "converter"],
        "s3": ["ar", "augmented", "reality", "visually", "visual", "inspect",
               "afficher", "lunettes", "glasses", "overlay"],
        "s4": ["content", "contenu", "show", "affichage", "display", "displayer"],
        "s5": ["error", "erreur", "detect", "fault", "detector", "defect"],
        "s6": ["analyze", "analyser", "anomaly", "anomalie", "diagnos",
               "analyzer", "vibration", "current", "motor"],
    }

    # Mots-clés des intentions directement
    intent_keywords = {
        "overheating": ["overheat", "surchauffe", "temperature", "hot", "chaud", "noise", "bruit"],
        "ar":          ["ar", "augmented", "visual", "inspect", "visually"],
        "anomaly":     ["anomaly", "anomalie", "analyzer", "annamal", "annamali", "detect"],
        "error":       ["error", "erreur", "fault", "detector", "identify", "problem"],
        "voice":       ["voice", "vocal", "conversion", "speak"],
        "maintenance": ["maintenance", "repair", "fix", "broken"],
        "diagnostic":  ["diagnos", "full", "complete", "entire"],
    }

    scored = []

    for intent in intentions:
        score = 0
        desc_words = set(intent["description"].lower().split())

        # Score 1 — mots en commun avec la description
        score += len(desc_words & text_words) * 2

        # Score 2 — mots-clés des services
        for svc_id in intent["services"]:
            kws = service_keywords.get(svc_id, [])
            for kw in kws:
                if kw in text_lower:
                    score += 3

        # Score 3 — mots-clés directs
        for group, kws in intent_keywords.items():
            for kw in kws:
                if kw in text_lower:
                    score += 1

        if score >= 3:
            scored.append((score, intent))

    # Trier par score décroissant
    scored.sort(key=lambda x: x[0], reverse=True)

    # Éviter les doublons de services
    selected  = []
    used_svcs = set()

    for score, intent in scored:
        intent_svcs = set(intent["services"])
        # Accepter si au moins un service nouveau
        new_svcs = intent_svcs - used_svcs
        if new_svcs:
            selected.append(intent)
            used_svcs.update(intent_svcs)
        if len(selected) >= 4:  # max 4 intentions par paragraphe
            break

    return selected

# ═══════════════════════════════════════════════════════
# WHISPER
# ═══════════════════════════════════════════════════════
print(f"⏳ Chargement Whisper '{WHISPER_MODEL}'...")
whisper_model = whisper.load_model(WHISPER_MODEL)
print("✅ Whisper prêt")

def transcribe(audio_array) -> tuple[str, str]:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp.name, SAMPLE_RATE, audio_array)
    result = whisper_model.transcribe(temp.name, fp16=False)
    try:
        os.remove(temp.name)
    except:
        pass
    return result["text"].strip(), result.get("language", "en")

# ═══════════════════════════════════════════════════════
# TTS
# ═══════════════════════════════════════════════════════
tts = pyttsx3.init()
tts.setProperty("rate", 155)
tts.setProperty("volume", 1.0)

def speak(text: str, lang: str = "en"):
    short = text[:350]
    voices = tts.getProperty("voices")
    for v in voices:
        name = v.name.lower()
        if lang == "fr" and ("french" in name or "fr_" in name):
            tts.setProperty("voice", v.id)
            break
        elif lang == "en" and "english" in name:
            tts.setProperty("voice", v.id)
            break
    tts.say(short)
    tts.runAndWait()

# ═══════════════════════════════════════════════════════
# FASTAPI + WEBSOCKET DASHBOARD
# ═══════════════════════════════════════════════════════
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Liste des clients WebSocket connectés
ws_clients: list[WebSocket] = []

async def broadcast(message: dict):
    """Envoyer une mise à jour à tous les clients dashboard."""
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
    # Envoyer l'état initial
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
# BOUCLE VOCALE (thread séparé)
# ═══════════════════════════════════════════════════════
def voice_loop():
    """Boucle principale d'écoute vocale."""
    print("\n" + "═"*55)
    print("  IBN Voice — Placement de services")
    print(f"  Dataset : {len(intentions)} intentions, {len(nodes)} nœuds")
    print(f"  Dashboard : http://localhost:{SERVER_PORT}")
    print("═"*55)
    print("\n💡 Exemples de commandes vocales :")
    print("   'Deploy AR service on the field'")
    print("   'I need voice conversion'")
    print("   'Activate anomaly analysis'")
    print("   'Je veux déployer le service AR'\n")

    loop = asyncio.new_event_loop()

    while True:
        try:
            input("\n⏎  Appuie sur ENTRÉE pour parler...")
            state["listening"] = True

            # Notifier le dashboard
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

            # Transcription
            print("⏳ Transcription...")
            text, lang = transcribe(audio)

            if not text:
                print("❌ Aucun texte détecté")
                continue

            print(f"\n🗣️  [{lang.upper()}] : {text}")
            state["last_text"] = text

            # Notifier dashboard
            loop.run_until_complete(broadcast({
                "type": "transcript",
                "text": text,
                "lang": lang
            }))

            # ── Détecter PLUSIEURS intentions ──
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

                # Calcul ressources
                req = total_resources(intent["services"])
                print(f"   Ressources: CPU={req['CPU']} MEM={req['MEM']} BW={req['BW']}Mbps")

                # Placement hybride
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

            # ── Réponse vocale résumée ──
            if all_responses:
                if lang == "fr":
                    response = f"{len(all_responses)} service(s) placé(s) : {'. '.join(all_responses)}."
                else:
                    response = f"{len(all_responses)} service(s) placed: {'. '.join(all_responses)}."
            else:
                if lang == "fr":
                    response = "Impossible de placer les services. Ressources insuffisantes."
                else:
                    response = "Cannot place services. Insufficient resources."

            print(f"\n🔊 {response}")
            speak(response, lang)

            # ── Mise à jour dashboard ──
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

/* TOPBAR */
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

/* LAYOUT */
.main{display:flex;height:calc(100vh - 53px);gap:0;position:relative;z-index:1}

/* LEFT */
.left{flex:1;display:flex;flex-direction:column;gap:10px;padding:14px;overflow-y:auto}
.left::-webkit-scrollbar{width:3px}
.left::-webkit-scrollbar-thumb{background:var(--b);border-radius:2px}

/* CARDS */
.card{background:var(--s);border:1px solid var(--b);border-radius:6px;padding:14px;transition:background .3s,border .3s}
.card-title{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:var(--a);letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:7px}
.card-title::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--a);box-shadow:0 0 6px var(--a);flex-shrink:0}

/* TRANSCRIPT */
.transcript-box{background:var(--s2);border:1px solid var(--b);border-radius:4px;padding:12px;font-size:13px;color:var(--t);min-height:48px;font-style:italic;line-height:1.6}
.transcript-box.listening{border-color:rgba(248,113,113,.4);animation:borderPulse 1.5s ease-in-out infinite}
@keyframes borderPulse{0%,100%{border-color:rgba(248,113,113,.2)}50%{border-color:rgba(248,113,113,.6)}}

/* INTENT + NODE */
.intent-node{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px}
.info-box{background:var(--s2);border:1px solid var(--b);border-radius:4px;padding:10px}
.info-label{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--t3);letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px}
.info-val{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:var(--a)}
.info-sub{font-size:10px;color:var(--t3);margin-top:2px}

/* KPI */
.kpi-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.kpi{background:var(--s2);border:1px solid var(--b);border-radius:4px;padding:10px}
.kpi-val{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:var(--t)}
.kpi-lbl{font-size:9px;color:var(--t3);letter-spacing:.08em;text-transform:uppercase;margin-top:2px}

/* RIGHT */
.right{width:360px;min-width:300px;display:flex;flex-direction:column;gap:10px;padding:14px;border-left:1px solid var(--b);overflow-y:auto}
.right::-webkit-scrollbar{width:3px}
.right::-webkit-scrollbar-thumb{background:var(--b);border-radius:2px}

/* NODE GRID */
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

/* LOG */
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

  <!-- LEFT -->
  <div class="left">

    <!-- Transcript -->
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

    <!-- KPIs -->
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

    <!-- Nœuds -->
    <div class="card">
      <div class="card-title">État des Nœuds en Temps Réel</div>
      <div class="node-grid" id="nodeGrid"></div>
    </div>

  </div>

  <!-- RIGHT -->
  <div class="right">

    <!-- Log placements -->
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
    const cpu  = pct(n.cpu_used,  n.cpu);
    const mem  = pct(n.mem_used,  n.mem);
    const disk = pct(n.disk_used, n.disk);
    const bw   = pct(n.bw_used,   n.bw);
    const isGW = n.type === 'gateway';
    return `
    <div class="node-card ${isGW?'gw':'cp'} ${n.active?'active':''}">
      <div class="node-id">
        ${n.id.toUpperCase()}
        <span class="node-badge ${isGW?'gw':'cp'}">${isGW?'GW':'CP'}</span>
      </div>
      ${[['CPU',cpu],['MEM',mem],['DISK',disk],['BW',bw]].map(([l,v])=>`
      <div class="res-row">
        <span class="res-lbl">${l}</span>
        <div class="bar"><div class="bar-fill" style="width:${v}%;background:${barColor(v)}"></div></div>
        <span class="res-pct">${v}%</span>
      </div>`).join('')}
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:${n.lat>70?'#f87171':n.lat>50?'#f59e0b':'#22d3ee'};margin-top:5px">
        lat ${n.lat}ms · ${n.intents.length} intent${n.intents.length!==1?'s':''}
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
          ? `<div class="log-detail">→ ${p.node.toUpperCase()}  (${p.lat}ms)</div>`
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
# MAIN — Lancer serveur + vocal en parallèle
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":

    # Thread vocal
    voice_thread = threading.Thread(target=voice_loop, daemon=True)
    voice_thread.start()

    # Serveur FastAPI (bloquant)
    print(f"\n🌐 Dashboard : http://localhost:{SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")