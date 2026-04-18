import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
import ollama
import json

# =====================
# Charger dataset
# =====================
with open("dataset_4.json", "r", encoding="utf-8") as f:
    data = json.load(f)

intentions  = data["intentions"]
services    = data["services"]
nodes       = data["nodes"]
latency_map = data["latency"]

print("Chargement Whisper...")
model = whisper.load_model("base")

fs = 16000

# =====================
# Détection intention via Ollama
# =====================
def detect_intention_with_ai(text):
    intentions_list = "\n".join(
        [f"{i['id']} : {i['description']}" for i in intentions]
    )

    prompt = f"""TASK: Match user text to intentions.
RULES:
- Return MAXIMUM 3 IDs
- Only IDs that DIRECTLY match the user request
- Format: i5,i10 (IDs only, comma separated, nothing else)

INTENTIONS:
{intentions_list}

USER: {text}

ANSWER (max 3 IDs):"""

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 20}  # ← limite la réponse à 20 tokens
    )

    answer = response["message"]["content"].strip().lower()
    print(f"   Ollama → {answer}")

    # Extraire uniquement les IDs qui apparaissent en premier
    import re
    found_ids = re.findall(r'i\d+', answer)
    found_ids = found_ids[:3]  # max 3

    detected = []
    for iid in found_ids:
        intent = next((i for i in intentions if i["id"].lower() == iid), None)
        if intent:
            detected.append(intent)

    return detected if detected else None

# =====================
# Calcul ressources
# =====================
def total_resources(service_ids):
    total = {"CPU": 0, "MEM": 0, "DISK": 0, "BW": 0}
    for s in services:
        if s["id"] in service_ids:
            for r in total:
                total[r] += s["resources"].get(r, 0)
    return total


# =====================
# Sélection nœud optimal
# =====================
def select_node_with_latency(required, qos_latency):
    best_node    = None
    best_latency = 9999

    for node in nodes:
        cap = node["capacity"]

        if not (
            cap["CPU"]  >= required["CPU"]  and
            cap["MEM"]  >= required["MEM"]  and
            cap["DISK"] >= required["DISK"] and
            cap["BW"]   >= required["BW"]
        ):
            continue

        node_id  = node["id"]
        lat_list = latency_map.get(node_id, [])

        if not lat_list:
            continue

        avg_latency = sum(lat_list) / len(lat_list)

        if avg_latency <= qos_latency and avg_latency < best_latency:
            best_latency = avg_latency
            best_node    = node_id

    return best_node, best_latency


# =====================
# BOUCLE PRINCIPALE
# =====================
print("🎤 Parle... appuie sur ENTER quand tu as fini")

while True:
    input("\nAppuie sur ENTER pour commencer à parler...")
    print("Enregistrement... parle maintenant")

    audio = sd.rec(int(60 * fs), samplerate=fs, channels=1)
    input("Appuie sur ENTER pour arrêter")
    sd.stop()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp.name, fs, audio)

    result = model.transcribe(temp.name)
    text   = result["text"].strip()

    if text:
        print(f"\n🗣️ Texte détecté : {text}")

        intents = detect_intention_with_ai(text)

        if intents:
            print(f"\n🎯 {len(intents)} intention(s) détectée(s) :")
            for intent in intents:
                print(f"\n   📌 {intent['id']} — {intent['description']}")
                print(f"   Services : {', '.join(intent['services'])}")

                required = total_resources(intent["services"])
                print(f"   Ressources : CPU={required['CPU']} MEM={required['MEM']} BW={required['BW']}Mbps")

                node, lat = select_node_with_latency(required, intent["QoS"]["latency"])

                if node:
                    print(f"   ✅ Nœud : {node.upper()} ({lat:.1f} ms)")
                else:
                    print(f"   ❌ Aucun nœud disponible")
        else:
            print("❌ Intention non trouvée")

    try:
        os.remove(temp.name)
    except:
        pass