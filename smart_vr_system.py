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
with open("dataset_3 .json", "r", encoding="utf-8") as f:
    data = json.load(f)

intentions = data["intentions"]
services = data["services"]
nodes = data["nodes"]
latency_map = data["latency"]

print("Chargement Whisper...")
model = whisper.load_model("base")

fs = 16000

# =====================
# IA intention
# =====================
def detect_intention_with_ai(text):

    intentions_list = "\n".join(
        [f"{i['id']} : {i['description']}" for i in intentions]
    )

    prompt = f"""
Tu es un système de classification.

Choisis l'intention EXACTE parmi cette liste.
Ne devine pas.
Retourne seulement l'ID.

INTENTIONS :
{intentions_list}

Texte utilisateur :
{text}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"].strip().lower()

    for intent in intentions:
        if intent["id"].lower() in answer:
            return intent

    return None


def total_resources(service_ids):
    total = {"CPU": 0, "MEM": 0, "DISK": 0, "BW": 0}
    for s in services:
        if s["id"] in service_ids:
            for r in total:
                total[r] += s["resources"].get(r, 0)
    return total


def select_node_with_latency(required, qos_latency):
    best_node = None
    best_latency = 9999

    for node in nodes:
        cap = node["capacity"]

        if not (
            cap["CPU"] >= required["CPU"]
            and cap["MEM"] >= required["MEM"]
            and cap["DISK"] >= required["DISK"]
            and cap["BW"] >= required["BW"]
        ):
            continue

        node_id = node["id"]
        lat_list = latency_map.get(node_id, [])

        if not lat_list:
            continue

        avg_latency = sum(lat_list) / len(lat_list)

        if avg_latency <= qos_latency and avg_latency < best_latency:
            best_latency = avg_latency
            best_node = node_id

    return best_node, best_latency


# =====================
# ENREGISTREMENT CONTINU
# =====================
print("🎤 Parle... appuie sur ENTER quand tu as fini")

while True:
    input("Appuie sur ENTER pour commencer à parler...")
    print("Enregistrement... parle maintenant")
    
    audio = sd.rec(int(60 * fs), samplerate=fs, channels=1)
    input("Appuie sur ENTER pour arrêter")
    sd.stop()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp.name, fs, audio)

    result = model.transcribe(temp.name, language="fr")
    text = result["text"].strip()

    if text:
        print("\n🗣️ Texte détecté :", text)

        intent = detect_intention_with_ai(text)

        if intent:
            print("🎯 Intention :", intent["description"])

            required = total_resources(intent["services"])
            print("⚙️ Ressources nécessaires :", required)

            qos_latency = intent["QoS"]["latency"]

            node, lat = select_node_with_latency(required, qos_latency)

            if node:
                print(f"🖥️ Nœud choisi : {node} (latence {lat:.1f} ms)")
            else:
                print("❌ Aucun nœud respecte la latence")

        else:
            print("❌ Intention non trouvée")

    try:
        os.remove(temp.name)
    except:
        pass