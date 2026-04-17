import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
import ollama
import json
import warnings

warnings.filterwarnings("ignore")

# =====================
# Charger dataset
# =====================

with open("dataset_3 .json", "r", encoding="utf-8") as f:
    data = json.load(f)

intentions = data["intentions"]
services = data["services"]
nodes = data["nodes"]
latency_map = data["latency"]

print("Chargement du modèle Whisper...")
model = whisper.load_model("base")

fs = 16000

# =====================
# Extraire JSON si l'IA ajoute du texte
# =====================

def extract_json(text):

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1:
        return text[start:end+1]

    return None

# =====================
# IA : détecter plusieurs intentions
# =====================

def detect_multiple_intentions_with_ai(text):

    intentions_list = "\n".join(
        [f"{i['id']} : {i['description']}" for i in intentions]
    )

    prompt = f"""
You are an AI system that detects tasks and maps them to intentions.

The user text may contain multiple tasks.

Your job:
1. Split the text into tasks
2. Map each task to ONE intention ID
3. Different tasks may have different intentions
4. Return ONLY JSON

Intentions:
{intentions_list}

Return format:

[
  {{"task":"...","intent":"i1"}},
  {{"task":"...","intent":"i2"}}
]

User text:
{text}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature":0}
    )

    answer = response["message"]["content"]

    print("\nRéponse brute IA :", answer)

    json_text = extract_json(answer)

    if json_text:

        try:
            result = json.loads(json_text)
            return result
        except:
            print("Erreur parsing JSON")

    return []

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
# Choix node optimal
# =====================

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

print("\n🎤 Speak in English or French")

while True:

    input("\nPress ENTER to start speaking...")
    print("Recording...")

    audio = sd.rec(int(60 * fs), samplerate=fs, channels=1)

    input("Press ENTER to stop recording")

    sd.stop()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    write(temp.name, fs, audio)

    result = model.transcribe(temp.name)

    text = result["text"].strip()

    if text:

        print("\n🗣️ Detected text :", text)

        tasks = detect_multiple_intentions_with_ai(text)

        if tasks:

            for t in tasks:

                task = t["task"]
                intent_id = t["intent"]

                print("\n-----------------------------")
                print("📌 Task :", task)
                print("🎯 Intention :", intent_id)

                intent = next((i for i in intentions if i["id"] == intent_id), None)

                if not intent:

                    print("❌ Unknown intention")
                    continue

                required = total_resources(intent["services"])

                print("⚙️ Required resources :", required)

                qos_latency = intent["QoS"]["latency"]

                node, lat = select_node_with_latency(required, qos_latency)

                if node:

                    print(f"🖥️ Selected node : {node} (latency {lat:.1f} ms)")

                else:

                    print("❌ No node satisfies QoS")

        else:

            print("❌ No intention detected")

    try:
        os.remove(temp.name)
    except:
        pass