import ollama
import json
from ibn_engine import DATA

def build_system_prompt():
    nodes_txt = "\n".join(
        f"- {n['id']} ({n['type']}): CPU={n['capacity']['CPU']}, "
        f"MEM={n['capacity']['MEM']}GB, BW={n['capacity']['BW']}Mbps"
        for n in DATA["nodes"]
    )
    services_txt = "\n".join(
        f"- {s['id']} {s['name']}: CPU={s['resources']['CPU']}, "
        f"MEM={s['resources']['MEM']}GB, BW={s['resources']['BW']}Mbps"
        for s in DATA["services"]
    )
    intentions_txt = "\n".join(
        f"- {i['id']}: \"{i['description']}\" → services {i['services']}, "
        f"latency≤{i['QoS']['latency']}ms, weight={i['weight']}"
        for i in DATA["intentions"]
    )
    return f"""Tu es un assistant expert en Edge Computing et Intent-Based Networking.

NŒUDS:
{nodes_txt}

SERVICES:
{services_txt}

INTENTIONS:
{intentions_txt}

Réponds en français ou anglais selon la langue du technicien.
Sois précis, cite les IDs, les valeurs de latence et ressources exactes."""

SYSTEM_PROMPT = build_system_prompt()

class Chatbot:
    def __init__(self, model="llama3"):
        self.model = model
        self.history = []
        # Injecter le system prompt comme premier message système
        self.history_with_system = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def ask(self, user_message: str) -> str:
        # Ajouter le message user
        self.history.append({
            "role": "user",
            "content": user_message
        })

        # Construire messages = system + historique
        messages = self.history_with_system + self.history

        # Appel Ollama
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.3}
        )

        reply = response["message"]["content"]

        # Sauvegarder la réponse
        self.history.append({
            "role": "assistant",
            "content": reply
        })

        return reply

    def reset(self):
        self.history = []
        print("Historique effacé.\n")


if __name__ == "__main__":
    # Choisir le modèle Ollama disponible
    model = "llama3"   # ou "mistral", "phi3", "gemma", etc.

    bot = Chatbot(model=model)
    print(f"=== Chatbot IBN démarré (modèle: {model}) ===")
    print("Commandes : 'quit' pour quitter, 'reset' pour effacer l'historique\n")

    while True:
        question = input("Technicien : ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "reset":
            bot.reset()
            continue

        print("⏳ Réflexion...")
        reponse = bot.ask(question)
        print(f"\nAssistant : {reponse}\n")