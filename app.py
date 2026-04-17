"""
app.py — Chatbot IBN Technicien (Ollama — gratuit, local, sans clé API)
Lancer : python app.py
Ouvrir : http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template_string
import ollama

app = Flask(__name__)

# ─────────────────────────────────────────
# CONFIGURATION — change le modèle ici
# ─────────────────────────────────────────
MODEL   = "llama3.2:1b"   # ou "tinyllama", "gemma2:2b", "llama3"
history = []

# ─────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un assistant expert en Edge Computing et Intent-Based Networking (IBN) pour un système industriel.

NŒUDS:
- g1 (gateway)  : CPU=2,  MEM=4GB,  DISK=2GB,  BW=100Mbps, latence≈42ms
- g2 (gateway)  : CPU=2,  MEM=3GB,  DISK=1GB,  BW=80Mbps,  latence≈50ms
- n1 (computing): CPU=8,  MEM=16GB, DISK=15GB, BW=150Mbps, latence≈57ms
- n2 (computing): CPU=12, MEM=24GB, DISK=25GB, BW=250Mbps, latence≈62ms
- n3 (computing): CPU=16, MEM=32GB, DISK=30GB, BW=300Mbps, latence≈72ms

SERVICES:
- s1 Voice_Converter    : CPU=1,  MEM=2GB,  BW=60Mbps
- s2 Protocol_Converter : CPU=2,  MEM=4GB,  BW=80Mbps
- s3 AR_Generator       : CPU=6,  MEM=12GB, BW=120Mbps
- s4 Content_Displayer  : CPU=2,  MEM=6GB,  BW=100Mbps
- s5 Error_Detector     : CPU=3,  MEM=8GB,  BW=40Mbps
- s6 Anomaly_Analyzer   : CPU=5,  MEM=10GB, BW=90Mbps

INTENTIONS:
- i1 → [s1]          QoS: lat≤60ms,  poids=5
- i2 → [s2,s3]       QoS: lat≤120ms, poids=8
- i3 → [s4,s5]       QoS: lat≤130ms, poids=7
- i4 → [s5,s6]       QoS: lat≤150ms, poids=9
- i5 → [s1,s2,s3,s4] QoS: lat≤200ms, poids=10
- i6 → [tous]        QoS: lat≤250ms, poids=12

Réponds en français ou anglais. Sois précis, cite les IDs et valeurs exactes. Réponds en 5 lignes max."""

# ─────────────────────────────────────────
# INTERFACE HTML
# ─────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IBN Assistant — Technicien</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      background: #f0f4f8;
      font-family: 'Segoe UI', Arial, sans-serif;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    #header {
      background: #1e3a5f;
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    #header .title   { color:#fff; font-size:17px; font-weight:700; letter-spacing:2px; }
    #header .subtitle{ color:#93c5fd; font-size:10px; letter-spacing:1px; margin-top:3px; }
    #status {
      display:flex; align-items:center; gap:7px;
      background:rgba(255,255,255,0.1);
      padding:6px 14px; border-radius:20px;
    }
    #status-dot {
      width:8px; height:8px; border-radius:50%;
      background:#22c55e; box-shadow:0 0 8px #22c55e;
      animation:pulse 2s infinite;
    }
    #status span { color:#86efac; font-size:11px; font-weight:600; }

    #nodes-bar {
      background:#fff; border-bottom:1px solid #e2e8f0;
      padding:8px 20px; display:flex; gap:10px; flex-wrap:wrap;
    }
    .node-chip {
      background:#f8fafc; border:1px solid #e2e8f0;
      border-radius:6px; padding:4px 10px; font-size:11px;
      display:flex; align-items:center; gap:5px;
    }
    .node-chip .nid { font-weight:700; }
    .node-chip.gateway .nid   { color:#ea580c; }
    .node-chip.computing .nid { color:#2563eb; }
    .node-chip .ninfo { color:#94a3b8; font-size:10px; }

    #messages {
      flex:1; overflow-y:auto; padding:20px;
      display:flex; flex-direction:column; gap:16px;
    }
    .msg { display:flex; gap:10px; align-items:flex-start; animation:fadeIn 0.3s ease; }
    .msg.user { flex-direction:row-reverse; }

    .avatar {
      width:36px; height:36px; border-radius:10px;
      display:flex; align-items:center; justify-content:center;
      font-size:16px; flex-shrink:0;
    }
    .avatar.bot  { background:linear-gradient(135deg,#1e3a5f,#2563eb); box-shadow:0 2px 8px rgba(37,99,235,0.3); }
    .avatar.user { background:linear-gradient(135deg,#16a34a,#15803d); box-shadow:0 2px 8px rgba(22,163,74,0.3); }

    .bubble {
      max-width:72%; padding:12px 16px; border-radius:12px;
      font-size:13px; line-height:1.7; white-space:pre-wrap;
    }
    .bubble.bot {
      background:#fff; border:1px solid #e2e8f0;
      border-radius:4px 12px 12px 12px; color:#1e293b;
      box-shadow:0 1px 4px rgba(0,0,0,0.06);
    }
    .bubble.user {
      background:#1e3a5f; border-radius:12px 4px 12px 12px; color:#fff;
    }

    #typing { display:none; align-items:flex-start; gap:10px; padding:0 20px 8px; }
    .typing-bubble {
      background:#fff; border:1px solid #e2e8f0;
      border-radius:4px 12px 12px 12px;
      padding:14px 18px; display:flex; gap:5px;
      box-shadow:0 1px 4px rgba(0,0,0,0.06);
    }
    .dot {
      width:7px; height:7px; border-radius:50%;
      background:#2563eb; animation:bounce 1.2s infinite;
    }
    .dot:nth-child(2){ animation-delay:0.2s; background:#1e3a5f; }
    .dot:nth-child(3){ animation-delay:0.4s; background:#16a34a; }

    #quick {
      background:#fff; border-top:1px solid #e2e8f0;
      padding:8px 16px; display:flex; gap:7px; flex-wrap:wrap;
    }
    .qbtn {
      background:#f8fafc; border:1px solid #cbd5e1;
      border-radius:16px; color:#2563eb; font-size:11px;
      padding:4px 12px; cursor:pointer; transition:all 0.2s; font-family:inherit;
    }
    .qbtn:hover { background:#eff6ff; border-color:#2563eb; }

    #input-area {
      background:#fff; border-top:1px solid #e2e8f0;
      padding:12px 16px; display:flex; gap:10px;
      box-shadow:0 -2px 8px rgba(0,0,0,0.05);
    }
    #input {
      flex:1; background:#f8fafc; border:1px solid #e2e8f0;
      border-radius:10px; color:#1e293b; padding:11px 16px;
      font-family:inherit; font-size:13px; outline:none; transition:border-color 0.2s;
    }
    #input:focus { border-color:#2563eb; background:#fff; }
    #input::placeholder { color:#94a3b8; }

    #btn-send {
      background:#1e3a5f; border:none; border-radius:10px;
      color:#fff; padding:11px 22px; cursor:pointer;
      font-family:inherit; font-size:13px; font-weight:600;
      transition:all 0.2s; letter-spacing:0.5px;
    }
    #btn-send:hover    { background:#2563eb; }
    #btn-send:disabled { background:#cbd5e1; cursor:not-allowed; }

    #btn-reset {
      background:#fff1f2; border:1px solid #fecdd3;
      border-radius:10px; color:#dc2626; padding:11px 14px;
      cursor:pointer; font-size:13px; font-family:inherit; transition:all 0.2s;
    }
    #btn-reset:hover { background:#fecdd3; }

    @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
    @keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }
    @keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:0.4} }

    ::-webkit-scrollbar { width:5px; }
    ::-webkit-scrollbar-track { background:#f1f5f9; }
    ::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:3px; }
  </style>
</head>
<body>

  <div id="header">
    <div>
      <div class="title">⚙ IBN ASSISTANT — TECHNICIEN</div>
      <div class="subtitle">Edge Computing · Service Orchestration · Ollama Local</div>
    </div>
    <div id="status">
      <div id="status-dot"></div>
      <span id="model-name">OLLAMA LOCAL</span>
    </div>
  </div>

  <div id="nodes-bar">
    <div class="node-chip gateway" ><span class="nid">G1</span><span class="ninfo">CPU:2 MEM:4 BW:100</span></div>
    <div class="node-chip gateway" ><span class="nid">G2</span><span class="ninfo">CPU:2 MEM:3 BW:80</span></div>
    <div class="node-chip computing"><span class="nid">N1</span><span class="ninfo">CPU:8 MEM:16 BW:150</span></div>
    <div class="node-chip computing"><span class="nid">N2</span><span class="ninfo">CPU:12 MEM:24 BW:250</span></div>
    <div class="node-chip computing"><span class="nid">N3</span><span class="ninfo">CPU:16 MEM:32 BW:300</span></div>
  </div>

  <div id="messages">
    <div class="msg">
      <div class="avatar bot">🤖</div>
      <div class="bubble bot">Bonjour technicien ! Je suis votre assistant IBN (Ollama local).

Je connais votre infrastructure : 5 nœuds, 6 services, 6 intentions.
Posez vos questions sur le placement, les ressources ou le dépannage.</div>
    </div>
  </div>

  <div id="typing">
    <div class="avatar bot">🤖</div>
    <div class="typing-bubble">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>
  </div>

  <div id="quick">
    <button class="qbtn" onclick="send('Quel nœud recommandes-tu pour i5 ?')">Nœud pour i5 ?</button>
    <button class="qbtn" onclick="send('Pourquoi i6 échoue sur g1 ?')">i6 sur g1 ?</button>
    <button class="qbtn" onclick="send('Quelles ressources libres sur n3 ?')">Ressources n3</button>
    <button class="qbtn" onclick="send('Compare n1 et n2 pour i4')">n1 vs n2</button>
    <button class="qbtn" onclick="send('Quel nœud a la latence la plus faible ?')">Latence min</button>
    <button class="qbtn" onclick="send('Peut-on déployer tous les services sur g1 ?')">⚡ Piège</button>
  </div>

  <div id="input-area">
    <input id="input" type="text"
           placeholder="Posez votre question... (FR / EN)"
           onkeydown="if(event.key==='Enter') sendMsg()" />
    <button id="btn-reset" onclick="resetChat()" title="Effacer l'historique">🔄</button>
    <button id="btn-send" onclick="sendMsg()">Envoyer →</button>
  </div>

  <script>
    const msgsDiv  = document.getElementById("messages");
    const input    = document.getElementById("input");
    const btnSend  = document.getElementById("btn-send");
    const typing   = document.getElementById("typing");

    function scrollBot() { msgsDiv.scrollTop = msgsDiv.scrollHeight; }

    function addMsg(text, role) {
      const d = document.createElement("div");
      d.className = `msg ${role}`;
      d.innerHTML = `
        <div class="avatar ${role==='user'?'user':'bot'}">${role==='user'?'👤':'🤖'}</div>
        <div class="bubble ${role==='user'?'user':'bot'}">${text}</div>`;
      msgsDiv.appendChild(d);
      scrollBot();
    }

    function showTyping(v) {
      typing.style.display = v ? "flex" : "none";
      if(v) scrollBot();
    }

    async function sendMsg() {
      const t = input.value.trim();
      if(!t || btnSend.disabled) return;
      send(t);
    }

    async function send(text) {
      addMsg(text, "user");
      input.value = "";
      btnSend.disabled = true;
      showTyping(true);
      try {
        const res  = await fetch("/chat", {
          method:"POST",
          headers:{"Content-Type":"application/json"},
          body: JSON.stringify({message: text})
        });
        const data = await res.json();
        showTyping(false);
        addMsg(data.reply || data.error, "bot");
      } catch(e) {
        showTyping(false);
        addMsg("❌ Erreur de connexion.", "bot");
      }
      btnSend.disabled = false;
      input.focus();
    }

    async function resetChat() {
      await fetch("/reset", {method:"POST"});
      msgsDiv.innerHTML = `
        <div class="msg">
          <div class="avatar bot">🤖</div>
          <div class="bubble bot">Historique effacé. Nouvelle conversation !</div>
        </div>`;
    }

    window.onload = () => input.focus();
  </script>
</body>
</html>"""

# ─────────────────────────────────────────
# ROUTES FLASK
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/chat", methods=["POST"])
def chat():
    data    = request.json
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Message vide"}), 400

    history.append({"role": "user", "content": message})

    # System prompt injecté comme premier message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        reply = ""
        for chunk in ollama.chat(
            model=MODEL,
            messages=messages,
            options={
                "temperature": 0.1,
                "num_predict": 300,
                "num_ctx":     512,
                "num_thread":  4,
            },
            stream=True
        ):
            reply += chunk["message"]["content"]

        history.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply})

    except Exception as e:
        history.pop()
        return jsonify({"error": f"Erreur Ollama : {str(e)}"}), 500

@app.route("/reset", methods=["POST"])
def reset():
    history.clear()
    return jsonify({"status": "ok"})

# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*45)
    print(f"  IBN Chatbot — Modèle : {MODEL}")
    print("  Ouvre : http://127.0.0.1:5000")
    print("="*45 + "\n")
    app.run(debug=False, port=5000)