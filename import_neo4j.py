"""
import_neo4j.py
═══════════════════════════════════════════════════════
Importe dataset_4.json dans Neo4j et calcule les placements

Installer : pip install neo4j
Lancer    : python import_neo4j.py

Connexion : neo4j://127.0.0.1:7687
Database  : Industry
"""

import json
from neo4j import GraphDatabase

# ═══════════════════════════════════════════════════════
# CONFIG — MODIFIER ICI
# ═══════════════════════════════════════════════════════
NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "emnakhouloudazizyoussef"       # ← Ton mot de passe ici
NEO4J_DATABASE = "neo4j"      # ← Nom de ta database
DATASET_FILE   = "dataset_4.json"

# ═══════════════════════════════════════════════════════
# CHARGEMENT DATASET
# ═══════════════════════════════════════════════════════
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

nodes      = data["nodes"]
services   = data["services"]
intentions = data["intentions"]
latency    = data["latency"]

print(f"✅ Dataset chargé :")
print(f"   {len(nodes)} nœuds")
print(f"   {len(services)} services")
print(f"   {len(intentions)} intentions")

# ═══════════════════════════════════════════════════════
# UTILITAIRES PLACEMENT
# ═══════════════════════════════════════════════════════
def avg(lst):
    return round(sum(lst) / len(lst), 1) if lst else 0

def sum_res(service_ids):
    t = {"CPU": 0, "MEM": 0, "DISK": 0, "BW": 0}
    for s in services:
        if s["id"] in service_ids:
            for r in t:
                t[r] += s["resources"].get(r, 0)
    return t

def find_best_node(req, qos_lat):
    best, blat = None, 9999
    for nd in nodes:
        cap = nd["capacity"]
        if all(cap.get(r, 0) >= req[r] for r in req):
            lat = avg(latency[nd["id"]])
            if lat <= qos_lat and lat < blat:
                blat, best = lat, nd["id"]
    return best, round(blat, 1) if best else None

# Calculer tous les placements
placements = []
for intent in intentions:
    req  = sum_res(intent["services"])
    node, lat = find_best_node(req, intent["QoS"]["latency"])
    placements.append({
        "id":          intent["id"],
        "description": intent["description"],
        "services":    intent["services"],
        "node":        node,
        "lat":         lat,
        "success":     node is not None,
        "cpu":         req["CPU"],
        "mem":         req["MEM"],
        "disk":        req["DISK"],
        "bw":          req["BW"],
        "qos_lat":     intent["QoS"]["latency"],
        "weight":      intent.get("weight", 1),
    })

ok   = sum(1 for p in placements if p["success"])
fail = len(placements) - ok
print(f"\n📊 Placements calculés :")
print(f"   ✅ {ok} succès / ❌ {fail} échecs ({round(ok/len(placements)*100)}%)")

# ═══════════════════════════════════════════════════════
# IMPORT NEO4J
# ═══════════════════════════════════════════════════════
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run(tx, query, **params):
    tx.run(query, **params)

with driver.session(database=NEO4J_DATABASE) as session:

    # ── 1. Nettoyer la base ──
    print("\n🗑️  Nettoyage de la base...")
    session.run("MATCH (n) DETACH DELETE n")

    # ── 2. Créer les contraintes d'unicité ──
    print("📌 Création des contraintes...")
    for label, prop in [("IbnNode","id"), ("Service","id"), ("Intention","id")]:
        try:
            session.run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
        except Exception:
            pass

    # ── 3. Créer les nœuds IBN ──
    print("🔵 Import des nœuds IBN...")
    for nd in nodes:
        lat_avg = avg(latency[nd["id"]])
        lat_vals = latency[nd["id"]]
        session.execute_write(run, """
            MERGE (n:IbnNode {id: $id})
            SET n.type       = $type,
                n.cpu        = $cpu,
                n.mem        = $mem,
                n.disk       = $disk,
                n.bw         = $bw,
                n.lat_avg    = $lat_avg,
                n.lat_min    = $lat_min,
                n.lat_max    = $lat_max,
                n.label      = $label
        """,
            id       = nd["id"],
            type     = nd["type"],
            cpu      = nd["capacity"]["CPU"],
            mem      = nd["capacity"]["MEM"],
            disk     = nd["capacity"]["DISK"],
            bw       = nd["capacity"]["BW"],
            lat_avg  = lat_avg,
            lat_min  = min(lat_vals),
            lat_max  = max(lat_vals),
            label    = nd["id"].upper(),
        )

    # ── 4. Créer les relations GATEWAY → COMPUTING ──
    print("🔗 Création des relations réseau...")
    gw_nodes = [n for n in nodes if n["type"] == "gateway"]
    cp_nodes = [n for n in nodes if n["type"] == "computing"]

    for gw in gw_nodes:
        for cp in cp_nodes:
            session.execute_write(run, """
                MATCH (g:IbnNode {id: $gw_id})
                MATCH (c:IbnNode {id: $cp_id})
                MERGE (g)-[:CONNECTED_TO {type: 'network'}]->(c)
            """, gw_id=gw["id"], cp_id=cp["id"])

    # ── 5. Créer les services ──
    print("⚙️  Import des services...")
    for svc in services:
        session.execute_write(run, """
            MERGE (s:Service {id: $id})
            SET s.name = $name,
                s.cpu  = $cpu,
                s.mem  = $mem,
                s.disk = $disk,
                s.bw   = $bw,
                s.label = $label
        """,
            id    = svc["id"],
            name  = svc["name"],
            cpu   = svc["resources"]["CPU"],
            mem   = svc["resources"]["MEM"],
            disk  = svc["resources"]["DISK"],
            bw    = svc["resources"]["BW"],
            label = svc["name"].replace("_", " "),
        )

    # ── 6. Créer les intentions + relations ──
    print("🎯 Import des intentions et placements...")
    for p in placements:
        # Créer l'intention
        session.execute_write(run, """
            MERGE (i:Intention {id: $id})
            SET i.description = $desc,
                i.success     = $success,
                i.qos_lat     = $qos_lat,
                i.weight      = $weight,
                i.cpu_req     = $cpu,
                i.mem_req     = $mem,
                i.bw_req      = $bw,
                i.label       = $id
        """,
            id      = p["id"],
            desc    = p["description"][:60],
            success = p["success"],
            qos_lat = p["qos_lat"],
            weight  = p["weight"],
            cpu     = p["cpu"],
            mem     = p["mem"],
            bw      = p["bw"],
        )

        # Relation Intention → Services (REQUIRES)
        for svc_id in p["services"]:
            session.execute_write(run, """
                MATCH (i:Intention {id: $i_id})
                MATCH (s:Service   {id: $s_id})
                MERGE (i)-[:REQUIRES]->(s)
            """, i_id=p["id"], s_id=svc_id)

        # Relation Intention → Nœud (PLACED_ON) si succès
        if p["success"]:
            session.execute_write(run, """
                MATCH (i:Intention {id: $i_id})
                MATCH (n:IbnNode   {id: $n_id})
                MERGE (i)-[:PLACED_ON {latency: $lat, success: true}]->(n)
            """, i_id=p["id"], n_id=p["node"], lat=p["lat"])
        else:
            # Relation FAILED (pas de nœud trouvé)
            session.execute_write(run, """
                MATCH (i:Intention {id: $i_id})
                SET i.failed = true
            """, i_id=p["id"])

    # ── 7. Statistiques finales ──
    print("\n📈 Vérification dans Neo4j...")
    result = session.run("""
        MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
    """)
    for rec in result:
        print(f"   {rec['label']:15s} : {rec['count']} nœuds")

    result2 = session.run("""
        MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS count
        ORDER BY count DESC
    """)
    for rec in result2:
        print(f"   {rec['rel']:20s} : {rec['count']} relations")

driver.close()

print("\n" + "="*55)
print("  ✅ Import terminé ! Ouvre Neo4j Browser :")
print("  http://localhost:7474")
print("="*55)
print("""
Requêtes Cypher à copier dans Neo4j Browser :

── Voir tout le graphe ──────────────────────────────
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100

── Nœuds IBN avec ressources ────────────────────────
MATCH (n:IbnNode) RETURN n

── Placements réussis ───────────────────────────────
MATCH (i:Intention)-[r:PLACED_ON]->(n:IbnNode)
RETURN i, r, n

── Nœud le plus chargé ──────────────────────────────
MATCH (i:Intention)-[:PLACED_ON]->(n:IbnNode)
RETURN n.id, count(i) AS intentions
ORDER BY intentions DESC

── Services requis par intention ────────────────────
MATCH (i:Intention)-[:REQUIRES]->(s:Service)
RETURN i.id, collect(s.id) AS services

── Intentions échouées ──────────────────────────────
MATCH (i:Intention) WHERE i.failed = true RETURN i
""")