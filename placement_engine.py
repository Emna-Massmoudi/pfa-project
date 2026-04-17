"""
placement_engine.py
────────────────────────────────────────
Calcule le plan de placement pour les 4 datasets
et génère placement_data.json utilisé par le générateur Word

Lancer : python placement_engine.py
"""

import json

# ─────────────────────────────────────────
# FICHIERS DATASETS
# ─────────────────────────────────────────
DATASETS = {
    "Dataset 1": "dataset_1.json",
    "Dataset 2": "dataset_2.json",
    "Dataset 3": "dataset_3 .json",
    "Dataset 4": "dataset_4.json",
}

# ─────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────

def latence_moyenne(valeurs):
    """Calcule la latence moyenne d'un nœud"""
    return round(sum(valeurs) / len(valeurs), 1)

def ressources_requises(services, service_ids):
    """
    Additionne les ressources de tous les services d'une intention
    Ex: i2 = [s2, s3] → CPU=2+6=8, MEM=4+12=16, etc.
    """
    total = {"CPU": 0, "MEM": 0, "DISK": 0, "BW": 0}
    for service in services:
        if service["id"] in service_ids:
            for ressource in total:
                total[ressource] += service["resources"].get(ressource, 0)
    return total

def trouver_meilleur_noeud(nodes, latency, ressources, qos_latence):
    """
    Trouve le nœud optimal pour une intention :
    1. Vérifie que les ressources sont suffisantes
    2. Vérifie que la latence respecte le QoS
    3. Retourne le nœud avec la latence la plus faible
    """
    meilleur_noeud = None
    meilleure_lat  = 9999

    for noeud in nodes:
        capacite = noeud["capacity"]

        # Étape 1 : vérifier toutes les ressources
        ressources_ok = all(
            capacite.get(r, 0) >= ressources[r]
            for r in ressources
        )
        if not ressources_ok:
            continue

        # Étape 2 : vérifier la latence QoS
        lat = latence_moyenne(latency[noeud["id"]])
        if lat > qos_latence:
            continue

        # Étape 3 : garder le nœud avec latence minimale
        if lat < meilleure_lat:
            meilleure_lat  = lat
            meilleur_noeud = noeud["id"]

    return meilleur_noeud, (round(meilleure_lat, 1) if meilleur_noeud else None)

# ─────────────────────────────────────────
# TRAITEMENT DES 4 DATASETS
# ─────────────────────────────────────────

tous_les_resultats = {}

for nom_dataset, fichier in DATASETS.items():

    print(f"\nTraitement de {nom_dataset}...")

    # Charger le dataset
    with open(fichier, "r", encoding="utf-8") as f:
        data = json.load(f)

    services   = data["services"]
    nodes      = data["nodes"]
    latency    = data["latency"]
    intentions = data["intentions"]

    # ── Calculer le placement pour chaque intention ──
    placements = []
    for intention in intentions:

        # Ressources totales nécessaires
        req = ressources_requises(services, intention["services"])

        # Meilleur nœud
        noeud, lat = trouver_meilleur_noeud(
            nodes, latency, req, intention["QoS"]["latency"]
        )

        placements.append({
            "id":          intention["id"],
            "description": intention["description"],
            "services":    intention["services"],
            "CPU":         req["CPU"],
            "MEM":         req["MEM"],
            "DISK":        req["DISK"],
            "BW":          req["BW"],
            "qos_lat":     intention["QoS"]["latency"],
            "node":        noeud,
            "lat_ms":      lat,
            "success":     noeud is not None
        })

        status = "✅" if noeud else "❌"
        print(f"  {status} {intention['id']:5s} → {noeud or 'ÉCHEC'}")

    # ── Résumé par nœud ──
    node_summary = {}
    for nd in nodes:
        nid = nd["id"]
        placed = [p["id"] for p in placements if p["node"] == nid]
        node_summary[nid] = {
            "type":    nd["type"],
            "CPU":     nd["capacity"]["CPU"],
            "MEM":     nd["capacity"]["MEM"],
            "DISK":    nd["capacity"]["DISK"],
            "BW":      nd["capacity"]["BW"],
            "lat_avg": latence_moyenne(latency[nid]),
            "placed":  placed
        }

    # ── Statistiques ──
    success = sum(1 for p in placements if p["success"])
    fail    = len(placements) - success
    taux    = round(success / len(placements) * 100)

    print(f"  → {success} succès / {fail} échecs / {taux}% taux")

    # ── Stocker les résultats ──
    tous_les_resultats[nom_dataset] = {
        "placements":    placements,
        "node_summary":  node_summary,
        "success":       success,
        "fail":          fail,
        "total":         len(intentions),
        "nb_services":   len(services),
        "nb_nodes":      len(nodes),
        "services_list": [
            {
                "id":   s["id"],
                "name": s["name"],
                "CPU":  s["resources"]["CPU"],
                "MEM":  s["resources"]["MEM"],
                "DISK": s["resources"]["DISK"],
                "BW":   s["resources"]["BW"]
            }
            for s in services
        ],
        "nodes_list": [
            {
                "id":   nd["id"],
                "type": nd["type"],
                "CPU":  nd["capacity"]["CPU"],
                "MEM":  nd["capacity"]["MEM"],
                "DISK": nd["capacity"]["DISK"],
                "BW":   nd["capacity"]["BW"],
                "lat":  latence_moyenne(latency[nd["id"]])
            }
            for nd in nodes
        ]
    }

# ── Sauvegarder le fichier intermédiaire ──
with open("placement_data.json", "w", encoding="utf-8") as f:
    json.dump(tous_les_resultats, f, ensure_ascii=False, indent=2)

print("\n" + "="*50)
print("  RÉSUMÉ FINAL")
print("="*50)
for nom, res in tous_les_resultats.items():
    taux = round(res["success"]/res["total"]*100)
    print(f"  {nom:12s} → {res['success']:2d}/{res['total']:2d} ({taux}%)")
print("\n✅ placement_data.json généré")
print("   Lance maintenant : node generate_word.js")