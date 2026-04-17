import json

# ── Charger le dataset ──
with open("dataset_1.json", "r") as f:
    data = json.load(f)

SERVICES   = data["services"]
NODES      = data["nodes"]
LATENCY    = data["latency"]
INTENTIONS = data["intentions"]

# ── Étape 1 : Calculer latence moyenne d'un nœud ──
def latence_moyenne(node_id):
    valeurs = LATENCY[node_id]        # ex: [67, 88, 32]
    return sum(valeurs) / len(valeurs) # ex: 62.3ms

# ── Étape 2 : Calculer ressources totales d'une intention ──
def ressources_requises(service_ids):
    total = {"CPU": 0, "MEM": 0, "DISK": 0, "BW": 0}
    for service in SERVICES:
        if service["id"] in service_ids:
            total["CPU"]  += service["resources"]["CPU"]
            total["MEM"]  += service["resources"]["MEM"]
            total["DISK"] += service["resources"]["DISK"]
            total["BW"]   += service["resources"]["BW"]
    return total

# ── Étape 3 : Trouver le meilleur nœud ──
def trouver_meilleur_noeud(ressources, qos_latence):
    meilleur_noeud = None
    meilleure_lat  = 9999

    for noeud in NODES:
        cap = noeud["capacity"]

        # Vérifier les ressources
        cpu_ok  = cap["CPU"]  >= ressources["CPU"]
        mem_ok  = cap["MEM"]  >= ressources["MEM"]
        disk_ok = cap["DISK"] >= ressources["DISK"]
        bw_ok   = cap["BW"]   >= ressources["BW"]

        # Vérifier la latence
        lat     = latence_moyenne(noeud["id"])
        lat_ok  = lat <= qos_latence

        # Si tout est OK → nœud éligible
        if cpu_ok and mem_ok and disk_ok and bw_ok and lat_ok:
            # Garder le nœud avec la latence la plus faible
            if lat < meilleure_lat:
                meilleure_lat  = lat
                meilleur_noeud = noeud["id"]

    return meilleur_noeud, round(meilleure_lat, 1)

# ── Étape 4 : Faire le placement pour toutes les intentions ──
print("=" * 55)
print("  PLAN DE PLACEMENT")
print("=" * 55)

for intention in INTENTIONS:
    print(f"\n📌 {intention['id']} — {intention['description']}")

    # Calculer ce dont l'intention a besoin
    res = ressources_requises(intention["services"])
    qos = intention["QoS"]["latency"]

    print(f"   Services  : {intention['services']}")
    print(f"   Ressources: CPU={res['CPU']} MEM={res['MEM']} "
          f"DISK={res['DISK']} BW={res['BW']}")
    print(f"   QoS       : latence ≤ {qos}ms")

    # Trouver le meilleur nœud
    noeud, lat = trouver_meilleur_noeud(res, qos)

    if noeud:
        print(f"   ✅ Placé sur : {noeud} (latence {lat}ms)")
    else:
        print(f"   ❌ ÉCHEC : aucun nœud ne satisfait le QoS")

print("\n" + "=" * 55)