"""
dashboard.py — IBN Node Monitor
Lancer : python dashboard.py
Ouvrir : http://127.0.0.1:8050

pip install dash plotly
"""

import json, random, time
from collections import deque
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════
# CHARGEMENT DES 4 DATASETS
# ═══════════════════════════════════════════════════════
DATASET_FILES = {
    "Dataset 1": "dataset_1.json",
    "Dataset 2": "dataset_2.json",
    "Dataset 3": "dataset_3_.json",
    "Dataset 4": "dataset_4.json",
}

DATASETS = {}
for ds_name, fname in DATASET_FILES.items():
    try:
        with open(fname, "r", encoding="utf-8") as f:
            DATASETS[ds_name] = json.load(f)
        print(f"✅ {ds_name} chargé")
    except FileNotFoundError:
        print(f"❌ {fname} introuvable")

# ═══════════════════════════════════════════════════════
# COULEURS
# ═══════════════════════════════════════════════════════
SVC_PALETTE = ["#38bdf8","#818cf8","#34d399","#f59e0b","#f472b6","#a78bfa","#22d3ee","#fb923c"]

def get_svc_color(idx):
    return SVC_PALETTE[idx % len(SVC_PALETTE)]

# ═══════════════════════════════════════════════════════
# UTILITAIRES PLACEMENT
# ═══════════════════════════════════════════════════════
def avg(lst):
    return round(sum(lst) / len(lst), 1) if lst else 0

def sum_res(services, sids):
    t = {"CPU":0, "MEM":0, "DISK":0, "BW":0}
    for s in services:
        if s["id"] in sids:
            for r in t:
                t[r] += s["resources"].get(r, 0)
    return t

def best_node(nodes, latency, req, qos_lat):
    best, blat = None, 9999
    for nd in nodes:
        cap = nd["capacity"]
        if all(cap.get(r, 0) >= req[r] for r in req):
            lat = avg(latency[nd["id"]])
            if lat <= qos_lat and lat < blat:
                blat, best = lat, nd["id"]
    return best, (round(blat, 1) if best else None)

def node_load(nid, placements, services):
    load = {"CPU":0, "MEM":0, "DISK":0, "BW":0}
    for p in placements:
        if p["node"] == nid:
            r = sum_res(services, p["services"])
            for k in load:
                load[k] += r[k]
    return load

def pct(used, cap):
    return min(100, round(used / cap * 100)) if cap else 0

# ═══════════════════════════════════════════════════════
# PRÉ-CALCULER LES PLACEMENTS PAR DATASET
# ═══════════════════════════════════════════════════════
DS_PLACEMENTS = {}

for ds_name, d in DATASETS.items():
    results = []
    for intent in d["intentions"]:
        req  = sum_res(d["services"], intent["services"])
        node, lat = best_node(d["nodes"], d["latency"], req, intent["QoS"]["latency"])
        results.append({
            "id":          intent["id"],
            "description": intent["description"],
            "services":    intent["services"],
            "node":        node,
            "lat":         lat,
            "success":     node is not None,
            "qos":         intent["QoS"]["latency"],
        })
    DS_PLACEMENTS[ds_name] = results
    ok = sum(1 for r in results if r["success"])
    print(f"  {ds_name}: {ok}/{len(results)} placements réussis")

# ═══════════════════════════════════════════════════════
# ÉTAT GLOBAL
# ═══════════════════════════════════════════════════════
G = {
    "ds":       "Dataset 1",
    "index":    0,
    "running":  False,
    "active":   [],
    "log":      deque(maxlen=30),
    "lat_hist": {},
    "tick":     0,
}

def init_state(ds_name):
    d = DATASETS[ds_name]
    G["ds"]      = ds_name
    G["index"]   = 0
    G["running"] = False
    G["active"]  = []
    G["log"]     = deque(maxlen=30)
    G["tick"]    = 0
    G["lat_hist"] = {
        n["id"]: deque([avg(d["latency"][n["id"]])] * 15, maxlen=30)
        for n in d["nodes"]
    }

init_state("Dataset 1")

def step_simulation():
    ds_name = G["ds"]
    results = DS_PLACEMENTS[ds_name]
    d       = DATASETS[ds_name]
    if G["index"] >= len(results):
        G["running"] = False
        return
    p  = results[G["index"]]
    ts = time.strftime("%H:%M:%S")
    G["index"] += 1
    G["tick"]  += 1
    if p["success"]:
        G["active"].append(p)
        G["log"].appendleft({"type":"success","ts":ts,
            "msg":f"✅ {p['id']} → {p['node'].upper()}  ({p['lat']} ms)"})
    else:
        G["log"].appendleft({"type":"error","ts":ts,
            "msg":f"❌ {p['id']} — QoS non satisfait  (lat≤{p['qos']}ms)"})
    for nid in G["lat_hist"]:
        base = avg(d["latency"][nid])
        G["lat_hist"][nid].append(round(base + random.uniform(-3, 3), 1))

# ═══════════════════════════════════════════════════════
# THÈME
# ═══════════════════════════════════════════════════════
C = {
    "bg":      "#080d16",
    "surface": "#0f1824",
    "surf2":   "#162030",
    "border":  "#1e3550",
    "accent":  "#38bdf8",
    "text":    "#e2e8f0",
    "muted":   "#64748b",
    "success": "#22d3ee",
    "warn":    "#f59e0b",
    "danger":  "#f87171",
}

CARD = {
    "background":   C["surface"],
    "border":       f"1px solid {C['border']}",
    "borderRadius": "6px",
    "padding":      "14px 16px",
}

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'IBM Plex Mono', monospace", color=C["muted"], size=10),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(56,189,248,0.06)", tickfont=dict(size=9), zerolinecolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor="rgba(56,189,248,0.06)", tickfont=dict(size=9), zerolinecolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=C["surf2"], font_family="IBM Plex Mono", font_size=11, bordercolor=C["border"]),
)

# ═══════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════
app = dash.Dash(__name__, title="IBN Dashboard", update_title=None)

def lbl(text):
    return html.Div(text, style={
        "fontFamily":"'IBM Plex Mono',monospace","fontSize":"9px","fontWeight":"600",
        "color":C["muted"],"letterSpacing":"0.12em","textTransform":"uppercase",
        "marginBottom":"10px","borderBottom":f"1px solid {C['border']}","paddingBottom":"6px",
    })

app.layout = html.Div([
    dcc.Interval(id="tmr", interval=1200, n_intervals=0),

    # ── Topbar ──
    html.Div([
        html.Div([
            html.Span("IBN", style={"color":C["accent"],"fontWeight":"700"}),
            html.Span(" · Node Monitor", style={"color":C["muted"]}),
        ], style={"fontFamily":"'IBM Plex Mono',monospace","fontSize":"13px",
                  "letterSpacing":"0.08em","marginRight":"auto"}),

        html.Div([
            html.Button(f"DS{i+1}", id=f"ds-btn-{i+1}", n_clicks=0,
                style={"fontFamily":"'IBM Plex Mono',monospace","fontSize":"11px",
                       "padding":"5px 14px",
                       "background":C["accent"]+"25" if i==0 else C["surf2"],
                       "color":C["accent"] if i==0 else C["muted"],
                       "border":f"1px solid {C['accent'] if i==0 else C['border']}",
                       "borderRadius":"3px","cursor":"pointer"})
            for i in range(4)
        ], style={"display":"flex","gap":"6px"}),

        html.Div(style={"width":"1px","height":"28px","background":C["border"],"margin":"0 12px"}),

        html.Button("▶ PLAY",  id="btn-play",  n_clicks=0,
            style={"fontFamily":"'IBM Plex Mono',monospace","fontSize":"11px","padding":"5px 14px",
                   "background":"rgba(34,211,238,0.1)","color":"#67e8f9",
                   "border":"1px solid rgba(34,211,238,0.3)","borderRadius":"3px","cursor":"pointer"}),
        html.Button("⏸ PAUSE", id="btn-pause", n_clicks=0,
            style={"fontFamily":"'IBM Plex Mono',monospace","fontSize":"11px","padding":"5px 14px",
                   "background":"rgba(245,158,11,0.1)","color":"#fcd34d",
                   "border":"1px solid rgba(245,158,11,0.3)","borderRadius":"3px","cursor":"pointer"}),
        html.Button("↺ RESET", id="btn-reset", n_clicks=0,
            style={"fontFamily":"'IBM Plex Mono',monospace","fontSize":"11px","padding":"5px 14px",
                   "background":"rgba(248,113,113,0.1)","color":"#fca5a5",
                   "border":"1px solid rgba(248,113,113,0.3)","borderRadius":"3px","cursor":"pointer"}),

        html.Div(style={"width":"1px","height":"28px","background":C["border"],"margin":"0 12px"}),

        html.Span("SPEED", style={"fontFamily":"'IBM Plex Mono',monospace","fontSize":"10px","color":C["muted"]}),
        dcc.Dropdown(id="speed-sel",
            options=[{"label":"0.5x","value":2400},{"label":"1x","value":1200},
                     {"label":"2x","value":600},{"label":"5x","value":250}],
            value=1200, clearable=False,
            style={"fontFamily":"'IBM Plex Mono',monospace","fontSize":"11px","width":"80px",
                   "background":C["surf2"],"color":C["text"],
                   "border":f"1px solid {C['border']}","borderRadius":"3px"},
        ),

        html.Div(style={"width":"1px","height":"28px","background":C["border"],"margin":"0 12px"}),

        html.Div(id="status-badge", children="■ STOPPED", style={
            "fontFamily":"'IBM Plex Mono',monospace","fontSize":"10px",
            "padding":"4px 10px","borderRadius":"2px","letterSpacing":"0.06em",
            "background":"rgba(100,116,139,0.12)","color":C["muted"],
            "border":f"1px solid rgba(100,116,139,0.25)",
        }),

    ], style={
        "display":"flex","alignItems":"center","gap":"8px",
        "padding":"10px 24px",
        "background":"rgba(8,13,22,0.95)",
        "borderBottom":f"1px solid {C['border']}",
        "position":"sticky","top":"0","zIndex":"100",
    }),

    # ── KPIs ──
    html.Div(id="kpi-row", style={
        "display":"grid","gridTemplateColumns":"repeat(5,1fr)",
        "gap":"10px","padding":"14px 24px 0",
    }),

    # ── Body ──
    html.Div([
        # Gauche
        html.Div([
            html.Div([lbl("Ressources des Nœuds (%)"),
                      dcc.Graph(id="chart-res",config={"displayModeBar":False},style={"height":"230px"})],
                     style={**CARD,"marginBottom":"10px"}),
            html.Div([lbl("Latence Temps Réel (ms)"),
                      dcc.Graph(id="chart-lat",config={"displayModeBar":False},style={"height":"200px"})],
                     style={**CARD,"marginBottom":"10px"}),
            html.Div([lbl("Topologie & Charge"),
                      dcc.Graph(id="chart-topo",config={"displayModeBar":False},style={"height":"220px"})],
                     style=CARD),
        ], style={"flex":"2","display":"flex","flexDirection":"column"}),

        # Droite
        html.Div([
            html.Div([lbl("Timeline des Intentions"),
                      html.Div(id="timeline",style={"display":"flex","flexWrap":"wrap","gap":"4px"})],
                     style={**CARD,"marginBottom":"10px"}),
            html.Div([lbl("Services Déployés"),
                      html.Div(id="services-panel",style={"display":"flex","flexWrap":"wrap","gap":"6px"})],
                     style={**CARD,"marginBottom":"10px"}),
            html.Div([lbl("Journal d'Événements"),
                      html.Div(id="journal",style={"maxHeight":"200px","overflowY":"auto",
                                                   "display":"flex","flexDirection":"column","gap":"3px"})],
                     style={**CARD,"marginBottom":"10px"}),
            html.Div([lbl("Intentions du Dataset"),
                      html.Div(id="intents-panel",style={"maxHeight":"200px","overflowY":"auto"})],
                     style=CARD),
        ], style={"width":"320px","display":"flex","flexDirection":"column"}),

    ], style={"display":"flex","gap":"10px","padding":"10px 24px 24px"}),

    dcc.Store(id="store-ctrl", data={"ds":"Dataset 1","running":False,"interval":1200}),

], style={
    "background":C["bg"],"minHeight":"100vh",
    "fontFamily":"'IBM Plex Sans',sans-serif",
    "backgroundImage":"linear-gradient(rgba(56,189,248,0.02) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,0.02) 1px,transparent 1px)",
    "backgroundSize":"40px 40px",
})

# ═══════════════════════════════════════════════════════
# CALLBACK CONTRÔLE
# ═══════════════════════════════════════════════════════
@app.callback(
    Output("store-ctrl","data"),
    Output("tmr","interval"),
    Input("btn-play","n_clicks"), Input("btn-pause","n_clicks"),
    Input("btn-reset","n_clicks"),
    Input("ds-btn-1","n_clicks"), Input("ds-btn-2","n_clicks"),
    Input("ds-btn-3","n_clicks"), Input("ds-btn-4","n_clicks"),
    Input("speed-sel","value"),
    State("store-ctrl","data"),
    prevent_initial_call=True,
)
def handle_controls(play,pause,reset,ds1,ds2,ds3,ds4,speed,state):
    ctx     = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    ds_map  = {"ds-btn-1":"Dataset 1","ds-btn-2":"Dataset 2",
               "ds-btn-3":"Dataset 3","ds-btn-4":"Dataset 4"}

    if trigger in ds_map:
        new_ds = ds_map[trigger]
        init_state(new_ds)
        state["ds"]      = new_ds
        state["running"] = False
    elif trigger == "btn-play":
        if G["index"] >= len(DS_PLACEMENTS[G["ds"]]):
            init_state(G["ds"])
        G["running"] = True
        state["running"] = True
    elif trigger == "btn-pause":
        G["running"] = False
        state["running"] = False
    elif trigger == "btn-reset":
        init_state(state.get("ds","Dataset 1"))
        state["running"] = False
    elif trigger == "speed-sel":
        state["interval"] = speed

    return state, state.get("interval",1200)

# ═══════════════════════════════════════════════════════
# CALLBACK PRINCIPAL
# ═══════════════════════════════════════════════════════
@app.callback(
    Output("kpi-row","children"),
    Output("chart-res","figure"), Output("chart-lat","figure"), Output("chart-topo","figure"),
    Output("timeline","children"), Output("services-panel","children"),
    Output("journal","children"), Output("intents-panel","children"),
    Output("status-badge","children"), Output("status-badge","style"),
    Output("ds-btn-1","style"), Output("ds-btn-2","style"),
    Output("ds-btn-3","style"), Output("ds-btn-4","style"),
    Input("tmr","n_intervals"), Input("store-ctrl","data"),
)
def update(_,ctrl):
    ds_name = G["ds"]
    d       = DATASETS[ds_name]
    results = DS_PLACEMENTS[ds_name]

    if G["running"] and G["index"] < len(results):
        step_simulation()

    active = G["active"]
    idx    = G["index"]
    svcs   = d["services"]
    nodes  = d["nodes"]
    nids   = [n["id"].upper() for n in nodes]

    # ── KPIs ──
    total   = len(results)
    success = sum(1 for r in results[:idx] if r["success"])
    fail    = idx - success
    taux    = round(success/total*100) if total else 0
    tc      = "#22d3ee" if taux>=90 else "#f59e0b" if taux>=60 else "#f87171"

    def kpi(val,lbl_,color):
        return html.Div([
            html.Div(str(val),style={"fontFamily":"'IBM Plex Mono',monospace",
                "fontSize":"26px","fontWeight":"700","color":color,"lineHeight":"1"}),
            html.Div(lbl_,style={"fontSize":"9px","color":C["muted"],
                "letterSpacing":"0.1em","textTransform":"uppercase","marginTop":"2px"}),
        ],style={**CARD,"padding":"12px 16px"})

    kpis = [
        kpi(len(nodes),"Nœuds","#7dd3fc"),
        kpi(len(svcs),"Services","#c084fc"),
        kpi(idx,"Traités",C["text"]),
        kpi(f"{success}/{total}","Succès",tc),
        kpi(f"{taux}%","Taux",tc),
    ]

    # ── Ressources ──
    res_colors = ["#38bdf8","#818cf8","#34d399","#f59e0b"]
    fig_res = go.Figure()
    for res, col in zip(["CPU","MEM","DISK","BW"], res_colors):
        vals = [pct(node_load(n["id"],active,svcs)[res], n["capacity"][res]) for n in nodes]
        fig_res.add_trace(go.Bar(
            name=res, x=nids, y=vals,
            marker=dict(color=col, opacity=0.85),
            text=[f"{v}%" for v in vals], textposition="outside",
            textfont=dict(color=col, size=9),
            hovertemplate=f"<b>%{{x}}</b> {res}: %{{y}}%<extra></extra>",
        ))
    fig_res.add_hline(y=80, line_dash="dot", line_color="rgba(248,113,113,0.4)",
                      annotation_text="⚠ 80%",
                      annotation_font=dict(color="#f87171",size=9))
    fig_res.update_layout(**PLOT_BASE,
        barmode="group",
        legend=dict(orientation="h",y=1.12,bgcolor="rgba(0,0,0,0)",font=dict(size=9)),
        yaxis=dict(range=[0,120],gridcolor="rgba(56,189,248,0.06)",
                   ticksuffix="%",tickfont=dict(size=9)),
        margin=dict(l=10,r=10,t=32,b=6),
    )

    # ── Latence ──
    lat_cols = SVC_PALETTE
    fig_lat  = go.Figure()
    for i, n in enumerate(nodes):
        col  = lat_cols[i % len(lat_cols)]
        hist = list(G["lat_hist"].get(n["id"],[avg(d["latency"][n["id"]])]*10))
        r,g_,b_ = int(col[1:3],16),int(col[3:5],16),int(col[5:7],16)
        fig_lat.add_trace(go.Scatter(
            x=list(range(len(hist))), y=hist,
            name=n["id"].upper(), mode="lines",
            line=dict(color=col, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({r},{g_},{b_},0.06)",
            hovertemplate=f"<b>{n['id'].upper()}</b>: %{{y:.1f}} ms<extra></extra>",
        ))
    fig_lat.update_layout(**PLOT_BASE,
        legend=dict(orientation="h",y=1.12,bgcolor="rgba(0,0,0,0)",font=dict(size=9)),
        xaxis=dict(showticklabels=False,gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(56,189,248,0.06)",ticksuffix=" ms",tickfont=dict(size=9)),
        hovermode="x unified",
        margin=dict(l=10,r=10,t=32,b=6),
    )

    # ── Topologie ──
    gw_nodes = [n for n in nodes if n["type"]=="gateway"]
    cp_nodes = [n for n in nodes if n["type"]=="computing"]
    vis_cp   = cp_nodes[:10]

    pos = {}
    for i,n in enumerate(gw_nodes):
        pos[n["id"]] = (0.5, (i+1)*(5.0/(len(gw_nodes)+1)))
    for i,n in enumerate(vis_cp):
        pos[n["id"]] = (3.5, (i+1)*(5.0/(len(vis_cp)+1)))

    fig_topo = go.Figure()
    for gn in gw_nodes:
        for cn in vis_cp:
            if gn["id"] not in pos or cn["id"] not in pos:
                continue
            act = any(p["node"]==cn["id"] for p in active)
            x0,y0 = pos[gn["id"]]
            x1,y1 = pos[cn["id"]]
            fig_topo.add_shape(type="line",x0=x0,y0=y0,x1=x1,y1=y1,
                line=dict(color="rgba(56,189,248,0.4)" if act else "rgba(56,189,248,0.1)",
                          width=2 if act else 1,dash="solid" if act else "dot"))

    for n in list(gw_nodes)+list(vis_cp):
        if n["id"] not in pos:
            continue
        x,y   = pos[n["id"]]
        load  = node_load(n["id"],active,svcs)
        cpu_p = pct(load["CPU"],n["capacity"]["CPU"])
        is_gw = n["type"]=="gateway"
        color = "#f59e0b" if is_gw else "#38bdf8"
        size  = 22 + cpu_p*0.14
        placed= [p["id"] for p in active if p["node"]==n["id"]]
        fig_topo.add_trace(go.Scatter(
            x=[x],y=[y],mode="markers+text",showlegend=False,
            marker=dict(size=size,color=color,opacity=0.9,
                        line=dict(color="white",width=1.5)),
            text=[n["id"].upper()],
            textposition="middle center",
            textfont=dict(color="white",size=8,family="IBM Plex Mono"),
            hovertext=(f"<b>{n['id'].upper()}</b> — {n['type']}<br>"
                       f"CPU:{pct(load['CPU'],n['capacity']['CPU'])}%  "
                       f"MEM:{pct(load['MEM'],n['capacity']['MEM'])}%<br>"
                       f"Intentions: {', '.join(placed) if placed else 'aucune'}"),
            hoverinfo="text",
        ))
    fig_topo.update_layout(**PLOT_BASE,
        xaxis=dict(showgrid=False,showticklabels=False,range=[0,4.5],zeroline=False),
        yaxis=dict(showgrid=False,showticklabels=False,range=[0,5.5],zeroline=False),
        showlegend=False,margin=dict(l=5,r=5,t=5,b=5),
    )
    fig_topo.add_annotation(x=0.02,y=0.98,xref="paper",yref="paper",
        text="● Gateway",font=dict(color="#f59e0b",size=9,family="IBM Plex Mono"),
        showarrow=False,xanchor="left")
    fig_topo.add_annotation(x=0.02,y=0.90,xref="paper",yref="paper",
        text="● Computing",font=dict(color="#38bdf8",size=9,family="IBM Plex Mono"),
        showarrow=False,xanchor="left")

    # ── Timeline ──
    tl = []
    for i,r in enumerate(results):
        if i < idx:
            col = "#22d3ee" if r["success"] else "#f87171"
            bg  = "rgba(34,211,238,0.15)" if r["success"] else "rgba(248,113,113,0.15)"
        elif i==idx and G["running"]:
            col,bg = C["accent"],"rgba(56,189,248,0.25)"
        else:
            col,bg = C["muted"],C["surf2"]
        tl.append(html.Div(r["id"].replace("i",""),
            title=f"{r['id']}: {r['description']}\n→ {r['node'] or 'ÉCHEC'}",
            style={"width":"22px","height":"22px","borderRadius":"3px",
                   "display":"flex","alignItems":"center","justifyContent":"center",
                   "fontFamily":"'IBM Plex Mono',monospace","fontSize":"9px",
                   "color":col,"background":bg,
                   "border":f"1px solid {col if i<=idx else C['border']}","cursor":"pointer"}))

    # ── Services ──
    deployed = {sid for p in active for sid in p["services"]}
    svc_items = []
    for i,svc in enumerate(svcs):
        on    = svc["id"] in deployed
        color = get_svc_color(i)
        where = list(set(p["node"] for p in active if svc["id"] in p["services"]))
        svc_items.append(html.Div([
            html.Div(style={"width":"8px","height":"8px","borderRadius":"50%","flexShrink":"0","marginTop":"2px",
                "background":color if on else C["muted"],
                "boxShadow":f"0 0 6px {color}" if on else "none"}),
            html.Div([
                html.Div(svc["id"].upper(),style={"fontFamily":"'IBM Plex Mono',monospace",
                    "fontSize":"10px","fontWeight":"600","color":color if on else C["muted"]}),
                html.Div(svc["name"].replace("_"," "),style={"fontSize":"9px","color":C["muted"]}),
                html.Div("→ "+", ".join(where) if where else "IDLE",
                    style={"fontSize":"9px","color":color if on else C["border"]}),
            ]),
        ],style={"display":"flex","gap":"7px","alignItems":"flex-start",
                 "padding":"7px 10px","borderRadius":"4px",
                 "background":color+"18" if on else C["surf2"],
                 "border":f"1px solid {color+'55' if on else C['border']}","minWidth":"100px"}))

    # ── Journal ──
    lc = {"success":"#22d3ee","error":"#f87171","warn":"#f59e0b"}
    lb = {"success":"rgba(34,211,238,0.05)","error":"rgba(248,113,113,0.05)","warn":"rgba(245,158,11,0.05)"}
    journal = [html.Div([
        html.Span(e["ts"],style={"fontFamily":"'IBM Plex Mono',monospace","fontSize":"9px",
            "color":C["muted"],"flexShrink":"0","marginRight":"8px"}),
        html.Span(e["msg"],style={"fontFamily":"'IBM Plex Mono',monospace","fontSize":"10px",
            "color":lc.get(e["type"],C["text"]),"lineHeight":"1.4"}),
    ],style={"display":"flex","padding":"5px 8px","borderRadius":"3px",
             "borderLeft":f"2px solid {lc.get(e['type'],C['border'])}",
             "background":lb.get(e["type"],C["surf2"]),"alignItems":"flex-start"})
    for e in list(G["log"])[:15]]

    # ── Intentions ──
    int_items = []
    for r in results:
        ri = results.index(r)
        placed  = r["id"] in [p["id"] for p in active]
        failed  = ri < idx and not r["success"]
        waiting = ri >= idx
        color = "#22d3ee" if placed else "#f87171" if failed else C["muted"]
        int_items.append(html.Div([
            html.Div([
                html.Span(r["id"].upper(),style={"fontFamily":"'IBM Plex Mono',monospace",
                    "fontSize":"10px","fontWeight":"700","color":color}),
                html.Span(
                    f"→ {r['node'].upper()}" if placed else "ÉCHEC" if failed else "EN ATTENTE",
                    style={"fontSize":"9px","padding":"1px 6px","borderRadius":"10px","marginLeft":"6px",
                           "background":"rgba(34,211,238,0.12)" if placed else "rgba(248,113,113,0.12)" if failed else C["surf2"],
                           "color":color,"fontFamily":"'IBM Plex Mono',monospace"}),
            ],style={"display":"flex","alignItems":"center"}),
            html.Div(r["description"],style={"fontSize":"9px","color":C["muted"],"marginTop":"2px"}),
        ],style={"marginBottom":"5px","padding":"7px 9px","borderRadius":"4px",
                 "background":"rgba(34,211,238,0.05)" if placed else "rgba(248,113,113,0.05)" if failed else C["surf2"],
                 "border":f"1px solid {color+'33' if ri<idx else C['border']}"}))

    # ── Status ──
    if G["running"]:
        s_txt = "● RUNNING"
        s_sty = {"fontFamily":"'IBM Plex Mono',monospace","fontSize":"10px","padding":"4px 10px",
                 "borderRadius":"2px","letterSpacing":"0.06em",
                 "background":"rgba(34,197,94,0.12)","color":"#86efac",
                 "border":"1px solid rgba(34,197,94,0.25)"}
    elif idx > 0:
        s_txt = "⏸ PAUSED"
        s_sty = {"fontFamily":"'IBM Plex Mono',monospace","fontSize":"10px","padding":"4px 10px",
                 "borderRadius":"2px","letterSpacing":"0.06em",
                 "background":"rgba(245,158,11,0.12)","color":"#fcd34d",
                 "border":"1px solid rgba(245,158,11,0.25)"}
    else:
        s_txt = "■ STOPPED"
        s_sty = {"fontFamily":"'IBM Plex Mono',monospace","fontSize":"10px","padding":"4px 10px",
                 "borderRadius":"2px","letterSpacing":"0.06em",
                 "background":"rgba(100,116,139,0.12)","color":C["muted"],
                 "border":f"1px solid rgba(100,116,139,0.25)"}

    if idx >= len(results) and not G["running"]:
        s_txt = "✅ TERMINÉ"
        s_sty["background"] = "rgba(34,211,238,0.12)"
        s_sty["color"]      = "#67e8f9"
        s_sty["border"]     = "1px solid rgba(34,211,238,0.25)"

    # ── DS tab styles ──
    ds_labels = ["Dataset 1","Dataset 2","Dataset 3","Dataset 4"]
    tab_styles = [{
        "fontFamily":"'IBM Plex Mono',monospace","fontSize":"11px","padding":"5px 14px",
        "background":C["accent"]+"25" if ds==G["ds"] else C["surf2"],
        "color":C["accent"] if ds==G["ds"] else C["muted"],
        "border":f"1px solid {C['accent'] if ds==G['ds'] else C['border']}",
        "borderRadius":"3px","cursor":"pointer",
    } for ds in ds_labels]

    return (kpis, fig_res, fig_lat, fig_topo,
            tl, svc_items, journal, int_items,
            s_txt, s_sty, *tab_styles)


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  IBN Node Monitor — Dashboard Amélioré")
    print("  Ouvrir : http://127.0.0.1:8050")
    print("="*50 + "\n")
    app.run(debug=False, port=8050)