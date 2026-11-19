#!/usr/bin/env python3
"""RG cluster monthly route planner.

Reads the RG Data Excel, schedules each site's required monthly visits
across Mon-Sun days (8 h per guard per day), builds road routes per day,
estimates fuel, and writes one map page (out/index.html) plus CSVs.

  python build.py RG_Data.xlsx --osrm http://localhost:5000
  python build.py RG_Data.xlsx --offline --cluster R6NGC2     # test only
"""
from concurrent.futures import ThreadPoolExecutor
import argparse, datetime, json, math, os, re, sys, time
import numpy as np
import pandas as pd
import requests

DAY_START_H = 8.0
DETOUR = 1.4
BAD_LEG_KM = 150
DELAY = 0.0


# get with a few retries, connections drop sometimes
def http_get(url, **kw):
    for k in range(4):
        try:
            return requests.get(url, headers={"User-Agent": "Mozilla/5.0 rg-route-planner"}, **kw)
        except requests.exceptions.RequestException:
            if k == 3: raise
            time.sleep(3 * (k + 1))


def hav_km(la1, lo1, la2, lo2):
    p = math.pi / 180
    a = (np.sin((la2 - la1) * p / 2) ** 2
         + np.cos(la1 * p) * np.cos(la2 * p) * np.sin((lo2 - lo1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


# load the excel
def load(path):
    # monthly tab
    names = pd.ExcelFile(path).sheet_names
    month_tab = next(n for n in names if not n.startswith("RG Cluster Schedule"))
    df = pd.read_excel(path, sheet_name=month_tab, header=2)
    df = df[df["FTAPID"].notna()].copy()
    df["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
    # swapped lat/lon
    sw = (df.lat > 30) & (df.lon < 30)
    df.loc[sw, ["lat", "lon"]] = df.loc[sw, ["lon", "lat"]].values
    bad = df.lat.isna() | df.lon.isna() | ~df.lat.between(4, 22) | ~df.lon.between(116, 128)
    rejects = df[bad].copy()
    rejects["reason"] = "no usable coordinates (cannot be placed on map)"
    df = df[~bad].copy()
    # group unassigned by province
    na = df["RG Cluster"].astype(str).str.contains("No Assigned", case=False)
    df.loc[na, "RG Cluster"] = "UNASSIGNED-" + df.loc[na, "Province"].astype(str)
    # blank or not required = 0 visits
    df["freq"] = pd.to_numeric(df["Visit Frequency per Month (Days)"], errors="coerce").fillna(0).astype(int)
    sch = pd.read_excel(path, sheet_name="RG Cluster Schedule & Budget", header=1)
    sch = sch[sch["RG Cluster (Unique)"].notna()].drop_duplicates("RG Cluster (Unique)")
    return df, rejects, sch.set_index("RG Cluster (Unique)")


def osrm_matrix(pts, base, block=50):
    n = len(pts)
    D = np.zeros((n, n)); T = np.zeros((n, n))
    # single site, nothing to route
    if n < 2:
        return D, T
    for i0 in range(0, n, block):
        for j0 in range(0, n, block):
            I = list(range(i0, min(i0 + block, n))); J = list(range(j0, min(j0 + block, n)))
            idx = I + [j for j in J if j not in I]
            coords = ";".join(f"{pts[k][1]:.6f},{pts[k][0]:.6f}" for k in idx)
            src = ";".join(str(idx.index(i)) for i in I)
            dst = ";".join(str(idx.index(j)) for j in J)
            r = http_get(f"{base}/table/v1/driving/{coords}",
                             params={"sources": src, "destinations": dst,
                                     "annotations": "distance,duration"}, timeout=120).json()
            time.sleep(DELAY)
            # bad block (unsnappable point etc), use straight-line guess
            if "distances" not in r:
                print(f"  table failed ({r.get('code')}), using straight-line for a block", file=sys.stderr)
                Do, To = offline_matrix([pts[k] for k in idx])
                a = [idx.index(i) for i in I]; b = [idx.index(j) for j in J]
                D[np.ix_(I, J)] = Do[np.ix_(a, b)]; T[np.ix_(I, J)] = To[np.ix_(a, b)]
                continue
            D[np.ix_(I, J)] = np.array(r["distances"], dtype=float) / 1000
            T[np.ix_(I, J)] = np.array(r["durations"], dtype=float) / 3600
    D = np.where(np.isnan(D), 9999, D); T = np.where(np.isnan(T), 999, T)
    return D, T


def offline_matrix(pts):
    la = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts])
    D = hav_km(la[:, None], lo[:, None], la[None, :], lo[None, :]) * DETOUR
    return D, D / 40.0


def osrm_route(pts, base):
    coords = ";".join(f"{p[1]:.6f},{p[0]:.6f}" for p in pts)
    r = http_get(f"{base}/route/v1/driving/{coords}",
                     params={"overview": "full", "geometries": "geojson"}, timeout=120).json()
    time.sleep(DELAY)
    pts_ll = [[round(c[1], 4), round(c[0], 4)] for c in r["routes"][0]["geometry"]["coordinates"]]
    out = [pts_ll[0]]
    for p in pts_ll[1:]:
        if p != out[-1]: out.append(p)
    return out


# 2-opt
def two_opt_open(order, T):
    order = list(order); n = len(order)

    def cost(o):
        return sum(T[o[k], o[k + 1]] for k in range(len(o) - 1))
    cur = cost(order)
    # cap passes, real roads are asymmetric
    for _ in range(50):
        improved = False
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                c = cost(cand)
                if c < cur - 1e-9:
                    order, cur, improved = cand, c, True
        if not improved: break
    return order


def best_insert(L, x, T):
    if not L:
        return 0, 0.0
    best = (0, T[x, L[0]])
    c = T[L[-1], x]
    if c < best[1]: best = (len(L), c)
    for p in range(1, len(L)):
        c = T[L[p - 1], x] + T[x, L[p]] - T[L[p - 1], L[p]]
        if c < best[1]: best = (p, c)
    return best


def fmt(h):
    m = int(round((h % 24) * 60)); return f"{m // 60:02d}:{m % 60:02d}"


def plan_cluster(cl, g, sch, args, days_in_month, first_day):
    g = g.reset_index(drop=True)
    n = len(g)
    pts = list(zip(g.lat, g.lon))
    D, T = offline_matrix(pts) if args.offline else osrm_matrix(pts, args.osrm)
    freq = np.minimum(g.freq.values, days_in_month)
    if args.route_not_required:
        freq = np.maximum(freq, 1)
    guards, visit_min, budget = 1, 60.0, None
    if cl in sch.index:
        wk = sch.loc[cl, "Total RG Hours Available for Rounds per Week"]
        if pd.notna(wk): guards = max(1, round(float(wk) / 35))
        vm = sch.loc[cl, "Routine Visit Duration (minutes)"]
        if pd.notna(vm): visit_min = float(vm)
        bd = sch.loc[cl, "Monthly Fuel Budget"]
        if pd.notna(bd): budget = float(bd)
    vh = visit_min / 60
    cap = args.hours_day
    routes = [[[] for _ in range(guards)] for _ in range(days_in_month)]
    load_h = [[0.0] * guards for _ in range(days_in_month)]
    sched = {i: [] for i in range(n)}
    # busiest sites first
    for i in sorted((i for i in range(n) if freq[i] > 0), key=lambda i: (-freq[i], pts[i][1])):
        f = int(freq[i])
        for k in range(f):
            lo = (k * days_in_month) // f
            hi = max(((k + 1) * days_in_month) // f, lo + 1)
            best = None
            for d in range(lo, hi):
                for s in range(guards):
                    pos, delta = best_insert(routes[d][s], i, T)
                    new = load_h[d][s] + delta + vh
                    cost = delta + max(0.0, new - cap) * 100 + 0.01 * load_h[d][s]
                    if best is None or cost < best[0]:
                        best = (cost, d, s, pos, delta)
            _, d, s, pos, delta = best
            routes[d][s].insert(pos, i)
            load_h[d][s] += delta + vh
            sched[i].append(d + 1)
    cache = {}
    # optimise all routes first, then fetch road lines in parallel
    opt = {}
    for d in range(days_in_month):
        for s_ in range(guards):
            if routes[d][s_]:
                opt[(d, s_)] = two_opt_open(routes[d][s_], T)
    if not args.offline:
        keys = {tuple(L) for (d, s_), L in opt.items() if len(L) > 1 and d < args.geom_days}
        def fetch(key):
            try:
                return key, osrm_route([pts[k] for k in key], args.osrm)
            except Exception as e:
                print(f"  road line failed ({cl}): {e}", file=sys.stderr)
                return key, None
        with ThreadPoolExecutor(max_workers=8) as ex:
            for key, val in ex.map(fetch, keys):
                cache[key] = val
    out_days, rows = [], []
    for d in range(days_in_month):
        date = first_day + datetime.timedelta(days=d)
        droutes = []
        for s in range(guards):
            if (d, s) not in opt: continue
            L = opt[(d, s)]
            dp = [pts[k] for k in L]
            line = [[round(a, 4), round(b, 4)] for a, b in dp]
            road = False
            if not args.offline and len(L) > 1 and d < args.geom_days:
                key = tuple(L)
                if cache.get(key):
                    line, road = cache[key], True
            clock = DAY_START_H; stops = []; km = dh = 0.0
            for q, k in enumerate(L):
                lkm, lh = (0.0, 0.0) if q == 0 else (D[L[q - 1], k], T[L[q - 1], k])
                km += lkm; dh += lh; clock += lh
                stops.append(dict(seq=q + 1, id=g.FTAPID[k], mno=str(g["MNO Site ID"][k]),
                                  lat=pts[k][0], lon=pts[k][1], muni=str(g.Municipality[k]),
                                  freq=int(freq[k]), visit_no=sched[k].index(d + 1) + 1,
                                  days=sched[k], leg_km=round(float(lkm), 1),
                                  leg_min=int(round(lh * 60)), arrive=fmt(clock),
                                  depart=fmt(clock + vh), flag=bool(lkm > BAD_LEG_KM)))
                clock += vh
            vis_h = len(L) * vh
            total = dh + vis_h
            km_r = float(km)
            droutes.append(dict(slot=s + 1, km=round(km_r, 1), drive_h=round(float(dh), 2),
                                visit_h=round(vis_h, 2), total_h=round(float(total), 2),
                                over=bool(total > cap + 1e-9), road=road, line=line, stops=stops))
            for st in stops:
                rows.append(dict(rg_cluster=cl, day=d + 1, date=date.isoformat(), route=s + 1,
                                 **{k: v for k, v in st.items() if k != "days"},
                                 visit_days=" ".join(map(str, st["days"]))))
        km_day = sum(r["km"] for r in droutes)
        fuel_l = km_day / args.kmpl
        out_days.append(dict(day=d + 1, label=date.strftime("%a %d %b"), date=date.isoformat(),
                             visits=sum(len(r["stops"]) for r in droutes), km=round(km_day, 1),
                             fuel_l=round(fuel_l, 2), fuel_php=round(fuel_l * args.fuel_price),
                             routes=droutes))
    worked = [x for x in out_days if x["routes"]]
    tot_km = sum(x["km"] for x in out_days)
    tot_fuel = sum(x["fuel_php"] for x in out_days)
    req_hours = sum(r["total_h"] for x in out_days for r in x["routes"])
    mix = {}
    for i in range(n):
        mix.setdefault(int(freq[i]), []).append(g.FTAPID[i])
    summary = dict(sites=n, required_visits=int(freq.sum()), not_required=int((freq == 0).sum()),
                   guards=guards, days_worked=len(worked), total_km=round(tot_km, 1),
                   avg_km_day=round(tot_km / max(len(worked), 1), 1),
                   avg_fuel_l_day=round(tot_km / args.kmpl / max(len(worked), 1), 2),
                   avg_fuel_php_day=round(tot_fuel / max(len(worked), 1)),
                   month_fuel_php=round(tot_fuel), budget=budget,
                   hours_needed=round(req_hours, 1),
                   hours_available=round(guards * cap * days_in_month, 1),
                   over_routes=sum(1 for x in out_days for r in x["routes"] if r["over"]))
    idle = [dict(id=g.FTAPID[i], lat=pts[i][0], lon=pts[i][1]) for i in range(n) if freq[i] == 0]
    res = dict(cluster=cl, agency=str(g["RG Agency"].iloc[0]),
               rg=", ".join(sorted(set(g["RG Name (New)"].dropna().astype(str)))),
               hours_day=cap, visit_min=visit_min, nd=days_in_month, kmpl=args.kmpl,
               price=args.fuel_price, summary=summary,
               mix={str(k): v for k, v in sorted(mix.items(), reverse=True)},
               idle=idle, days=out_days)
    return res, rows


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>RG Monthly Routes</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
body{font:14px system-ui,sans-serif;margin:0;color:#1d2433}
header{padding:10px 16px;background:#102a43;color:#fff;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
select,button{font-size:14px;padding:4px 8px}
#map{height:58vh}
#sum{padding:14px 18px;max-width:1150px}
.kpi{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:10px}
.kpi div{background:#f1f5f9;border-radius:8px;padding:8px 14px;min-width:110px}.kpi b{display:block;font-size:19px}
.rt{border-left:6px solid;padding:6px 12px;margin:8px 0;background:#fafbfc}
.warn{color:#b42318;font-weight:600}.muted{color:#667085;font-size:12.5px}
.stops{color:#475467;font-size:13px;line-height:1.6}
#bars{display:flex;gap:2px;align-items:flex-end;height:46px;margin:6px 0 12px}
#bars div{flex:1;background:#9db4cf;cursor:pointer;min-height:2px}#bars div.sel{background:#102a43}
h4{margin:14px 0 4px}
</style></head><body>
<header><b>RG monthly routes</b><label>Cluster <select id="cl"></select></label>
<button id="prev">&#9664;</button><select id="day"></select><button id="next">&#9654;</button>
<span id="meta"></span></header>
<div id="map"></div><div id="sum"></div>
<script>
let INDEX=[];const CACHE={};
const COL=['#1f77b4','#d62728','#2ca02c','#9467bd','#ff7f0e','#17becf'];
const map=L.map('map');L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; OpenStreetMap'}).addTo(map);
let layers=[],cur=null,di=0;
const cl=document.getElementById('cl'),dsel=document.getElementById('day');
const P=n=>'₱'+Math.round(n).toLocaleString();
async function showCluster(){
 const m=INDEX[cl.value];document.getElementById('sum').innerHTML='Loading...';
 if(!CACHE[m.file])CACHE[m.file]=await (await fetch('data/'+m.file)).json();
 cur=CACHE[m.file];dsel.innerHTML='';
 cur.days.forEach((d,i)=>dsel.add(new Option(`${d.label}: ${d.visits} visits, ${d.km} km`,i)));
 di=Math.max(0,cur.days.findIndex(d=>d.routes.length));dsel.value=di;
 document.getElementById('meta').textContent=`${cur.agency}${cur.rg?' | '+cur.rg:''}`;
 render();}
function render(){
 layers.forEach(l=>map.removeLayer(l));layers=[];const b=[];const d=cur.days[di];
 cur.idle.forEach(s=>{layers.push(L.circleMarker([s.lat,s.lon],{radius:4,color:'#98a2b3',weight:1,fillOpacity:.6}).bindPopup(`<b>${s.id}</b><br>Visit not required`).addTo(map));b.push([s.lat,s.lon]);});
 d.routes.forEach((r,i)=>{const c=COL[i%COL.length];
  layers.push(L.polyline(r.line,{color:c,weight:4,opacity:.85,dashArray:r.road?null:'6 6'}).addTo(map));
  r.stops.forEach(s=>{b.push([s.lat,s.lon]);
   layers.push(L.marker([s.lat,s.lon],{icon:L.divIcon({className:'',html:`<div style="background:${c};color:#fff;border-radius:50%;width:22px;height:22px;line-height:22px;text-align:center;font-size:11px;border:2px solid #fff">${s.seq}</div>`,iconSize:[22,22]})})
   .bindPopup(`<b>${s.id}</b> (${s.mno})<br>${s.muni}<br>${d.label}, route ${r.slot}, stop ${s.seq}<br>Arrive ${s.arrive} - leave ${s.depart}<br>From previous: ${s.leg_km} km, ${s.leg_min} min<br>Required ${s.freq}x per month (this is visit ${s.visit_no})<br>Visit days: ${s.days.join(', ')}`).addTo(map));});});
 if(b.length)map.fitBounds(b,{padding:[30,30]});
 text();}
function text(){
 const s=cur.summary,d=cur.days[di],mx=Math.max(...cur.days.map(x=>x.km),1);
 const pct=s.budget?Math.round(100*s.month_fuel_php/s.budget):null;
 const util=Math.round(100*s.hours_needed/s.hours_available);
 let h=`<div class="kpi"><div><b>${s.sites}</b>sites</div><div><b>${s.required_visits}</b>visits required per month</div>
 <div><b>${s.guards}</b>guard(s), ${cur.hours_day} h/day</div><div><b>${s.avg_km_day} km</b>avg per working day</div>
 <div><b>${cur.days.length?s.avg_fuel_l_day:0} L / ${P(s.avg_fuel_php_day)}</b>est. fuel per day</div>
 <div><b>${P(s.month_fuel_php)}</b>est. fuel per month${s.budget?`<br><span class="${pct>100?'warn':''}">${pct}% of ${P(s.budget)} budget</span>`:''}</div>
 <div><b>${util}%</b>of guard hours used (${s.hours_needed} of ${s.hours_available} h)</div></div>
 <div class="muted">Fuel assumes ${cur.kmpl} km/L at ${P(cur.price)} per liter and counts only travel between sites (not to or from the guard's home). Work days are Mon to Sun. Visit time ${cur.visit_min} min per site. Dashed lines are straight-line approximations.</div>`;
 if(s.over_routes)h+=`<p class="warn">${s.over_routes} route(s) go over ${cur.hours_day} h in the month: this cluster needs more guards, fewer visits, or longer days.</p>`;
 h+=`<h4>Distance per day (click a bar)</h4><div id="bars">${cur.days.map((x,i)=>`<div class="${i==di?'sel':''}" data-i="${i}" title="${x.label}: ${x.km} km" style="height:${Math.max(2,Math.round(44*x.km/mx))}px"></div>`).join('')}</div>`;
 h+=`<h4>${d.label}: ${d.visits} visits, ${d.km} km, est. fuel ${d.fuel_l} L (${P(d.fuel_php)})</h4>`;
 if(!d.routes.length)h+='<p>No visits scheduled this day.</p>';
 d.routes.forEach((r,i)=>{h+=`<div class="rt" style="border-color:${COL[i%COL.length]}"><b>Route ${r.slot}</b>: ${r.stops.length} sites, ${r.km} km, drive ${r.drive_h} h + visits ${r.visit_h} h = ${r.total_h} h${r.over?' <span class="warn">over '+cur.hours_day+' h</span>':''}${r.stops.some(x=>x.flag)?' <span class="warn">leg over 150 km: check coordinates or water crossing</span>':''}
  <div class="stops">${r.stops.map(x=>`${x.seq}. ${x.id} ${x.arrive}${x.seq>1?` (+${x.leg_km} km, ${x.leg_min} min)`:''}`).join(' &rarr; ')}</div></div>`;});
 h+=`<h4>Required visits per month</h4>`;
 Object.entries(cur.mix).forEach(([f,ids])=>{h+=`<div class="stops"><b>${f=='0'?'Not required':f+'x per month'}</b> (${ids.length} sites): ${ids.join(', ')}</div>`;});
 document.getElementById('sum').innerHTML=h;
 document.querySelectorAll('#bars div').forEach(el=>el.onclick=()=>{di=+el.dataset.i;dsel.value=di;render();});}
cl.onchange=showCluster;dsel.onchange=()=>{di=+dsel.value;render();};
document.getElementById('prev').onclick=()=>{if(di>0){di--;dsel.value=di;render();}};
document.getElementById('next').onclick=()=>{if(di<cur.days.length-1){di++;dsel.value=di;render();}};
fetch('data/index.json').then(r=>r.json()).then(j=>{INDEX=j;INDEX.forEach((c,i)=>cl.add(new Option(`${c.cluster} (${c.sites} sites)`,i)));showCluster();})
.catch(()=>{document.getElementById('sum').innerHTML='Could not load the data. Open this page through a web server (add --serve), not by double-clicking it.';});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx"); ap.add_argument("--osrm", default="http://localhost:5000")
    ap.add_argument("--offline", action="store_true"); ap.add_argument("--cluster", nargs="+")
    ap.add_argument("--out", default="out"); ap.add_argument("--serve", action="store_true")
    ap.add_argument("--public", action="store_true"); ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--month", default="2026-11")
    ap.add_argument("--hours-day", type=float, default=8.0)
    ap.add_argument("--kmpl", type=float, default=35.0)
    ap.add_argument("--fuel-price", type=float, default=70.0)
    ap.add_argument("--geom-days", type=int, default=999)
    ap.add_argument("--route-not-required", action="store_true")
    args = ap.parse_args()
    global DELAY
    DELAY = args.delay
    os.makedirs(args.out, exist_ok=True)
    src = args.xlsx
    if src.startswith("http"):
        saved = os.path.join(args.out, "_source.xlsx")
        try:
            r = http_get(src, timeout=120); r.raise_for_status()
            open(saved, "wb").write(r.content)
        except requests.exceptions.RequestException as e:
            # use the last downloaded copy if there is one
            if not os.path.exists(saved):
                sys.exit(f"Could not download the sheet ({e}). Try again in a minute.")
            print(f"Download failed, using the last saved copy ({e})", file=sys.stderr)
        src = saved
    # check osrm is up
    if not args.offline:
        try:
            http_get(f"{args.osrm}/route/v1/driving/121.0,14.6;121.01,14.61", timeout=15).raise_for_status()
        except Exception as e:
            sys.exit(f"Cannot reach OSRM at {args.osrm} ({e}). Start OSRM or use --osrm https://router.project-osrm.org")
    first = datetime.date.fromisoformat(args.month + "-01")
    nxt = (first.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
    nd = (nxt - first).days
    df, rejects, sch = load(src)
    rejects[["FTAPID", "RG Cluster", "Latitude", "Longitude", "reason"]].to_csv(
        f"{args.out}/excluded_sites.csv", index=False)
    results, rows = [], []
    groups = [(c, g) for c, g in df.groupby("RG Cluster") if not args.cluster or c in args.cluster]
    for k, (cl, g) in enumerate(groups, 1):
        res, r = plan_cluster(cl, g, sch, args, nd, first)
        results.append(res); rows += r
        s = res["summary"]
        print(f"[{k}/{len(groups)}] {cl}: {s['sites']} sites, {s['required_visits']} visits, "
              f"{s['avg_km_day']} km/day, fuel P{s['month_fuel_php']}/mo", file=sys.stderr)
    pd.DataFrame(rows).to_csv(f"{args.out}/route_stops.csv", index=False)
    pd.DataFrame([dict(rg_cluster=r["cluster"], day=d["day"], date=d["date"], route=x["slot"],
                       sites=len(x["stops"]), km=x["km"], drive_h=x["drive_h"], visit_h=x["visit_h"],
                       total_h=x["total_h"], over_hours=x["over"])
                  for r in results for d in r["days"] for x in d["routes"]]
                 ).to_csv(f"{args.out}/route_days.csv", index=False)
    pd.DataFrame([dict(rg_cluster=r["cluster"], **r["summary"]) for r in results]
                 ).to_csv(f"{args.out}/cluster_summary.csv", index=False)
    if args.public:
        for r in results:
            r["rg"] = ""; r["agency"] = "hidden"
            for s in r["idle"]:
                s["lat"] = round(s["lat"], 3); s["lon"] = round(s["lon"], 3)
            for d in r["days"]:
                for x in d["routes"]:
                    x["line"] = [[round(a, 3), round(b, 3)] for a, b in x["line"]]
                    for st in x["stops"]:
                        st["mno"] = "hidden"; st["muni"] = ""
                        st["lat"] = round(st["lat"], 3); st["lon"] = round(st["lon"], 3)
    # web files go in out/site, csv files stay in out
    site = os.path.join(args.out, "site")
    os.makedirs(os.path.join(site, "data"), exist_ok=True)
    dump = lambda o: o.item() if hasattr(o, "item") else str(o)
    index = []
    for r in results:
        fn = re.sub(r"[^A-Za-z0-9_-]", "_", r["cluster"]) + ".json"
        with open(os.path.join(site, "data", fn), "w", encoding="utf-8") as f:
            json.dump(r, f, separators=(",", ":"), default=dump)
        index.append(dict(cluster=r["cluster"], sites=r["summary"]["sites"], file=fn))
    with open(os.path.join(site, "data", "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))
    with open(os.path.join(site, "index.html"), "w", encoding="utf-8") as f:
        f.write(HTML)
    print("done:", site, file=sys.stderr)
    # osm tiles need http
    if args.serve:
        import functools, http.server, webbrowser
        h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=site)
        url = "http://localhost:8000/index.html"
        print("open", url, "(Ctrl+C to stop)", file=sys.stderr)
        webbrowser.open(url)
        http.server.ThreadingHTTPServer(("localhost", 8000), h).serve_forever()


if __name__ == "__main__":
    main()
