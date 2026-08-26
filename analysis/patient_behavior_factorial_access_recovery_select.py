#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

POLICIES = ("horizon_only","reservation_only","both_flexible")
CELL = ["horizon_days","Q","window"]
UTIL_TOL = 0.005
WIN = 0.005
NEUTRAL = 0.005
MIN_W = 2

def dedupe(x):
    if x.empty: return x.copy()
    x=x.copy()
    rank={"exact":0,"coarse":1,"fine_average_utilization":2,
          "fine_priority_weighted_utilization":3,"fine_access_balance":4,
          "fine_baseline_access":5}
    x["_r"]=x["arm"].map(rank).fillna(99) if "arm" in x else 99
    return (x.sort_values(CELL+["seed","_r"],kind="stable")
             .drop_duplicates(CELL+["seed"],keep="first")
             .drop(columns="_r"))

def agg(x,n):
    if x.empty: return pd.DataFrame()
    g=x.groupby(CELL,as_index=False).agg(
        U=("average_utilization","mean"),
        S1=("class_1_percent_serviced","mean"),
        S2=("class_2_percent_serviced","mean"),
        n=("seed","nunique"))
    return g[g.n==n].copy()

def ref_pool(raw,p):
    b=raw[raw.stage=="baseline"]
    h=raw[raw.stage=="horizon_only"]
    if p=="horizon_only": return dedupe(h)
    if p=="reservation_only":
        return dedupe(pd.concat([raw[raw.stage==p],b],ignore_index=True))
    return dedupe(pd.concat([raw[raw.stage==p],h],ignore_index=True))

def search_pool(raw,p):
    if p=="horizon_only":
        return dedupe(raw[raw.stage==p])
    return dedupe(raw[(raw.stage==p)&(raw.Q>0)&(raw.window>=MIN_W)])

def baseline(raw,n):
    b=dedupe(raw[raw.stage=="baseline"])
    if b.seed.nunique()!=n: raise ValueError("incomplete baseline")
    return dict(U=b.average_utilization.mean(),
                S1=b.class_1_percent_serviced.mean(),
                S2=b.class_2_percent_serviced.mean())

def avgopt(raw,p,ref,n):
    g=agg(ref_pool(raw,p),n)
    h,q,w=int(ref.selected_horizon_days),int(ref.selected_Q),int(ref.selected_window)
    z=g[(g.horizon_days==h)&(g.Q==q)&(g.window==w)]
    if len(z)!=1: raise ValueError(f"cannot recover avg-opt {p} {h,q,w}")
    r=z.iloc[0]
    return dict(U=float(r.U),S1=float(r.S1),S2=float(r.S2),H=h,Q=q,W=w)

def rank_access(g):
    if g.empty:return None
    return g.sort_values(["S1","S2","U","Q","window","horizon_days"],
                         ascending=[False,False,False,True,True,True],
                         kind="stable").iloc[0]

def rank_fav(g):
    if g.empty:return None
    return g.sort_values(["d1b","d2b","U","Q","window","horizon_days"],
                         ascending=[False,False,False,True,True,True],
                         kind="stable").iloc[0]

def row(bg,p,typ,r,a,b,nf):
    base=dict(background_id=bg,policy=p,candidate_type=typ,
              candidate_exists=r is not None,
              utilization_tolerance=UTIL_TOL,win_threshold=WIN,neutral_band=NEUTRAL,
              avg_opt_horizon_days=a["H"],avg_opt_Q=a["Q"],avg_opt_window=a["W"],
              avg_opt_average_utilization=a["U"],
              avg_opt_class_1_percent_serviced=a["S1"],
              avg_opt_class_2_percent_serviced=a["S2"],
              baseline_average_utilization=b["U"],
              baseline_class_1_percent_serviced=b["S1"],
              baseline_class_2_percent_serviced=b["S2"],
              n_feasible_cells=nf)
    if r is None:
        return {**base,**{k:np.nan for k in [
            "selected_horizon_days","selected_Q","selected_window",
            "search_average_utilization","search_class_1_percent_serviced",
            "search_class_2_percent_serviced","delta_utilization_vs_avg_opt",
            "delta_sr1_vs_avg_opt","delta_sr2_vs_avg_opt",
            "delta_utilization_vs_baseline","delta_sr1_vs_baseline",
            "delta_sr2_vs_baseline"]},
            "same_as_avg_utilization_optimum":False,
            "genuinely_active_reservation":False}
    h,q,w=int(r.horizon_days),int(r.Q),int(r.window)
    return {**base,
        "selected_horizon_days":h,"selected_Q":q,"selected_window":w,
        "search_average_utilization":float(r.U),
        "search_class_1_percent_serviced":float(r.S1),
        "search_class_2_percent_serviced":float(r.S2),
        "delta_utilization_vs_avg_opt":float(r.U-a["U"]),
        "delta_sr1_vs_avg_opt":float(r.S1-a["S1"]),
        "delta_sr2_vs_avg_opt":float(r.S2-a["S2"]),
        "delta_utilization_vs_baseline":float(r.U-b["U"]),
        "delta_sr1_vs_baseline":float(r.S1-b["S1"]),
        "delta_sr2_vs_baseline":float(r.S2-b["S2"]),
        "same_as_avg_utilization_optimum":bool((h,q,w)==(a["H"],a["Q"],a["W"])),
        "genuinely_active_reservation":bool(q>0 and w>=MIN_W)}

def select(raw,bg,p,ref):
    n=int(ref.search_n_seeds)
    b=baseline(raw,n); a=avgopt(raw,p,ref,n)
    g=agg(search_pool(raw,p),n)
    g=g[g.U>=a["U"]-UTIL_TOL].copy()
    if not g.empty:
        g["d1b"]=g.S1-b["S1"]; g["d2b"]=g.S2-b["S2"]

    access=rank_access(g)

    ww = g[(g.d1b>=WIN-1e-12)&(g.d2b>=WIN-1e-12)] if not g.empty else g
    wn = g[(g.d1b>=WIN-1e-12)&(g.d2b.abs()<=NEUTRAL+1e-12)&
           (g.d2b<WIN-1e-12)] if not g.empty else g
    bestww=rank_fav(ww); bestwn=rank_fav(wn)

    keepwn = bestwn is not None and (
        bestww is None or float(bestwn.d1b)>float(bestww.d1b)+1e-12
    )

    return [
        row(bg,p,"access_recovery",access,a,b,len(g)),
        row(bg,p,"best_win_win",bestww,a,b,len(g)),
        row(bg,p,"c1_win_c2_neutral_if_better",bestwn if keepwn else None,a,b,len(g)),
    ]

def summary(d):
    out=[]
    for p in POLICIES:
        x=d[d.policy==p]
        a=x[x.candidate_type=="access_recovery"]
        ww=x[x.candidate_type=="best_win_win"]
        wn=x[x.candidate_type=="c1_win_c2_neutral_if_better"]
        ae=a[a.candidate_exists]
        out.append(dict(
            policy=p,backgrounds=x.background_id.nunique(),
            access_recovery_exists=int(a.candidate_exists.sum()),
            median_delta_u_vs_avg_opt=ae.delta_utilization_vs_avg_opt.median(),
            median_delta_sr1_vs_avg_opt=ae.delta_sr1_vs_avg_opt.median(),
            median_delta_sr2_vs_avg_opt=ae.delta_sr2_vs_avg_opt.median(),
            share_c1_positive_vs_avg_opt=float((ae.delta_sr1_vs_avg_opt>0).mean()),
            share_c1_ge_0_5pp_vs_avg_opt=float((ae.delta_sr1_vs_avg_opt>=0.005).mean()),
            best_win_win_exists=int(ww.candidate_exists.sum()),
            extra_c1_win_c2_neutral_exists=int(wn.candidate_exists.sum())))
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()
    root=args.output_dir.resolve()
    rawdir=root/"search"/"raw"
    sel=pd.read_csv(root/"selection"/"selected_cells.csv")
    refs=(sel[(sel.selection_objective=="average_utilization")&
              sel.policy.isin(POLICIES)]
          .set_index(["background_id","policy"]))
    rows=[]
    files=sorted(rawdir.glob("*.csv"))
    for i,f in enumerate(files,1):
        raw=pd.read_csv(f); bg=str(raw.source_background_id.iloc[0])
        for p in POLICIES:
            ref=refs.loc[(bg,p)]
            if isinstance(ref,pd.DataFrame): ref=ref.iloc[0]
            rows.extend(select(raw,bg,p,ref))
        if i%50==0 or i==len(files): print(f"Processed {i}/{len(files)}")
    d=pd.DataFrame(rows)
    dest=root/"access_recovery_optimization"; dest.mkdir(parents=True,exist_ok=True)
    d.to_csv(dest/"access_recovery_candidates.csv",index=False)
    sm=summary(d); sm.to_csv(dest/"selection_summary.csv",index=False)
    print("\\nSEARCH-SEED DIAGNOSTIC ONLY:")
    print(sm.to_string(index=False))
    print("\\nWrote:",dest/"access_recovery_candidates.csv")
    print("Wrote:",dest/"selection_summary.csv")

if __name__=="__main__":
    main()
