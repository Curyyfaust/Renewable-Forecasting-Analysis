"""31省两阶段 PyPSA-China 全年风电预测调度。"""
from __future__ import annotations
import json, logging
from datetime import datetime, timezone
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

P=Path(__file__).resolve().parent; ROOT=P.parent
DATA=ROOT/"dataset"; POWER=DATA/"provincial_hourly_2018_2023"
CAP=DATA/"pypsa_china_thu_existing"; FACTOR=DATA/"carbon_emission_factors"
FORECAST=ROOT/"results"/"wind_forecast_2020_all_models.csv"
OUT=ROOT/"results"/"pypsa_two_stage_2020"; FIG=ROOT/"results"/"diagnostics"
MODELS=("CNN","LSTM","Transformer","DNN")
KEY={"CNN":"cnn","LSTM":"lstm","Transformer":"transformer","DNN":"dnn"}
MC={"hydro":20.,"nuclear":60.,"coal":280.,"gas":550.}
UP={"hydro":30.,"nuclear":120.,"coal":80.,"gas":120.}
DOWN={"hydro":15.,"nuclear":60.,"coal":40.,"gas":60.}
TARGET={"wind":281530.,"solar":253430.,"hydro":370160.,"nuclear":49890.,"coal_gas":1245170.}

def power(name):
    f=pd.read_csv(POWER/f"provincial_hourly_{name}_2018_2023.csv",parse_dates=["timestamp_beijing"])
    return f.set_index("timestamp_beijing").sort_index().astype(float)

def cap_file(name, provinces):
    d=pd.read_csv(CAP/name,index_col=0).apply(pd.to_numeric,errors="coerce")
    cols=[c for c in d.columns if str(c).isdigit() and int(c)<=2020]
    return (d[cols].sum(1) if cols else d.sum(1)).reindex(provinces).fillna(0.)

def capacities(provinces):
    c=pd.DataFrame(index=provinces)
    c["wind"]=cap_file("onwind_capacity.csv",provinces); c["solar"]=cap_file("solar_capacity.csv",provinces)
    c["hydro"]=cap_file("hydroelectricity_capacity.csv",provinces); c["nuclear"]=cap_file("nuclear_capacity.csv",provinces)
    c["coal"]=cap_file("coal_capacity.csv",provinces)+cap_file("chp_coal_capacity.csv",provinces)
    c["gas"]=cap_file("ocgt_capacity.csv",provinces)+cap_file("chp_gas_capacity.csv",provinces)
    for t in ("wind","solar","hydro","nuclear"): c[t]*=TARGET[t]/c[t].sum()
    c[["coal","gas"]]*=TARGET["coal_gas"]/c[["coal","gas"]].to_numpy().sum()
    return c

def inputs():
    fc=pd.read_csv(FORECAST,parse_dates=["forecast_origin_beijing","valid_time_beijing"])
    meta={"model","forecast_origin_beijing","lead_hour","valid_time_beijing"}
    provinces=[c for c in fc.columns if c not in meta]
    times=pd.DatetimeIndex(sorted(fc.valid_time_beijing.unique())); times=times[~((times.month==2)&(times.day==29))]
    if len(times)!=8760: raise ValueError(f"需要8760小时，实际{len(times)}")
    wind=power("wind").reindex(times)[provinces]; solar=power("solar").reindex(times)[provinces]; load=power("load").reindex(times)[provinces]
    if wind.isna().any().any() or solar.isna().any().any() or load.isna().any().any(): raise ValueError("真实风光负荷存在缺失")
    scenarios={m:fc.loc[fc.model==KEY[m]].set_index("valid_time_beijing").reindex(times)[provinces].clip(lower=0.) for m in MODELS}
    if any(x.isna().any().any() for x in scenarios.values()): raise ValueError("预测存在缺失")
    caps=capacities(provinces)
    for p in provinces:
        caps.at[p,"wind"]=max(caps.at[p,"wind"],1.001*max(wind[p].max(),*(x[p].max() for x in scenarios.values())))
        caps.at[p,"solar"]=max(caps.at[p,"solar"],1.001*solar[p].max())
    edges=pd.read_csv(P/"resources/data/grids/edges_current.csv",header=None,names=["bus0","bus1","p_nom"])
    edges=edges.query("bus0 in @provinces and bus1 in @provinces").reset_index(drop=True)
    return dict(times=times,provinces=provinces,wind=wind,solar=solar,load=load,scenarios=scenarios,caps=caps,edges=edges)

def base(times, provinces, load, edges):
    n=pypsa.Network(); n.set_snapshots(times); n.snapshot_weightings.loc[:,:]=1.
    for p in provinces: n.add("Bus",p); n.add("Load",f"load::{p}",bus=p,p_set=load[p])
    for i,e in edges.iterrows(): n.add("Link",f"line::{i}",bus0=e.bus0,bus1=e.bus1,p_nom=float(e.p_nom),p_min_pu=-1.,efficiency=1.)
    return n

def day_ahead(times, provinces, load, solar, forecast, caps, edges):
    n=base(times,provinces,load,edges)
    for p in provinces:
        for t,x in (("wind",forecast),("solar",solar)):
            c=caps.at[p,t]; n.add("Generator",f"{t}::{p}",bus=p,carrier=t,p_nom=c,marginal_cost=0.,p_max_pu=(x[p]/c).clip(0,1))
        for t,cost in MC.items():
            c=caps.at[p,t]
            if c>0: n.add("Generator",f"{t}::{p}",bus=p,carrier=t,p_nom=c,marginal_cost=cost)
        n.add("Generator",f"shed::{p}",bus=p,carrier="shed",p_nom=2*load[p].max(),marginal_cost=10000.)
    return n

def real_time(da, provinces, load, solar, observed, caps, edges):
    n=base(da.snapshots,provinces,load,edges)
    for p in provinces:
        for t,x,value in (("wind",observed,-200.),("solar",solar,-150.)):
            c=caps.at[p,t]; n.add("Generator",f"{t}::{p}",bus=p,carrier=t,p_nom=c,marginal_cost=value,p_max_pu=(x[p]/c).clip(0,1))
        for t in MC:
            old=f"{t}::{p}"
            if old not in da.generators.index: continue
            c=float(da.generators.at[old,"p_nom"]); q=da.generators_t.p[old].clip(0,c)
            # 日前计划q固定；实时阶段只能使用显式上调/下调变量修正。
            n.add("Generator",f"base::{old}",bus=p,carrier=f"base_{t}",p_nom=c,marginal_cost=0.,p_min_pu=q/c,p_max_pu=q/c)
            n.add("Generator",f"up::{old}",bus=p,carrier=f"up_{t}",p_nom=c,marginal_cost=UP[t],p_max_pu=(c-q)/c)
            n.add("Generator",f"down::{old}",bus=p,carrier=f"down_{t}",p_nom=c,marginal_cost=-DOWN[t],p_min_pu=-q/c,p_max_pu=0.)
        n.add("Generator",f"shed::{p}",bus=p,carrier="shed",p_nom=2*load[p].max(),marginal_cost=10000.)
    return n

def solve(n):
    status,condition=n.optimize(solver_name="highs",log_to_console=False)
    if status!="ok": raise RuntimeError((status,condition))

def chunk(model,times,d,factors):
    ps=d["provinces"]; load=d["load"].loc[times]; solar=d["solar"].loc[times]
    obs=d["wind"].loc[times]; pred=obs if model=="Perfect" else d["scenarios"][model].loc[times]
    da=day_ahead(times,ps,load,solar,pred,d["caps"],d["edges"]); solve(da)
    rt=real_time(da,ps,load,solar,obs,d["caps"],d["edges"]); solve(rt)
    up=pd.Series(0.,index=times); down=up.copy(); cost=up.copy(); co2=up.copy(); gen={t:up.copy() for t in MC}
    for p in ps:
        for t in MC:
            old=f"{t}::{p}"
            if old not in da.generators.index: continue
            u=rt.generators_t.p[f"up::{old}"]; dn=rt.generators_t.p[f"down::{old}"]; b=rt.generators_t.p[f"base::{old}"]
            actual=(b+u+dn).clip(lower=0); up+=u; down+=-dn; cost+=u*UP[t]+dn*(-DOWN[t]); co2+=actual*float(factors[t]); gen[t]+=actual
    shed=rt.generators_t.p.loc[:,rt.generators.carrier=="shed"].sum(1)
    fuel=sum(gen[t]*MC[t] for t in MC)
    out=pd.DataFrame(dict(model=model,timestamp_beijing=times,forecast_wind_mw=pred.sum(1),observed_wind_mw=obs.sum(1),forecast_error_mw=pred.sum(1)-obs.sum(1),load_mw=load.sum(1),solar_mw=solar.sum(1),up_balancing_mw=up,down_balancing_mw=down,balancing_cost_cny=cost,realised_fuel_cost_cny=fuel,total_operating_cost_cny=fuel+cost,direct_co2_t=co2,load_shedding_mw=shed))
    for t,x in gen.items(): out[f"{t}_generation_mw"]=x.to_numpy()
    return out

def summaries(h):
    x=h.copy(); x["date"]=pd.to_datetime(x.timestamp_beijing).dt.date
    if "total_cost_difference_cny" not in x: x["total_cost_difference_cny"]=np.nan
    daily=x.groupby(["model","date"],as_index=False).agg(forecast_mae_mw=("forecast_error_mw",lambda z:z.abs().mean()),balancing_cost_million_cny=("balancing_cost_cny",lambda z:z.sum()/1e6),fuel_cost_million_cny=("realised_fuel_cost_cny",lambda z:z.sum()/1e6),total_cost_million_cny=("total_operating_cost_cny",lambda z:z.sum()/1e6),total_cost_difference_million_cny=("total_cost_difference_cny",lambda z:z.sum()/1e6),direct_co2_mt=("direct_co2_t",lambda z:z.sum()/1e6),up_balancing_gwh=("up_balancing_mw",lambda z:z.sum()/1e3),down_balancing_gwh=("down_balancing_mw",lambda z:z.sum()/1e3),load_shedding_mwh=("load_shedding_mw","sum"))
    annual=daily.groupby("model",as_index=False).agg(forecast_mae_mw=("forecast_mae_mw","mean"),balancing_cost_million_cny=("balancing_cost_million_cny","sum"),fuel_cost_million_cny=("fuel_cost_million_cny","sum"),total_cost_million_cny=("total_cost_million_cny","sum"),total_cost_difference_million_cny=("total_cost_difference_million_cny","sum"),direct_co2_mt=("direct_co2_mt","sum"),up_balancing_twh=("up_balancing_gwh",lambda z:z.sum()/1e3),down_balancing_twh=("down_balancing_gwh",lambda z:z.sum()/1e3),load_shedding_gwh=("load_shedding_mwh",lambda z:z.sum()/1e3))
    annual.insert(1,"hours",8760); return daily,annual

def plot(h,daily,annual):
    colors={"CNN":"#4C88CF","LSTM":"#22B18A","Transformer":"#F39C3D","DNN":"#A45FD2"}; pos=np.arange(4)
    plt.rcParams.update({"font.family":"Arial","font.size":10.5,"axes.titlesize":14,"axes.labelsize":11.5,"xtick.labelsize":9.5,"ytick.labelsize":9.5,"axes.linewidth":1.1,"pdf.fonttype":42})
    fig=plt.figure(figsize=(15.5,5.4)); outer=fig.add_gridspec(1,3,width_ratios=[1.25,1,1],wspace=.48)
    axa=fig.add_subplot(outer[0,0])
    gb=outer[0,1].subgridspec(1,2,width_ratios=[.68,.32],wspace=.08)
    gc=outer[0,2].subgridspec(1,2,width_ratios=[.68,.32],wspace=.08)
    axb=fig.add_subplot(gb[0,0]); axbv=fig.add_subplot(gb[0,1])
    axc=fig.add_subplot(gc[0,0]); axcv=fig.add_subplot(gc[0,1])

    err=[h.loc[h.model==m,"forecast_error_mw"].to_numpy()/1e3 for m in MODELS]
    rng=np.random.default_rng(42)
    for i,(values,m) in enumerate(zip(err,MODELS)):
        ids=rng.choice(len(values),size=min(120,len(values)),replace=False)
        jitter=rng.normal(0,.035,len(ids))
        axa.scatter(np.full(len(ids),i-.24)+jitter,values[ids],s=8,color=colors[m],alpha=.62,edgecolors="none",zorder=2)
        violin=axa.violinplot([values],positions=[i],widths=.76,showextrema=False)
        body=violin["bodies"][0]; verts=body.get_paths()[0].vertices
        verts[:,0]=np.maximum(verts[:,0],i+.02)
        body.set_facecolor(colors[m]); body.set_edgecolor("none"); body.set_alpha(.25)
    boxes=axa.boxplot(err,positions=pos,widths=.27,patch_artist=True,showfliers=False,
                      whiskerprops={"linewidth":1},capprops={"linewidth":1})
    for patch,m in zip(boxes["boxes"],MODELS): patch.set_facecolor(colors[m]); patch.set_edgecolor(colors[m]); patch.set_alpha(.88)
    for line in boxes["whiskers"]+boxes["caps"]: line.set_color("#555555")
    for med in boxes["medians"]: med.set_color("white"); med.set_linewidth(1.4)
    axa.axhline(0,color="#666666",ls="--",lw=1)
    axa.set_ylabel("Wind-power forecast error (GW)"); axa.set_xlabel("Forecast model")

    specs=[
        (axb,axbv,"balancing_cost_million_cny","Balancing cost","Annual balancing cost (million CNY)",1.0),
        (axc,axcv,"total_cost_difference_million_cny","Total cost difference","Annual cost difference (million CNY)",1.0),
    ]
    annual_idx=annual.set_index("model")
    for ax,vax,metric,title,ylabel,scale in specs:
        values=annual_idx.loc[list(MODELS),metric]/scale
        bars=ax.bar(pos,values,color=[colors[m] for m in MODELS],width=.68,zorder=2)
        ax.set_title(title,loc="left",fontsize=10,pad=6); ax.set_ylabel(ylabel)
        ymax=float(values.max()); ax.set_ylim(0,ymax*1.10)
        for bar,val in zip(bars,values):
            ax.annotate(f"{val:.1f}",(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),textcoords="offset points",ha="center",va="bottom",fontsize=9)
        dist=[daily.loc[daily.model==m,metric].to_numpy() for m in MODELS]
        y=np.arange(3,-1,-1)
        parts=vax.violinplot(dist,positions=y,vert=False,widths=.72,showextrema=False)
        for body,m in zip(parts["bodies"],MODELS): body.set_facecolor(colors[m]); body.set_edgecolor("none"); body.set_alpha(.72)
        for yi,vals in zip(y,dist): vax.vlines(np.median(vals),yi-.27,yi+.27,color="#222222",lw=1.25,zorder=3)
        vax.axvline(0,color="#666666",ls="--",lw=.9)
        vax.set_ylim(-.65,3.65); vax.set_yticks([]); vax.set_xlabel("million CNY\nper day",fontsize=10)
        vax.set_title("Daily distribution",fontsize=12,pad=9)
        vax.tick_params(labelsize=9); vax.spines[["top","right","left"]].set_visible(False)

    for ax in (axa,axb,axc):
        ax.set_xticks(pos,MODELS,rotation=30,ha="right")
        ax.spines[["top","right"]].set_visible(False); ax.grid(False)
    fig.subplots_adjust(left=.065,right=.99,bottom=.25,top=.86); FIG.mkdir(exist_ok=True)
    fig.savefig(FIG/"pypsa_two_stage_8760h.png",dpi=300,bbox_inches="tight"); fig.savefig(FIG/"pypsa_two_stage_8760h.pdf",bbox_inches="tight"); plt.close(fig)

def run():
    logging.getLogger("pypsa").setLevel(logging.ERROR); logging.getLogger("linopy").setLevel(logging.ERROR)
    d=inputs(); factors=pd.read_csv(FACTOR/"carrier_operational_co2_factors.csv").set_index("carrier").factor_t_co2_per_mwh
    period_items=[(period,pd.DatetimeIndex(values.to_numpy())) for period,values in pd.Series(d["times"],index=d["times"]).groupby(d["times"].to_period("M"))]
    frames=[]
    for model in MODELS:
        for period,times in period_items:
            print(f"{model} {period}: {len(times)} h",flush=True); frames.append(chunk(model,times,d,factors))
    perfect_frames=[]
    for period,times in period_items:
        print(f"Perfect {period}: {len(times)} h",flush=True); perfect_frames.append(chunk("Perfect",times,d,factors))
    hourly=pd.concat(frames,ignore_index=True); perfect=pd.concat(perfect_frames,ignore_index=True)
    baseline=perfect.set_index("timestamp_beijing")["total_operating_cost_cny"]
    hourly["perfect_total_operating_cost_cny"]=hourly["timestamp_beijing"].map(baseline)
    hourly["total_cost_difference_cny"]=hourly["total_operating_cost_cny"]-hourly["perfect_total_operating_cost_cny"]
    daily,annual=summaries(hourly); OUT.mkdir(parents=True,exist_ok=True)
    combined=ROOT/"results"/"pypsa_dispatch_2020_all_models.csv"; hourly.to_csv(combined,index=False,float_format="%.6f")
    perfect.to_csv(OUT/"perfect_forecast_hourly.csv",index=False,float_format="%.6f")
    daily.to_csv(OUT/"daily_metrics.csv",index=False,float_format="%.6f"); annual.to_csv(OUT/"annual_metrics.csv",index=False,float_format="%.6f"); plot(hourly,daily,annual)
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"pypsa_version":pypsa.__version__,"solver":"HiGHS","network":"31省节点和PyPSA-China省际线路","stage_1":"真实省级负荷和光伏+预测风电","stage_2":"固定日前机组计划+真实风电+上/下调","hours":8760,"models":list(MODELS)}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); print(annual.to_string(index=False)); print(combined)

if __name__=="__main__": run()
