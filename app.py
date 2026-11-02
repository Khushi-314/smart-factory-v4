import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from simulator import (
    PRODUCTS, MACHINES, WS_SEQUENCE, WS_RESOURCES, SCENARIOS,
    weibull_third, run_four, summarize_run, monte_carlo
)

st.set_page_config(page_title="Smart Factory Reliability Simulator", page_icon="🏭", layout="wide")
BASE=os.path.dirname(__file__)
PARAM_PATH=os.path.join(BASE,"data","processing_params.csv")
params=pd.read_csv(PARAM_PATH)

st.title("Smart Factory Stochastic Simulation")
st.caption("Workstation-first DES • product-specific processing distributions • configurable WIP/buffers • four reliability scenarios")

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("Simulation inputs")
    st.subheader("Order quantities")
    quantities={p:st.number_input(p,min_value=0,max_value=10000,value=10,step=1,key=f"qty_{p}") for p in PRODUCTS}
    st.divider()
    wip=st.number_input("Maximum WIP",min_value=1,max_value=10000,value=1,step=1,
                        help="Maximum number of admitted but unfinished products. Default = 1.")
    seed=st.number_input("Random seed",min_value=0,max_value=2_000_000_000,value=42,step=1)
    st.divider()
    st.subheader("Machine buffers")
    st.caption("0 = no waiting product in front of that machine.")
    buf_df=pd.DataFrame({"Machine":MACHINES,"Buffer capacity":[0]*len(MACHINES)})
    edited_buf=st.data_editor(buf_df,hide_index=True,use_container_width=True,
                              column_config={"Machine":st.column_config.TextColumn(disabled=True),
                                             "Buffer capacity":st.column_config.NumberColumn(min_value=0,step=1)},
                              key="buffers")
    buffers={r.Machine:int(r["Buffer capacity"]) for _,r in edited_buf.iterrows()}

    st.divider()
    st.subheader("Failure model")
    pair=st.selectbox("Weibull input pair",["MTTF + beta","eta + beta","MTTF + eta"],index=0,
                      help="Any two parameters are entered; the third is derived automatically.")
    if pair=="MTTF + beta":
        fail_df=pd.DataFrame({"Machine":MACHINES,"MTTF (h)":[5.0]*9,"beta":[1.5]*9})
    elif pair=="eta + beta":
        fail_df=pd.DataFrame({"Machine":MACHINES,"eta (h)":[5.54]*9,"beta":[1.5]*9})
    else:
        fail_df=pd.DataFrame({"Machine":MACHINES,"MTTF (h)":[5.0]*9,"eta (h)":[5.54]*9})
    edited_fail=st.data_editor(fail_df,hide_index=True,use_container_width=True,
                               column_config={"Machine":st.column_config.TextColumn(disabled=True)},key=f"fail_{pair}")

    st.subheader("Repair model")
    st.caption("Repair time is Lognormal. These inputs are configurable per machine.")
    rep_df=pd.DataFrame({"Machine":MACHINES,"Mean repair (min)":[5.0]*9,"Repair CV":[0.30]*9})
    edited_rep=st.data_editor(rep_df,hide_index=True,use_container_width=True,
                              column_config={"Machine":st.column_config.TextColumn(disabled=True),
                                             "Mean repair (min)":st.column_config.NumberColumn(min_value=0.0,step=0.5),
                                             "Repair CV":st.column_config.NumberColumn(min_value=0.0,step=0.05)},key="repair")

    reliability={}
    errors=[]
    for i,m in enumerate(MACHINES):
        try:
            if pair=="MTTF + beta": w=weibull_third(pair,edited_fail.loc[i,"MTTF (h)"],edited_fail.loc[i,"beta"])
            elif pair=="eta + beta": w=weibull_third(pair,edited_fail.loc[i,"eta (h)"],edited_fail.loc[i,"beta"])
            else: w=weibull_third(pair,edited_fail.loc[i,"MTTF (h)"],edited_fail.loc[i,"eta (h)"])
            reliability[m]={**w,"repair_mean_min":float(edited_rep.loc[i,"Mean repair (min)"]),
                            "repair_cv":float(edited_rep.loc[i,"Repair CV"])}
        except Exception as e: errors.append(f"{m}: {e}")
    if errors:
        st.error("Invalid Weibull inputs: " + "; ".join(errors[:3]))
    else:
        with st.expander("Derived Weibull parameter"):
            d=pd.DataFrame([{"Machine":m,"MTTF (h)":reliability[m]["mttf_h"],"eta (h)":reliability[m]["eta_h"],"beta":reliability[m]["beta"]} for m in MACHINES])
            st.dataframe(d.style.format({"MTTF (h)":"{:.3f}","eta (h)":"{:.3f}","beta":"{:.3f}"}),hide_index=True,use_container_width=True)

    st.divider()
    do_mc=st.toggle("Run Monte Carlo",value=False)
    n_mc=st.number_input("Monte Carlo runs",min_value=10,max_value=500,value=50,step=10,disabled=not do_mc)
    run_btn=st.button("Run all four scenarios",type="primary",use_container_width=True,disabled=bool(errors) or sum(quantities.values())==0)

if run_btn:
    with st.spinner("Running four scenarios..."):
        st.session_state["results"]=run_four(params,quantities,int(wip),buffers,reliability,int(seed),event_log=True)
        st.session_state["config"]={"quantities":quantities,"wip":int(wip),"buffers":buffers,"reliability":reliability,"seed":int(seed)}
        if do_mc:
            mc,hi=monte_carlo(params,quantities,int(wip),buffers,reliability,int(n_mc),int(seed)+101)
            st.session_state["mc"]=(mc,hi)
        else: st.session_state.pop("mc",None)

# ---------- Reference model ----------
with st.expander("Product routing used by the model",expanded=False):
    st.markdown("**Workstation route (all products):**  `WS1 → WS2 → WS3 → WS4 → WS5 → WS6 → WS7 → WS9`")
    route=pd.DataFrame([
        ["WS1","Kitting / human","single"],
        ["WS2","MC1, MC3","global alternating assignment"],
        ["WS3","MC2, MC4","global alternating assignment"],
        ["WS4","Human","manual; no mechanical failure"],
        ["WS5","MC5","single machine"],
        ["WS6","MC6, MC7","global alternating assignment"],
        ["WS7","MC8, MC9","global alternating assignment"],
        ["WS9","Human","manual; no mechanical failure"],
    ],columns=["Workstation","Resource(s)","Rule"])
    st.dataframe(route,hide_index=True,use_container_width=True)
    st.info("Product type does not choose the machine. At every two-machine workstation, successive arrivals alternate; unavailable machines are skipped.")

if "results" not in st.session_state:
    st.info("Configure the inputs in the sidebar and run the simulation. Default WIP = 1 and every machine buffer = 0.")
    st.stop()

results=st.session_state["results"]
config=st.session_state["config"]

# ---------- 1. Factory performance ----------
st.header("1. Factory performance — four-scenario comparison")
summary=pd.DataFrame({label:summarize_run(res) for label,res in results.items()}).T.reset_index(names="Scenario")
st.dataframe(summary.style.format({"Makespan (h)":"{:.3f}","Throughput (/h)":"{:.2f}","Mean lead (min)":"{:.2f}","P50 lead (min)":"{:.2f}","P95 lead (min)":"{:.2f}"},na_rep="—"),hide_index=True,use_container_width=True)

c1,c2,c3,c4=st.columns(4)
for col,(label,res) in zip([c1,c2,c3,c4],results.items()):
    with col:
        st.subheader(label)
        st.metric("Completed",f'{res["completed"]}/{res["total"]}')
        st.metric("Makespan","STALLED" if res["stalled"] else f'{res["makespan_s"]/3600:.3f} h')
        st.metric("Throughput",f'{res["throughput_h"]:.2f}/h')
        st.metric("Failures",res["failures"])

valid=summary.dropna(subset=["Makespan (h)"])
if len(valid):
    fig=px.bar(valid,x="Scenario",y="Makespan (h)",title="Makespan by reliability scenario",text_auto=".3f")
    st.plotly_chart(fig,use_container_width=True)

st.caption(f'Configured WIP limit: **{config["wip"]}**. WIP is an input constraint, so it is not shown as a “Peak WIP” headline KPI.')

# ---------- 2. Flow ----------
st.header("2. Factory flow & WIP")
scenario=st.selectbox("Scenario for detailed analysis",list(results.keys()),index=3,key="detail_scenario")
res=results[scenario]
wipdf=res["wip"]
fig=go.Figure()
fig.add_trace(go.Scatter(x=wipdf.time_s,y=wipdf.wip,mode="lines",name="Actual WIP",line_shape="hv"))
fig.add_hline(y=config["wip"],line_dash="dash",annotation_text=f'Configured WIP = {config["wip"]}')
fig.update_layout(title=f"WIP over simulated time — {scenario}",xaxis_title="Simulation time (s)",yaxis_title="Products")
st.plotly_chart(fig,use_container_width=True)

# ---------- 3. Resource analytics ----------
st.header("3. Resource analytics")
rdf=res["resources"].copy()
a,b=st.columns(2)
with a:
    f=px.bar(rdf.sort_values("utilization_pct"),x="utilization_pct",y="resource",orientation="h",title="Resource utilization",labels={"utilization_pct":"Utilization (%)","resource":"Resource"})
    st.plotly_chart(f,use_container_width=True)
with b:
    state_cols=[c for c in rdf.columns if c.startswith("state::")]
    if state_cols:
        sd=rdf[["resource"]+state_cols].copy()
        for c in state_cols: sd[c]=100*sd[c]/max(res["end_time_s"],1e-9)
        long=sd.melt(id_vars="resource",var_name="State",value_name="% of run")
        long["State"]=long["State"].str.replace("state::","",regex=False)
        f=px.bar(long,x="resource",y="% of run",color="State",title="Canonical machine state-time composition")
        st.plotly_chart(f,use_container_width=True)

qcol,failcol=st.columns(2)
with qcol:
    q=rdf[rdf.resource.isin(MACHINES)][["resource","avg_queue","max_queue","buffer_cap"]]
    qlong=q.melt(id_vars=["resource","buffer_cap"],value_vars=["avg_queue","max_queue"],var_name="Metric",value_name="Products")
    f=px.bar(qlong,x="resource",y="Products",color="Metric",barmode="group",title="Input queue pressure")
    st.plotly_chart(f,use_container_width=True)
with failcol:
    f=px.bar(rdf[rdf.resource.isin(MACHINES)],x="resource",y="repair_downtime_min",title="Repair downtime by machine",labels={"repair_downtime_min":"Downtime (min)"})
    st.plotly_chart(f,use_container_width=True)
    st.dataframe(rdf[rdf.resource.isin(MACHINES)][["resource","failures","repair_downtime_min","permanent_failed"]].rename(columns={"resource":"Machine"}),hide_index=True,use_container_width=True)

# ---------- 4. Product lead time ----------
st.header("4. Product lead time & fulfillment")
pdf=res["products"]
if pdf.empty:
    st.warning("No products completed in this scenario.")
else:
    prod_stats=[]
    for p,g in pdf.groupby("product"):
        prod_stats.append({"Product":p,"Completed":len(g),"Mean lead (min)":g.lead_s.mean()/60,"P50 lead (min)":g.lead_s.quantile(.5)/60,"P95 lead (min)":g.lead_s.quantile(.95)/60,"Mean fulfillment (min)":g.fulfillment_s.mean()/60})
    ps=pd.DataFrame(prod_stats)
    long=ps.melt(id_vars=["Product","Completed"],value_vars=["Mean lead (min)","P50 lead (min)","P95 lead (min)"],var_name="Statistic",value_name="Minutes")
    f=px.bar(long,x="Product",y="Minutes",color="Statistic",barmode="group",title=f"In-factory lead-time distribution by product — {scenario}")
    st.plotly_chart(f,use_container_width=True)
    st.dataframe(ps.style.format({c:"{:.2f}" for c in ps.columns if "min" in c}),hide_index=True,use_container_width=True)
    if res["stalled"]: st.warning("This scenario stalled. Lead-time statistics include completed products only; unfinished products are not silently assigned a finite lead time.")

# ---------- 5. Reliability impact ----------
st.header("5. Reliability impact")
base=results["No failure"]
impact=[]
for label,r in results.items():
    impact.append({"Scenario":label,"Completion %":100*r["completed"]/max(1,r["total"]),"Makespan (h)":np.nan if pd.isna(r["makespan_s"]) else r["makespan_s"]/3600,"Throughput (/h)":r["throughput_h"],"Failures":r["failures"],"Repair downtime (min)":r["resources"].repair_downtime_min.sum()})
impact=pd.DataFrame(impact)
st.dataframe(impact.style.format({"Completion %":"{:.1f}%","Makespan (h)":"{:.3f}","Throughput (/h)":"{:.2f}","Repair downtime (min)":"{:.2f}"},na_rep="—"),hide_index=True,use_container_width=True)

# ---------- 6. Monte Carlo ----------
st.header("6. Monte Carlo uncertainty")
if "mc" not in st.session_state:
    st.info("Enable ‘Run Monte Carlo’ in the sidebar and rerun to generate uncertainty analysis across all four scenarios.")
else:
    mc,hi=st.session_state["mc"]
    mc_summary=[]
    for s,g in mc.groupby("scenario"):
        completed=g["Status"].eq("Complete")
        finite=g.loc[completed,"Makespan (h)"].dropna()
        mc_summary.append({
            "Scenario":s,"Mean makespan (h)":finite.mean() if len(finite) else np.nan,
            "Median makespan (h)":finite.median() if len(finite) else np.nan,
            "P05 makespan (h)":finite.quantile(.05) if len(finite) else np.nan,
            "P95 makespan (h)":finite.quantile(.95) if len(finite) else np.nan,
            "Mean throughput (/h)":g["Throughput (/h)"].mean(),"Mean failures":g["Failures"].mean(),
            "P(≥1 failure)":(g["Failures"]>=1).mean(),"P(zero failures)":(g["Failures"]==0).mean(),
            "P(order complete)":completed.mean(),"P(stall)":1-completed.mean(),
        })
    mcs=pd.DataFrame(mc_summary)
    st.dataframe(mcs.style.format({c:"{:.3f}" for c in mcs.columns if "makespan" in c or "throughput" in c or "failures" in c}).format({c:"{:.1%}" for c in ["P(≥1 failure)","P(zero failures)","P(order complete)","P(stall)"]}),hide_index=True,use_container_width=True)
    a,b=st.columns(2)
    with a:
        finite_mc=mc[mc["Status"].eq("Complete") & mc["Makespan (h)"].notna()]
        if len(finite_mc):
            f=px.histogram(finite_mc,x="Makespan (h)",color="scenario",barmode="overlay",opacity=.55,title="Makespan distributions across Monte Carlo runs")
            st.plotly_chart(f,use_container_width=True)
    with b:
        freq=hi.groupby(["scenario","resource"]).size().reset_index(name="count")
        freq["probability"]=freq.groupby("scenario")["count"].transform(lambda x:x/x.sum())
        f=px.bar(freq,x="resource",y="probability",color="scenario",barmode="group",title="Highest-utilization resource frequency")
        st.plotly_chart(f,use_container_width=True)

# ---------- 7. Model and data ----------
st.header("7. Model & data provenance")
st.caption("Processing parameters are intentionally isolated from the DES so a future rolling/live telemetry pipeline can replace them without rewriting simulation logic.")
st.dataframe(params,hide_index=True,use_container_width=True)
with st.expander("Reliability parameters currently used"):
    rr=pd.DataFrame([{"Machine":m,**config["reliability"][m]} for m in MACHINES])
    st.dataframe(rr,hide_index=True,use_container_width=True)

# ---------- 8. Event log ----------
st.header("8. Technical event log")
elog=res["event_log"]
if len(elog):
    evs=sorted(elog.event.dropna().unique().tolist())
    chosen=st.multiselect("Event types",evs,default=[e for e in evs if e in ["FAILURE","REPAIR_START","REPAIR_COMPLETE","PRODUCT_COMPLETE","BLOCKED_TRANSFER"]])
    show=elog[elog.event.isin(chosen)] if chosen else elog
    st.dataframe(show,hide_index=True,use_container_width=True,height=420)
