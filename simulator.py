from __future__ import annotations
import heapq, math, hashlib
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.optimize import brentq

PRODUCTS = ["PR1","PR2","PR3","PR4"]
WS_SEQUENCE = ["WS1","WS2","WS3","WS4","WS5","WS6","WS7","WS9"]
WS_RESOURCES = {
    "WS1":["WS1"], "WS2":["MC1","MC3"], "WS3":["MC2","MC4"],
    "WS4":["WS4"], "WS5":["MC5"], "WS6":["MC6","MC7"],
    "WS7":["MC8","MC9"], "WS9":["WS9"],
}
MACHINES = [f"MC{i}" for i in range(1,10)]
HUMANS = ["WS1","WS4","WS9"]
SCENARIOS = {
    "No failure":"no_failure",
    "Instant repair":"instant_repair",
    "No repair":"no_repair",
    "Failure + repair":"repair",
}


def stable_seed(*parts) -> int:
    h=hashlib.sha256("|".join(map(str,parts)).encode()).digest()
    return int.from_bytes(h[:8],"little") % (2**32-1)


def weibull_third(pair: str, a: float, b: float):
    """Returns dict with mttf_h, eta_h, beta given a selected pair."""
    if pair == "MTTF + beta":
        mttf, beta = float(a), float(b)
        eta = mttf / gamma(1.0 + 1.0/beta)
    elif pair == "eta + beta":
        eta, beta = float(a), float(b)
        mttf = eta * gamma(1.0 + 1.0/beta)
    elif pair == "MTTF + eta":
        mttf, eta = float(a), float(b)
        ratio=mttf/eta
        def f(beta): return gamma(1.0+1.0/beta)-ratio
        # Gamma minimum means not every MTTF/eta pair has a positive beta solution.
        grid=np.logspace(-2,3,500)
        vals=[f(x) for x in grid]
        roots=[]
        for x1,x2,y1,y2 in zip(grid[:-1],grid[1:],vals[:-1],vals[1:]):
            if y1==0 or y1*y2<0:
                try: roots.append(brentq(f,x1,x2))
                except Exception: pass
        if not roots:
            raise ValueError("This MTTF/eta pair does not define a valid Weibull beta.")
        beta=max(roots)  # prefer conventional wear/random-failure branch when two roots exist
    else:
        raise ValueError(pair)
    if min(mttf,eta,beta)<=0: raise ValueError("Weibull parameters must be positive")
    return {"mttf_h":mttf,"eta_h":eta,"beta":beta}


def lognormal_mu_sigma(mean: float, cv: float):
    s2=math.log(1+cv*cv)
    return math.log(mean)-0.5*s2, math.sqrt(s2)

@dataclass
class Product:
    pid: int
    ptype: str
    stage: int = 0
    admit_t: Optional[float] = None
    complete_t: Optional[float] = None
    current_resource: Optional[str] = None
    target_resource: Optional[str] = None
    remaining_service: Optional[float] = None
    stage_start_t: Optional[float] = None

@dataclass
class Resource:
    name: str
    ws: str
    buffer_cap: int = 0
    current: Optional[Product] = None
    queue: deque = field(default_factory=deque)
    permanent_failed: bool = False
    under_repair: bool = False
    life_remaining: float = math.inf
    failures: int = 0
    repair_downtime: float = 0.0
    state: str = "Idle (B1-MCx)"
    state_since: float = 0.0
    state_durations: dict = field(default_factory=lambda: defaultdict(float))
    q_area: float = 0.0
    q_last_t: float = 0.0
    q_last_len: int = 0
    q_max: int = 0

    def set_state(self, new_state: str, t: float):
        self.state_durations[self.state] += max(0.0,t-self.state_since)
        self.state=new_state; self.state_since=t
    def queue_update(self,t:float):
        self.q_area += self.q_last_len * max(0.0,t-self.q_last_t)
        self.q_last_t=t; self.q_last_len=len(self.queue); self.q_max=max(self.q_max,len(self.queue))

class FactorySim:
    def __init__(self, params: pd.DataFrame, quantities: Dict[str,int], wip_limit:int,
                 buffers:Dict[str,int], reliability:Dict[str,dict], scenario:str,
                 seed:int=42, event_log:bool=True):
        self.params=params.set_index(["resource","product"])
        self.quantities=quantities; self.wip_limit=max(1,int(wip_limit)); self.buffers=buffers
        self.reliability=reliability; self.scenario=scenario; self.seed=int(seed); self.keep_log=event_log
        self.t=0.0; self.seq=0; self.events=[]; self.event_rows=[]; self.wip_series=[(0.0,0)]
        self.completed=[]; self.released=0; self.total=sum(quantities.values()); self.stalled=False
        self.next_choice={ws:0 for ws in WS_RESOURCES}
        self.resources={}
        for ws,names in WS_RESOURCES.items():
            for n in names:
                cap=int(buffers.get(n,0)) if n in MACHINES else 0
                self.resources[n]=Resource(n,ws,cap)
        self.recovery={ws:deque() for ws in WS_RESOURCES}
        self.failure_rng={m:np.random.default_rng(stable_seed(seed,"failure",m)) for m in MACHINES}
        self.repair_rng={m:np.random.default_rng(stable_seed(seed,"repair",m)) for m in MACHINES}
        for m in MACHINES: self.resources[m].life_remaining=self._sample_life(m)
        self.backlog=self._build_backlog()

    def _build_backlog(self):
        rem={p:int(self.quantities.get(p,0)) for p in PRODUCTS}; out=[]; pid=1
        while sum(rem.values()):
            for p in PRODUCTS:
                if rem[p]>0:
                    out.append(Product(pid,p)); pid+=1; rem[p]-=1
        return deque(out)

    def _log(self,event,product=None,resource=None,**kw):
        if not self.keep_log: return
        self.event_rows.append({"time_s":self.t,"event":event,"pid":getattr(product,"pid",None),
                                "product":getattr(product,"ptype",None),"resource":resource,**kw})

    def _push(self,t,event,payload):
        self.seq+=1; heapq.heappush(self.events,(float(t),self.seq,event,payload))

    def _sample_process(self,res,p:Product):
        try:r=self.params.loc[(res,p.ptype)]
        except KeyError: raise KeyError(f"No processing parameters for {res} × {p.ptype}")
        mu=float(r.mean_s); sd=max(float(r.sd_s),0.01)
        rng=np.random.default_rng(stable_seed(self.seed,"process",p.pid,p.stage,res))
        # positive normal by rejection; very small chance of repeat for observed data
        for _ in range(100):
            x=float(rng.normal(mu,sd))
            if x>0:return x
        return max(0.01,mu)

    def _sample_life(self,m):
        r=self.reliability[m]; eta_s=float(r["eta_h"])*3600.0; beta=float(r["beta"])
        return max(1e-6, eta_s*self.failure_rng[m].weibull(beta))

    def _sample_repair(self,m):
        r=self.reliability[m]; mean=max(float(r["repair_mean_min"])*60.0,0.0); cv=max(float(r["repair_cv"]),0.0)
        if mean<=0:return 0.0
        if cv<=0:return mean
        lm,ls=lognormal_mu_sigma(mean,cv)
        return float(self.repair_rng[m].lognormal(lm,ls))

    def _can_accept(self,resname):
        r=self.resources[resname]
        if r.permanent_failed:return False
        if r.current is None and not r.under_repair:return True
        return len(r.queue)<r.buffer_cap

    def _eligible_resources(self,ws):
        return [n for n in WS_RESOURCES[ws] if not self.resources[n].permanent_failed]

    def _choose_resource(self,ws):
        names=WS_RESOURCES[ws]
        if len(names)==1:return None if self.resources[names[0]].permanent_failed else names[0]
        start=self.next_choice[ws]%len(names)
        # strict alternating when possible; skip currently unavailable/down resources
        for k in range(len(names)):
            idx=(start+k)%len(names); n=names[idx]; r=self.resources[n]
            if not r.permanent_failed and not r.under_repair:
                self.next_choice[ws]=(idx+1)%len(names); return n
        # all temporarily unavailable: reserve next non-permanent in cyclic order
        for k in range(len(names)):
            idx=(start+k)%len(names); n=names[idx]
            if not self.resources[n].permanent_failed:
                self.next_choice[ws]=(idx+1)%len(names); return n
        return None

    def _accept(self,resname,p:Product,remaining_override=None):
        r=self.resources[resname]; p.current_resource=resname; p.target_resource=None
        if r.current is None and not r.under_repair and not r.permanent_failed:
            r.current=p
            if remaining_override is not None:p.remaining_service=remaining_override
            self._start_or_resume(r,p)
        else:
            r.queue_update(self.t); r.queue.append((p,remaining_override)); r.queue_update(self.t)
            self._log("BUFFER_ENTER",p,resname,queue_len=len(r.queue))

    def _start_or_resume(self,r:Resource,p:Product):
        if p.remaining_service is None:p.remaining_service=self._sample_process(r.name,p)
        p.stage_start_t = self.t if p.stage_start_t is None else p.stage_start_t
        r.set_state("Processing (B3)",self.t)
        self._log("PROCESS_START",p,r.name,remaining_s=p.remaining_service)
        if r.name in MACHINES and self.scenario!="no_failure":
            dt=min(p.remaining_service,r.life_remaining)
            if r.life_remaining < p.remaining_service-1e-9:
                self._push(self.t+dt,"FAILURE",r.name)
            else:self._push(self.t+dt,"COMPLETE",r.name)
        else:self._push(self.t+p.remaining_service,"COMPLETE",r.name)

    def _start_next(self,r:Resource):
        if r.current is not None or r.permanent_failed or r.under_repair:return
        if r.queue:
            r.queue_update(self.t); p,rem=r.queue.popleft(); r.queue_update(self.t)
            self._log("BUFFER_LEAVE",p,r.name,queue_len=len(r.queue))
            r.current=p
            if rem is not None:p.remaining_service=rem
            self._start_or_resume(r,p)
        else:r.set_state("Idle (B1-MCx)",self.t)

    def _attempt_next_stage(self,p:Product,upstream:Optional[Resource]):
        if p.stage >= len(WS_SEQUENCE)-1:
            p.complete_t=self.t; self.completed.append(p); self._log("PRODUCT_COMPLETE",p,upstream.name if upstream else None)
            if upstream:
                upstream.current=None; upstream.set_state("Idle (B1-MCx)",self.t); self._start_next(upstream)
            self.wip_series.append((self.t,self.released-len(self.completed)))
            return True
        next_stage=p.stage+1; ws=WS_SEQUENCE[next_stage]
        if p.target_resource is None:
            p.target_resource=self._choose_resource(ws)
        target=p.target_resource
        if target is None:
            if upstream: upstream.set_state("Waiting for machine (B6)",self.t)
            return False
        if self._can_accept(target):
            if upstream:
                upstream.current=None; upstream.set_state("Idle (B1-MCx)",self.t)
            p.stage=next_stage; p.remaining_service=None; p.stage_start_t=None
            self._log("TRANSFER",p,target,workstation=ws)
            self._accept(target,p)
            if upstream:self._start_next(upstream)
            return True
        if upstream: upstream.set_state("Waiting for machine (B6)",self.t)
        self._log("BLOCKED_TRANSFER",p,upstream.name if upstream else None,target=target)
        return False

    def _retry_blocked(self):
        changed=True; loops=0
        while changed and loops<50:
            changed=False; loops+=1
            # failed-machine recovery jobs first
            for ws,q in self.recovery.items():
                if not q: continue
                p,remaining=q[0]
                p.target_resource=None
                target=self._choose_resource(ws)
                if target and self._can_accept(target):
                    q.popleft(); p.current_resource=target
                    self._accept(target,p,remaining); changed=True
            # resources holding completed jobs
            for r in self.resources.values():
                if r.current and r.state=="Waiting for machine (B6)":
                    if self._attempt_next_stage(r.current,r): changed=True

    def _release(self):
        while self.backlog and (self.released-len(self.completed))<self.wip_limit:
            ws="WS1"; target="WS1"
            if not self._can_accept(target):break
            p=self.backlog.popleft(); p.admit_t=self.t; p.stage=0; self.released+=1
            self.wip_series.append((self.t,self.released-len(self.completed)))
            self._log("SOURCE_ADMIT",p,target); self._accept(target,p)

    def _complete(self,resname):
        r=self.resources[resname]; p=r.current
        if p is None:return
        elapsed=p.remaining_service
        if resname in MACHINES and self.scenario!="no_failure": r.life_remaining=max(0.0,r.life_remaining-elapsed)
        p.remaining_service=0.0
        self._log("PROCESS_COMPLETE",p,resname)
        self._attempt_next_stage(p,r)

    def _failure(self,resname):
        r=self.resources[resname]; p=r.current
        if p is None:return
        elapsed=r.life_remaining
        p.remaining_service=max(0.0,p.remaining_service-elapsed)
        r.life_remaining=0.0; r.failures+=1
        self._log("FAILURE",p,resname,remaining_s=p.remaining_service)
        if self.scenario=="instant_repair":
            r.life_remaining=self._sample_life(resname)
            self._log("REPAIR_COMPLETE",p,resname,repair_s=0.0)
            self._start_or_resume(r,p)
        elif self.scenario=="repair":
            r.under_repair=True; r.set_state("Under maintainence(B5)",self.t)
            dt=self._sample_repair(resname); r.repair_downtime+=dt
            self._log("REPAIR_START",p,resname,repair_s=dt); self._push(self.t+dt,"REPAIR_COMPLETE",resname)
        elif self.scenario=="no_repair":
            r.permanent_failed=True; r.set_state("Off ( B4)",self.t)
            # evacuate current + queued work to workstation recovery; preserve remaining work for interrupted job
            r.current=None; self.recovery[r.ws].appendleft((p,p.remaining_service))
            while r.queue:
                r.queue_update(self.t); qp,rem=r.queue.popleft(); r.queue_update(self.t)
                self.recovery[r.ws].append((qp,rem))
            self._retry_blocked()

    def _repair_complete(self,resname):
        r=self.resources[resname]; p=r.current
        r.under_repair=False; r.life_remaining=self._sample_life(resname)
        self._log("REPAIR_COMPLETE",p,resname)
        if p is not None:self._start_or_resume(r,p)
        else:self._start_next(r)

    def run(self,max_time_s=7*24*3600):
        self._release()
        while self.events and len(self.completed)<self.total and self.t<=max_time_s:
            t,_,ev,payload=heapq.heappop(self.events); self.t=t
            if ev=="COMPLETE": self._complete(payload)
            elif ev=="FAILURE": self._failure(payload)
            elif ev=="REPAIR_COMPLETE": self._repair_complete(payload)
            self._retry_blocked(); self._release()
        if len(self.completed)<self.total:
            self.stalled=True
        # close accounting
        end=self.t
        for r in self.resources.values():
            r.state_durations[r.state]+=max(0,end-r.state_since)
            r.q_area += r.q_last_len*max(0,end-r.q_last_t)
        return self.results()

    def results(self):
        completed=len(self.completed); makespan=self.t if completed==self.total else np.nan
        throughput=(completed/(self.t/3600.0)) if self.t>0 else np.nan
        prod_rows=[]
        for p in self.completed:
            prod_rows.append({"pid":p.pid,"product":p.ptype,"admit_s":p.admit_t,"complete_s":p.complete_t,
                              "lead_s":p.complete_t-p.admit_t,"fulfillment_s":p.complete_t})
        prod_df=pd.DataFrame(prod_rows)
        station=[]
        for n,r in self.resources.items():
            proc=r.state_durations.get("Processing (B3)",0.0)
            station.append({"resource":n,"workstation":r.ws,"utilization_pct":100*proc/self.t if self.t else 0,
                            "failures":r.failures,"repair_downtime_min":r.repair_downtime/60,
                            "avg_queue":r.q_area/self.t if self.t else 0,"max_queue":r.q_max,
                            "buffer_cap":r.buffer_cap,"permanent_failed":r.permanent_failed,
                            **{f"state::{k}":v for k,v in r.state_durations.items()}})
        station_df=pd.DataFrame(station)
        wip=pd.DataFrame(self.wip_series,columns=["time_s","wip"]).drop_duplicates(subset=["time_s"],keep="last")
        return {
            "scenario":self.scenario,"completed":completed,"total":self.total,"stalled":self.stalled,
            "makespan_s":makespan,"end_time_s":self.t,"throughput_h":throughput,
            "failures":sum(r.failures for r in self.resources.values()),
            "products":prod_df,"resources":station_df,"wip":wip,
            "event_log":pd.DataFrame(self.event_rows),
        }

def run_four(params, quantities, wip_limit, buffers, reliability, seed=42, event_log=True):
    out={}
    for label,key in SCENARIOS.items():
        out[label]=FactorySim(params,quantities,wip_limit,buffers,reliability,key,seed,event_log).run()
    return out

def summarize_run(res):
    pdf=res["products"]
    return {
        "Completed":f'{res["completed"]}/{res["total"]}',
        "Status":"STALLED" if res["stalled"] else "Complete",
        "Makespan (h)":np.nan if pd.isna(res["makespan_s"]) else res["makespan_s"]/3600,
        "Throughput (/h)":res["throughput_h"],
        "Failures":res["failures"],
        "Mean lead (min)":np.nan if pdf.empty else pdf.lead_s.mean()/60,
        "P50 lead (min)":np.nan if pdf.empty else pdf.lead_s.quantile(.5)/60,
        "P95 lead (min)":np.nan if pdf.empty else pdf.lead_s.quantile(.95)/60,
    }

def monte_carlo(params, quantities, wip_limit, buffers, reliability, n_runs=50, seed=123):
    rows=[]; highest=[]
    for i in range(int(n_runs)):
        run_seed=stable_seed(seed,"mc",i)
        results=run_four(params,quantities,wip_limit,buffers,reliability,run_seed,event_log=False)
        for label,res in results.items():
            s=summarize_run(res); s.update({"run":i,"scenario":label})
            rows.append(s)
            rdf=res["resources"]
            if len(rdf): highest.append({"run":i,"scenario":label,"resource":rdf.loc[rdf.utilization_pct.idxmax(),"resource"]})
    return pd.DataFrame(rows),pd.DataFrame(highest)
