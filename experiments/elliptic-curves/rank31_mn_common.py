"""Support code for the curve #302 Mestre-Nagao GPU experiment."""

from __future__ import annotations

import concurrent.futures
import hashlib
import math
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "build"
RESULTS_DIR = ROOT / "experiments" / "elliptic-curves" / "results"
CUDA_SOURCE_PATH = ROOT / "experiments" / "elliptic-curves" / "mestre_nagao_score.cu"
ICARM_URL = "https://elliptic-rank.icarm.cloud/curve/302"
MESTRE_NAGAO_REFERENCE_URL = "https://math.mit.edu/~drew/ANTSXIV/NewRankRecordsSlides.pdf"
RECORD_T = Fraction(164518, 924945)

MINIMAL_A = (
    Fraction(1), Fraction(1), Fraction(1),
    Fraction(-1284727764113567728281797636015784768866707681415849262157224232063),
    Fraction(560368321454261339256859338901915312332769858684945406858043869199456710681989058863306170127006181),
)

WITNESS_TEXT = r"""
-343746913367434053557936715792581|-31005994945795167281668477280091616990476695982462
739387066521976456185309925500635|3830813927243125109605210052773411116315836476962
794918160232596023344129483097739|6435806665987309972375508567418781686192466975138
552229821725543064455253103966115|-4394327660511778936811414681478594663009127495638
621111741565922967463663441261795|1421683191996488160880565213880200774834445145522
672709190481033101277453068263195|739179814158401859498464392317452223287910224482
821399009280840681555588652225649|-7699942068198872209289633789793873375706107955662
765203485098820658379256355481245|-5034345308760298854369394902951465889284652680168
5399595603851507511243908479925915|388654067273650918227195921742660550010310029109122
752668954339285868300409092384315|4448429560131722871335630396999897920170009189762
665267573831105472345147950403835|-339632380726246052348178275348181915170728600638
2348702476455858806523697264250795|102466166054736815527265002023727452468648322685682
-485301239340016399925702578908055|-32703997667874232639739344802658052943027850955518
1671562150318197588290891477490065|-55528472789639333688258829014227823359584879796238
533792265728561313885832952129951|-5165872676030535509137532850965718653070997396662
8050101189939018648004785213625795/9|-304090504548899861983493167851611965128494136836666/27
641126276756517817380587191947425|475393761225519880788251703870437551199359754892
-28607312564278696457116932283279229/25|2883492798944948226520935391601885431994599305099066/125
56549444812087071231733196839837255/4|13405577698133557868624072930089297336164276021592851/8
1436747523773762064481823443661979|40991896134294415817647888149334972652525806317538
-5064387018792502839844249474841045/9|897570094862437631568619794896775730629375504169894/27
487104273264503086273608258117432335/484|-180314887995146178717423515157548125006198119570993099/10648
25599342024506626201702677512314955|4091893764954118703319829997327215311497327967339042
980266989498097787981788009573917995/961|-525575897314714907955968902277850024344706090034709538/29791
5818507410725525932023407879789875/9|1789182708478717961552943758166081786352797832214/27
19197683320190396803412613672365855/49|-3712724452088495818457900090458774587078999890353974/343
887913128901643530184527456121547975/841|-475948997371817062287059043149845391545859899478008302/24389
1816718417495680977851120075801270195/1849|1248545591609034704022269662335921779441905481482388694/79507
4528903098191425769774358894938128668457595/3630906049|-6558125371552387528524400913253424783234119422079260125534495614/218787505794593
33880100347989327554317013369507240405/52441|-1625428840401701290033088199518465040555146308891387272/12008989
-11477003798992551861310315487329257523535/8922169|-244939776343182008186576362205394985355490153552116110249914/26650518803
""".strip()
WITNESSES = tuple((Fraction(x), Fraction(y)) for x, y in (line.split("|", 1) for line in WITNESS_TEXT.splitlines()))

@dataclass(frozen=True)
class ScoreRow:
    index: int
    t: Fraction
    score: float
    good_primes: int


def family_values(t: Fraction):
    L = 446667*t**2 + 471466*t + 239031
    p = 318552*t**2 + 368554*t - 72570
    q = 733413*t**2 - 45082*t - 14960
    D = 5*(7174492962*t**4 - 7114589515*t**3 - 22069002960*t**2 + 3909144679*t - 205134150)
    E = 882769396002*t**4 + 811447034567*t**3 - 1174040743*t**2 - 32493137198*t - 2386325360
    B = p*q*(L+p+q) - p*E - q*D
    return L, p, q, D, E, B


def family_ainvs(t: Fraction):
    L, _p, _q, D, E, B = family_values(t)
    return -L, D+E, -B, D*E, Fraction(0)


def discriminant(a):
    a1,a2,a3,a4,a6 = a
    b2 = a1*a1 + 4*a2
    b4 = 2*a4 + a1*a3
    b6 = a3*a3 + 4*a6
    b8 = a1*a1*a6 + 4*a2*a6 - a1*a3*a4 + a2*a3*a3 - a4*a4
    return -b2*b2*b8 - 8*b4**3 - 27*b6**2 + 9*b2*b4*b6


def on_curve(point, a):
    x,y = point; a1,a2,a3,a4,a6 = a
    return y*y + a1*x*y + a3*y == x**3 + a2*x*x + a4*x + a6


def exact_nth_root(value: int, n: int) -> int:
    if value < 0 or n < 1: raise ValueError("invalid exact root input")
    if value in (0,1): return value
    lo, hi = 0, 1 << ((value.bit_length()+n-1)//n + 1)
    while lo+1 < hi:
        mid=(lo+hi)//2
        if mid**n < value: lo=mid
        else: hi=mid
    if hi**n == value: return hi
    if lo**n == value: return lo
    raise ArithmeticError(f"value is not an exact {n}th power")


def transformed_ainvs(old,u,r,s,t):
    a1,a2,a3,a4,a6=old
    return (
        (a1+2*s)/u,
        (a2-s*a1+3*r-s*s)/u**2,
        (a3+r*a1+2*t)/u**3,
        (a4-s*a3+2*r*a2-(t+r*s)*a1+3*r*r-2*s*t)/u**4,
        (a6+r*a4+r*r*a2+r**3-t*a3-r*t*a1-t*t)/u**6,
    )


def calibration() -> dict:
    if len(WITNESSES) != 31: raise AssertionError("expected 31 witnesses")
    for point in WITNESSES:
        if not on_curve(point, MINIMAL_A): raise AssertionError(f"bad public witness: {point}")
    family_a=family_ainvs(RECORD_T)
    df, dm = discriminant(family_a), discriminant(MINIMAL_A)
    ratio=df/dm
    u=Fraction(exact_nth_root(ratio.numerator,12), exact_nth_root(ratio.denominator,12))
    a1,a2,a3,_,_=family_a; ta1,ta2,ta3,_,_=MINIMAL_A
    s=(u*ta1-a1)/2
    r=(u**2*ta2-a2+s*a1+s*s)/3
    tt=(u**3*ta3-a3-r*a1)/2
    if transformed_ainvs(family_a,u,r,s,tt) != MINIMAL_A:
        raise AssertionError("change of variables does not recover minimal model")
    rows=[]
    for i,(xn,yn) in enumerate(WITNESSES,1):
        xo=u*u*xn+r; yo=u**3*yn+s*u*u*xn+tt
        if not on_curve((xo,yo),family_a): raise AssertionError(f"transport failed for witness {i}")
        h=max(abs(xo.numerator),xo.denominator)
        rows.append({"index":i,"minimal_x":str(xn),"minimal_y":str(yn),
                     "family_x_numerator":str(xo.numerator),"family_x_denominator":str(xo.denominator),
                     "family_x_projective_height":str(h),"family_x_projective_height_digits":len(str(h))})
    hs=sorted(int(row["family_x_projective_height"]) for row in rows)
    return {"source":ICARM_URL,"record_t":str(RECORD_T),
            "family_ainvariants":[str(v) for v in family_a],"minimal_ainvariants":[str(v) for v in MINIMAL_A],
            "family_discriminant":str(df),"minimal_discriminant":str(dm),
            "change_of_variables":{"u":str(u),"r":str(r),"s":str(s),"t":str(tt)},"witnesses":rows,
            "minimum_family_x_projective_height":str(hs[0]),"median_family_x_projective_height":str(hs[len(hs)//2]),
            "maximum_family_x_projective_height":str(hs[-1]),
            "reachable_at_h_2m":sum(h<=2_000_000 for h in hs),"reachable_at_h_20m":sum(h<=20_000_000 for h in hs)}


def splitmix64(v:int)->int:
    v=(v+0x9E3779B97F4A7C15)&((1<<64)-1)
    v=(v^(v>>30))*0xBF58476D1CE4E5B9&((1<<64)-1)
    v=(v^(v>>27))*0x94D049BB133111EB&((1<<64)-1)
    return v^(v>>31)


def generate_candidates(count:int,seed:int,dmin:int,dmax:int,span:Fraction):
    if count<1 or dmin<1 or dmax<dmin or span<=0: raise ValueError("invalid candidate generation parameters")
    result=[RECORD_T]; seen={RECORD_T}; state=seed&((1<<64)-1); attempts=0
    while len(result)<=count:
        attempts+=1
        if attempts>count*100: raise RuntimeError("candidate generator stalled")
        state=splitmix64(state); den=dmin+state%(dmax-dmin+1)
        state=splitmix64(state); span_num=max(1,(span.numerator*den)//span.denominator)
        offset=int(state%(2*span_num+1))-span_num
        center=(RECORD_T.numerator*den+RECORD_T.denominator//2)//RECORD_T.denominator
        t=Fraction(center+offset,den)
        if t not in seen and -(1<<63)<t.numerator<(1<<63) and 0<t.denominator<(1<<63):
            seen.add(t); result.append(t)
    return result


def sha256_file(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_helper(force=False):
    nvcc=shutil.which("nvcc")
    if nvcc is None: raise RuntimeError("nvcc is required")
    if not CUDA_SOURCE_PATH.exists(): raise RuntimeError(f"missing CUDA source {CUDA_SOURCE_PATH}")
    BUILD_DIR.mkdir(parents=True,exist_ok=True)
    source_hash=sha256_file(CUDA_SOURCE_PATH); binary=BUILD_DIR/f"mestre_nagao_{source_hash[:16]}"
    cmd=[nvcc,"-O3","-std=c++14","-arch=native","-lineinfo","-o",str(binary),str(CUDA_SOURCE_PATH)]
    if force or not binary.exists():
        print("Compiling CUDA scorer...",flush=True); subprocess.run(cmd,cwd=ROOT,check=True)
    version=subprocess.check_output([nvcc,"--version"],text=True).strip().splitlines()[-1]
    return binary,{"source_path":str(CUDA_SOURCE_PATH.relative_to(ROOT)),"source_sha256":source_hash,
                   "binary_sha256":sha256_file(binary),"compile_command":cmd,"nvcc_version":version}


def helper_input(path,indexed):
    with path.open("w") as h:
        for i,t in indexed: h.write(f"{i}\t{t.numerator}\t{t.denominator}\n")


def parse_scores(path,cmap):
    out=[]
    for line in path.read_text().splitlines():
        if line.strip():
            i,s,g=line.split("\t"); i=int(i); out.append(ScoreRow(i,cmap[i],float(s),int(g)))
    return out


def score_candidates(binary:Path,candidates,devices,prime_bound,label,batch_size):
    cmap={i:t for i,t in enumerate(candidates)}; assignments={d:[] for d in devices}
    for i,t in cmap.items(): assignments[devices[i%len(devices)]].append((i,t))
    lock=threading.Lock(); started=time.perf_counter(); total=len(candidates); completed_global=0
    def worker(device):
        nonlocal completed_global
        rows=[]; stats=[]; assigned=assignments[device]
        for b,start in enumerate(range(0,len(assigned),batch_size),1):
            batch=assigned[start:start+batch_size]
            inp=BUILD_DIR/f"mn-{label}-gpu{device}-batch{b}.tsv"; outp=BUILD_DIR/f"mn-{label}-gpu{device}-batch{b}.out"
            helper_input(inp,batch); cmd=[str(binary),"--device",str(device),"--input",str(inp),"--output",str(outp),"--prime-bound",str(prime_bound)]
            t0=time.perf_counter(); done=subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True); t1=time.perf_counter(); sec=t1-t0
            rows.extend(parse_scores(outp,cmap)); stats.append({"device":device,"batch":b,"candidate_count":len(batch),"seconds":sec,
                "start_seconds_from_stage":t0-started,"end_seconds_from_stage":t1-started,"helper_stderr":done.stderr.strip()})
            with lock:
                completed_global+=len(batch); elapsed=time.perf_counter()-started; rate=completed_global/elapsed if elapsed else 0; eta=(total-completed_global)/rate if rate else 0
                print(f"[{label}] GPU {device} batch {b}: {completed_global:,}/{total:,} ({100*completed_global/total:5.1f}%) rate={rate:,.0f}/s ETA={eta:,.1f}s",flush=True)
        return rows,stats
    all_rows=[]; all_stats=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for future in [pool.submit(worker,d) for d in devices]:
            rows,stats=future.result(); all_rows.extend(rows); all_stats.extend(stats)
    all_rows.sort(key=lambda r:r.index)
    if [row.index for row in all_rows]!=list(range(len(candidates))):
        raise RuntimeError("GPU scorer returned missing or duplicate candidate rows")
    if len(devices)>1 and not any(
        a["device"]!=b["device"] and
        max(a["start_seconds_from_stage"],b["start_seconds_from_stage"]) <
        min(a["end_seconds_from_stage"],b["end_seconds_from_stage"])
        for i,a in enumerate(all_stats) for b in all_stats[i+1:]
    ):
        raise RuntimeError("GPU worker intervals did not overlap")
    return all_rows,all_stats


def family_mod(num,den,prime):
    if den%prime==0:return None
    t=(num%prime)*pow(den%prime,prime-2,prime)%prime; t2=t*t%prime; t3=t2*t%prime; t4=t2*t2%prime
    L=(446667*t2+471466*t+239031)%prime; pv=(318552*t2+368554*t-72570)%prime; qv=(733413*t2-45082*t-14960)%prime
    D=5*(7174492962*t4-7114589515*t3-22069002960*t2+3909144679*t-205134150)%prime
    E=(882769396002*t4+811447034567*t3-1174040743*t2-32493137198*t-2386325360)%prime
    B=(pv*qv*(L+pv+qv)-pv*E-qv*D)%prime
    return L,pv,qv,D,E,B


def discriminant_mod(v,prime):
    L,_p,_q,D,E,B=v; a1=-L%prime; a2=(D+E)%prime; a3=-B%prime; a4=D*E%prime
    b2=(a1*a1+4*a2)%prime; b4=(2*a4+a1*a3)%prime; b6=a3*a3%prime
    b8=(-a1*a3*a4+a2*a3*a3-a4*a4)%prime
    return (-b2*b2*b8-8*b4**3-27*b6*b6+9*b2*b4*b6)%prime


def cpu_np(t:Fraction,prime:int):
    v=family_mod(t.numerator,t.denominator,prime)
    if v is None or discriminant_mod(v,prime)==0:return None
    L,_p,_q,D,E,B=v; residues={y*y%prime for y in range(1,prime)}; total=0
    for x in range(prime):
        disc=((L*x+B)**2+4*x*(x+D)*(x+E))%prime
        total += 0 if disc==0 else (1 if disc in residues else -1)
    return prime+1+total


def odd_primes_up_to(bound:int):
    sieve=bytearray(b"\x01")*(bound+1)
    sieve[:2]=b"\x00\x00"
    for p in range(2,math.isqrt(bound)+1):
        if sieve[p]:
            sieve[p*p:bound+1:p]=b"\x00"*(((bound-p*p)//p)+1)
    return [p for p in range(3,bound+1,2) if sieve[p]]


def validate_gpu_counts(binary:Path,device:int,candidates,prime_bound:int):
    indices=sorted({0,1,len(candidates)//2,len(candidates)-1})
    sample=[(i,candidates[i]) for i in indices]
    inp=BUILD_DIR/f"mn-validation-gpu{device}.tsv"; outp=BUILD_DIR/f"mn-validation-gpu{device}.out"
    counts=BUILD_DIR/f"mn-validation-gpu{device}-counts.tsv"; helper_input(inp,sample)
    cmd=[str(binary),"--device",str(device),"--input",str(inp),"--output",str(outp),"--prime-bound",str(prime_bound),"--counts",str(counts)]
    done=subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True); smap=dict(sample); checked=0; mismatches=[]
    primes=odd_primes_up_to(prime_bound)
    scores={i:[0.0,0] for i in indices}
    lines=counts.read_text().splitlines()
    if len(lines)!=len(sample)*len(primes):
        raise RuntimeError("GPU validation count output is incomplete")
    for line,(expected_i,expected_p) in zip(lines,((i,p) for i in indices for p in primes)):
        i,p,n=map(int,line.split("\t"))
        if (i,p)!=(expected_i,expected_p):
            raise RuntimeError(f"GPU validation count row is missing, duplicated or out of order: {(i,p)}")
        expected=cpu_np(smap[i],p)
        expected=-1 if expected is None else expected
        checked+=1
        if n!=expected:
            mismatches.append({"candidate_index":i,"t":str(smap[i]),"prime":p,"gpu_np":n,"cpu_np":expected})
        if n!=-1:
            if n<=0: raise RuntimeError(f"invalid GPU point count {n} at candidate {i}, prime {p}")
            scores[i][0]+=(1.0-(p-1)/n)*math.log(p)
            scores[i][1]+=1
    if mismatches: raise RuntimeError(f"GPU/CPU point-count mismatch: {mismatches[:3]}")
    gpu_scores={row.index:row for row in parse_scores(outp,smap)}
    if set(gpu_scores)!=set(indices): raise RuntimeError("GPU validation score output is incomplete")
    for i,(score,good) in scores.items():
        row=gpu_scores[i]
        if row.good_primes!=good or not math.isclose(row.score,score,rel_tol=1e-10,abs_tol=1e-9):
            raise RuntimeError(f"GPU/CPU score mismatch at candidate {i}: GPU={row}, CPU={(score,good)}")
    return {"device":device,"prime_bound":prime_bound,"candidate_indices":indices,
            "candidate_count":len(sample),"candidate_prime_pairs_checked":checked,
            "all_gpu_count_rows_aggregated":len(lines),"oracle_prime_count":len(primes),
            "oracle_prime_min":primes[0],"oracle_prime_max":primes[-1],
            "scores_checked":len(sample),"mismatches":0,"helper_stderr":done.stderr.strip()}


def percentile(rows,record_index=0):
    record=next(r for r in rows if r.index==record_index); below=sum(r.score<=record.score for r in rows); greater=sum(r.score>record.score for r in rows)
    return {"score":record.score,"good_primes":record.good_primes,"rank_higher_score_is_better":greater+1,"population":len(rows),"percentile":100.0*below/len(rows)}


def device_listing():
    binary=ROOT/"ratpoints_gpu"
    if binary.exists():
        try:return subprocess.check_output([str(binary),"--list-devices"],text=True)
        except subprocess.SubprocessError:pass
    smi=shutil.which("nvidia-smi")
    if smi:return subprocess.check_output([smi,"--query-gpu=index,name,pci.bus_id,compute_cap,memory.total","--format=csv,noheader"],text=True)
    return "unavailable"


def git(args,check=True):return subprocess.run(["git",*args],cwd=ROOT,check=check,capture_output=True,text=True)


def commit_and_push(paths,message):
    for path in paths:git(["add","--",str(path.relative_to(ROOT))])
    if git(["diff","--cached","--quiet"],check=False).returncode==0:
        head=git(["rev-parse","HEAD"]).stdout.strip();return {"committed":False,"head":head,"pushed":False,"reason":"result files unchanged"}
    git(["commit","-m",message]);head=git(["rev-parse","HEAD"]).stdout.strip();branch=git(["branch","--show-current"]).stdout.strip()
    if branch!="master":raise RuntimeError(f"refusing automatic push from branch {branch!r}")
    git(["push","origin","master"]);remote=git(["ls-remote","origin","refs/heads/master"]).stdout.split()[0]
    if remote!=head:raise RuntimeError(f"push verification failed: HEAD={head}, origin/master={remote}")
    return {"committed":True,"head":head,"pushed":True,"origin_master":remote}


def score_json(row):return {"index":row.index,"t":f"{row.t.numerator}/{row.t.denominator}","score":row.score,"good_primes":row.good_primes}

def environment():return {"python":sys.version.split()[0],"implementation":platform.python_implementation(),"platform":platform.platform(),"device_listing":device_listing()}
