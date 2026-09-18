#!/usr/bin/env python3
"""Search sparse degree-four x(T) expressions and their specializations.

The finite ansatz is x=(sum c_i*b_i)/d, where d is 1..4, at most three
coefficients c_i are nonzero and each lies in [-12,12]. The eight b_i are
D,E,pq,p^2,q^2,L^2,pL,qL. Modular square tests are necessary conditions;
survivors receive an exact integer-square and curve-equation check.
"""
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

from rank31_mn_common import family_values, on_curve

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
NAMES=('D','E','pq','p2','q2','L2','pL','qL')
PRIMES=(11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107)
COEFFS=tuple(c for c in range(-12,13) if c)

def terms():
    for d in (1,2,3,4):
        for size in (1,2,3):
            for indices in itertools.combinations(range(len(NAMES)),size):
                for coefficients in itertools.product(COEFFS,repeat=size):
                    if math.gcd(d,math.gcd(*coefficients))==1:
                        yield d,indices,coefficients

def bases(values):
    L,p,q,D,E,_=values
    return D,E,p*q,p*p,q*q,L*L,p*L,q*L

def modular_rows(ainvs,basis,primes=PRIMES):
    a1,a2,a3,a4,a6=ainvs
    b2=a1*a1+4*a2;b4=2*a4+a1*a3;b6=a3*a3+4*a6
    return [(p,tuple(x%p for x in basis),b2%p,b4%p,b6%p,
             {x*x%p for x in range(p)}) for p in primes]

def survives(d,indices,coefficients,rows):
    for p,basis,b2,b4,b6,squares in rows:
        x=sum(c*basis[i] for i,c in zip(indices,coefficients))%p
        value=(4*x*x*x*d+b2*x*x*d*d+2*b4*x*d**3+b6*d**4)%p
        if value not in squares:return False
    return True

def scan_fiber(index,row):
    model=row['model'];scale=int(model['scale']);ainvs=tuple(map(int,model['ainvariants']))
    values=family_values(Fraction(row['t']))
    basis=tuple(int(v*scale**2) for v in bases(values))
    rows=modular_rows(ainvs,basis)
    a1,a2,a3,a4,a6=ainvs
    b2=a1*a1+4*a2;b4=2*a4+a1*a3;b6=a3*a3+4*a6
    known={Fraction(p['x']) for p in model['sections']}
    count=0;modular=0;exact=[];seen=set()
    for d,indices,coefficients in terms():
        count+=1
        if not survives(d,indices,coefficients,rows):continue
        modular+=1
        n=sum(c*basis[i] for i,c in zip(indices,coefficients))
        value=4*n**3*d+b2*n*n*d*d+2*b4*n*d**3+b6*d**4
        if value<0:continue
        root=math.isqrt(value)
        if root*root!=value:continue
        x=Fraction(n,d)
        if x in seen:continue
        y=(Fraction(root,d*d)-a1*x-a3)/2
        if not on_curve((x,y),ainvs):raise ArithmeticError('exact survivor off curve')
        seen.add(x)
        exact.append({'x':str(x),'y':str(y),'known_section_x':x in known,
                      'formula':{'denominator':d,'terms':[(NAMES[i],c) for i,c in zip(indices,coefficients)]}})
    return {'index':index,'t':row['t'],'model_sha256':model['model_sha256'],
            'tested_expressions':count,'modular_survivors':modular,
            'exact_points':exact,'new_x_count':sum(not p['known_section_x'] for p in exact)}

def scan_generic():
    sample=((t,p) for t in (-3,-2,-1,0,1,2,3) for p in PRIMES)
    rows=[]
    for t,p in sample:
        L,pp,q,D,E,B=map(int,family_values(Fraction(t)))
        rows.extend(modular_rows((-L,D+E,-B,D*E,0),bases((L,pp,q,D,E,B)),(p,)))
    count=0;survivors=[]
    for d,indices,coefficients in terms():
        count+=1
        if survives(d,indices,coefficients,rows):
            survivors.append({'denominator':d,'terms':[(NAMES[i],c) for i,c in zip(indices,coefficients)]})
    return {'tested_expressions':count,'modular_survivors':survivors,
            'sample_T':[-3,-2,-1,0,1,2,3],'sample_primes':list(PRIMES),
            'interpretation':'A square discriminant over Q(T) must survive every sample; the surviving expressions are the known x=-D,-E,-pq sections.'}

def main():
    inventory=json.loads((RESULTS/'rank31-multifiber-inventory.json').read_text())
    selected=inventory['selected']
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(scan_fiber,i,row) for i,row in enumerate(selected)]
        fibers=[future.result() for future in as_completed(futures)]
    fibers.sort(key=lambda x:x['index'])
    output={'ansatz':'x=(sum c_i*b_i)/d; d=1..4; 1..3 nonzero c_i in [-12,12]; b_i=D,E,pq,p^2,q^2,L^2,pL,qL',
            'modular_filter':'For each odd test prime, d^4 times the discriminant cubic must be a quadratic residue; no rational point in the finite ansatz is rejected.',
            'generic':scan_generic(),'fibers':fibers}
    path=RESULTS/'rank31-sparse-coordinate-scan.json'
    path.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(path,'fibers',len(fibers),'new x',sum(f['new_x_count'] for f in fibers))

if __name__=='__main__':main()
