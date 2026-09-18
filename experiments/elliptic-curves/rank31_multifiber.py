#!/usr/bin/env python3
"""Inventory campaign finalists and run bounded fixed-fiber GPU searches.

The search rectangle is |n| <= height, 1 <= d <= denominators for
x=center+n/d on the stated integral Weierstrass model. Results are exact
square-checked in Python; Sage certification is performed separately.
"""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from rank31_mn_common import RECORD_T, RESULTS_DIR, calibration, discriminant, family_ainvs, family_values, on_curve
from rank31_gpu_fixed_search import run_command

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ('-47/80', '-191989/4040887', '532929/2579219', '-47/500', '-353682/460195')


def inventory(limit=24, standalone_extra=6):
    seen = {}
    sources = sorted(RESULTS_DIR.glob('rank31-mestre-nagao-campaign-*/global-stage-b.json.gz'))
    for source in sources:
        summary_path=source.parent/'campaign.summary.json'
        parameters=json.loads(summary_path.read_text())['parameters'] if summary_path.exists() else {}
        data = json.loads(gzip.decompress(source.read_bytes()))
        for row in data['finalists']:
            t = str(Fraction(row['t']))
            if t == str(RECORD_T):
                continue
            old = seen.get(t)
            item = {'t': t, 'score': row['score'], 'prime_bound': parameters.get('second_prime_bound'),
                    'good_primes': row['good_primes'], 'stage_a_score': row.get('stage_a_score'),
                    'stage_a_prime_bound': parameters.get('first_prime_bound'), 'source_campaigns': [],
                    'source_seeds': [], 'control': False}
            if old is None:
                seen[t] = item
                old = item
            if source.parent.name not in old['source_campaigns']:
                old['source_campaigns'].append(source.parent.name)
            if row.get('source_seed') is not None and row['source_seed'] not in old['source_seeds']:
                old['source_seeds'].append(row['source_seed'])
            if row['score'] > old['score']:
                for key in ('score', 'good_primes', 'stage_a_score'):
                    old[key] = item[key]
    ranked = sorted(seen.values(), key=lambda r: (-r['score'], Fraction(r['t'])))
    chosen = []
    for t in CORE:
        if str(Fraction(t)) in seen:
            chosen.append(seen[str(Fraction(t))])
    for row in ranked:
        if row not in chosen and len(chosen) < limit:
            chosen.append(row)
    standalone_sources=sorted(RESULTS_DIR.glob('rank31-mestre-nagao-20*.json.gz'))
    for source in standalone_sources:
        data=json.loads(gzip.decompress(source.read_bytes()))
        params=data['parameters']
        refined=data['refined_results']
        targets={row['t'] for row in refined}
        broad={row['t']:row for row in data['broad_results'] if row['t'] in targets}
        for row in refined:
            t=str(Fraction(row['t']))
            if t==str(RECORD_T):continue
            broad_row=broad.get(t,{})
            item={'t':t,'score':row['score'],'prime_bound':params['refine_prime_bound'],
                  'good_primes':row['good_primes'],'stage_a_score':broad_row.get('score'),
                  'stage_a_prime_bound':params['broad_prime_bound'],
                  'source_campaigns':[source.name], 'source_seeds':[params['seed']], 'control':False}
            old=seen.get(t)
            if old is None:
                seen[t]=item
            else:
                if source.name not in old['source_campaigns']:old['source_campaigns'].append(source.name)
                if params['seed'] not in old['source_seeds']:old['source_seeds'].append(params['seed'])
                if (item['prime_bound']>(old['prime_bound'] or 0) or
                    (item['prime_bound']==old['prime_bound'] and item['score']>old['score'])):
                    for key in ('score','prime_bound','good_primes','stage_a_score','stage_a_prime_bound'):
                        old[key]=item[key]
    extras=sorted((row for row in seen.values() if row not in chosen and row['prime_bound']==100000
                   and any(source.startswith('rank31-mestre-nagao-20') for source in row['source_campaigns'])),
                  key=lambda row:(-row['score'],Fraction(row['t'])))[:standalone_extra]
    chosen.extend(extras)
    for row in chosen:
        row['selection_reason'] = ('required strong lead' if row['t'] in CORE else
                                   'standalone refined frontier' if row in extras else 'global Stage B frontier')
    control=calibration()
    return {'campaign_files': [str(p.relative_to(RESULTS_DIR)) for p in sources+standalone_sources],
            'deduplicated_count': len(seen), 'selected': chosen,
            'control': {'t': str(RECORD_T), 'role': 'published rank-31 calibration',
                        'exact_curve_witnesses_verified':len(control['witnesses']),
                        'verified_witness_rows_sha256':hashlib.sha256(json.dumps(control['witnesses'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                        'source':control['source'],'minimal_ainvariants':control['minimal_ainvariants'],
                        'median_family_x_projective_height':control['median_family_x_projective_height']}}


def integral_model(t):
    t = Fraction(t)
    a = family_ainvs(t)
    scale = t.denominator**2
    # Weierstrass weights, rather than the coefficient positions.
    ints = [a[i] * scale**w for i, w in enumerate((1, 2, 3, 4, 6))]
    if any(z.denominator != 1 for z in ints):
        raise ArithmeticError('nonintegral specialized model')
    ints = tuple(map(int, ints))
    L, p, q, D, E, B = family_values(t)
    sections = [('P0', Fraction(0), Fraction(0)), ('PD', -D, Fraction(0)),
                ('PE', -E, Fraction(0)), ('PQ', -p*q, p*(p*q-E))]
    points = []
    for label, x, y in sections:
        if not on_curve((x,y), a):
            raise ArithmeticError('section off family curve: ' + label)
        xi, yi = x*scale**2, y*scale**3
        if xi.denominator != 1 or yi.denominator != 1 or not on_curve((xi,yi), ints):
            raise ArithmeticError('section off integral curve: ' + label)
        points.append({'label': label, 'x': str(xi), 'y': str(yi)})
    digest = hashlib.sha256(json.dumps(list(ints), separators=(',', ':')).encode()).hexdigest()
    return {'t': str(t), 'ainvariants': list(map(str, ints)), 'model_sha256': digest,
            'sections': points, 'scale': str(scale)}


def torsion_triviality_certificate(model):
    """Use good reductions with coprime group orders to certify P0 infinite."""
    a=tuple(map(int,model['ainvariants']))
    delta=int(discriminant(a))
    gcd_orders=0
    evidence=[]
    for prime in (3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97):
        if delta%prime==0:continue
        a1,a2,a3,a4,a6=a
        b2=(a1*a1+4*a2)%prime;b4=(a1*a3+2*a4)%prime;b6=(a3*a3+4*a6)%prime
        count=prime+1
        for x in range(prime):
            z=(4*x*x*x+b2*x*x+2*b4*x+b6)%prime
            if z:count+=1 if pow(z,(prime-1)//2,prime)==1 else -1
        evidence.append({'prime':prime,'order':count})
        gcd_orders=math.gcd(gcd_orders,count)
        if gcd_orders==1:break
    return {'trivial_torsion':gcd_orders==1,'good_reductions':evidence,
            'torsion_order_divides':gcd_orders}


def centers(model):
    a1,a2,a3,a4,a6 = map(int, model['ainvariants'])
    result = {p['label']: int(p['x']) for p in model['sections']}
    # Largest real root of the cubic discriminant, rounded upward.
    b2 = a1*a1 + 4*a2
    b4 = 2*a4 + a1*a3
    b6 = a3*a3 + 4*a6
    f = lambda x: 4*x*x*x+b2*x*x+2*b4*x+b6
    bound = 1 << max(4, max(abs(b2),abs(b4),abs(b6)).bit_length())
    lo, hi = -bound, bound
    if f(lo) < 0 < f(hi):
        while hi-lo > 1:
            mid=(hi+lo)//2
            if f(mid) <= 0: lo=mid
            else: hi=mid
        result['R'] = hi
    return result


def coefficients(model, center, stride=1):
    a1,a2,a3,a4,a6 = map(int, model['ainvariants'])
    b2=a1*a1+4*a2; b4=2*a4+a1*a3; b6=a3*a3+4*a6
    c=center
    return (4*c**3+b2*c*c+2*b4*c+b6,
            (12*c*c+2*b2*c+2*b4)*stride, (12*c+b2)*stride**2, 4*stride**3)


def exact_points(stdout, model, center, coeffs, stride=1):
    a=tuple(map(int,model['ainvariants']))
    a1,a2,a3,a4,a6=a
    found={}
    for line in stdout.splitlines():
        n,yroot,d=map(int,line.split())
        if d<1 or math.gcd(n,d)!=1: raise ArithmeticError('nonreduced output')
        value=coeffs[3]*n**3*d+coeffs[2]*n*n*d*d+coeffs[1]*n*d**3+coeffs[0]*d**4
        if yroot*yroot!=value: raise ArithmeticError('false exact square')
        x=Fraction(center*d+stride*n,d)
        y=(Fraction(yroot,d*d)-a1*x-a3)/2
        if not on_curve((x,y),a): raise ArithmeticError('point off curve')
        found[(x,y)]={'x':str(x),'y':str(y),'n':n,'d':d}
    return sorted(found.values(),key=lambda p:(Fraction(p['x']),Fraction(p['y'])))


def search(model, label, height, denominators, timeout=90, stride=1,
           center_override=None, square_denominators=False):
    if square_denominators and not 1<=denominators<=46340:
        raise ValueError('square-denominator root bound must be in 1..46340')
    center=centers(model)[label] if center_override is None else int(center_override)
    coeffs=coefficients(model,center,stride)
    command=[str(ROOT/'ratpoints_gpu'),' '.join(map(str,coeffs)),str(height),
             '-dl','1','-du',str(denominators),'--devices','0,1','-i','-v','-f','%x %y %z\\n']
    if square_denominators:command.append('--square-denominators')
    run=run_command(command,timeout)
    points=exact_points(run['stdout'],model,center,coeffs,stride) if run['returncode']==0 else []
    metrics_line=next((line for line in run['stderr'].splitlines() if line.startswith('wall_ms=')), '')
    metrics=dict(re.findall(r'([a-z_]+)=([0-9.]+)',metrics_line))
    utilization={'0':[],'1':[]}
    for sample in run['utilization_samples']:
        for line in sample.splitlines():
            fields=[field.strip() for field in line.split(',')]
            if len(fields)==2 and fields[0] in utilization:
                try: utilization[fields[0]].append(int(fields[1]))
                except ValueError:pass
    sites=(2*height+1)*denominators
    return {'t':model['t'],'model_sha256':model['model_sha256'],'center_label':label,
            'center':str(center),'stride':str(stride),'height':height,'denominators':denominators,
            'denominator_mode':'squares' if square_denominators else 'consecutive',
            'sites':sites,'elapsed_seconds':run['elapsed_seconds'],
            'sites_per_second':sites/run['elapsed_seconds'],'devices':'0,1',
            'returncode':run['returncode'],'timed_out':run['timed_out'],
            'metrics':metrics,'points':points,
            'gpu_utilization_percent':{device:{'samples':len(values),
                'mean':sum(values)/len(values) if values else None,'max':max(values) if values else None}
                for device,values in utilization.items()},
            'stderr_tail':run['stderr'][-1200:]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=('inventory','search'))
    ap.add_argument('--index',type=int,default=0)
    ap.add_argument('--center',default='P0')
    ap.add_argument('--center-x',type=int,help='explicit integral-model x center')
    ap.add_argument('--height',type=int,default=1000000)
    ap.add_argument('--denominators',type=int,default=64)
    ap.add_argument('--stride',default='1',help='integer x increment per n/d, or family')
    ap.add_argument('--square-denominators',action='store_true',help='search d=k^2, 1<=k<=--denominators')
    ap.add_argument('--tag',default='screen')
    args=ap.parse_args()
    path=RESULTS_DIR/'rank31-multifiber-inventory.json'
    if args.mode=='inventory':
        data=inventory()
        for row in data['selected']:
            row['model']=integral_model(row['t'])
            row['already_deeply_searched']=row['t']=='-47/80'
            row['existing_certified_rank_lower_bound']=3 if row['t']=='-47/80' else None
            row['known_rational_points']=row['model']['sections']
        path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
        print(path, len(data['selected']), data['deduplicated_count'])
    else:
        data=json.loads(path.read_text())
        row=data['selected'][args.index]
        stride=int(row['model']['scale'])**2 if args.stride=='family' else int(args.stride)
        if stride<1:ap.error('--stride must be positive')
        output=search(row['model'],args.center,args.height,args.denominators,stride=stride,
                      center_override=args.center_x,square_denominators=args.square_denominators)
        target=RESULTS_DIR/f'rank31-multifiber-{args.tag}-{args.index:02d}-{args.center}.json'
        target.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
        print(target, len(output['points']), round(output['sites_per_second']/1e12,3),'Tsites/s')
        if output['returncode']!=0: raise SystemExit(1)

if __name__=='__main__': main()
