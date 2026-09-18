#!/usr/bin/env python3
"""Exact Sage verification and subgroup saturation for one inventory fiber."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from sage.all import EllipticCurve, QQ

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'

def point_json(p):
    return {'x':str(p[0]),'y':str(p[1])} if not p.is_zero() else {'infinity':True}

def run(index, reports, output=None, verify_only=False, max_prime=11, descent=False, pairing=False, height_proof=False, rank_bound=False):
    inventory=json.loads((RESULTS/'rank31-multifiber-inventory.json').read_text())
    row=inventory['selected'][index]
    model=row['model']
    ainvs=[QQ(z) for z in model['ainvariants']]
    digest=hashlib.sha256(json.dumps(list(map(int,ainvs)),separators=(',',':')).encode()).hexdigest()
    assert digest==model['model_sha256']
    E=EllipticCurve(QQ,ainvs)
    if E.discriminant()==0: raise ArithmeticError('singular specialization')
    candidates=[]
    for section in model['sections']:
        p=E(QQ(section['x']),QQ(section['y']))
        candidates.append((section['label'],p))
    gpu=[]
    for report_path in reports:
        report=json.loads(Path(report_path).read_text())
        if report['t']!=model['t'] or report['model_sha256']!=digest:
            raise ValueError('GPU report belongs to another fiber')
        for point in report['points']:
            p=E(QQ(point['x']),QQ(point['y']))
            if all(p!=old for _,old in candidates):
                candidates.append((f"GPU:{Path(report_path).name}",p))
            gpu.append(point_json(p))
    if descent:
        known=[p for label,p in candidates if label in ('P0','PD','PQ')]
        lower,upper,discovered=E.simon_two_descent(known_points=known,limbigprime=0)
        return {'t':model['t'],'model_sha256':digest,'mode':'simon_two_descent',
                'lower_bound':int(lower),'upper_bound':int(upper),
                'points':[point_json(p) for p in discovered],
                'note':'bounded by outer Docker runner; inspect returned bounds before rank claims'}
    if rank_bound:
        return {'t':model['t'],'model_sha256':digest,'mode':'mwrank_upper_bound',
                'rigorous_upper_bound':int(E.rank_bound(algorithm='mwrank'))}
    if pairing:
        known=[p for label,p in candidates if label in ('P0','PD','PQ')]
        matrix=E.height_pairing_matrix(known)
        return {'t':model['t'],'model_sha256':digest,'mode':'numerical_height_pairing',
                'matrix':[[str(x) for x in row] for row in matrix.rows()],
                'determinant':str(matrix.det()),
                'classification':'numerical heuristic; not a rank certificate'}
    if height_proof:
        from rank31_height_certificate import certify
        known=[p for label,p in candidates if label in ('P0','PD','PQ')]
        proof=certify(E,known)
        dependent=[p for label,p in candidates if label in ('P0','PD','PE')]
        assert dependent[0]+dependent[1]+dependent[2]==E(0)
        negative=certify(E,dependent,max_doublings=4)
        if negative['certified_rank'] is not None:
            raise ArithmeticError('height certificate falsely proved dependent control')
        negative_four=certify(E,known+[dependent[2]],max_doublings=4)
        if negative_four['certified_rank'] is not None:
            raise ArithmeticError('height certificate falsely proved dependent four-point control')
        growth=[]
        for label,p in candidates[4:]:
            if any(p==q or p==-q for _,q in candidates[:4]):continue
            growth.append({'label':label,'point':point_json(p),
                           'proof':certify(E,known+[p])})
        return {'t':model['t'],'model_sha256':digest,'mode':'exact_height_certificate',
                'sections_verified':[{'label':label,**point_json(p)} for label,p in candidates[:4]],
                'proof':proof,'dependent_control':{'relation':'P0+PD+PE=O',
                                                   'interval_test':'inconclusive for both dependent triples and quadruples'},
                'gpu_rank_growth_tests':growth}
    if verify_only:
        return {'t':model['t'],'model_sha256':digest,'sections_verified':[
            {'label':label,**point_json(p)} for label,p in candidates[:4]],
            'gpu_points_verified':gpu,'gpu_point_count':len(gpu),'status':'verified'}
    basis=[]
    steps=[{'label':'PE','status':'dependent','proof':'P0 + PD + PE = O, exact group law'}]
    started=time.monotonic()
    if output:
        Path(output).write_text(json.dumps({'t':model['t'],'status':'running',
            'current_label':'P0,PD,PQ','certified_subgroup_rank':0,'steps':steps},indent=2)+'\n')
    try:
        sat,index_sat,reg=E.saturation([p for label,p in candidates if label in ('P0','PD','PQ')], max_prime=max_prime)
        basis=list(sat)
        steps.append({'label':'P0,PD,PQ','status':'certified','saturation_index':str(index_sat),
                      'regulator':str(reg),'elapsed_seconds':time.monotonic()-started})
    except (ValueError,ArithmeticError) as exc:
        steps.append({'label':'P0,PD,PQ','status':'uncertified','error':str(exc),
                      'elapsed_seconds':time.monotonic()-started})
    for label,p in candidates:
        if not label.startswith('GPU:'):continue
        if any(p==q or p==-q for _,q in candidates[:4]):
            steps.append({'label':label,'status':'known section or inverse'});continue
        if not basis:
            steps.append({'label':label,'status':'awaiting baseline certification'});continue
        started=time.monotonic()
        if output:
            Path(output).write_text(json.dumps({'t':model['t'],'status':'running',
                'current_label':label,'certified_subgroup_rank':len(basis),'steps':steps},indent=2)+'\n')
        try:
            sat,index_sat,reg=E.saturation(basis+[p], max_prime=max_prime)
            grew=len(sat)>len(basis)
            if grew:basis=list(sat)
            steps.append({'label':label,'status':'independent' if grew else 'dependent',
                          'saturation_index':str(index_sat),'regulator':str(reg),
                          'elapsed_seconds':time.monotonic()-started})
        except (ValueError,ArithmeticError) as exc:
            steps.append({'label':label,'status':'uncertified','error':str(exc),
                          'elapsed_seconds':time.monotonic()-started})
    return {'t':model['t'],'model_sha256':digest,'ainvariants':list(map(str,E.ainvs())),
            'certified_subgroup_rank':len(basis),'independent_generators':[point_json(p) for p in basis],
            'sections_verified':[{'label':label,**point_json(p)} for label,p in candidates[:4]],
            'gpu_points_verified':gpu,'saturation':('full computed saturation bound (Sage/eclib)' if max_prime==-1
                                                  else f'through prime {max_prime} (Sage/eclib)'),
            'steps':steps}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true')
    ap.add_argument('--descent',action='store_true')
    ap.add_argument('--pairing',action='store_true')
    ap.add_argument('--height-proof',action='store_true')
    ap.add_argument('--rank-bound',action='store_true')
    ap.add_argument('--max-prime',type=int,default=11)
    ap.add_argument('index',type=int);ap.add_argument('output');ap.add_argument('reports',nargs='*')
    args=ap.parse_args();Path(args.output).write_text(json.dumps(run(args.index,args.reports,args.output,args.verify_only,args.max_prime,args.descent,args.pairing,args.height_proof,args.rank_bound),indent=2,sort_keys=True)+'\n')
