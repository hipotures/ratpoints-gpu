#!/usr/bin/env python3
"""Exact Sage verification and subgroup saturation for one inventory fiber."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from sage.all import EllipticCurve, QQ, pari

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'

def point_json(p):
    return {'x':str(p[0]),'y':str(p[1])} if not p.is_zero() else {'infinity':True}

def run(index, reports, output=None, verify_only=False, max_prime=11, descent=False, pairing=False, height_proof=False, rank_bound=False, minimal_info=False, pari_rank=False, pari_effort=0, geometry=False):
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
    if geometry:
        from itertools import product
        known=[p for label,p in candidates if label in ('P0','PD','PQ')]
        section_x={p[0] for _,p in candidates[:4]}
        rows=[]
        for coefficients in product(range(-2,3),repeat=3):
            if coefficients==(0,0,0):continue
            p=sum((coefficient*q for coefficient,q in zip(coefficients,known)),E(0))
            if p.is_zero():continue
            x=p[0]
            rows.append({'coefficients':coefficients,'x':str(x),'y':str(p[1]),
                         'x_height_bits':max(abs(x.numerator()).nbits(),x.denominator().nbits()),
                         'x_denominator_bits':x.denominator().nbits(),
                         'section_x':x in section_x})
        unseen=[row for row in rows if not row['section_x']]
        unseen.sort(key=lambda row:(row['x_height_bits'],row['x_denominator_bits'],row['coefficients']))
        return {'t':model['t'],'model_sha256':digest,'mode':'small_subgroup_geometry',
                'coefficient_range':[-2,2],'nonzero_combinations_checked':len(rows),
                'min_nonsection_x_height_bits':min(row['x_height_bits'] for row in unseen),
                'min_nonsection_x_denominator_bits':min(row['x_denominator_bits'] for row in unseen),
                'smallest_nonsection_points':unseen[:12]}
    if minimal_info:
        minimal=E.minimal_model()
        iso=E.isomorphism_to(minimal)
        mapped=[(label,iso(p)) for label,p in candidates[:4]]
        return {'t':model['t'],'model_sha256':digest,'mode':'minimal_model_diagnostic',
                'integral_ainvariants':[str(a) for a in E.ainvs()],
                'minimal_ainvariants':[str(a) for a in minimal.ainvs()],
                'isomorphism_parameters':[str(z) for z in iso.tuple()],
                'minimal_sections':[{'label':label,**point_json(p)} for label,p in mapped],
                'integral_coefficient_max_bits':max(max(abs(a.numerator()).nbits(),a.denominator().nbits()) for a in E.ainvs()),
                'minimal_coefficient_max_bits':max(max(abs(a.numerator()).nbits(),a.denominator().nbits()) for a in minimal.ainvs())}
    if pari_rank:
        known=[p for label,p in candidates if label in ('P0','PD','PQ')]
        result=pari.ellrank(pari.ellinit([int(a) for a in E.ainvs()]),pari_effort,
                            [[p[0],p[1]] for p in known])
        discovered=[]
        for point in result[3]:
            p=E(QQ(point[0]),QQ(point[1]))
            discovered.append(point_json(p))
        return {'t':model['t'],'model_sha256':digest,'mode':'pari_ellrank',
                'effort':pari_effort,'lower_bound':int(result[0]),
                'upper_bound':int(result[1]),'sha_information':str(result[2]),
                'points':discovered,'known_points_supplied':[point_json(p) for p in known],
                'note':'PARI rank bounds are unconditional; independently certify any new lower bound before reporting it'}
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
    ap.add_argument('--minimal-info',action='store_true')
    ap.add_argument('--pari-rank',action='store_true')
    ap.add_argument('--pari-effort',type=int,default=0)
    ap.add_argument('--geometry',action='store_true')
    ap.add_argument('--max-prime',type=int,default=11)
    ap.add_argument('index',type=int);ap.add_argument('output');ap.add_argument('reports',nargs='*')
    args=ap.parse_args();Path(args.output).write_text(json.dumps(run(args.index,args.reports,args.output,args.verify_only,args.max_prime,args.descent,args.pairing,args.height_proof,args.rank_bound,args.minimal_info,args.pari_rank,args.pari_effort,args.geometry),indent=2,sort_keys=True)+'\n')
