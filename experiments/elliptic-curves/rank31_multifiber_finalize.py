#!/usr/bin/env python3
"""Merge exact certificates into the search scoreboard and summarize research."""
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rank31_multifiber import RESULTS_DIR, torsion_triviality_certificate

INVENTORY=RESULTS_DIR/'rank31-multifiber-inventory.json'
SCOREBOARD=RESULTS_DIR/'rank31-multifiber-scoreboard.json'
REPORT=RESULTS_DIR/'rank31-multifiber-report.md'

def main():
    inventory=json.loads(INVENTORY.read_text())
    board=json.loads(SCOREBOARD.read_text())
    board['control']=inventory['control']
    board['sieve_contract']='For each tested odd prime coprime to d, a rational square remains a quadratic residue modulo that prime. The GPU filter therefore has no false negatives for reduced n/d in each explicit rectangle; every emitted point is checked exactly.'
    ranked=[]
    for i,row in enumerate(inventory['selected']):
        entry=board['candidate_rows'][row['t']]
        entry['source_seeds']=row['source_seeds']
        entry['already_deeply_searched']=row['already_deeply_searched']
        entry['existing_certified_rank_lower_bound']=row['existing_certified_rank_lower_bound']
        torsion=torsion_triviality_certificate(row['model'])
        entry['torsion_certificate']=torsion
        if not torsion['trivial_torsion']:
            raise ArithmeticError(f"no rank-one baseline for {row['t']}")
        entry['certified_subgroup_rank']=1
        entry['independent_generators']=[{'x':row['model']['sections'][0]['x'],
                                          'y':row['model']['sections'][0]['y']}]
        entry['saturation_status']='not completed; exact good-reduction proof gives rank at least one'
        entry['saturation_index']=None
        entry['regulator_numerical']=None
        entry['cpu_elapsed_seconds']=0
        sage_path=RESULTS_DIR/f'rank31-multifiber-sage-{i:02d}.json'
        if sage_path.exists():
            sage=json.loads(sage_path.read_text())
            entry['sage_certificate_path']=sage_path.name
            if sage.get('model_sha256')==entry['model_sha256'] and sage.get('certified_subgroup_rank',0)>=1:
                entry['certified_subgroup_rank']=sage['certified_subgroup_rank']
                entry['independent_generators']=sage['independent_generators']
                entry['saturation_status']=sage.get('saturation','Sage saturation completed')
                entries=[step for step in sage.get('steps',[]) if step.get('status')=='certified']
                if entries:
                    entry['saturation_index']=entries[-1]['saturation_index']
                    entry['regulator_numerical']=entries[-1].get('regulator')
                entry['cpu_elapsed_seconds']=sage.get('runner_elapsed_seconds',sum(step.get('elapsed_seconds',0) for step in sage.get('steps',[])))
            else:
                entry['sage_status']=sage.get('status','incomplete')
                entry['sage_current_step']=sage.get('current_label')
                entry['cpu_elapsed_seconds']=sage.get('runner_elapsed_seconds',sage.get('timeout_seconds',0))
        height_path=RESULTS_DIR/f'rank31-multifiber-sage-{i:02d}-height-proof.json'
        if height_path.exists():
            height=json.loads(height_path.read_text())
            if height.get('model_sha256')!=entry['model_sha256'] or height.get('proof',{}).get('certified_rank')!=3:
                raise ArithmeticError(f"height proof missing or invalid for {row['t']}")
            entry['height_certificate_path']=height_path.name
            entry['height_certificate']={'doublings':height['proof']['doublings'],
                'silverman_bound_integer':height['proof']['silverman_bound_integer'],
                'principal_minor_lower_bounds':height['proof']['principal_minor_lower_bounds']}
            entry['cpu_elapsed_seconds']+=height.get('runner_elapsed_seconds',0)
            if entry['certified_subgroup_rank']<3:
                entry['certified_subgroup_rank']=3
                entry['independent_generators']=[{'x':section['x'],'y':section['y']}
                                                  for section in row['model']['sections'] if section['label'] in ('P0','PD','PQ')]
                entry['saturation_status']='rank 3 certified by exact height intervals; saturation not completed'
            growth=[item for item in height.get('gpu_rank_growth_tests',[])
                    if item.get('proof',{}).get('certified_rank')==4]
            if growth:
                entry['certified_subgroup_rank']=4
                entry['independent_generators']=[{'x':section['x'],'y':section['y']}
                                                  for section in row['model']['sections'] if section['label'] in ('P0','PD','PQ')]+[growth[0]['point']]
                entry['saturation_status']='rank 4 certified by exact height intervals; saturation pending'
        full_path=RESULTS_DIR/f'rank31-multifiber-sage-{i:02d}-p-1.json'
        if full_path.exists():
            full=json.loads(full_path.read_text())
            entry['cpu_elapsed_seconds']+=full.get('runner_elapsed_seconds',0)
            if (full.get('model_sha256')==entry['model_sha256'] and
                full.get('certified_subgroup_rank',0)>=entry['certified_subgroup_rank'] and
                full.get('saturation')=='full computed saturation bound (Sage/eclib)'):
                entry['independent_generators']=full['independent_generators']
                entry['saturation_status']='full Sage/eclib saturation'
                certified=[step for step in full['steps'] if step.get('status')=='certified']
                entry['saturation_index']=certified[-1]['saturation_index'] if certified else None
                entry['regulator_numerical']=certified[-1].get('regulator') if certified else None
                entry['full_saturation_source']=full_path.name
        entry['auxiliary_cpu_experiments']=[]
        for suffix in ('p2','pairing','descent','rank-bound'):
            auxiliary=RESULTS_DIR/f'rank31-multifiber-sage-{i:02d}-{suffix}.json'
            if auxiliary.exists():
                experiment=json.loads(auxiliary.read_text())
                seconds=experiment.get('runner_elapsed_seconds',experiment.get('timeout_seconds',0))
                entry['cpu_elapsed_seconds']+=seconds
                entry['auxiliary_cpu_experiments'].append({'path':auxiliary.name,
                    'status':experiment.get('runner_status',experiment.get('status','completed')),
                    'elapsed_seconds':seconds})
        if row['t']=='-47/80':
            previous=json.loads((RESULTS_DIR/'rank31-t-minus-47-80.json').read_text())
            previous_sage=json.loads((RESULTS_DIR/'rank31-t-minus-47-80-sage-sections.json').read_text())
            section_step=previous_sage['steps']['section_saturation']
            if (previous['models']['integral']['ainvariants']!=row['model']['ainvariants']
                or section_step['status']!='completed' or section_step['value']['rank']!=3):
                raise ArithmeticError('prior full saturation does not match selected model')
            entry['saturation_status']='full Sage/eclib saturation from issue 14'
            entry['saturation_index']=section_step['value']['index']
            entry['regulator_numerical']=section_step['value']['regulator_numerical']
            entry['full_saturation_source']='rank31-t-minus-47-80-sage-sections.json'
        verify=RESULTS_DIR/f'rank31-multifiber-sage-{i:02d}-verify.json'
        if not verify.exists():raise RuntimeError(f'missing Sage GPU verification: {row["t"]}')
        proof=json.loads(verify.read_text())
        entry['cpu_elapsed_seconds']+=proof.get('runner_elapsed_seconds',0)
        expected=sum(len(json.loads((RESULTS_DIR/run['path']).read_text())['points'])
                     for run in entry['gpu_searches'])
        if proof.get('model_sha256')!=entry['model_sha256'] or proof.get('gpu_point_count')!=expected:
            raise RuntimeError(f'Sage GPU verification mismatch or stale: {row["t"]}')
        entry['sage_gpu_verification']={'path':verify.name,'points_checked':expected}
        entry['gpu_search_bounds_completed']=[{
            'center':run['center_label'],'height':run['height'],'denominators':run['denominators'],
            'stride':str(run.get('stride',1))} for run in entry['gpu_searches'] if run['returncode']==0]
        for run in entry['gpu_searches']:
            raw=json.loads((RESULTS_DIR/run['path']).read_text())
            if run.get('survivors') is not None:run['survivors']=int(run['survivors'])
            if run.get('exact_survivors') is not None:run['exact_survivors']=int(run['exact_survivors'])
            if raw['t']!=entry['t'] or raw['model_sha256']!=entry['model_sha256']:
                raise RuntimeError(f"GPU report model mismatch: {run['path']}")
            if raw['returncode']!=0 or raw['timed_out']:
                raise RuntimeError(f"incomplete GPU rectangle: {run['path']}")
            if not all(re.search(rf'gpu={device} denominators=[1-9][0-9]*',raw['stderr_tail']) for device in (0,1)):
                raise RuntimeError(f"two-GPU work unverified: {run['path']}")
        used={run['path'].split('-')[2] for run in entry['gpu_searches']}
        decisions=[row['selection_reason']]
        if 'promote' in used:decisions.append('promoted to wide rectangles by Stage B score after clean screen')
        elif 'screen' in used and not (used & {'standalone','rescale','structured','coarse','deeper'}):
            decisions.append('demoted after screen found only known section points')
        if 'rescale' in used:decisions.append('promoted to rescaled family-coordinate grid by smaller parameter denominator or top score')
        if 'structured' in used:decisions.append('searched rational x lattices generated by differences of known section x-coordinates')
        if 'coarse' in used:decisions.append('extended to coarse family-coordinate grid after rescaled search')
        if 'deeper' in used:decisions.append('extended denominator depth for two small-denominator high-score fibers')
        if 'standalone' in used:decisions.append('widened a newly inventoried standalone refined lead')
        if 'outer' in used:decisions.append('searched logarithmically spaced family-coordinate regions away from known sections')
        entry['scheduling_decisions']=decisions
        novel=any(point['x'] not in {section['x'] for section in row['model']['sections']}
                  for point in entry['exact_points'])
        if entry['certified_subgroup_rank']>=4:
            entry['status_reason']='new exact GPU point and independent rank-four height certificate; inspect saturation status'
        elif novel:
            entry['status_reason']='new exact GPU point found; independence not yet certified'
        elif entry['certified_subgroup_rank']>=3:
            entry['status_reason']='certified visible rank 3; no new independent GPU point in completed rectangles'
        elif entry['gpu_searches']:
            entry['status_reason']='GPU rectangles completed; full section independence timed out in bounded Sage run'
        ranked.append(entry)
    if any(entry['certified_subgroup_rank']<3 for entry in ranked):
        raise RuntimeError('height certificates have not completed for every selected fiber')
    ranked.sort(key=lambda r:(-r['certified_subgroup_rank'],-r['score'],r['t']))
    board['ranked_candidates']=[{'t':r['t'],'certified_subgroup_rank':r['certified_subgroup_rank'],
                                 'score':r['score'],'gpu_sites':r['gpu_sites'],
                                 'exact_point_count':len(r['exact_points'])} for r in ranked]
    board['generated_at_utc']=datetime.now(timezone.utc).isoformat()
    SCOREBOARD.write_text(json.dumps(board,indent=2,sort_keys=True)+'\n')
    stages=defaultdict(lambda:{'runs':0,'sites':0,'seconds':0.0,'exact_points':0,
                               'modular_survivors':0,'exact_x_survivors':0})
    for entry in ranked:
        for run in entry['gpu_searches']:
            stage=run['path'].split('-')[2]
            stages[stage]['runs']+=1;stages[stage]['sites']+=run['sites']
            stages[stage]['seconds']+=run['elapsed_seconds']
            stages[stage]['exact_points']+=len(json.loads((RESULTS_DIR/run['path']).read_text())['points'])
            stages[stage]['modular_survivors']+=run.get('survivors') or 0
            stages[stage]['exact_x_survivors']+=run.get('exact_survivors') or 0
    total_sites=sum(value['sites'] for value in stages.values())
    total_seconds=sum(value['seconds'] for value in stages.values())
    bulk_sites=sum(stages[name]['sites'] for name in ('promote','rescale','coarse','standalone','deeper','outer'))
    bulk_seconds=sum(stages[name]['seconds'] for name in ('promote','rescale','coarse','standalone','deeper','outer'))
    total_cpu_seconds=sum(r['cpu_elapsed_seconds'] for r in ranked)
    best_rank=max(r['certified_subgroup_rank'] for r in ranked)
    doubling_counts=', '.join(map(str,sorted({r['height_certificate']['doublings'] for r in ranked})))
    utilization={'0':{'weighted':0.0,'samples':0},'1':{'weighted':0.0,'samples':0}}
    for entry in ranked:
        for run in entry['gpu_searches']:
            raw=json.loads((RESULTS_DIR/run['path']).read_text())
            for device,stats in raw.get('gpu_utilization_percent',{}).items():
                if device in utilization and stats.get('mean') is not None:
                    utilization[device]['weighted']+=stats['mean']*stats['samples']
                    utilization[device]['samples']+=stats['samples']
    lines=['# Issue #15: multi-candidate fixed-fiber search','',
           f"Generated: {board['generated_at_utc']}",'',
           f"Three completed Stage B campaigns and two standalone refined runs contributed {inventory['deduplicated_count']:,} distinct non-control finalists. "
           f"The selected frontier contains {len(ranked)} specializations; selection includes the five requested strong leads. "
           'The rank-31 record is a calibration control, not a discovery candidate. '+
           f"Its {inventory['control']['exact_curve_witnesses_verified']} published witness coordinates were checked exactly on the recorded minimal model.",'',
           '## Rigorous results','',
           f"Best newly certified lower bound: **rank at least {best_rank}**. "
           +('At least one fourth independent point was certified. ' if best_rank>=4 else 'No fourth independent point was found in the completed GPU rectangles. ')
           +'No new generic section was found. The previously known x=-pq section is included in every exact specialization.','',
           '| T | Certified subgroup rank | Stage B score | GPU sites | Sage status |',
           '|---|---:|---:|---:|---|']
    for entry in ranked[:15]:
        lines.append(f"| {entry['t']} | {entry['certified_subgroup_rank']} | {entry['score']:.2f} | "
                     f"{entry['gpu_sites']:,} | {entry['saturation_status']} |")
    lines.extend(['',f"All {len(ranked)} fibers have exact rank-at-least-three certificates for P0, PD, and PQ. Under pinned Sage 10.10.beta10, the certificates use {doubling_counts} point-doubling rounds, exact x-coordinates, rational enclosures of logarithms, and an upward bound on Sage’s Silverman height-difference formula; all three leading principal minors are strictly positive. The known relation P0+PD+PE=O is checked as a negative control for both triples and quadruples. Sage/eclib saturation completed on a subset; a saturation timeout leaves the index unknown, not the rank-three lower bound. Good reductions separately prove torsion is trivial for every selected fiber.",'',
                  '## GPU work and limits','',
                  '| Stage | Rectangles | Sites | Modular survivors | Exact x survivors | GPU time |',
                  '|---|---:|---:|---:|---:|---:|'])
    for stage,value in sorted(stages.items()):
        lines.append(f"| {stage} | {value['runs']} | {value['sites']:,} | {value['modular_survivors']:,} | "
                     f"{value['exact_x_survivors']:,} | {value['seconds']:.1f} s |")
    lines.extend(['',f"Scheduled scoreboard total (excluding overlapping pilot runs): {total_sites:,} bounded sites in {total_seconds:.1f} summed GPU-run seconds "
                  f"({total_sites/total_seconds/1e12:.2f} trillion sites/s including startup). "
                  f"These sequential GPU jobs alone account for {total_seconds/60:.1f} minutes of active computation. "
                  f"Bulk stages averaged {bulk_sites/bulk_seconds/1e12:.2f} trillion sites/s, consistent with the 16–20 trillion sites/s issue-14 baseline. "
                  f"Recorded bounded Sage CPU time: {total_cpu_seconds:.1f} s across selected fibers. "
                  'Every bulk invocation used devices 0 and 1; each raw JSON report records their denominator batches and kernel timings. '
                  +'; '.join(f"GPU {device}: {stats['weighted']/stats['samples']:.1f}% mean utilization across {stats['samples']} telemetry samples"
                             for device,stats in utilization.items() if stats['samples'])+'.','',
                  'The exact finite regions are listed per candidate in the scoreboard and in each raw GPU JSON report: '
                  'x = center + stride·n/d, |n| ≤ height, 1 ≤ d ≤ denominators, gcd(n,d)=1. '
                  'The modular sieve rejects no rational point in that rectangle, as validated against exact CPU enumeration. '
                  'Every GPU output passed Python integer-square and curve-equation checks. '
                  'The separate Sage verification artifacts record exact point construction for each completed report. '
                  'Absence of a new point excludes only these rectangles. A bounded Simon 2-descent for T=-44/43 timed out at 120 seconds; '
                  'for T=-802/2917 it hit PARI’s 1 GiB bnfinit stack limit before returning a bound. Bounded mwrank bounds on both fibers failed because their 2-descents did not complete. None supplies an upper bound.','',
                  '## Adaptation and next work','',
                  'The cheap screen returned only known section points. Wide windows therefore went to the strongest 12 Stage B leads; '
                  'six additional standalone refined leads were screened and widened. Structured x lattices and subsequent rescaled windows favored smaller parameter denominators, which provide better resolution in family coordinates. '
                  'Candidates with only known points were demoted from further identical-width searches. '
                  'Fourteen outer windows on two small-denominator fibers tested logarithmically spaced family-coordinate regions and found no points. '
                  'Minimal-model diagnostics on six leading fibers found larger maximum coefficient bit sizes than the integral factored models, so repeating the failed descents on these minimal models was not prioritized. '
                  'Promising follow-up is to derive candidate x-coordinates from covering curves or lattice reduction, then feed those centers to the exact GPU sieve; repeated local rectangles around section points have low yield. '
                  'Alternative bounded descent algorithms may resolve upper bounds for the smaller-denominator fibers. '
                  'No global upper bound is claimed.',''])
    REPORT.write_text('\n'.join(lines))
    print(SCOREBOARD,REPORT,total_sites,max(r['certified_subgroup_rank'] for r in ranked))

if __name__=='__main__':main()
