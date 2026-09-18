#!/usr/bin/env python3
"""Adaptive, checkpointed two-GPU fixed-fiber campaign."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

from rank31_multifiber import RESULTS_DIR, centers, search

INVENTORY=RESULTS_DIR/'rank31-multifiber-inventory.json'
SCOREBOARD=RESULTS_DIR/'rank31-multifiber-scoreboard.json'

def write_board(board):
    SCOREBOARD.write_text(json.dumps(board,indent=2,sort_keys=True)+'\n')

def candidate_entry(row):
    return {'t':row['t'],'score':row['score'],'prime_bound':row['prime_bound'],
            'stage_a_score':row['stage_a_score'],'stage_a_prime_bound':row['stage_a_prime_bound'],
            'source_campaigns':row['source_campaigns'],'source_seeds':row['source_seeds'],
            'model_sha256':row['model']['model_sha256'],'model_ainvariants':row['model']['ainvariants'],
            'sections':row['model']['sections'],'certified_subgroup_rank':None,
            'independent_generators':[],'saturation_status':'pending Sage certification',
            'gpu_searches':[],'gpu_sites':0,'gpu_elapsed_seconds':0,'exact_points':[],
            'status_reason':row['selection_reason']}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=('screen','promote','rescale','coarse','structured','deeper','standalone'),required=True)
    ap.add_argument('--limit',type=int,default=24)
    args=ap.parse_args()
    inventory=json.loads(INVENTORY.read_text())
    if SCOREBOARD.exists():board=json.loads(SCOREBOARD.read_text())
    else:board={'inventory_count':inventory['deduplicated_count'],'control':inventory['control'],
                'candidate_rows':{}}
    board['inventory_count']=inventory['deduplicated_count']
    board['control']=inventory['control']
    for row in inventory['selected']:
        entry=board['candidate_rows'].setdefault(row['t'],candidate_entry(row))
        entry.update(score=row['score'],prime_bound=row['prime_bound'],
                     stage_a_score=row['stage_a_score'],stage_a_prime_bound=row['stage_a_prime_bound'],
                     source_campaigns=row['source_campaigns'],source_seeds=row['source_seeds'])
    selected=inventory['selected'][:args.limit]
    if args.stage=='screen':
        jobs=[(i,label,100_000_000,4096,1,None) for i,row in enumerate(selected) for label in ('P0','PD','PE','PQ')]
    elif args.stage=='promote':
        # Prioritize observed non-section points, then heuristic score and model size.
        ranked=sorted(enumerate(selected),key=lambda pair:(
            -sum(1 for p in board['candidate_rows'][pair[1]['t']]['exact_points']
                 if p['x'] not in {s['x'] for s in pair[1]['model']['sections']}),
            -pair[1]['score'],pair[0]))
        promote=[i for i,_ in ranked[:min(args.limit,12)]]
        jobs=[(i,label,2_000_000_000,65536,1,None) for i in promote
              for label in ('P0','PD','PE','PQ','R') if label in centers(selected[i]['model'])]
    elif args.stage=='structured':
        promote=sorted(range(len(selected)),key=lambda i:(Fraction(selected[i]['t']).denominator,i))[:12]
        jobs=[]
        for i in promote:
            section_x={p['label']:int(p['x']) for p in selected[i]['model']['sections']}
            for label in ('P0','PQ'):
                for source in ('PD','PE'):
                    stride=abs(section_x[source])
                    if stride:jobs.append((i,label,100_000_000,4096,stride,source))
    elif args.stage=='deeper':
        promote=[i for i,row in enumerate(selected) if row['t'] in ('-44/43','-47/500')]
        jobs=[(i,label,2_000_000_000,262144,int(selected[i]['model']['scale'])**2,None)
              for i in promote for label in ('P0','PD','PE','PQ','R')
              if label in centers(selected[i]['model'])]
    elif args.stage=='standalone':
        jobs=[(i,label,2_000_000_000,65536,1,None)
              for i in range(24,len(selected)) for label in ('P0','PD','PE','PQ','R')
              if label in centers(selected[i]['model'])]
    else:
        # Small parameter denominators yield the most useful family-coordinate
        # resolution. Keep the strongest two large-denominator leads as controls.
        ranked=sorted(range(len(selected)),key=lambda i:(Fraction(selected[i]['t']).denominator,i))
        promote=ranked[:min(args.limit,10)]
        for i in (1,2):
            if i<len(selected) and i not in promote:promote.append(i)
        if args.stage=='coarse':promote=promote[:8]
        jobs=[(i,label,2_000_000_000,65536,
               int(selected[i]['model']['scale'])**(1 if args.stage=='rescale' else 2),None)
              for i in promote for label in ('P0','PD','PE','PQ','R')
              if label in centers(selected[i]['model'])]
    for number,(i,label,h,d,stride,stride_label) in enumerate(jobs,1):
        row=selected[i]; entry=board['candidate_rows'][row['t']]
        key=(label,h,d,str(stride))
        if any((run['center_label'],run['height'],run['denominators'],str(run.get('stride',1)))==key
               for run in entry['gpu_searches']):continue
        try: result=search(row['model'],label,h,d,timeout=180,stride=stride)
        except Exception as exc:
            entry['status_reason']=f'{args.stage} {label} failed: {type(exc).__name__}: {exc}'
            write_board(board);raise
        path=RESULTS_DIR/f'rank31-multifiber-{args.stage}-{i:02d}-{label}'
        if stride_label:path=Path(str(path)+f'-{stride_label}')
        path=Path(str(path)+'.json')
        path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        entry['gpu_searches'].append({'path':path.name,'center_label':label,'center':result['center'],
                                     'stride':str(stride),
                                     'stride_source':stride_label,
                                     'height':h,'denominators':d,'sites':result['sites'],
                                     'survivors':result['metrics'].get('survivors'),
                                     'exact_survivors':result['metrics'].get('exact_survivors'),
                                     'elapsed_seconds':result['elapsed_seconds'],
                                     'returncode':result['returncode'],'timed_out':result['timed_out']})
        entry['gpu_sites']+=result['sites'];entry['gpu_elapsed_seconds']+=result['elapsed_seconds']
        seen={(p['x'],p['y']) for p in entry['exact_points']}
        for p in result['points']:
            if (p['x'],p['y']) not in seen:
                entry['exact_points'].append({'x':p['x'],'y':p['y']});seen.add((p['x'],p['y']))
        novel=[p for p in entry['exact_points'] if p['x'] not in {s['x'] for s in row['model']['sections']}]
        entry['status_reason']=('promoted: non-section exact point; needs subgroup test' if novel else
            'screened: only section x-coordinates found; wider search scheduled' if args.stage=='screen' else
            f'{args.stage} rectangle completed; only section x-coordinates found')
        write_board(board)
        if novel:
            reports=[str(RESULTS_DIR/item['path']) for item in entry['gpu_searches']]
            base=[sys.executable,str(Path(__file__).with_name('run_multifiber_sage.py')),
                  str(i),'--timeout','90']
            for report in reports:base.extend(['--report',report])
            subprocess.run(base+['--height-proof'],check=False)
            proof_path=RESULTS_DIR/f'rank31-multifiber-sage-{i:02d}-height-proof.json'
            if proof_path.exists():
                certificate=json.loads(proof_path.read_text())
                growth=[item for item in certificate.get('gpu_rank_growth_tests',[])
                        if item.get('proof',{}).get('certified_rank')==4]
                if growth:
                    entry['certified_subgroup_rank']=4
                    entry['status_reason']='new point exactly verified and rank-four height certificate completed; saturating'
                    write_board(board)
                    subprocess.run(base,check=False)
                    entry['status_reason']='new independent point certified; bounded saturation attempted'
                    write_board(board)
        print(f'{args.stage} {number}/{len(jobs)} T={row["t"]} {label} stride={stride} '
              f'{result["sites"]/1e12:.2f}T sites {result["elapsed_seconds"]:.1f}s '
              f'{len(result["points"])} points {len(novel)} novel',flush=True)
        if result['returncode']!=0:raise RuntimeError(f'GPU search failed: {path}')

if __name__=='__main__':main()
