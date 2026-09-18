#!/usr/bin/env python3
"""Bounded, cleanup-safe Sage container for multifiber certification."""
import argparse
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
IMAGE='sagemath/sagemath:10.10.beta10'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('index',type=int);ap.add_argument('--timeout',type=int,default=30)
    ap.add_argument('--report',action='append',default=[])
    ap.add_argument('--verify-only',action='store_true')
    ap.add_argument('--descent',action='store_true')
    ap.add_argument('--pairing',action='store_true')
    ap.add_argument('--height-proof',action='store_true')
    ap.add_argument('--rank-bound',action='store_true')
    ap.add_argument('--minimal-info',action='store_true')
    ap.add_argument('--pari-rank',action='store_true')
    ap.add_argument('--pari-effort',type=int,default=0)
    ap.add_argument('--geometry',action='store_true')
    ap.add_argument('--max-prime',type=int,default=11)
    args=ap.parse_args()
    name='sage-mf-'+uuid.uuid4().hex
    suffix=('-geometry' if args.geometry else '-pari-rank-e'+str(args.pari_effort) if args.pari_rank else '-minimal-info' if args.minimal_info else '-height-proof' if args.height_proof else '-rank-bound' if args.rank_bound else '-descent' if args.descent else '-pairing' if args.pairing else '-verify' if args.verify_only else ('-p'+str(args.max_prime) if args.max_prime!=11 else ''))
    out=HERE/'results'/f'rank31-multifiber-sage-{args.index:02d}{suffix}.json'
    with tempfile.NamedTemporaryFile(dir=out.parent,prefix='sage-mf-',suffix='.json',delete=False) as stream:
        temporary=Path(stream.name)
    script='/work/experiments/elliptic-curves/rank31_multifiber_sage.py'
    paths=['/work/'+str(Path(p).resolve().relative_to(ROOT)) for p in args.report]
    cmd=['docker','run','--rm','--name',name,'--user','0:0','--mount',f'type=bind,src={ROOT},dst=/work',
         '--workdir','/work','--entrypoint','/usr/bin/sage',IMAGE,'-python',script,
         *(['--verify-only'] if args.verify_only else []),
         *(['--descent'] if args.descent else []),
         *(['--pairing'] if args.pairing else []),
         *(['--height-proof'] if args.height_proof else []),
         *(['--rank-bound'] if args.rank_bound else []),
         *(['--minimal-info'] if args.minimal_info else []),
         *(['--pari-rank','--pari-effort',str(args.pari_effort)] if args.pari_rank else []),
         *(['--geometry'] if args.geometry else []),
         '--max-prime',str(args.max_prime),
         str(args.index),'/work/'+str(temporary.relative_to(ROOT)),*paths]
    started=time.monotonic()
    try:
        result=subprocess.run(cmd,timeout=args.timeout,capture_output=True,text=True)
        print('status',result.returncode,'seconds',round(time.monotonic()-started,2),'output',out)
        if result.stdout: print(result.stdout[-2000:])
        if result.stderr: print(result.stderr[-2000:])
        if result.returncode:
            temporary.write_text(json.dumps({'runner_status':'error','returncode':result.returncode,
                'runner_elapsed_seconds':time.monotonic()-started,'stderr_tail':result.stderr[-3000:]},
                indent=2,sort_keys=True)+'\n')
            raise SystemExit(result.returncode)
        completed=json.loads(temporary.read_text())
        completed['runner_elapsed_seconds']=time.monotonic()-started
        completed['runner_status']='completed'
        temporary.write_text(json.dumps(completed,indent=2,sort_keys=True)+'\n')
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
        try: partial=json.loads(temporary.read_text())
        except (OSError,ValueError):partial={'t':None}
        partial.update({'status':'timeout','timeout_seconds':args.timeout,
                        'runner_elapsed_seconds':time.monotonic()-started,'runner_status':'timeout'})
        temporary.write_text(json.dumps(partial,indent=2,sort_keys=True)+'\n')
        print('timeout',args.timeout,'seconds',out)
        raise SystemExit(124)
    finally:
        subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
        if temporary.exists():os.replace(temporary,out)

if __name__=='__main__':main()
