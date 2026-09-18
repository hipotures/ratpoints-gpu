import json
import math
import os
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch
import subprocess
import sys

from rank31_multifiber import coefficients, exact_points, integral_model, inventory, search, torsion_triviality_certificate
import run_multifiber_sage


class MultifiberTests(unittest.TestCase):
    def test_inventory_and_sections(self):
        data=inventory(24)
        rows=data['selected']
        self.assertEqual(len({Fraction(r['t']) for r in rows}),30)
        self.assertTrue({'-47/80','-191989/4040887','532929/2579219','-47/500','-353682/460195'}
                        <= {r['t'] for r in rows})
        self.assertEqual(data['control']['exact_curve_witnesses_verified'],31)
        self.assertEqual(json.dumps(inventory(24),sort_keys=True),json.dumps(inventory(24),sort_keys=True))
        for row in rows:
            model=integral_model(row['t'])
            self.assertEqual(len(model['sections']),4)
            self.assertEqual(model['model_sha256'],integral_model(row['t'])['model_sha256'])
            self.assertTrue(torsion_triviality_certificate(model)['trivial_torsion'])

    def test_gpu_vs_exact_cpu(self):
        structured=int(integral_model('-47/80')['sections'][1]['x'])
        for t,stride in [('-47/80',1),('-47/80',6400**4),('-47/80',structured),('-47/500',1)]:
            model=integral_model(t)
            gpu=search(model,'P0',200,8,timeout=30,stride=stride)
            self.assertEqual(gpu['returncode'],0)
            c=int(gpu['center']);coeffs=coefficients(model,c,stride)
            expected=set()
            for d in range(1,9):
                for n in range(-200,201):
                    if math.gcd(n,d)!=1:continue
                    z=coeffs[3]*n**3*d+coeffs[2]*n*n*d*d+coeffs[1]*n*d**3+coeffs[0]*d**4
                    if z<0:continue
                    root=math.isqrt(z)
                    if root*root==z:
                        for sign in (-1,1):
                            lines=f'{n} {sign*root} {d}'
                            for p in exact_points(lines,model,c,coeffs,stride):
                                expected.add((p['x'],p['y']))
            actual={(p['x'],p['y']) for p in gpu['points']}
            self.assertEqual(actual,expected)

    def test_point_deduplication(self):
        model=integral_model('-47/80')
        coeffs=coefficients(model,0)
        self.assertEqual(len(exact_points('0 1817195034915611951338416978 1\n'
                                          '0 1817195034915611951338416978 1',
                                          model,0,coeffs)),1)
        with self.assertRaises(ArithmeticError):
            exact_points('0 1 1',model,0,coeffs)

    @unittest.skipUnless(os.environ.get('RANK31_TEST_SAGE')=='1','opt-in Docker integration')
    def test_sage_subgroup_growth(self):
        script=Path(__file__).with_name('run_multifiber_sage.py')
        report=script.with_name('results')/'rank31-multifiber-screen-00-P0.json'
        result=subprocess.run([sys.executable,str(script),'0','--timeout','30','--report',str(report)],
                              capture_output=True,text=True,timeout=40)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads((script.with_name('results')/'rank31-multifiber-sage-00.json').read_text())
        self.assertEqual(data['certified_subgroup_rank'],3)
        self.assertTrue(any(step['status']=='known section or inverse' for step in data['steps']))
        proof_run=subprocess.run([sys.executable,str(script),'1','--height-proof','--timeout','30'],
                                 capture_output=True,text=True,timeout=40)
        self.assertEqual(proof_run.returncode,0,proof_run.stderr)
        proof=json.loads((script.with_name('results')/'rank31-multifiber-sage-01-height-proof.json').read_text())
        self.assertEqual(proof['proof']['certified_rank'],3)
        self.assertEqual(proof['dependent_control']['relation'],'P0+PD+PE=O')

    def test_sage_timeout_removes_container(self):
        calls=[]
        def fake_run(command,**kwargs):
            calls.append(command)
            if command[:2]==['docker','run']:
                raise subprocess.TimeoutExpired(command,1)
            return subprocess.CompletedProcess(command,0)
        with patch.object(sys,'argv',['run_multifiber_sage.py','0','--timeout','1']), \
             patch.object(run_multifiber_sage.subprocess,'run',side_effect=fake_run), \
             patch.object(run_multifiber_sage.os,'replace',side_effect=lambda src,dst:Path(src).unlink()):
            with self.assertRaises(SystemExit) as caught:run_multifiber_sage.main()
        self.assertEqual(caught.exception.code,124)
        self.assertEqual(calls[-1][:3],['docker','rm','-f'])
        self.assertEqual(calls[-1][-1],calls[0][calls[0].index('--name')+1])


if __name__=='__main__':unittest.main()
