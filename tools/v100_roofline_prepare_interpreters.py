#!/usr/bin/env python3
"""Bind explicit existing interpreters in fresh ANS output-overlay plans.

No process execution, kernel changes, source edits or input/output redirection.
Use after v100_roofline_prepare_overlay.py in a NEW attempt root. Existing failed
attempts are immutable. Runtime availability is recorded by the later collector.
"""
import argparse
import hashlib
import json
from pathlib import Path

PYTHON = Path('/home/imo/noback/conda/bin/python3.9')
JULIA = Path('/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan-root',type=Path,required=True)
    a=ap.parse_args();base=a.plan_root.resolve()
    if not base.is_relative_to('/tmp') or base == Path('/tmp'):
        ap.error('Fresh staged attempt tree strictly under /tmp required')
    for path in sorted(base.glob('ans/*/plan.json')):
        if (path.parent/'profile_provenance.json').exists():
            raise ValueError('Refuse mutation of a started attempt')
        before=path.read_bytes();plan=json.loads(before)
        if plan.get('explicit_interpreters'):
            raise ValueError('Already transformed')
        command=plan['profile_target_command'];original=list(command)
        if plan['benchmark']!='ans' or command[:1]!=['python3']:
            raise ValueError('Unexpected original interpreter command')
        command[0]=str(PYTHON)
        if plan['model']!='cuda':
            if command[2]!='julia':raise ValueError('Unexpected ANS child launcher')
            command[2]=str(JULIA)
        changed=[p for p in [PYTHON,JULIA] if str(p) in command]
        plan['execution_bindings'].update({str(p):sha(p) for p in changed})
        plan['execution_bindings'][str(Path(__file__).resolve())]=sha(Path(__file__))
        plan['explicit_interpreters']={
            'previous_plan_sha256':hashlib.sha256(before).hexdigest(),
            'before':original,'after':command,
            'bindings':{str(p):sha(p) for p in changed},
            'reason':'Existing NumPy-capable Python for the unchanged independent CPU audit; direct installed Julia1.12.6 avoids nested julialauncher under NCU. This is a profiling environment intervention, not proof that the launcher caused prior failures.',
            'original_failed_attempts_retained':True,
            'kernel_math_inputs_outputs_and_selected_contract_unchanged':True}
        (path.parent/'plan_before_interpreters.json').write_bytes(before)
        (path.parent/'interpreter_plan_diff.json').write_text(json.dumps(plan['explicit_interpreters'],indent=2)+'\n')
        path.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
        print(plan['benchmark'],plan['model'],'explicit interpreter plan prepared; not executed')


if __name__=='__main__':main()
