"""Finish calculations beside hosted artifacts, then publish only review evidence."""
import argparse,gc,json,os,shutil,subprocess,sys,traceback
from pathlib import Path
from recranklab.data.prepare import digest
from recranklab.experiments.hybrid import run_hybrid
from recranklab.utils.repro import seed_all
import yaml


def command(*args,**kwargs):return subprocess.run(list(args),check=True,**kwargs)

def hydrate():
    root=Path('reports/v4');m=json.loads((root/'experiment_manifest.json').read_text())
    for entry in m['runs'].values():
        r=json.loads((root/entry['evidence']).read_text());out=Path('outputs/v4')/r['dataset']/r['experiment']/str(r['seed']);out.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/entry['evidence'],out/'result.json')
        for name,file in entry['files'].items():
            assert digest(root/file['path'])==file['sha256'];shutil.copyfile(root/file['path'],out/name)


def restore_artifacts(run_id):
    stage=Path(os.environ['RUNNER_TEMP'])/'v4_download';stage.mkdir(exist_ok=True)
    command('gh','run','download',run_id,'-D',str(stage))
    found=0
    for path in stage.rglob('result.json'):
        r=json.loads(path.read_text())
        if r.get('dataset')!='ml-10m':continue
        assert r['environment']['execution']['run_id']==run_id and r['environment']['git_commit']=='a2b7dd8c05849a23afd34a0a90c71243cf06789f'
        assert r['config']['checkpoint_data_reference'] and digest(path.with_name('model.pt'))==r['checkpoint_sha256']
        target=Path('outputs/v4')/'ml-10m'/r['experiment']/str(r['seed']);target.mkdir(parents=True,exist_ok=True)
        if (target/'result.json').exists():assert digest(target/'result.json')==digest(path),'Refuse changed completed final record'
        for file in path.parent.iterdir():
            if file.is_file():shutil.copyfile(file,target/file.name)
        found+=1
    assert found==9,found
    for path in stage.rglob('metadata.json'):
        metadata=json.loads(path.read_text());assert digest(path.with_name('model.pt'))==metadata['checkpoint_sha256']
        cfg=metadata['config'];target=Path('outputs/v4_validation_components')/cfg['dataset']/cfg['experiment']/str(cfg['seed']);target.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target/'metadata.json');shutil.copyfile(path.with_name('model.pt'),target/'model.pt')


def restore_reference(branch):
    command('git','fetch','origin',branch)
    base=Path('outputs/v4/ml-10m/sgdmf_v4/17');cache=Path('outputs/v4_validation_components/ml-10m/sgdmf_v4/17');cache.mkdir(parents=True,exist_ok=True)
    for name,target in [('final_model.pt',base/'model.pt'),('validation_model.pt',cache/'model.pt'),('metadata.json',cache/'metadata.json')]:
        with target.open('wb') as output:command('git','show','FETCH_HEAD:reference/sgdmf_v4_seed17/'+name,stdout=output)
    r=json.loads((base/'result.json').read_text());metadata=json.loads((cache/'metadata.json').read_text())
    assert digest(base/'model.pt')==r['checkpoint_sha256'] and digest(cache/'model.pt')==metadata['checkpoint_sha256']


def publish():
    command('git','config','user.name','github-actions[bot]')
    command('git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com')
    branch=f"evidence/v4-finish-{os.environ['GITHUB_RUN_ID']}-a{os.environ.get('GITHUB_RUN_ATTEMPT','1')}"
    command('git','switch','-c',branch);command('git','add','reports/v4')
    command('git','commit','-m','Publish all available V4 raw evidence, context stress and independent audit')
    command('git','push','origin','HEAD:'+branch);print('Published evidence branch:',branch,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training-run',required=True);parser.add_argument('--reference-branch',required=True);a=parser.parse_args()
    failure=None;seed_all(17,2)
    try:
        hydrate();restore_artifacts(a.training_run);restore_reference(a.reference_branch)
        study=yaml.safe_load(Path('configs/v4/study.yaml').read_text())
        # Completed logical signatures are never rerun to fix reporting/transport.
        for name,spec in study['hybrids'].items():
            for seed in study['seeds']:
                out=Path('outputs/v4/ml-10m')/name/str(seed)/'result.json'
                if out.exists():print('Preserve existing final hybrid:',out,flush=True);continue
                run_hybrid('ml-10m',name,spec,seed,study)
        command(sys.executable,'scripts/stress_v4.py','--dataset','ml-10m')
        command(sys.executable,'scripts/report_v4.py')
        command(sys.executable,'scripts/handoff_v4.py')
        command(sys.executable,'scripts/seal_v4.py')
        with Path('reports/v4/audit_output.txt').open('w') as output:
            command(sys.executable,'scripts/audit_v4.py',stdout=output,stderr=subprocess.STDOUT)
        command(sys.executable,'scripts/seal_v4.py');command(sys.executable,'scripts/audit_v4.py')
        command(sys.executable,'scripts/audit_results.py');command(sys.executable,'scripts/audit_v3.py')
        command(sys.executable,'reports/external_benchmark/verify.py')
    except Exception as error:
        failure=error;traceback.print_exc()
        # Preserve every produced final record even when postprocessing fails.
        try:
            import report_v4
            from recranklab.experiments.study_v4 import load_study_v4
            report_v4.collect(load_study_v4())
        except Exception:traceback.print_exc()
        Path('reports/v4/execution_failure.json').write_text(json.dumps({'error':str(error),'scope':'available raw final evidence preserved; no performance-based discard or new test selection'},indent=2))
        if Path('reports/v4/experiment_manifest.json').exists():command(sys.executable,'scripts/seal_v4.py')
    finally:
        publish()
    if failure:raise SystemExit(1)

if __name__=='__main__':main()
