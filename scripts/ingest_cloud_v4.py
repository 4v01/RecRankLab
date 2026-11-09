"""Download completed parameter-only workflow artifacts and verify before adoption."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
import torch
from recranklab.data.prepare import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    artifacts=json.loads(subprocess.check_output(['gh','api',f'repos/4v01/RecRankLab/actions/runs/{a.run_id}/artifacts?per_page=100'],text=True))['artifacts']
    ledger_path=Path('/workspace/work/v4_transfer_ledger.json')
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    completed={x['artifact_id'] for x in ledger};new=0
    for artifact in artifacts:
        if artifact['id'] in completed:continue
        assert artifact['name'].startswith('v4-ml10m-') and not artifact['expired']
        staging=Path('/workspace/work/v4_artifact_staging')/artifact['name'];staging.mkdir(parents=True,exist_ok=True)
        subprocess.run(['gh','run','download',a.run_id,'-n',artifact['name'],'-D',str(staging)],check=True)
        files=list(staging.rglob('result.json'))
        if not files:print('No final result in partial artifact; retain for inspection:',artifact['name']);continue
        assert len(files)==1;root=files[0].parent;r=json.loads(files[0].read_text())
        assert r['dataset']=='ml-10m' and r['model'] in ('sgdmf','mf_dcn','sparse_ease','history_tower')
        assert r['experiment']==r['model']+'_v4' and r['seed'] in (17,29,43)
        assert r['environment']['git_commit']=='a2b7dd8c05849a23afd34a0a90c71243cf06789f' and not r['environment']['git_dirty']
        assert r['environment']['execution']['run_id']==a.run_id
        assert digest(root/'model.pt')==r['checkpoint_sha256']
        for filename,key in [('per_user.csv','per_user_sha256'),('per_user_rating.csv','per_user_rating_sha256')]:
            if key in r:assert digest(root/filename)==r[key]
        def check_context(path,partition):
            payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            assert not {'training','compact_context','history_ids'}&set(payload)
            assert payload['external_context']=={'dataset':'ml-10m','fingerprint':r['data']['fingerprint'],'partition':partition}
        check_context(root/'model.pt','trainval')
        metadata_files=list(staging.rglob('metadata.json'));assert len(metadata_files)==1
        cache=metadata_files[0].parent;metadata=json.loads(metadata_files[0].read_text())
        assert metadata['dependency_run_signature']==r['signature'] and metadata['config']==r['selected_config']
        assert digest(cache/'model.pt')==metadata['checkpoint_sha256'];check_context(cache/'model.pt','train')
        targets=[(root,Path('outputs/v4')/'ml-10m'/r['experiment']/str(r['seed'])),(cache,Path('outputs/v4_validation_components')/'ml-10m'/r['experiment']/str(r['seed']))]
        for src,dst in targets:
            dst.mkdir(parents=True,exist_ok=True)
            if (dst/'result.json').exists():
                assert digest(dst/'result.json')==digest(src/'result.json'),'Refusing a different completed final result'
            for item in sorted(src.iterdir(),key=lambda x:x.name in ('result.json','metadata.json')):
                if not item.is_file():continue
                target=dst/item.name
                if target.exists() and digest(target)!=digest(item):
                    assert not (dst/'result.json').exists(),'Refusing completed overwrite'
                    archive=Path('outputs/v4_interrupted_validation')/'ml-10m'/r['experiment']/str(r['seed']);archive.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(target,archive/(digest(target)[:12]+'_'+target.name))
                item.replace(target)
        entry={'run_id':a.run_id,'artifact_id':artifact['id'],'artifact_name':artifact['name'],'artifact_digest':artifact.get('digest'),'result_signature':r['signature'],'source_commit':r['environment']['git_commit'],'checkpoint_sha256':r['checkpoint_sha256'],'validation_checkpoint_sha256':metadata['checkpoint_sha256']}
        ledger.append(entry);ledger_path.write_text(json.dumps(ledger,indent=2));shutil.rmtree(staging);new+=1
        print('Verified and adopted:',artifact['name'],flush=True)
    print(f'{new} new artifacts; {len(ledger)} verified total',flush=True)

if __name__=='__main__':main()
