"""Verify review artifact/source hashes and recompute metric counterexamples.

No external model code is executed. This checks provenance bookkeeping, not the
truth or reproducibility of an upstream author's reported performance.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostics():
    return {
        'kind': 'synthetic arithmetic examples, not measured model performance',
        'ndcg': {'truth_positives': 10, 'k': 10, 'hits': [1]+[0]*9,
                 'truth_based_idcg': sum(1/math.log2(j+2) for j in range(10)),
                 'recranklab_definition': 1/sum(1/math.log2(j+2) for j in range(10)),
                 'returned_hits_only_idcg_definition': 1.0},
        'recall': {'truth_positives': 40, 'k': 20, 'hits': 10,
                   'recranklab_definition': 10/40,
                   'min_k_truth_denominator_definition': 10/min(20,40)}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--refresh',action='store_true')
    p.add_argument('--source-root',type=Path,help='Optional local bench_sources directory; verifies actual downloaded bytes')
    a=p.parse_args();root=Path(__file__).resolve().parent
    manifest_path=root/'source_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if a.refresh:
        (root/'metric_diagnostic.json').write_text(json.dumps(diagnostics(),indent=2)+'\n')
        manifest['artifacts']={str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*'))
                               if f.is_file() and f.name!='source_manifest.json' and '__pycache__' not in f.parts}
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    for name,expected in manifest['artifacts'].items():
        assert digest(root/name)==expected,name
    assert json.loads((root/'metric_diagnostic.json').read_text())==diagnostics()
    rows=list(csv.DictReader((root/'reported_results.csv').open()))
    assert len({r['id'] for r in rows})==len(rows)
    assert {r['source_id'] for r in rows}==set(manifest['sources'])
    for row in rows:
        assert math.isfinite(float(row['value']))
        assert row['locally_reproduced']=='False' and row['directly_comparable_with_v2']=='False'
        if row['metric'] in ('NDCG','Recall'):assert 0<=float(row['value'])<=1
    checked=0
    if a.source_root:
        for entry in manifest['sources'].values():
            for file in entry['files']:
                assert digest(a.source_root/entry['local_folder']/'extracted'/file['path'])==file['sha256'],file['path']
                checked+=1
    print(f"PASS: {len(manifest['sources'])} source snapshots, {len(rows)} externally reported values, artifact hashes and metric diagnostics; {checked} downloaded source files checked.")


if __name__=='__main__':main()
