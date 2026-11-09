"""Refresh report-file hashes after adding audit/handoff text; never alter runs."""
import json
from datetime import datetime,timezone
from pathlib import Path
from recranklab.data.prepare import digest
from recranklab.utils.repro import provenance
root=Path('reports/v4');path=root/'experiment_manifest.json';m=json.loads(path.read_text())
m['artifacts']={str(p.relative_to(root)):{'sha256':digest(p)} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='experiment_manifest.json'}
m['sealed_at_utc']=datetime.now(timezone.utc).isoformat();m['sealing_environment']=provenance(require_clean=False)
path.write_text(json.dumps(m,indent=2))
