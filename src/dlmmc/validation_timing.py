"""Timing-only exclusions for interrupted validation jobs; never filter draws."""
import hashlib
import json
from pathlib import Path


def timing_exclusions(root):
    """Return timing-only exclusions from process pauses and observed sleep."""
    root = Path(root)
    paths = set()
    provenance = {}
    manifests = [*(root / 'reports').glob('resumed-processes*.json'),
                 *(root / 'reports').glob('timing-interruptions*.json')]
    for manifest in sorted(manifests):
        record = json.loads(manifest.read_text())
        if record['status'] not in {'RESUMED', 'OBSERVED_SYSTEM_SLEEP'}:
            continue
        entries = list(record.get('timing_exclusions', []))
        if record.get('paused_cmdstan_timing_exclusion'):
            entries.append(record['paused_cmdstan_timing_exclusion'])
        for entry in entries:
            path = Path(entry)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Timing exclusions must be project-relative paths')
            paths.add(path.as_posix())
        provenance[str(manifest.relative_to(root))] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return paths, provenance
