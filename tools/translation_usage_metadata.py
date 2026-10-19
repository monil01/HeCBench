"""Read only configured-model context records from the registered usage stream.

No prompt or conversation text is exported. Searches backward within a fixed byte
snapshot, so metadata later than the actual usage event cannot label its counters.
"""
import datetime,hashlib,json
from pathlib import Path

def configured_model_at_usage(path,usage_timestamp,byte_size=None):
    path=Path(path)
    limit=min(path.stat().st_size,byte_size) if byte_size is not None else path.stat().st_size
    target=datetime.datetime.fromisoformat(usage_timestamp.replace('Z','+00:00'))
    end=limit;carry=b'';reviewed=0
    with path.open('rb') as stream:
        while end:
            start=max(0,end-2**21);stream.seek(start);data=stream.read(end-start);reviewed+=len(data)
            lines=(data+carry).splitlines()
            carry=lines.pop(0) if start and lines else b''
            for raw in reversed(lines):
                if b'"turn_context"' not in raw:continue
                try:event=json.loads(raw)
                except ValueError:continue
                if event.get('type')!='turn_context' or not event.get('timestamp'):continue
                when=datetime.datetime.fromisoformat(event['timestamp'].replace('Z','+00:00'))
                if when>target:continue
                payload=event.get('payload') or {}
                if not payload.get('model'):return None
                return {'model':payload['model'],'reasoning_effort':payload.get('effort') or payload.get('reasoning_effort'),'metadata_timestamp':event['timestamp'],'context_event_sha256':hashlib.sha256(raw).hexdigest(),'source_snapshot_bytes':limit,'lookup_bytes_reviewed':reviewed,'lookup_method':'Latest complete same-stream turn_context not later than the usage event, backward bounded-memory scan.','scope':'Observed configured identifier only; exact serving snapshot/seed/temperature unknown. No cross-stream inference and no historical token recovery.'}
            end=start
    return None
