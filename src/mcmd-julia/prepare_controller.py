#!/usr/bin/env python3
"""Add only an out-of-ROI full-state observer to the original CPU controller."""
from pathlib import Path
import re,json,hashlib
B=Path(__file__).resolve().parent;p=B/'reference_history/main.cpp';s=p.read_text();s=re.sub(r'#include "([^"]+)"',r'#include "reference_history/\1"',s);s=s.replace('int main(int argc, char **argv) {','void mcmd_observe_system(System &system);\nint mcmd_source_main(int argc, char **argv) {');assert s.count('      count_md_steps++;')==1;s=s.replace('      count_md_steps++;','      mcmd_observe_system(system); // study observation outside reported ROI\n      count_md_steps++;');(B/'configured_main.cpp').write_text(s)
(B/'controller_provenance.json').write_text(json.dumps(dict(original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),configured_sha256=hashlib.sha256(s.encode()).hexdigest(),changes=['Relativeincludeadapters','Renameentrypointforinprocesswarmup+fivefreshSystemruns','OnefullstateobservationaftereachcompleteMDstep; observationtimeexcludedfromROI']),indent=2)+'\n')
