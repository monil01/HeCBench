from pathlib import Path
import json
p=Path(__file__).resolve().parent;m=json.loads((p/'data/fixture_manifest.json').read_text());out=['#pragma once','#include <vector>','#include <string>','struct Case {int n,nr,njk;std::vector<int> jk;std::vector<std::string> seq;};','static const std::vector<Case> CASES={']
for c in m['cases']:out.append('{'+','.join(map(str,[c['n'],c['nr'],c['njk']]))+',{'+','.join(map(str,c['jk_sizes']))+'},{'+','.join('"'+s+'"'for s in c['source_H2D_frame_sequence'])+'}},')
out+=['};',f'static constexpr size_t PREFIX_VALUES={m["prefix_count_total"]};','static constexpr size_t TOTAL_VALUES=PREFIX_VALUES+8794;'];(p/'study_cases.h').write_text('\n'.join(out)+'\n')
