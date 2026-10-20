#!/usr/bin/env python3
"""Plot per-kernel precision-specific roofline projections; CPU-only, no benchmark aggregation.

Default CSV is the campaign export. Only stdlib and matplotlib are required.
Unknown stage identities remain visibly pending. Explicit setup/audit/non-ROI
stages are excluded. A figure is not evidence of full-ROI or steady-state coverage.
"""
import argparse
import csv
import hashlib
import io
import json
import gzip
import fcntl
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('cuda', 'cuda_jl', 'ka', 'jacc')
TITLES = {'cuda': 'Native CUDA', 'cuda_jl': 'CUDA.jl', 'ka': 'KA', 'jacc': 'JACC'}
COLORS = {'cuda': '#32688e', 'cuda_jl': '#c66928', 'ka': '#33856a', 'jacc': '#8d5a99'}
MARKERS = {'fp16': '^', 'fp32': 'o', 'fp64': 's', 'tensor_fp16_fp32':'D'}
PENDING = {'', 'unassigned', 'unknown', 'unreviewed', 'pending', 'none', 'null'}
# Stage metadata is authoritative; never infer exclusion from a kernel name.
EXCLUDED_STAGE = re.compile(
    r'(^|[_\s:/-])(non[_\s-]?roi|outside[_\s-]?roi|excluded|audit|validation|'
    r'setup|initializer|initialization)([_\s:/-]|$)', re.I)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def finite_positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def truth(value):
    return str(value).strip().lower() in {'true', '1', 'yes'}


def exclusion(row):
    if row.get('dram_roofline_applicability','').startswith('not_applicable_host_mapped'):
        return 'host_mapped_primary_data_requires_unmeasured_PCIe_tier'
    status=row.get('profiling_status','')
    if status and status!='profile_process_completed':
        return 'profile_process_not_completed'
    for key in ('in_roi', 'roi_member', 'semantic_family_in_primary_ROI'):
        if str(row.get(key, '')).strip().lower() in {'false', '0', 'no'}:
            return 'explicit_nonROI_flag'
    stage = row.get('stage', '').strip()
    proved_member = truth(row.get('semantic_family_in_primary_ROI'))
    if EXCLUDED_STAGE.search(stage) and not proved_member:
        return 'explicit_setup_audit_or_nonROI_stage'
    if row.get('model') not in MODELS:
        return 'unsupported_model'
    if row.get('precision', '').lower() not in MARKERS:
        return 'unsupported_scalar_precision'
    for key in ('duration_seconds', 'dram_bytes', 'executed_scalar_flops',
                'modeled_clock_adjusted_peak_flops_per_second',
                'dram_arithmetic_intensity_flop_per_byte',
                'achieved_flops_per_second', 'ridge_flop_per_byte'):
        if finite_positive(row.get(key)) is None:
            return 'missing_nonpositive_or_nonfinite_' + key
    return None


def select_points(rows):
    points, exclusions = [], []
    for index, row in enumerate(rows, start=2):
        reason = exclusion(row)
        identity = {key: row.get(key) for key in
                    ('benchmark', 'model', 'configuration_id', 'kernel_id', 'process_id','context_id','stream_id',
                     'kernel', 'grid', 'block', 'precision', 'stage',
                     'roi_coverage', 'sample_phase', 'semantic_family_in_primary_ROI', 'evidence_manifest')}
        if reason:
            exclusions.append(dict(csv_line=index, reason=reason, identity=identity))
            continue
        x = float(row['dram_arithmetic_intensity_flop_per_byte']) / float(row['ridge_flop_per_byte'])
        y = float(row['achieved_flops_per_second']) / float(row['modeled_clock_adjusted_peak_flops_per_second'])
        if not (math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0):
            exclusions.append(dict(csv_line=index, reason='invalid_normalized_coordinates', identity=identity))
            continue
        stage = row.get('stage', '').strip().lower()
        pending = (stage in PENDING or 'unreviewed' in stage or 'pending' in stage
                   or ';' in stage or 'ambiguous' in stage or 'candidate' in stage
                   or ('semantic_family_in_primary_ROI' in row
                       and not truth(row['semantic_family_in_primary_ROI'])))
        pending = pending or not truth(row.get('sample_in_primary_timed_invocation'))
        points.append(dict(identity, csv_line=index, normalized_intensity=x,
                           normalized_throughput=y,
                           pending_stage_review=pending,
                           projection_only=truth(row.get('projection_only'))))
    return points, exclusions


def limits(values, minimum_low, minimum_high):
    if not values:
        return minimum_low, minimum_high
    low = min(minimum_low, 10 ** math.floor(math.log10(min(values))))
    high = max(minimum_high, 10 ** math.ceil(math.log10(max(values))))
    return low, high


def draw(points, output_prefix):
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8), sharex=True, sharey=True)
    xlim = limits([p['normalized_intensity'] for p in points], 1e-3, 10)
    ylim = limits([p['normalized_throughput'] for p in points], 1e-5, 2)
    # Always show both roof branches and the ridge. No points are clipped to a roof.
    roof_x = [xlim[0], 1, xlim[1]]
    for ax, model in zip(axes.flat, MODELS):
        selected = [p for p in points if p['model'] == model]
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.plot(roof_x, [min(x, 1) for x in roof_x], color='#343434', lw=1.4, zorder=1)
        ax.axvline(1, color='#888888', lw=.8, ls=':', zorder=0)
        ax.grid(which='major', color='#dddddd', lw=.55, zorder=0)
        for precision in MARKERS:
            for pending in (True, False):
                for projection in (False, True):
                    subset = [p for p in selected if p['precision'] == precision
                              and p['pending_stage_review'] == pending
                              and p['projection_only'] == projection]
                    if not subset:
                        continue
                    ax.scatter([p['normalized_intensity'] for p in subset],
                               [p['normalized_throughput'] for p in subset],
                               s=29, marker=MARKERS[precision],
                               facecolors='none' if projection else COLORS[model],
                               edgecolors=COLORS[model], linewidths=.9,
                               alpha=.28 if pending else .85, zorder=3)
        ax.set_title(TITLES[model], color=COLORS[model], loc='left', fontweight='bold')
        pending_count = sum(p['pending_stage_review'] for p in selected)
        ax.text(.98, .04, f'{len(selected)} precision projection rows\n'
                f'{pending_count} warmup or phase/stage pending', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='#555555')
        if not selected:
            ax.text(.5, .5, 'No compatible precision points', transform=ax.transAxes,
                    ha='center', color='#777777')
    handles = [Line2D([], [], color='#343434', lw=1.4, label='Modeled roof: min(x, 1)')]
    for precision, marker in MARKERS.items():
        if any(p['precision'] == precision for p in points):
            handles.append(Line2D([], [], marker=marker, ls='none', color='#555555',
                                  markersize=5, label=('Tensor FP16→FP32' if precision=='tensor_fp16_fp32' else precision.upper())))
    handles.extend([
        Line2D([], [], marker='o', ls='none', color='#555555', markersize=5,
               label='Solid: sole precision'),
        Line2D([], [], marker='o', ls='none', color='#555555', markerfacecolor='none',
               markersize=5, label='Hollow: mixed projection'),
        Line2D([], [], marker='o', ls='none', color='#555555', alpha=.28,
               markersize=5, label='Faint: warmup or phase/stage pending')])
    fig.suptitle('V100 kernel observations — normalized precision rooflines', y=.98, fontsize=13)
    fig.text(.5, .938, 'Partial first-launch / warmup samples; source stages may be unassigned',
             ha='center', fontsize=9, color='#555555')
    fig.supxlabel('DRAM intensity / precision-compatible ridge  (I / R)', y=.19)
    fig.supylabel('Achieved FLOP/s / precision peak  (P / C)', x=.012)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .085),
               frameon=False, ncol=3, fontsize=8, handletextpad=.5, columnspacing=1.3)
    fig.text(.5, .025, 'Each point is one kernel–precision observation; no benchmark averaging or causal diagnosis.\n'
             'Clock-adjusted modeled ceilings; scalar add/mul/FMA and reviewed tensor work. Other costs omitted.',
             ha='center', va='bottom', fontsize=8, color='#555555')
    fig.subplots_adjust(left=.105, right=.98, top=.88, bottom=.26, hspace=.24, wspace=.13)
    outputs = []
    for suffix in ('.pdf', '.svg'):
        path = output_prefix.with_suffix(suffix)
        fig.savefig(path, metadata={'Creator': 'v100_roofline_plot.py'})
        outputs.append(path)
    plt.close(fig)
    return outputs, {'x_limits': list(xlim), 'y_limits': list(ylim)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'ontology/studies/v100_roofline_537/kernel_roofline_points.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'ontology/studies/v100_roofline_537/figures')
    args = parser.parse_args()
    # One byte snapshot prevents a concurrent summarizer refresh mixing rows/hashes.
    tensor_path=args.input.with_name('kernel_tensor_roofline_points.csv')
    tensor_raw=None
    with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        raw = args.input.read_bytes()
        if tensor_path.is_file():tensor_raw=tensor_path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if tensor_raw:
        for item in csv.DictReader(io.StringIO(tensor_raw.decode('utf-8-sig'))):
            if item['memory_tier']!='dram':continue
            rows.append({**item,'precision':'tensor_fp16_fp32',
                'executed_scalar_flops':item['executed_flops'],
                'dram_arithmetic_intensity_flop_per_byte':item['arithmetic_intensity_flop_per_byte']})
    points, exclusions = select_points(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / 'normalized_kernel_rooflines'
    input_snapshot = prefix.with_suffix('.input.csv.gz')
    input_snapshot.write_bytes(gzip.compress(raw,mtime=0))
    tensor_input=None
    if tensor_raw:
        tensor_snapshot=prefix.with_suffix('.tensor.input.csv.gz')
        tensor_snapshot.write_bytes(gzip.compress(tensor_raw,mtime=0))
        tensor_input={'path':str(tensor_path.resolve()),'sha256':sha256_bytes(tensor_raw),
            'snapshot':str(tensor_snapshot.resolve()),
            'snapshot_compressed_sha256':sha256_bytes(tensor_snapshot.read_bytes()),
            'precision':'FP16inputs_FP32accumulation_HMMA884_sm70',
            'selected_target_SASS_observed':False,
            'basis':'Source/API-mode proof with calibration and prospective exact input bindings; not target SASS observation.'}
    outputs, plot_limits = draw(points, prefix)
    manifests = []
    for value in sorted({p['evidence_manifest'] for p in points if p.get('evidence_manifest')}):
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        manifests.append({'path': str(path), 'sha256': sha256_bytes(path.read_bytes()) if path.is_file() else None})
    metadata = {
        'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'input': {'path': str(args.input.resolve()), 'sha256': sha256_bytes(raw),
                  'snapshot':str(input_snapshot.resolve()),
                  'snapshot_compressed_sha256':sha256_bytes(input_snapshot.read_bytes())},
        'script': {'path': str(Path(__file__).resolve()), 'sha256': sha256_bytes(Path(__file__).read_bytes())},
        'tensor_input':tensor_input,
        'matplotlib_version': matplotlib.__version__,
        'outputs': [{'path': str(p.resolve()), 'sha256': sha256_bytes(p.read_bytes())} for p in outputs],
        'evidence_manifests': manifests,
        'counts': {'input_rows': len(rows), 'plotted_scalar_projection_rows': sum(p['precision']!='tensor_fp16_fp32'for p in points),
                   'plotted_tensor_projection_rows':sum(p['precision']=='tensor_fp16_fp32'for p in points),
                   'excluded_rows': len(exclusions),
                   'by_model': dict(Counter(p['model'] for p in points)),
                   'by_precision': dict(Counter(p['precision'] for p in points)),
                   'pending_stage_review': sum(p['pending_stage_review'] for p in points),
                   'projection_only': sum(p['projection_only'] for p in points),
                   'exclusion_reasons': dict(Counter(e['reason'] for e in exclusions))},
        'formulas': {'x': 'dram_arithmetic_intensity_flop_per_byte / ridge_flop_per_byte',
                     'y': 'achieved_flops_per_second / modeled_clock_adjusted_peak_flops_per_second',
                     'universal_roof': 'min(x, 1)'},
        'plot_thresholds': {**plot_limits, 'pending_alpha': .28, 'reviewed_alpha': .85,
                            'positive_finite_values_only': True, 'clipping_to_roof': False,
                            'excluded_stage_regex': EXCLUDED_STAGE.pattern,
                            'ridge_guide': 1, 'no_near_ridge_classification_added': True},
        'scope': 'Kernel precision projections; phase/coverage metadata distinguishes source-counted retained windows, warmup and unassigned samples. No complete elapsed-ROI, benchmark-average or causal bottleneck claims.',
        'counter_scope': 'Conventional scalar add+mul+2FMA; approved tensor points separately labeled; SFU, integer, address/control and atomic costs omitted. Mixed precision remains separate hollow projections.',
        'ceiling_kind': 'modeled_theoretical_clock_adjusted; not empirically calibrated sustained throughput',
        'plotted_points': points, 'excluded_points': exclusions,
    }
    destination = prefix.with_suffix('.metadata.json')
    destination.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'metadata': str(destination), 'counts': metadata['counts']}))


if __name__ == '__main__':
    main()
