#!/usr/bin/env python3
"""Run the guide's fixed examples with PAW and an unadapted prompt baseline."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path


def load_cases(data, selection):
    cases = []
    if selection in ('all', 'existing'):
        groups = [('urgency', data['original_policy']),
                  ('deadline', data['deadline_policy'] + data['additional_deadline_checks']),
                  ('router', data['routing'])]
        groups += [(group['name'], group['results']) for group in data['gallery']]
        for task, rows in groups:
            for index, row in enumerate(rows, 1):
                cases.append({'set': 'existing', 'id': f'{task}-{index}', 'task': task,
                              'input': row['input'], 'expected': row['expected']})
    if selection in ('all', 'exploratory'):
        for row in data['exploratory_comparison']['cases']:
            cases.append({'set': 'exploratory', **{key: row[key] for key in ('id', 'task', 'input', 'expected')}})
    if selection in ('all', 'everyday'):
        for row in data['everyday_comparison']['cases']:
            cases.append({'set': 'everyday', **{key: row[key] for key in ('id', 'task', 'input', 'expected')}})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('verified-results.json'))
    parser.add_argument('--set', choices=('all', 'existing', 'exploratory', 'everyday'), default='all')
    parser.add_argument('--offline', action='store_true', help='Use previously downloaded model and adapter assets')
    parser.add_argument('--output', type=Path, required=True, help='New JSON output file; existing files are preserved')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new filename.')
    data = json.loads(args.cases.read_text())
    cases = load_cases(data, args.set)
    import programasweights as paw

    report = {'started_at': datetime.now(timezone.utc).isoformat(),
              'interpreter': 'Qwen/Qwen3-0.6B', 'temperature': 0,
              'context_tokens': 2048, 'max_tokens': None,
              'sdk_version': importlib.metadata.version('programasweights'),
              'offline': args.offline, 'results': cases}
    for task in dict.fromkeys(row['task'] for row in cases):
        with paw.function(data['programs'][task]['program_id'], n_ctx=2048, offline=args.offline) as fn:
            for row in cases:
                if row['task'] == task:
                    row['paw_output'] = fn(row['input'], temperature=0).strip()
    with paw.function(None, interpreter='Qwen/Qwen3-0.6B', n_ctx=2048, offline=args.offline) as fn:
        for row in cases:
            task = row['task']
            heading = {'reranking': 'Candidate description', 'navigation': 'Visitor request'}.get(task, 'Email')
            prompt = data['programs'][task]['specification'] + '\n\n' + heading + ':\n' + row['input'] + '\n\nLabel:'
            row['baseline_input'] = prompt
            row['prompt_output'] = fn(prompt, temperature=0).strip()
            row['paw_correct'] = row['paw_output'] == row['expected']
            row['prompt_correct'] = row['prompt_output'] == row['expected']
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report['scores'] = {}
    for name in dict.fromkeys(row['set'] for row in cases):
        rows = [row for row in cases if row['set'] == name]
        report['scores'][name] = {'cases': len(rows),
                                'paw_correct': sum(row['paw_correct'] for row in rows),
                                'prompt_correct': sum(row['prompt_correct'] for row in rows)}
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({'scores': report['scores']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
