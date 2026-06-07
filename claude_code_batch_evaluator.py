"""
Claude Code-Generation Batch Evaluator
========================================
4-step workflow using Anthropic Message Batches API.
The server only generates code; execution happens locally.

  submit  → ask Claude to write Python code (single-turn, no tool)
  status  → poll Anthropic Batches API
  fetch   → download generated code → results/codes_{model}.json
  run     → execute code locally, match stdout to closest MCQ option, score

Usage:
  python claude_code_batch_evaluator.py submit --model claude-sonnet-4-6
  python claude_code_batch_evaluator.py status
  python claude_code_batch_evaluator.py fetch --model claude-sonnet-4-6
  python claude_code_batch_evaluator.py run   --model claude-sonnet-4-6
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from code_runner import execute_python

QUESTIONS_FILE = 'questions/all_questions_with_E.json'
ANSWERS_FILE   = 'answers/all_answers_with_E.json'
STATE_FILE     = '.claude_batch_state.json'

SYSTEM_BATCH_CODE = (
    'You are an applied mathematics expert. '
    'Solve the following problem by writing Python code. '
    'Available libraries: numpy (import as np), scipy, sympy. '
    'Your code must print() the final numerical answer as the last output. '
    'Write ONLY a Python code block:\n'
    '```python\n'
    '# your solution here\n'
    'print(answer)\n'
    '```\n'
    'No explanation outside the code block.'
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def format_question(q):
    opts = '\n'.join(f"{k}. {v}" for k, v in sorted(q['options'].items()))
    return f"{q['question']}\n\n{opts}"


def make_codes_path(model):
    safe = model.replace('/', '_').replace('.', '-')
    Path('results').mkdir(exist_ok=True)
    return f"results/codes_{safe}.json"


def make_output_path(model):
    safe = model.replace('/', '_').replace('.', '-')
    return f"results/results_{safe}_code_generation_batch.json"


def client():
    return anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY', ''))


def extract_code(text):
    m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

def parse_stdout_value(stdout):
    if not stdout:
        return None
    lines = [l.strip() for l in stdout.strip().split('\n') if l.strip()]
    for line in reversed(lines):
        try:
            return float(line)
        except ValueError:
            m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if m:
                try:
                    return float(m.group())
                except ValueError:
                    pass
    return None


def pick_closest_option(stdout, options):
    value = parse_stdout_value(stdout)
    if value is None:
        return 'E'
    numeric = {k: float(v) for k, v in options.items()
               if k != 'E' and isinstance(v, (int, float))}
    if not numeric:
        return 'E'
    return min(numeric, key=lambda k: abs(numeric[k] - value))


# ── Step 1: submit ────────────────────────────────────────────────────────────

def submit(model, questions_file, max_tokens, limit, start=0, force=False):
    if Path(STATE_FILE).exists() and not force:
        try:
            prev  = load_json(STATE_FILE)
            c     = client()
            batch = c.beta.messages.batches.retrieve(prev['batch_id'])
            if batch.processing_status == 'in_progress':
                print(f"\n  A batch is still active (id: {batch.id}).")
                print(f"  Check with:   python {sys.argv[0]} status")
                print(f"  To force a new submission: --force")
                return
        except Exception:
            pass

    questions = load_json(questions_file)[start:]
    if limit:
        questions = questions[:limit]
    if not questions:
        print(f"  No questions to send.")
        return

    print(f"  Sending questions {start + 1}–{start + len(questions)} ({len(questions)} total)")

    c = client()
    requests = [
        {
            'custom_id': q['id'],
            'params': {
                'model':      model,
                'max_tokens': max_tokens,
                'system':     SYSTEM_BATCH_CODE,
                'messages':   [{'role': 'user', 'content': format_question(q)}],
            },
        }
        for q in questions
    ]

    batch = c.beta.messages.batches.create(requests=requests)
    print(f"  Created batch id: {batch.id}  status: {batch.processing_status}")

    Path(STATE_FILE).write_text(json.dumps({
        'batch_id':      batch.id,
        'model':         model,
        'num_questions': len(questions),
        'submitted_ids': [q['id'] for q in questions],
    }, indent=2))
    print(f"\n  State saved → {STATE_FILE}")
    print(f"  Run 'python {sys.argv[0]} status' to check progress.")


# ── Step 2: status ────────────────────────────────────────────────────────────

def status():
    state = load_json(STATE_FILE)
    c     = client()
    batch = c.beta.messages.batches.retrieve(state['batch_id'])

    print(f"\n  Batch:    {batch.id}")
    print(f"  Status:   {batch.processing_status}")
    rc = batch.request_counts
    if rc:
        print(f"  Progress: {rc.processing} processing  "
              f"{rc.succeeded} succeeded  {rc.errored} errored")

    if batch.processing_status == 'ended':
        print(f"\n  Done! Run 'python {sys.argv[0]} fetch --model {state['model']}' to download code.")
    else:
        print(f"\n  Still processing — check again later.")


# ── Step 3: fetch ─────────────────────────────────────────────────────────────

def fetch(model, questions_file):
    state = load_json(STATE_FILE)
    c     = client()
    batch = c.beta.messages.batches.retrieve(state['batch_id'])

    if batch.processing_status != 'ended':
        print(f"  Batch not ready (status: {batch.processing_status}). Try 'status' first.")
        return

    questions = load_json(questions_file)
    q_by_id   = {q['id']: q for q in questions}
    submitted = set(state.get('submitted_ids', []))

    raw_by_id   = {}
    usage_by_id = {}
    for result in c.beta.messages.batches.results(state['batch_id']):
        if result.result.type == 'succeeded':
            raw_by_id[result.custom_id]   = result.result.message.content[0].text.strip()
            usage_by_id[result.custom_id] = result.result.message.usage
        else:
            raw_by_id[result.custom_id] = None

    codes = []
    missing = []
    for qid in submitted:
        q     = q_by_id.get(qid)
        raw   = raw_by_id.get(qid)
        usage = usage_by_id.get(qid)
        if q is None:
            continue
        if raw is None:
            missing.append(qid)
            continue
        codes.append({
            'id':            qid,
            'domain':        q.get('domain'),
            'code':          extract_code(raw),
            'options':       q['options'],
            'input_tokens':  usage.input_tokens  if usage else 0,
            'output_tokens': usage.output_tokens if usage else 0,
        })

    codes_file = make_codes_path(model)
    Path(codes_file).write_text(json.dumps({
        'model':   model,
        'count':   len(codes),
        'entries': codes,
    }, indent=2))

    print(f"\n  Fetched {len(codes)} code snippets → {codes_file}")
    if missing:
        print(f"  ⚠ {len(missing)} question(s) errored/expired: {', '.join(missing)}")
    print(f"  Run 'python {sys.argv[0]} run --model {model}' to execute locally.")


# ── Step 4: run ───────────────────────────────────────────────────────────────

def run(model, answers_file, output_file):
    codes_file = make_codes_path(model)
    if not Path(codes_file).exists():
        print(f"  Codes file not found: {codes_file}. Run 'fetch' first.")
        return

    data    = load_json(codes_file)
    answers = load_json(answers_file)
    entries = data['entries']
    total   = len(entries)

    print(f"\n{'='*62}")
    print(f"  Model:   {model}")
    print(f"  Running: {total} code snippets locally")
    print(f"{'='*62}\n")

    results = []
    correct = 0
    total_in_tokens, total_out_tokens = 0, 0

    for entry in entries:
        qid      = entry['id']
        expected = answers.get(qid, '?')
        options  = entry['options']
        in_tok   = entry.get('input_tokens', 0)
        out_tok  = entry.get('output_tokens', 0)
        total_in_tokens  += in_tok
        total_out_tokens += out_tok

        result = execute_python(entry['code'])

        if result['error']:
            letter = 'E'
            print(f"  {qid:<12} exec=ERROR  model={letter}  expected={expected}  {'✓' if letter == expected else '✗'}  [in={in_tok} out={out_tok}]")
        else:
            letter = pick_closest_option(result['stdout'], options)
            print(f"  {qid:<12} stdout={result['stdout'][:40]!r}  model={letter}  expected={expected}  {'✓' if letter == expected else '✗'}  [in={in_tok} out={out_tok}]")

        is_correct = letter == expected
        if is_correct:
            correct += 1

        results.append({
            'id':            qid,
            'domain':        entry.get('domain'),
            'model_answer':  letter,
            'expected':      expected,
            'correct':       is_correct,
            'stdout':        result['stdout'],
            'error':         result['error'],
            'input_tokens':  in_tok,
            'output_tokens': out_tok,
        })

    accuracy = correct / total * 100 if total else 0.0
    print(f"\n{'='*62}")
    print(f"  {correct}/{total} correct — {accuracy:.1f}%")
    print(f"  Tokens — input: {total_in_tokens}  output: {total_out_tokens}  total: {total_in_tokens + total_out_tokens}")
    print()
    for domain in ['stochastic_reactor', 'ode_blowup', 'laplace_method', 'dominant_balance', 'nondim']:
        dr = [r for r in results if r['domain'] == domain]
        if dr:
            dc = sum(1 for r in dr if r['correct'])
            print(f"  {domain:<22} {dc}/{len(dr)}")
    print(f"{'='*62}\n")

    Path(output_file).write_text(json.dumps({
        'model':               model,
        'mode':                'code_generation_batch',
        'total':               total,
        'correct':             correct,
        'accuracy_pct':        round(accuracy, 2),
        'total_input_tokens':  total_in_tokens,
        'total_output_tokens': total_out_tokens,
        'total_tokens':        total_in_tokens + total_out_tokens,
        'details':             results,
    }, indent=2))
    print(f"  Saved to: {output_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Claude Code-Generation Batch MCQ Evaluator')
    sub = p.add_subparsers(dest='command', required=True)

    p_sub = sub.add_parser('submit')
    p_sub.add_argument('--model',      default='claude-opus-4-8')
    p_sub.add_argument('--questions',  default=QUESTIONS_FILE)
    p_sub.add_argument('--limit',      type=int, default=None)
    p_sub.add_argument('--start',      type=int, default=0)
    p_sub.add_argument('--max-tokens', type=int, default=4096)
    p_sub.add_argument('--force', action='store_true')

    sub.add_parser('status')

    p_fetch = sub.add_parser('fetch')
    p_fetch.add_argument('--model',     default='claude-opus-4-8')
    p_fetch.add_argument('--questions', default=QUESTIONS_FILE)

    p_run = sub.add_parser('run')
    p_run.add_argument('--model',   default='claude-opus-4-8')
    p_run.add_argument('--answers', default=ANSWERS_FILE)
    p_run.add_argument('--output',  default=None)

    args = p.parse_args()

    if args.command == 'submit':
        submit(args.model, args.questions, args.max_tokens,
               args.limit, args.start, args.force)
    elif args.command == 'status':
        status()
    elif args.command == 'fetch':
        fetch(args.model, args.questions)
    elif args.command == 'run':
        out = args.output or make_output_path(args.model)
        run(args.model, args.answers, out)

# 1. Submit 2 questions to the batch API
python claude_code_batch_evaluator.py submit --limit 2

# 2. Check if batch is done
python claude_code_batch_evaluator.py status

# 3. Download the generated code (once status shows "ended")
python claude_code_batch_evaluator.py fetch --model claude-sonnet-4-6

# 4. Execute code locally and score
python claude_code_batch_evaluator.py run --model claude-sonnet-4-6
