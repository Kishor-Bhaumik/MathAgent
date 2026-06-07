"""
No-Tool-Access Batch Evaluator (Claude + GPT)
===============================================
3-step workflow. Provider auto-detected from model name.
Model picks A-E directly from knowledge — no code execution.

  submit  → send questions to Batch API
  status  → poll API
  fetch   → download answers, score, save results

Usage:
  python no_tool_batch_evaluator.py submit --model claude-opus-4-8
  python no_tool_batch_evaluator.py submit --model gpt-5-5
  python no_tool_batch_evaluator.py status --model claude-opus-4-8
  python no_tool_batch_evaluator.py fetch  --model claude-opus-4-8
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

QUESTIONS_FILE = 'questions/all_questions_with_E.json'
ANSWERS_FILE   = 'answers/all_answers_with_E.json'

SYSTEM_NO_TOOL = (
    'You are an applied mathematics expert answering multiple-choice questions. '
    'Think through the problem step by step. '
    'End your response with "Answer: X" where X is A, B, C, D, or E.'
)


# ── Provider detection ────────────────────────────────────────────────────────

def detect_provider(model):
    m = model.lower()
    if m.startswith('gpt') or m.startswith('o1') or m.startswith('o3') or m.startswith('o4'):
        return 'openai'
    return 'anthropic'


def make_state_file(model):
    """প্রতিটা model-এর আলাদা state file।"""
    safe = model.replace('/', '_').replace('.', '-')
    return f".no_tool_batch_state_{safe}.json"


def make_output_path(model):
    safe = model.replace('/', '_').replace('.', '-')
    Path('results').mkdir(exist_ok=True)
    return f"results/results_{safe}_no_tool_batch.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def format_question(q):
    opts = '\n'.join(f"{k}. {v}" for k, v in sorted(q['options'].items()))
    return f"{q['question']}\n\n{opts}"


def extract_letter(text):
    text = (text or '').strip().upper()
    for pattern in [
        r'(?:ANSWER|OPTION)\s*(?:IS\s*)?[:\-]?\s*\**\b([A-E])\b',
        r'\b(?:CHOOSE|SELECT|PICK)\s+(?:OPTION\s+)?\**([A-E])\b',
        r'\*\*([A-E])\*\*\s*$',
        r'\b([A-E])\b\s*$',
    ]:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    m = re.search(r'\b([A-E])\b', text)
    if m:
        return m.group(1)
    return '?'


def print_summary(results, model, total_in, total_out):
    total    = len(results)
    correct  = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100 if total else 0.0

    print(f"\n{'='*62}")
    print(f"  {correct}/{total} correct — {accuracy:.1f}%")
    print(f"  Tokens — input: {total_in}  output: {total_out}  total: {total_in + total_out}")
    print()
    for domain in ['stochastic_reactor', 'ode_blowup', 'laplace_method',
                   'dominant_balance', 'nondim']:
        dr = [r for r in results if r['domain'] == domain]
        if dr:
            dc = sum(1 for r in dr if r['correct'])
            print(f"  {domain:<22} {dc}/{len(dr)}")
    print(f"{'='*62}\n")
    return correct, total, accuracy


# ── ANTHROPIC ─────────────────────────────────────────────────────────────────

def anthropic_submit(model, questions, max_tokens, state_file, force):
    import anthropic
    c = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY', ''))

    if Path(state_file).exists() and not force:
        try:
            prev  = load_json(state_file)
            batch = c.beta.messages.batches.retrieve(prev['batch_id'])
            if batch.processing_status == 'in_progress':
                print(f"\n  Batch still active (id: {batch.id}). Use --force to override.")
                return
        except Exception:
            pass

    requests = [
        {
            'custom_id': q['id'],
            'params': {
                'model':      model,
                'max_tokens': max_tokens,
                'system':     SYSTEM_NO_TOOL,
                'messages':   [{'role': 'user', 'content': format_question(q)}],
            },
        }
        for q in questions
    ]
    batch = c.beta.messages.batches.create(requests=requests)
    print(f"  Created batch id: {batch.id}  status: {batch.processing_status}")
    return batch.id


def anthropic_status(model, state_file):
    import anthropic
    state = load_json(state_file)
    c     = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY', ''))
    batch = c.beta.messages.batches.retrieve(state['batch_id'])

    print(f"\n  Batch:  {batch.id}")
    print(f"  Status: {batch.processing_status}")
    rc = batch.request_counts
    if rc:
        print(f"  Progress: {rc.processing} processing  "
              f"{rc.succeeded} succeeded  {rc.errored} errored")
    if batch.processing_status == 'ended':
        print(f"\n  Done! Run fetch --model {model}")
    else:
        print(f"\n  Still processing — check again later.")


def anthropic_fetch(model, state_file, q_by_id, answers, output_file):
    import anthropic
    state = load_json(state_file)
    c     = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY', ''))
    batch = c.beta.messages.batches.retrieve(state['batch_id'])

    if batch.processing_status != 'ended':
        print(f"  Not ready (status: {batch.processing_status}). Run status first.")
        return

    raw_by_id, usage_by_id = {}, {}
    for result in c.beta.messages.batches.results(state['batch_id']):
        if result.result.type == 'succeeded':
            raw_by_id[result.custom_id]   = result.result.message.content[0].text.strip()
            usage_by_id[result.custom_id] = result.result.message.usage
        else:
            raw_by_id[result.custom_id] = None

    return raw_by_id, {
        qid: (u.input_tokens, u.output_tokens)
        for qid, u in usage_by_id.items()
    }


# ── OPENAI ────────────────────────────────────────────────────────────────────

def openai_submit(model, questions, max_tokens, state_file, force):
    from openai import OpenAI
    c = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))

    if Path(state_file).exists() and not force:
        try:
            prev  = load_json(state_file)
            batch = c.batches.retrieve(prev['batch_id'])
            if batch.status in ('validating', 'in_progress', 'finalizing'):
                print(f"\n  Batch still active (id: {batch.id}). Use --force to override.")
                return
        except Exception:
            pass

    lines = [
        json.dumps({
            'custom_id': q['id'],
            'method':    'POST',
            'url':       '/v1/chat/completions',
            'body': {
                'model':                 model,
                'max_completion_tokens': max_tokens,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_NO_TOOL},
                    {'role': 'user',   'content': format_question(q)},
                ],
            },
        })
        for q in questions
    ]

    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as tmp:
        tmp.write('\n'.join(lines).encode('utf-8'))
        tmp_path = tmp.name

    try:
        with open(tmp_path, 'rb') as f:
            uploaded = c.files.create(file=f, purpose='batch')
    finally:
        os.unlink(tmp_path)

    batch = c.batches.create(
        input_file_id     = uploaded.id,
        endpoint          = '/v1/chat/completions',
        completion_window = '24h',
    )
    print(f"  Created batch id: {batch.id}  status: {batch.status}")
    return batch.id


def openai_status(model, state_file):
    from openai import OpenAI
    state = load_json(state_file)
    c     = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))
    batch = c.batches.retrieve(state['batch_id'])

    print(f"\n  Batch:  {batch.id}")
    print(f"  Status: {batch.status}")
    rc = batch.request_counts
    if rc:
        print(f"  Progress: {rc.completed} completed  "
              f"{rc.failed} failed  {rc.total} total")
    if batch.status == 'completed':
        print(f"\n  Done! Run fetch --model {model}")
    else:
        print(f"\n  Still processing — check again later.")


def openai_fetch(model, state_file, q_by_id, answers, output_file):
    from openai import OpenAI
    state = load_json(state_file)
    c     = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))
    batch = c.batches.retrieve(state['batch_id'])

    if batch.status != 'completed':
        print(f"  Not ready (status: {batch.status}). Run status first.")
        return None, None

    content   = c.files.content(batch.output_file_id)
    raw_by_id = {}
    tok_by_id = {}

    for line in content.text.strip().splitlines():
        try:
            item   = json.loads(line)
            qid    = item['custom_id']
            resp   = item.get('response', {})
            if resp.get('status_code') == 200:
                body = resp['body']
                raw_by_id[qid] = body['choices'][0]['message']['content'].strip()
                u = body.get('usage', {})
                tok_by_id[qid] = (
                    u.get('prompt_tokens', 0),
                    u.get('completion_tokens', 0),
                )
            else:
                raw_by_id[qid] = None
        except Exception:
            continue

    return raw_by_id, tok_by_id


# ── Shared fetch + score ──────────────────────────────────────────────────────

def score_and_save(model, provider, raw_by_id, tok_by_id,
                   submitted, q_by_id, answers, output_file):
    results  = []
    correct  = 0
    missing  = []
    total_in, total_out = 0, 0

    print(f"\n{'='*62}")
    print(f"  Model:    {model}  ({provider})")
    print(f"  Mode:     no_tool_access (batch)")
    print(f"{'='*62}\n")

    for qid in submitted:
        q     = q_by_id.get(qid)
        raw   = raw_by_id.get(qid) if raw_by_id else None
        toks  = (tok_by_id or {}).get(qid, (0, 0))

        if q is None:
            continue
        if raw is None:
            missing.append(qid)
            results.append({
                'id': qid, 'domain': q.get('domain'),
                'model_answer': '?', 'expected': answers.get(qid, '?'),
                'correct': False, 'raw': None,
                'input_tokens': 0, 'output_tokens': 0,
            })
            continue

        letter     = extract_letter(raw)
        expected   = answers.get(qid, '?')
        in_tok, out_tok = toks
        total_in  += in_tok
        total_out += out_tok
        is_correct = letter == expected
        if is_correct:
            correct += 1

        print(f"  {qid:<12} model={letter}  expected={expected}  "
              f"{'✓' if is_correct else '✗'}  [in={in_tok} out={out_tok}]")

        results.append({
            'id': qid, 'domain': q.get('domain'),
            'model_answer': letter, 'expected': expected,
            'correct': is_correct, 'raw': raw,
            'input_tokens': in_tok, 'output_tokens': out_tok,
        })

    correct_n, total, accuracy = print_summary(results, model, total_in, total_out)

    if missing:
        print(f"  ⚠ {len(missing)} errored/expired: {', '.join(missing)}")

    Path(output_file).write_text(json.dumps({
        'model':               model,
        'mode':                'no_tool_access_batch',
        'total':               total,
        'correct':             correct_n,
        'accuracy_pct':        round(accuracy, 2),
        'total_input_tokens':  total_in,
        'total_output_tokens': total_out,
        'total_tokens':        total_in + total_out,
        'details':             results,
    }, indent=2))
    print(f"  Saved to: {output_file}")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_submit(args):
    provider   = detect_provider(args.model)
    state_file = make_state_file(args.model)
    questions  = load_json(args.questions)[args.start:]
    if args.limit:
        questions = questions[:args.limit]

    print(f"  Provider:  {provider}")
    print(f"  Sending:   {len(questions)} questions")

    if provider == 'anthropic':
        batch_id = anthropic_submit(args.model, questions,
                                    args.max_tokens, state_file, args.force)
    else:
        batch_id = openai_submit(args.model, questions,
                                 args.max_tokens, state_file, args.force)

    if batch_id:
        Path(state_file).write_text(json.dumps({
            'batch_id':      batch_id,
            'model':         args.model,
            'provider':      provider,
            'num_questions': len(questions),
            'submitted_ids': [q['id'] for q in questions],
        }, indent=2))
        print(f"  State saved → {state_file}")


def cmd_status(args):
    provider   = detect_provider(args.model)
    state_file = make_state_file(args.model)
    if not Path(state_file).exists():
        print(f"  No state file found for {args.model}. Run submit first.")
        return
    if provider == 'anthropic':
        anthropic_status(args.model, state_file)
    else:
        openai_status(args.model, state_file)


def cmd_fetch(args):
    provider   = detect_provider(args.model)
    state_file = make_state_file(args.model)
    if not Path(state_file).exists():
        print(f"  No state file found for {args.model}. Run submit first.")
        return

    state     = load_json(state_file)
    questions = load_json(args.questions)
    answers   = load_json(args.answers)
    q_by_id   = {q['id']: q for q in questions}
    submitted = set(state.get('submitted_ids', []))
    output    = args.output or make_output_path(args.model)

    if provider == 'anthropic':
        raw_by_id, tok_by_id = anthropic_fetch(
            args.model, state_file, q_by_id, answers, output)
    else:
        raw_by_id, tok_by_id = openai_fetch(
            args.model, state_file, q_by_id, answers, output)

    if raw_by_id is None:
        return

    score_and_save(args.model, provider, raw_by_id, tok_by_id,
                   submitted, q_by_id, answers, output)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='No-Tool Batch Evaluator — Claude & GPT auto-detected'
    )
    sub = p.add_subparsers(dest='command', required=True)

    # submit
    ps = sub.add_parser('submit')
    ps.add_argument('--model',      default='claude-opus-4-8')
    ps.add_argument('--questions',  default=QUESTIONS_FILE)
    ps.add_argument('--limit',      type=int, default=None)
    ps.add_argument('--start',      type=int, default=0)
    ps.add_argument('--max-tokens', type=int, default=2048)
    ps.add_argument('--force',      action='store_true')

    # status
    pst = sub.add_parser('status')
    pst.add_argument('--model', default='claude-opus-4-8')

    # fetch
    pf = sub.add_parser('fetch')
    pf.add_argument('--model',     default='claude-opus-4-8')
    pf.add_argument('--questions', default=QUESTIONS_FILE)
    pf.add_argument('--answers',   default=ANSWERS_FILE)
    pf.add_argument('--output',    default=None)

    args = p.parse_args()

    if args.command == 'submit':
        cmd_submit(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'fetch':
        cmd_fetch(args)