"""
Applied-math MCQ Evaluator (No-Tool + Code-Generation + Tool-Augmented Agent)
==============================================================================
Three modes:
  no_tool_access   — pure LLM reasoning, picks A-E from knowledge
  code_generation  — model writes Python code, executed locally, model picks A-E
  tool_augmented   — full ReAct agent loop, T_max=10, option E hidden from model

Usage:
  python code_evaluator.py --model claude-sonnet-4-6 --mode no_tool_access
  python code_evaluator.py --model claude-sonnet-4-6 --mode code_generation
  python code_evaluator.py --model claude-opus-4-8   --mode tool_augmented
  python code_evaluator.py --model gpt-4o            --mode tool_augmented
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from code_runner import execute_python
from agent_runner import run_agent          # ← নতুন import

QUESTIONS_FILE = 'questions/all_questions_with_E.json'
ANSWERS_FILE   = 'answers/all_answers_with_E.json'

SYSTEM_NO_TOOL = (
    'You are an applied mathematics expert answering multiple-choice questions. '
    'Think through the problem step by step, then end your response with '
    '"Answer: X" where X is A, B, C, D, or E.'
)

SYSTEM_CODE = (
    'You are an applied mathematics expert solving multiple-choice questions. '
    'Write Python code using numpy (import as np), scipy, or sympy to solve the problem numerically. '
    'Your code must print() the final numerical answer and nothing else. '
    'Call the execute_python tool with your complete, runnable code. '
    'After seeing the execution result, pick the option (A, B, C, D, or E) '
    'whose value is closest to the printed answer. '
    'Reply with ONLY the option letter.'
)

EXECUTE_TOOL_CLAUDE = {
    'name': 'execute_python',
    'description': (
        'Execute Python code to solve the problem numerically. '
        'Available: numpy (as np), scipy, sympy. '
        'Code must print() the final numerical answer.'
    ),
    'input_schema': {
        'type': 'object',
        'properties': {
            'code': {
                'type': 'string',
                'description': 'Complete, runnable Python code that prints the numerical answer.',
            }
        },
        'required': ['code'],
    },
}

EXECUTE_TOOL_OPENAI = {
    'type': 'function',
    'function': {
        'name':        EXECUTE_TOOL_CLAUDE['name'],
        'description': EXECUTE_TOOL_CLAUDE['description'],
        'parameters':  EXECUTE_TOOL_CLAUDE['input_schema'],
    },
}


# ── Provider helpers ──────────────────────────────────────────────────────────

def detect_provider(model):
    m = model.lower()
    if m.startswith('gpt') or m.startswith('o1') or m.startswith('o3') or m.startswith('o4'):
        return 'openai'
    return 'anthropic'


def create_client(provider):
    if provider == 'openai':
        return OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))
    return anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY', ''))


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def format_question(q):
    opts = '\n'.join(f"{k}. {v}" for k, v in sorted(q['options'].items()))
    return f"{q['question']}\n\n{opts}"


def _serialize_block(block):
    if block.type == 'text':
        return {'type': 'text', 'text': block.text}
    if block.type == 'tool_use':
        return {'type': 'tool_use', 'id': block.id, 'name': block.name, 'input': block.input}
    return {'type': 'text', 'text': str(block)}


# ── No-tool mode ──────────────────────────────────────────────────────────────

def answer_no_tool(client, model, question_text, max_tokens=512, provider='anthropic'):
    if provider == 'openai':
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[
                {'role': 'system', 'content': SYSTEM_NO_TOOL},
                {'role': 'user',   'content': question_text},
            ],
        )
        return resp.choices[0].message.content.strip(), resp.usage.prompt_tokens, resp.usage.completion_tokens

    resp = client.messages.create(
        model=model,
        system=SYSTEM_NO_TOOL,
        max_tokens=max_tokens,
        messages=[{'role': 'user', 'content': question_text}],
    )
    return resp.content[0].text.strip(), resp.usage.input_tokens, resp.usage.output_tokens


# ── Code-generation mode ──────────────────────────────────────────────────────

def run_code_agent(client, model, question_text, max_tokens=2048, provider='anthropic'):
    """Returns (letter, stdout, error, in_tok, out_tok)."""
    stdout_captured = None
    total_in, total_out = 0, 0

    if provider == 'openai':
        messages = [
            {'role': 'system', 'content': SYSTEM_CODE},
            {'role': 'user',   'content': question_text},
        ]
        for _ in range(3):
            resp = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                tools=[EXECUTE_TOOL_OPENAI],
                messages=messages,
            )
            total_in  += resp.usage.prompt_tokens
            total_out += resp.usage.completion_tokens
            msg        = resp.choices[0].message
            tool_calls = msg.tool_calls or []

            if tool_calls:
                messages.append(msg)
                for tc in tool_calls:
                    code   = json.loads(tc.function.arguments).get('code', '')
                    result = execute_python(code)
                    print(f"      [exec] {'OK' if not result['error'] else 'ERROR'}")
                    if result['error']:
                        return 'E', '', result['error'], total_in, total_out
                    if not result['stdout']:
                        return 'E', '', 'Code executed but produced no output.', total_in, total_out
                    stdout_captured = result['stdout']
                    messages.append({
                        'role':         'tool',
                        'tool_call_id': tc.id,
                        'content':      result['stdout'],
                    })
            else:
                raw    = (msg.content or '').strip()
                letter = extract_letter(raw)
                return letter, stdout_captured or '', None, total_in, total_out

        return 'E', stdout_captured or '', 'Max iterations reached', total_in, total_out

    # Anthropic
    messages = [{'role': 'user', 'content': question_text}]
    for _ in range(3):
        resp = client.messages.create(
            model=model,
            system=SYSTEM_CODE,
            max_tokens=max_tokens,
            tools=[EXECUTE_TOOL_CLAUDE],
            messages=messages,
        )
        total_in  += resp.usage.input_tokens
        total_out += resp.usage.output_tokens

        tool_blocks = [b for b in resp.content if b.type == 'tool_use']
        text_blocks = [b for b in resp.content if b.type == 'text']

        if tool_blocks:
            messages.append({'role': 'assistant', 'content': [_serialize_block(b) for b in resp.content]})
            results = []
            for b in tool_blocks:
                code   = b.input.get('code', '')
                result = execute_python(code)
                print(f"      [exec] {'OK' if not result['error'] else 'ERROR'}")
                if result['error']:
                    return 'E', '', result['error'], total_in, total_out
                if not result['stdout']:
                    return 'E', '', 'Code executed but produced no output.', total_in, total_out
                stdout_captured = result['stdout']
                results.append({
                    'type':        'tool_result',
                    'tool_use_id': b.id,
                    'content':     result['stdout'],
                })
            messages.append({'role': 'user', 'content': results})

        elif text_blocks:
            raw    = '\n'.join(b.text for b in text_blocks).strip()
            letter = extract_letter(raw)
            return letter, stdout_captured or '', None, total_in, total_out
        else:
            return 'E', '', 'Unexpected agent state', total_in, total_out

    return 'E', stdout_captured or '', 'Max iterations reached', total_in, total_out


# ── Tool-augmented agent mode ─────────────────────────────────────────────────

def make_call_model_anthropic(client, model, max_tokens=2048):
    """Anthropic-এর জন্য call_model function বানায়।"""
    def call_model(messages):
        system  = next((m['content'] for m in messages if m['role'] == 'system'), '')
        history = [m for m in messages if m['role'] != 'system']
        resp = client.messages.create(
            model      = model,
            max_tokens = max_tokens,
            system     = system,
            messages   = history,
        )
        return resp.content[0].text
    return call_model


def make_call_model_openai(client, model, max_tokens=2048):
    """OpenAI-এর জন্য call_model function বানায়।"""
    def call_model(messages):
        resp = client.chat.completions.create(
            model                 = model,
            max_completion_tokens = max_tokens,
            messages              = messages,
        )
        return resp.choices[0].message.content.strip()
    return call_model


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def make_output_path(model, mode):
    safe = model.replace('/', '_').replace('.', '-')
    Path('results').mkdir(exist_ok=True)
    return f"results/results_{safe}_{mode}.json"


def load_checkpoint(output_file):
    p = Path(output_file)
    if not p.exists():
        return set(), [], 0, 0
    try:
        data      = json.loads(p.read_text())
        results   = [r for r in data.get('details', []) if r.get('model_answer', '?') != '?']
        done_ids  = {r['id'] for r in results}
        total_in  = sum(r.get('input_tokens', 0) for r in results)
        total_out = sum(r.get('output_tokens', 0) for r in results)
        return done_ids, results, total_in, total_out
    except Exception:
        return set(), [], 0, 0


def save_checkpoint(output_file, model, mode, results, total_in, total_out):
    correct = sum(1 for r in results if r.get('correct'))
    total   = len(results)
    Path(output_file).write_text(json.dumps({
        'model':               model,
        'mode':                mode,
        'total_answered':      total,
        'correct':             correct,
        'accuracy_pct':        round(correct / total * 100 if total else 0.0, 2),
        'total_input_tokens':  total_in,
        'total_output_tokens': total_out,
        'total_tokens':        total_in + total_out,
        'details':             results,
    }, indent=2))


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(model, mode, questions_file, answers_file, output_file,
             delay=0.0, limit=None, max_tokens=512, print_reasoning=False, t_max=3):

    questions = json.loads(Path(questions_file).read_text())
    answers   = json.loads(Path(answers_file).read_text())

    if limit:
        questions = questions[:limit]

    done_ids, results, total_in_tokens, total_out_tokens = load_checkpoint(output_file)
    if done_ids:
        print(f"\n  Resuming: {len(done_ids)} already answered.")

    remaining = [q for q in questions if q['id'] not in done_ids]
    total_all = len(questions)
    provider  = detect_provider(model)
    client    = create_client(provider)
    correct   = sum(1 for r in results if r.get('correct'))

    # tool_augmented-এর জন্য call_model function বানাও
    if mode == 'tool_augmented':
        if provider == 'openai':
            call_model = make_call_model_openai(client, model, max_tokens=2048)
        else:
            call_model = make_call_model_anthropic(client, model, max_tokens=2048)

    print(f"\n{'='*62}")
    print(f"  Model:     {model}  ({provider})")
    print(f"  Mode:      {mode}")
    print(f"  Total Qs:  {total_all}  |  Remaining: {len(remaining)}")
    print(f"  Output:    {output_file}")
    print(f"{'='*62}\n")

    for q in remaining:
        qid      = q['id']
        expected = answers.get(qid, '?')
        qtext    = format_question(q)
        in_tok, out_tok = 0, 0

        try:
            # ── mode 1: no_tool_access ────────────────────────────────────────
            if mode == 'no_tool_access':
                raw, in_tok, out_tok = answer_no_tool(
                    client, model, qtext, max_tokens=max_tokens, provider=provider
                )
                letter = extract_letter(raw)
                entry = {
                    'id':            qid,
                    'domain':        q.get('domain'),
                    'model_answer':  letter,
                    'expected':      expected,
                    'correct':       letter == expected,
                    'tool_calls':    0,
                    'input_tokens':  in_tok,
                    'output_tokens': out_tok,
                    'raw':           raw,
                }

            # ── mode 2: code_generation ───────────────────────────────────────
            elif mode == 'code_generation':
                letter, stdout, error, in_tok, out_tok = run_code_agent(
                    client, model, qtext, max_tokens=max_tokens, provider=provider
                )
                entry = {
                    'id':            qid,
                    'domain':        q.get('domain'),
                    'model_answer':  letter,
                    'expected':      expected,
                    'correct':       letter == expected,
                    'stdout':        stdout,
                    'error':         error,
                    'input_tokens':  in_tok,
                    'output_tokens': out_tok,
                }

            # ── mode 3: tool_augmented (নতুন) ────────────────────────────────
            elif mode == 'tool_augmented':
                agent_result = run_agent(q, call_model, t_max=t_max)
                letter = agent_result['model_answer']   # A/B/C/D অথবা E (T_max)
                entry = {
                    'id':            qid,
                    'domain':        q.get('domain'),
                    'model_answer':  letter,
                    'expected':      expected,
                    'correct':       letter == expected,
                    'termination':   agent_result['termination'],   # submitted / hit_T_max
                    'steps':         agent_result['steps'],
                    'executions':    agent_result['executions'],
                    'input_tokens':  0,    # ReAct mode-এ token count আলাদাভাবে track হয় না
                    'output_tokens': 0,
                }

        except Exception as e:
            letter = '?'
            entry = {
                'id':            qid,
                'domain':        q.get('domain'),
                'model_answer':  '?',
                'expected':      expected,
                'correct':       False,
                'error':         str(e),
                'input_tokens':  in_tok,
                'output_tokens': out_tok,
            }
            print(f"  ERROR on {qid}: {e}")

        total_in_tokens  += in_tok
        total_out_tokens += out_tok

        if entry['correct']:
            correct += 1

        answered_so_far = len(results) + 1
        status = '✓' if entry['correct'] else '✗'

        # tool_augmented-এ steps দেখাও, বাকিতে token দেখাও
        if mode == 'tool_augmented':
            extra = f"  [steps={entry.get('steps',0)} exec={entry.get('executions',0)} term={entry.get('termination','')}]"
        else:
            extra = f"  [in={in_tok} out={out_tok}]"

        print(f"  [{answered_so_far}/{total_all}] {qid:<12} model={letter}  expected={expected}  {status}{extra}")

        if print_reasoning:
            hint = entry.get('raw') or entry.get('stdout') or ''
            if hint:
                print(f"    {hint[:240]}\n")

        results.append(entry)
        save_checkpoint(output_file, model, mode, results, total_in_tokens, total_out_tokens)

        if delay > 0:
            time.sleep(delay)

    total_answered = len(results)
    accuracy = correct / total_answered * 100 if total_answered else 0.0

    print(f"\n{'='*62}")
    print(f"  {correct}/{total_answered} correct — {accuracy:.1f}%")
    print(f"  Tokens — input: {total_in_tokens}  output: {total_out_tokens}  total: {total_in_tokens + total_out_tokens}")
    print()
    for domain in ['stochastic_reactor', 'ode_blowup', 'laplace_method', 'dominant_balance', 'nondim']:
        dr = [r for r in results if r['domain'] == domain]
        if dr:
            dc = sum(1 for r in dr if r['correct'])
            print(f"  {domain:<22} {dc}/{len(dr)}")
    print(f"{'='*62}\n")
    print(f"  Saved to: {output_file}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Applied-math MCQ Evaluator')
    p.add_argument('--model',      default='claude-sonnet-4-6')
    p.add_argument('--mode',       choices=['no_tool_access', 'code_generation', 'tool_augmented'],
                   default='no_tool_access')
    p.add_argument('--questions',  default=QUESTIONS_FILE)
    p.add_argument('--answers',    default=ANSWERS_FILE)
    p.add_argument('--output',     default=None)
    p.add_argument('--delay',      type=float, default=0.0)
    p.add_argument('--limit',      type=int,   default=None)
    p.add_argument('--max-tokens', type=int,   default=512,
                   help='Max tokens per API call (use 2048+ for code_generation / tool_augmented)')
    p.add_argument('--print-reasoning', action='store_true', dest='print_reasoning')
    p.add_argument('--t-max', type=int, default=3, dest='t_max')
    args = p.parse_args()

    output_file = args.output or make_output_path(args.model, args.mode)
    evaluate(
        model           = args.model,
        mode            = args.mode,
        questions_file  = args.questions,
        answers_file    = args.answers,
        output_file     = output_file,
        delay           = args.delay,
        limit           = args.limit,
        max_tokens      = args.max_tokens,
        print_reasoning = args.print_reasoning,
        t_max           = args.t_max,
    )