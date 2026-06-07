"""
Local LLM Applied-math MCQ Evaluator (No-Tool + Code-Generation + Tool-Augmented Agent)
=========================================================================================
Three modes:
  no_tool_access   — pure LLM reasoning, picks A-E from knowledge
  code_generation  — model writes Python code, executed locally; retry up to
                     MAX_ATTEMPTS times on crash, then picks E
  tool_augmented   — full ReAct agent loop, T_max=10, option E hidden from model

Usage:
  python local_code_evaluator.py --model Qwen/Qwen2.5-7B-Instruct --mode no_tool_access
  python local_code_evaluator.py --model Qwen/Qwen2.5-7B-Instruct --mode code_generation
  python local_code_evaluator.py --model Qwen/Qwen2.5-7B-Instruct --mode tool_augmented
  python local_code_evaluator.py --model Qwen/Qwen2.5-7B-Instruct --mode no_tool_access --no-4bit
"""

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent))
from code_runner import execute_python
from agent_runner import run_agent          # ← পরিবর্তন ১: নতুন import

QUESTIONS_FILE = 'questions/all_questions_with_E.json'
ANSWERS_FILE   = 'answers/all_answers_with_E.json'
DEFAULT_MODEL  = 'Qwen/Qwen2.5-7B-Instruct'
MAX_ATTEMPTS   = 3

OPENAI_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'execute_python',
            'description': (
                'Execute Python code to solve the problem numerically. '
                'Available: numpy (as np), scipy, sympy. '
                'Code must print() the final numerical answer.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': 'Complete, runnable Python code that prints the answer.',
                    }
                },
                'required': ['code'],
            },
        },
    }
]

SYSTEM_NO_TOOL = (
    'You are an applied mathematics expert answering multiple-choice questions. '
    'Think through the problem step by step, then end your response with '
    '"Answer: X" where X is A, B, C, D or E.'
)

SYSTEM_CODE = (
    'You are an applied mathematics expert solving multiple-choice questions. '
    'Write Python code using numpy (as np), scipy, or sympy to solve the problem numerically. '
    'Your code must print() the final numerical answer and nothing else. '
    'Call the execute_python tool with your complete, runnable code. '
    'After seeing the result, pick the option (A, B, C, D, or E) whose value is closest. '
    'Reply with ONLY the option letter.'
)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name, load_in_4bit=True, trust_remote_code=False):
    print(f"  Loading {model_name} ({'4-bit NF4' if load_in_4bit else 'bfloat16'})...")
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if load_in_4bit else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    kwargs = dict(device_map='auto', trust_remote_code=trust_remote_code)
    if bnb_config is not None:
        kwargs['quantization_config'] = bnb_config
    else:
        kwargs['torch_dtype'] = 'auto'
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    print(f"  Model loaded on: {next(model.parameters()).device}\n")
    return model, tokenizer


# ── Generation helper ─────────────────────────────────────────────────────────

def generate(model, tokenizer, messages, max_new_tokens=512, tools=None):
    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    in_tok = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][in_tok:]
    out_tok = generated.shape[0]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), in_tok, out_tok


# ── Tool call parser ──────────────────────────────────────────────────────────

def parse_tool_call(text):
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data['name'], data.get('arguments', data.get('parameters', {}))
        except (json.JSONDecodeError, KeyError):
            pass
    m = re.search(r'<\|python_tag\|>\s*(\{.*?\})', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data['name'], data.get('parameters', data.get('arguments', {}))
        except (json.JSONDecodeError, KeyError):
            pass
    m = re.search(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|parameters)"\s*:\s*(\{.*?\})\s*\}',
        text, re.DOTALL,
    )
    if m:
        try:
            return m.group(1), json.loads(m.group(2))
        except json.JSONDecodeError:
            pass
    return None, None


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


# ── No-tool mode ──────────────────────────────────────────────────────────────

def answer_no_tool(model, tokenizer, question_text, max_new_tokens=4096):
    messages = [
        {'role': 'system', 'content': SYSTEM_NO_TOOL},
        {'role': 'user',   'content': question_text},
    ]
    text, in_tok, out_tok = generate(model, tokenizer, messages, max_new_tokens=max_new_tokens)
    return text, in_tok, out_tok


# ── Code-generation mode ──────────────────────────────────────────────────────

def run_code_agent(model, tokenizer, question_text, max_new_tokens=4096):
    """Returns (letter, stdout, error, attempts, total_in_tok, total_out_tok)."""
    messages = [
        {'role': 'system', 'content': SYSTEM_CODE},
        {'role': 'user',   'content': question_text},
    ]
    stdout_captured = None
    error_captured  = None
    total_in_tok, total_out_tok = 0, 0

    for attempt in range(MAX_ATTEMPTS):
        response, in_tok, out_tok = generate(model, tokenizer, messages,
                                             max_new_tokens=max_new_tokens, tools=OPENAI_TOOLS)
        total_in_tok += in_tok
        total_out_tok += out_tok
        name, args = parse_tool_call(response)

        if name == 'execute_python' and args is not None:
            code   = args.get('code', '')
            result = execute_python(code)
            messages.append({'role': 'assistant', 'content': response})

            print(f"      [code]\n{code}\n")
            if result['error']:
                error_captured = result['error']
                print(f"      [exec] ERROR (attempt {attempt + 1}/{MAX_ATTEMPTS})")
                if attempt < MAX_ATTEMPTS - 1:
                    messages.append({
                        'role':    'tool',
                        'name':    'execute_python',
                        'content': f"ERROR:\n{result['error']}",
                    })
                    continue
                else:
                    return 'E', '', error_captured, MAX_ATTEMPTS, total_in_tok, total_out_tok
            elif not result['stdout']:
                error_captured = 'Code executed but produced no output.'
                print(f"      [exec] EMPTY (attempt {attempt + 1}/{MAX_ATTEMPTS})")
                if attempt < MAX_ATTEMPTS - 1:
                    messages.append({
                        'role':    'tool',
                        'name':    'execute_python',
                        'content': 'Your code ran but produced no output. Make sure to print() the final numerical answer.',
                    })
                    continue
                else:
                    return 'E', '', error_captured, MAX_ATTEMPTS, total_in_tok, total_out_tok
            else:
                stdout_captured = result['stdout']
                print(f"      [exec] OK  stdout={stdout_captured[:60]!r}")
                messages.append({
                    'role':    'tool',
                    'name':    'execute_python',
                    'content': result['stdout'],
                })
                final, in_tok2, out_tok2 = generate(model, tokenizer, messages, max_new_tokens=256)
                total_in_tok += in_tok2
                total_out_tok += out_tok2
                letter = extract_letter(final)
                return letter, stdout_captured, None, attempt + 1, total_in_tok, total_out_tok
        else:
            letter = extract_letter(response)
            return letter, '', None, attempt + 1, total_in_tok, total_out_tok

    return 'E', '', error_captured, MAX_ATTEMPTS, total_in_tok, total_out_tok


# ── Tool-augmented mode ───────────────────────────────────────────────────────
# পরিবর্তন ২: নতুন function

def make_local_call_model(model, tokenizer, max_new_tokens=2048):
    """Local model-এর জন্য call_model wrapper — agent_runner-এ pass করা হবে।"""
    def call_model(messages):
        text, _, _ = generate(model, tokenizer, messages,
                               max_new_tokens=max_new_tokens)
        return text
    return call_model


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def make_output_path(model_name, mode):
    safe = model_name.replace('/', '_').replace('.', '-')
    Path('results').mkdir(exist_ok=True)
    return f"results/results_{safe}_{mode}.json"


def load_checkpoint(output_file):
    p = Path(output_file)
    if not p.exists():
        return set(), [], 0
    try:
        data     = json.loads(p.read_text())
        results  = [r for r in data.get('details', []) if r.get('model_answer', '?') != '?']
        done_ids = {r['id'] for r in results}
        correct  = sum(1 for r in results if r.get('correct'))
        return done_ids, results, correct
    except Exception:
        return set(), [], 0


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(model_name, mode, questions_file, answers_file, output_file,
             delay=0.0, load_in_4bit=True, trust_remote_code=False,
             print_reasoning=False, limit=None, max_new_tokens=4096):
    questions = json.loads(Path(questions_file).read_text())
    answers   = json.loads(Path(answers_file).read_text())

    if limit:
        questions = questions[:limit]

    done_ids, results, correct = load_checkpoint(output_file)
    if done_ids:
        print(f"\n  Resuming: {len(done_ids)} already answered.")

    remaining = [q for q in questions if q['id'] not in done_ids]
    total     = len(questions)

    if not remaining:
        print("\n  All questions already answered.")
        return results

    model, tokenizer = load_model(model_name, load_in_4bit, trust_remote_code)

    # পরিবর্তন ৩ক: model load-এর পরে call_model বানাও
    if mode == 'tool_augmented':
        call_model = make_local_call_model(model, tokenizer,
                                           max_new_tokens=max_new_tokens)

    print(f"\n{'='*62}")
    print(f"  Model:      {model_name}")
    print(f"  Mode:       {mode}")
    print(f"  Quant:      {'4-bit NF4' if load_in_4bit else 'bfloat16'}")
    if mode == 'code_generation':
        print(f"  Max tries:  {MAX_ATTEMPTS}")
    if mode == 'tool_augmented':
        print(f"  T_max:      10")
    print(f"  Total Qs:   {total}  |  Remaining: {len(remaining)}")
    print(f"  Max tokens: {max_new_tokens}")
    print(f"  Output:     {output_file}")
    print(f"{'='*62}\n")

    for i, q in enumerate(remaining, len(done_ids) + 1):
        qid      = q['id']
        expected = answers.get(qid, '?')
        qtext    = format_question(q)

        in_tok, out_tok = 0, 0
        try:
            # ── mode 1: no_tool_access ────────────────────────────────────────
            if mode == 'no_tool_access':
                raw, in_tok, out_tok = answer_no_tool(
                    model, tokenizer, qtext, max_new_tokens=max_new_tokens
                )
                letter = extract_letter(raw)
                entry  = {
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
                letter, stdout, error, attempts, in_tok, out_tok = run_code_agent(
                    model, tokenizer, qtext, max_new_tokens=max_new_tokens
                )
                entry = {
                    'id':            qid,
                    'domain':        q.get('domain'),
                    'model_answer':  letter,
                    'expected':      expected,
                    'correct':       letter == expected,
                    'stdout':        stdout,
                    'error':         error,
                    'attempts':      attempts,
                    'input_tokens':  in_tok,
                    'output_tokens': out_tok,
                }

            # ── mode 3: tool_augmented (নতুন) ────────────────────────────────
            elif mode == 'tool_augmented':
                agent_result = run_agent(q, call_model, t_max=args.t_max)
                letter = agent_result['model_answer']
                entry = {
                    'id':            qid,
                    'domain':        q.get('domain'),
                    'model_answer':  letter,
                    'expected':      expected,
                    'correct':       letter == expected,
                    'termination':   agent_result['termination'],
                    'steps':         agent_result['steps'],
                    'executions':    agent_result['executions'],
                    'input_tokens':  0,
                    'output_tokens': 0,
                }

        except Exception as e:
            letter = '?'
            entry  = {
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

        is_correct = entry['correct']
        correct   += is_correct
        status     = '✓' if is_correct else '✗'

        # পরিবর্তন ৩খ: extra print — mode অনুযায়ী
        if mode == 'code_generation':
            extra = f"  [{entry.get('attempts')} attempt(s)]"
        elif mode == 'tool_augmented':
            extra = f"  [steps={entry.get('steps',0)} exec={entry.get('executions',0)} term={entry.get('termination','')}]"
        else:
            extra = ''

        tok_str = f"  [in={in_tok} out={out_tok}]"
        print(f"  [{i}/{total}] {qid:<12} model={letter}  expected={expected}  {status}{extra}{tok_str}")
        if print_reasoning:
            hint = entry.get('raw') or entry.get('stdout') or ''
            if hint:
                print(f"    {hint[:120]}\n")

        results.append(entry)

        answered = len(results)
        Path(output_file).write_text(json.dumps({
            'model':        model_name,
            'mode':         mode,
            'total':        total,
            'correct':      correct,
            'accuracy_pct': round(correct / answered * 100, 2),
            'details':      results,
        }, indent=2))

        if delay > 0:
            time.sleep(delay)

    accuracy = correct / total * 100 if total else 0.0
    print(f"\n{'='*62}")
    print(f"  {correct}/{total} correct — {accuracy:.1f}%")
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
    p = argparse.ArgumentParser(description='Local LLM Applied-math MCQ Evaluator')
    p.add_argument('--model',     default=DEFAULT_MODEL)
    p.add_argument('--mode',      choices=['no_tool_access', 'code_generation', 'tool_augmented'],
                   default='no_tool_access')           # ← পরিবর্তন ৪
    p.add_argument('--questions', default=QUESTIONS_FILE)
    p.add_argument('--answers',   default=ANSWERS_FILE)
    p.add_argument('--output',    default=None)
    p.add_argument('--delay',     type=float, default=0.0)
    p.add_argument('--limit',     type=int,   default=None)
    p.add_argument('--max-tokens',        type=int, default=4096, dest='max_tokens')
    p.add_argument('--no-4bit',           action='store_true', dest='no_4bit')
    p.add_argument('--trust-remote-code', action='store_true', dest='trust_remote_code')
    p.add_argument('--print-reasoning',   action='store_true', dest='print_reasoning')
    #tmax
    p.add_argument('--t-max', type=int, default=10, dest='t_max',
                   help='Tool-augmented mode-এর জন্য agent loop-এর T_max (default: 10)')
    args = p.parse_args()

    output_file = args.output or make_output_path(args.model, args.mode)
    evaluate(
        model_name        = args.model,
        mode              = args.mode,
        questions_file    = args.questions,
        answers_file      = args.answers,
        output_file       = output_file,
        delay             = args.delay,
        load_in_4bit      = not args.no_4bit,
        trust_remote_code = args.trust_remote_code,
        print_reasoning   = args.print_reasoning,
        limit             = args.limit,
        max_new_tokens    = args.max_tokens,
    )