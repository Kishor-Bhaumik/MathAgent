from code_runner import execute_python

# ── Tool descriptions (ReAct text format) ─────────────────────────────────────

TOOL_DESCRIPTIONS = """
You have access to two tools:

1. run_python
   Write Python code using numpy, scipy, sympy.
   Your code MUST print() the final answer as the last line.
   Format:
   ACTION: run_python
   ```python
   # your code here
   print(answer)
   ```

2. submit_answer
   When you are confident, submit your final answer.
   Format:
   ACTION: submit_answer
   CHOICE: <A or B or C or D>
"""

# ── Parser ────────────────────────────────────────────────────────────────────

def parse_action(text: str) -> dict:
    """
    মডেলের output থেকে ACTION parse করে।

    Returns one of:
        {'type': 'run_python',     'code': str}
        {'type': 'submit_answer',  'choice': str}
        {'type': 'unknown',        'raw': str}
    """
    if 'ACTION: run_python' in text:
        code = ''
        if '```python' in text:
            start = text.index('```python') + len('```python')
            end   = text.index('```', start)
            code  = text[start:end].strip()
        elif '```' in text:
            start = text.index('```') + 3
            end   = text.index('```', start)
            code  = text[start:end].strip()
        return {'type': 'run_python', 'code': code}

    if 'ACTION: submit_answer' in text:
        choice = ''
        for line in text.splitlines():
            if line.strip().startswith('CHOICE:'):
                choice = line.split('CHOICE:')[-1].strip().upper()
                break
        if choice in ('A', 'B', 'C', 'D'):
            return {'type': 'submit_answer', 'choice': choice}

    return {'type': 'unknown', 'raw': text}


# ── Dispatcher ────────────────────────────────────────────────────────────────

def dispatch(action: dict) -> str:
    """
    action dict নিয়ে tool চালায়, observation string ফেরত দেয়।
    submit_answer-এর জন্য কিছু করে না — agent_runner handle করবে।
    """
    if action['type'] != 'run_python':
        return "OBSERVATION:\nUnknown action."

    result = execute_python(action['code'])

    if result['timed_out']:
        return (
            "OBSERVATION:\n"
            "Error: Execution timed out (30s). "
            "Your code may contain an infinite loop or a diverging solver. "
            "Rewrite with a finite stopping condition."
        )

    if result['error']:
        lines = result['error'].strip().splitlines()
        short = '\n'.join(lines[-15:])
        return f"OBSERVATION:\nError:\n{short}"

    if not result['stdout']:
        stderr_hint = ''
        if result.get('stderr'):
            stderr_hint = f"\nSolver warnings:\n{result['stderr'][:300]}"
        return "OBSERVATION:\nNo output. Did you forget to print() the answer?" + stderr_hint

    # success — stdout + stderr (lsoda warnings etc.) দুইটাই দাও
    obs = f"OBSERVATION:\nstdout: {result['stdout']}"
    if result.get('stderr'):
        obs += f"\nSolver warnings:\n{result['stderr'][:500]}"
    return obs