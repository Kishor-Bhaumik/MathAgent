"""
agent_runner.py
===============
Standard ReAct agentic loop for mathagent.

- T_max = 10 steps
- Tools: run_python, submit_answer
- Option E is HIDDEN from the model during the loop.
  If T_max is exhausted without an answer → E is assigned by this code.
- Works for both:
    * Anthropic / OpenAI real-time API  (pass call_model=your_api_function)
    * Local HuggingFace models          (pass call_model=your_local_function)
"""

from tools import TOOL_DESCRIPTIONS, parse_action, dispatch

T_MAX = 10

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are an applied mathematics expert.
Solve the given problem by writing and executing Python code.
Available libraries: numpy (np), scipy, sympy (sp).

{TOOL_DESCRIPTIONS}

Rules:
- You MUST use run_python for any numerical computation. Do NOT answer from memory.
- Call only ONE tool per turn.
- After you see the output, decide: is this the correct answer? If yes, submit_answer.
- The answer is definitely one of A, B, C, or D.
- If your result does not match any option, your code is wrong. Fix it and try again.
"""

# ── Question formatter ────────────────────────────────────────────────────────

def format_question(question: dict) -> str:
    """
    Option E লুকিয়ে A B C D দেখায়।
    """
    opts = {k: v for k, v in sorted(question['options'].items()) if k != 'E'}
    opts_str = '\n'.join(f"{k}. {v}" for k, v in opts.items())
    return f"{question['question']}\n\n{opts_str}"


# ── Main agent loop ───────────────────────────────────────────────────────────

def run_agent(question: dict, call_model, t_max: int = 10) -> dict:
    """
    একটা প্রশ্নের জন্য পুরো agent loop চালায়।

    Parameters
    ----------
    question   : question dict (id, question, options, domain)
    call_model : function(messages: list[dict]) -> str
                 মডেলকে ডাকার function — API বা local, যেটাই হোক।
                 messages = [{'role': 'system'|'user'|'assistant', 'content': str}]

    Returns
    -------
    {
        'id'          : str,
        'domain'      : str,
        'model_answer': 'A'|'B'|'C'|'D'|'E',   # E মানে T_max শেষ
        'termination' : 'submitted'|'hit_T_max',
        'steps'       : int,
        'executions'  : int,
        'trajectory'  : list of step dicts       # failure taxonomy-র জন্য
    }
    """
    # ── ইতিহাস শুরু করো
    messages = [
        {'role': 'system',  'content': SYSTEM_PROMPT},
        {'role': 'user',    'content': format_question(question)},
    ]

    trajectory  = []   # প্রতিটা ধাপের log
    executions  = 0    # কতবার run_python চলল
    final_answer = None
    termination  = 'hit_T_max'

    for step in range(1, t_max + 1):

        # ── মডেলকে ডাকো
        model_output = call_model(messages)
        messages.append({'role': 'assistant', 'content': model_output})

        # ── কী করতে চাইছে parse করো
        action = parse_action(model_output)

        step_log = {
            'step'          : step,
            'action_type'   : action['type'],
            'model_output'  : model_output,
            'observation'   : None,
        }

        # ── submit_answer → থামো
        if action['type'] == 'submit_answer':
            final_answer = action['choice']
            termination  = 'submitted'
            trajectory.append(step_log)
            break

        # ── run_python → চালাও, observation ফেরত দাও
        if action['type'] == 'run_python':
            executions += 1
            observation = dispatch(action)
            step_log['code']        = action.get('code', '')
            step_log['observation'] = observation

            messages.append({'role': 'user', 'content': observation})

        # ── unknown → মডেলকে remind করো
        else:
            reminder = (
                "OBSERVATION:\nI could not parse your action. "
                "Please use exactly:\n"
                "ACTION: run_python\n```python\n...\n```\n"
                "or\nACTION: submit_answer\nCHOICE: <A|B|C|D>"
            )
            step_log['observation'] = reminder
            messages.append({'role': 'user', 'content': reminder})

        trajectory.append(step_log)

    # ── T_max শেষ, উত্তর নেই → E assign করো
    if final_answer is None:
        final_answer = 'E'
        termination  = 'hit_T_max'

    return {
        'id'          : question['id'],
        'domain'      : question['domain'],
        'model_answer': final_answer,
        'termination' : termination,
        'steps'       : len(trajectory),
        'executions'  : executions,
        'trajectory'  : trajectory,
    }
