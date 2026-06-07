import random
import numpy as np
from scipy.integrate import solve_ivp


def find_blowup(
    ode_rhs,
    initial_conditions,
    t_span=(0, 5.0),
    blowup_index=1,
    threshold=1e6,
    has_origin_singularity=False,
    x0_bootstrap=1e-6,
    max_step=1e-4,
):
    order = len(initial_conditions)

    def system(x, u):
        dudt = list(u[1:])
        dudt.append(ode_rhs(x, u))
        return dudt

    def blowup_event(x, u):
        return abs(u[blowup_index]) - threshold
    blowup_event.terminal  = True
    blowup_event.direction = 1

    if has_origin_singularity:
        x0      = x0_bootstrap
        highest = ode_rhs(x0, initial_conditions)
        u0      = []
        for i in range(order):
            val = initial_conditions[i]
            if i + 1 < order:
                val += initial_conditions[i + 1] * x0
            if i + 2 < order:
                val += 0.5 * initial_conditions[i + 2] * x0 ** 2
            val += (1.0 / 6.0) * highest * x0 ** (3 - i) if (3 - i) > 0 else 0.0
            u0.append(val)
        t_start = x0
    else:
        u0      = list(initial_conditions)
        t_start = t_span[0]

    sol = solve_ivp(
        system,
        [t_start, t_span[1]],
        u0,
        method='Radau',
        events=blowup_event,
        rtol=1e-10,
        atol=1e-12,
        max_step=max_step,
    )

    if sol.t_events[0].size > 0:
        return round(float(sol.t_events[0][0]), 6)
    return round(float(sol.t[-1]), 6)


RHS_TEMPLATES = [
    (
        "x - {a}*{u0} + {b}*{u1}**2 + {c}*{u2}**5",
        ["a","b","c"], [(0.1,1.0),(0.5,3.0),(0.5,3.0)],
        3, False
    ),
    (
        "{a}*x**2 - {b}*{u0}**3 + {c}*{u1}**4",
        ["a","b","c"], [(0.1,1.0),(0.1,1.0),(0.5,3.0)],
        2, False
    ),
    (
        "x + {a}*{u0}**2 + {b}*{u1}**3 + {c}*{u2}**3",
        ["a","b","c"], [(0.1,1.0),(0.1,1.0),(0.1,1.0)],
        3, False
    ),
    (
        "{a}*{u0}**2 + {b}*{u1}**2 + {c}*{u2}**2 + {d}*{u3}**3",
        ["a","b","c","d"], [(0.1,1.0),(0.1,1.0),(0.1,1.0),(0.1,1.0)],
        4, False
    ),
    (
        "x**2 + {a}*{u1}**3 + {b}*{u2}**4",
        ["a","b"], [(0.1,1.0),(0.1,1.0)],
        3, False
    ),
    (
        "{a}*{u0}*{u1} + {b}*{u2}**3",
        ["a","b"], [(0.1,2.0),(0.5,3.0)],
        3, False
    ),
    (
        "x - {a}*{u0}*{u1} + {b}*{u1}**2 + {c}*{u2}**3",
        ["a","b","c"], [(0.1,1.0),(0.1,1.0),(0.5,3.0)],
        3, False
    ),
    (
        "{a}*{u0}**2*{u1} + {b}*{u2}**3",
        ["a","b"], [(0.1,1.0),(0.5,3.0)],
        3, False
    ),
    (
        "{a}*{u1}*{u2} + {b}*{u0}**3 + {c}*x",
        ["a","b","c"], [(0.5,3.0),(0.1,1.0),(0.1,1.0)],
        3, False
    ),
    (
        "{a}*{u0}*{u2} + {b}*{u1}**4",
        ["a","b"], [(0.1,1.0),(0.1,2.0)],
        3, False
    ),
    (
        "{a}*{u1}**2*{u3} + {b}*{u2}**3",
        ["a","b"], [(0.1,1.0),(0.1,1.0)],
        4, False
    ),
    (
        "np.sin({a}*{u1}) + {b}*{u2}**4",
        ["a","b"], [(0.5,2.0),(0.5,3.0)],
        3, False
    ),
    (
        "x + np.cos({a}*{u0}) + {b}*{u1}**3",
        ["a","b"], [(0.5,2.0),(0.5,3.0)],
        2, False
    ),
    (
        "np.sin({a}*x)*{u0} + {b}*{u1}**3 + {c}*{u2}**2",
        ["a","b","c"], [(0.5,2.0),(0.5,3.0),(0.1,1.0)],
        3, False
    ),
    (
        "np.tanh({a}*{u1}) + {b}*{u2}**5",
        ["a","b"], [(0.5,2.0),(0.5,3.0)],
        3, False
    ),
    (
        "np.sin({a}*{u0})*np.cos({b}*{u1}) + {c}*{u2}**3",
        ["a","b","c"], [(0.5,2.0),(0.5,2.0),(0.5,3.0)],
        3, False
    ),
    (
        "np.exp({a}*{u1}) - {b}*{u0} + {c}*x",
        ["a","b","c"], [(0.1,0.5),(0.1,1.0),(0.1,1.0)],
        2, False
    ),
    (
        "{a}*np.exp({b}*{u2}) + {c}*{u1}**2",
        ["a","b","c"], [(0.1,1.0),(0.1,0.5),(0.1,1.0)],
        3, False
    ),
    (
        "np.exp({a}*{u1}) + {b}*{u2}**3 - {c}*{u0}",
        ["a","b","c"], [(0.1,0.4),(0.1,1.0),(0.1,1.0)],
        3, False
    ),
    (
        "x - {a}*{u0}/(x**2 + {b}) + {c}*{u1}**2 + {d}*{u2}**3",
        ["a","b","c","d"], [(0.1,1.0),(0.5,2.0),(0.5,2.0),(0.5,3.0)],
        3, False
    ),
    (
        "{a}*{u1}**2 / ({b} + {u0}**2) + {c}*{u2}**4",
        ["a","b","c"], [(0.5,2.0),(0.5,2.0),(0.5,3.0)],
        3, False
    ),
    (
        "{a}*x / (x**2 + {b}) + {c}*{u1}**3 + {d}*{u2}**2",
        ["a","b","c","d"], [(0.5,2.0),(0.5,2.0),(0.1,1.0),(0.5,3.0)],
        3, False
    ),
    (
        "{a}/{u0_safe} + {b}*{u1}**3",
        ["a","b"], [(0.1,1.0),(0.5,3.0)],
        2, True
    ),
    (
        "x - {a}*{u0}/({b}*x + {eps}) + {c}*{u1}**2 + {d}*{u2}**5",
        ["a","b","c","d","eps"], [(0.1,1.0),(1.0,3.0),(0.5,2.0),(0.5,3.0),(1e-6,1e-6)],
        3, True
    ),
    (
        "{a}/{xsafe} + {b}*{u0}**2 + {c}*{u2}**3",
        ["a","b","c"], [(0.1,1.0),(0.1,1.0),(0.5,3.0)],
        3, True
    ),
]

DERIV_NOTATION = {2: "''", 3: "'''", 4: "''''"}
ORDER_NAMES    = {2: "second", 3: "third", 4: "fourth"}


def _build_rhs_expr(tmpl_str, coeffs, order):
    expr = tmpl_str
    expr = expr.replace("{u0_safe}", "(u[0] + 1e-12)")
    expr = expr.replace("{xsafe}",   "(x + 1e-12)")
    for i in reversed(range(order)):
        expr = expr.replace(f"{{u{i}}}", f"u[{i}]")
    expr = expr.format(**coeffs)
    return expr


def _sample_rhs(order):
    compatible = [t for t in RHS_TEMPLATES if t[3] <= order]
    if not compatible:
        raise RuntimeError(f"No templates for order {order}")
    tmpl_str, names, ranges, _, has_sing = random.choice(compatible)
    coeffs = {n: round(random.uniform(*r), 4) for n, r in zip(names, ranges)}
    expr   = _build_rhs_expr(tmpl_str, coeffs, order)
    return expr, has_sing


def _build_question_text(params):
    order       = len(params["initial_conditions"])
    deriv_marks = DERIV_NOTATION[order]
    blowup_idx  = params["blowup_index"]
    blowup_marks = {0: "", 1: "'", 2: "''", 3: "'''"}[blowup_idx]
    ic_parts = []
    prime = ""
    for i, v in enumerate(params["initial_conditions"]):
        ic_parts.append(f"y{prime}(0)={v}")
        prime += "'"
    ic_str = ", ".join(ic_parts)
    return (
        f"Consider the {ORDER_NAMES[order]}-order ODE: "
        f"y{deriv_marks} = {params['ode_rhs']}, "
        f"with initial conditions {ic_str}. "
        f"The solution y{blowup_marks}(x) blows up at a finite point x*. "
        f"Find x*, rounded to 6 decimal places."
    )


def _make_wrong_options(correct, n=3, min_gap=0.001):
    wrong    = []
    attempts = 0
    sign     = 1.0 if correct >= 0 else -1.0
    while len(wrong) < n and attempts < 2000:
        attempts += 1
        factor    = random.uniform(0.3, 0.7)
        direction = random.choice([-1, 1])
        candidate = correct * (1 + direction * factor)
        if sign > 0 and candidate <= 0:
            candidate = abs(candidate)
        elif sign < 0 and candidate >= 0:
            candidate = -abs(candidate)
        candidate = round(candidate, 6)
        if abs(candidate - correct) < min_gap:
            continue
        if any(abs(candidate - w) < min_gap for w in wrong):
            continue
        wrong.append(candidate)
    if len(wrong) < n:
        raise RuntimeError(f"Could not generate {n} distinct wrong options for correct={correct}")
    return wrong


def _build_question(qid, params, correct):
    wrong          = _make_wrong_options(correct, n=3)
    options_values = wrong + [correct]
    random.shuffle(options_values)
    labels  = ["A", "B", "C", "D"]
    options = {lbl: val for lbl, val in zip(labels, options_values)}
    answer  = next(lbl for lbl, val in options.items() if val == correct)
    question = {
        "id":       qid,
        "domain":   "ode_blowup",
        "question": _build_question_text(params),
        "params":   params,
        "options":  options,
    }
    return question, answer


def generate(n=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    questions = []
    answers   = {}
    idx       = 1
    attempts  = 0
    max_total = n * 50

    print(f"Generating {n} ode_blowup questions ...")

    while len(questions) < n and attempts < max_total:
        attempts += 1
        order            = random.choice([2, 3, 4])
        rhs_expr, has_sing = _sample_rhs(order)
        ics              = [round(random.uniform(0.5, 3.0), 4) for _ in range(order)]
        t_end            = round(random.uniform(2.0, 8.0), 2)
        t_span           = (0, t_end)
        blowup_index     = random.randint(0, order - 1)
        threshold        = 10 ** random.uniform(4, 8)
        max_step         = {2: 1e-3, 3: 1e-4, 4: 1e-4}[order]

        try:
            rhs_fn   = eval(f"lambda x, u: {rhs_expr}", {"np": np})
            test_val = rhs_fn(0.1, ics)
            if not np.isfinite(test_val):
                continue
        except Exception:
            continue

        params = {
            "ode_rhs":                rhs_expr,
            "initial_conditions":     ics,
            "blowup_index":           blowup_index,
            "has_origin_singularity": has_sing,
            "t_span":                 list(t_span),
            "threshold":              threshold,
        }

        try:
            x_star = find_blowup(
                ode_rhs=rhs_fn,
                initial_conditions=ics,
                t_span=t_span,
                blowup_index=blowup_index,
                threshold=threshold,
                has_origin_singularity=has_sing,
                max_step=max_step,
            )
        except Exception as e:
            print(f"  [skip] solver error: {e}")
            continue

        if x_star >= t_span[1] - 1e-3:
            continue
        if abs(x_star) < 0.01:
            continue

        qid = f"BLOWUP_{idx:02d}"
        try:
            q, answer = _build_question(qid, params, x_star)
        except RuntimeError as e:
            print(f"  [skip] option generation error: {e}")
            continue

        questions.append(q)
        answers[qid] = answer
        print(f"  [{idx:02d}/{n}]  x*={x_star:.6f}  order={order}  answer={answer}")
        idx += 1

    if len(questions) < n:
        raise RuntimeError(f"Only generated {len(questions)}/{n} questions after {attempts} attempts.")

    return questions, answers
