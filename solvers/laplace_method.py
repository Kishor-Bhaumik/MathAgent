import random
import numpy as np
import sympy as sp
from scipy.optimize import dual_annealing
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

SP_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def solve_laplace_method(integral_data):
    t    = sp.symbols('t', real=True)
    locs = {'t': t, 'sin': sp.sin, 'cos': sp.cos, 'atan': sp.atan}

    g_expr = parse_expr(integral_data['g'], local_dict=locs, transformations=SP_TRANSFORMS)
    f_expr = parse_expr(integral_data['f'], local_dict=locs, transformations=SP_TRANSFORMS)

    a = float(integral_data['a'])
    b = float(integral_data['b'])
    k = int(integral_data['k'])

    f_func = sp.lambdify(t, f_expr, 'numpy')
    g_func = sp.lambdify(t, g_expr, 'numpy')

    def objective(tv):
        return -k * float(f_func(tv[0]))

    result      = dual_annealing(objective, bounds=[(a, b)], seed=42, maxiter=200)
    t0_interior = float(result.x[0])

    candidates = [a, b, t0_interior]
    cand_vals  = [-k * float(f_func(c)) for c in candidates]
    t0         = candidates[int(np.argmin(cand_vals))]

    tol         = 1e-6
    at_boundary = abs(t0 - a) < tol or abs(t0 - b) < tol

    g_at = float(g_func(t0))
    f_at = float(f_func(t0))

    fprime_expr = sp.diff(f_expr, t)
    fp_func     = sp.lambdify(t, fprime_expr, 'numpy')

    if at_boundary:
        fp_at = float(fp_func(t0))
        if abs(fp_at) > 1e-12:
            return {'A': g_at / fp_at, 'c': k * f_at, 'form': 'boundary', 't0': t0}

    fdouble_expr = sp.diff(f_expr, t, 2)
    fpp_func     = sp.lambdify(t, fdouble_expr, 'numpy')
    fpp_at       = float(fpp_func(t0))

    return {
        'A':    g_at * np.sqrt(2.0 / abs(fpp_at)),
        'c':    k * f_at,
        'form': 'interior',
        't0':   t0,
    }


G_TEMPLATES = [
    ("{a}*t**5 + {b}*t**3 + {c}*t**2 + {d}*sin(t)",
     ["a","b","c","d"], [(-1.5,1.5),(-2.0,2.0),(-2.0,2.0),(0.5,3.0)]),
    ("{a}*t**4 + {b}*t**2 + {c}*cos(t)",
     ["a","b","c"], [(-2.0,2.0),(-2.0,2.0),(0.5,3.0)]),
    ("{a}*t**3 + {b}*t + {c}*sin(t) + {d}*cos(t)",
     ["a","b","c","d"], [(-2.0,2.0),(-2.0,2.0),(0.5,2.0),(0.5,2.0)]),
    ("{a}*t**2 + {b}*sin(2*t)",
     ["a","b"], [(-2.0,2.0),(0.5,3.0)]),
    ("{a}*t**4 + {b}*t**3 + {c}*sin(t)",
     ["a","b","c"], [(-1.5,1.5),(-1.5,1.5),(0.5,3.0)]),
    ("{a}*t**3 + {b}*cos(t) + {c}*t",
     ["a","b","c"], [(-2.0,2.0),(0.5,3.0),(-2.0,2.0)]),
    ("{a}*t**5 + {b}*t**2 + {c}*cos(t) + {d}*t",
     ["a","b","c","d"], [(-1.5,1.5),(-2.0,2.0),(0.5,2.0),(-1.5,1.5)]),
    ("{a}*sin(t) + {b}*t**2 + {c}*cos(2*t)",
     ["a","b","c"], [(0.5,3.0),(-2.0,2.0),(0.5,2.0)]),
    ("{a}*t**6 + {b}*t**3 + {c}*sin(t)",
     ["a","b","c"], [(-1.0,1.0),(-1.5,1.5),(0.5,3.0)]),
    ("{a}*t + {b}*sin(t)*cos(t)",
     ["a","b"], [(-2.0,2.0),(0.5,3.0)]),
]

F_TEMPLATES = [
    ("{a}*t**5 + {b}*t**4 + {c}*t**2 + {d}*cos(t)",
     ["a","b","c","d"], [(-2.5,2.5),(-1.5,1.5),(-2.0,2.0),(0.5,1.5)]),
    ("{a}*t**4 + {b}*t**3 + {c}*t**2",
     ["a","b","c"], [(-2.0,2.0),(-2.0,2.0),(-3.0,3.0)]),
    ("{a}*t**3 + {b}*t**2 + {c}*sin(t)",
     ["a","b","c"], [(-2.0,2.0),(-2.0,2.0),(0.5,2.0)]),
    ("{a}*t**5 + {b}*t**3 + {c}*cos(t)",
     ["a","b","c"], [(-2.5,2.5),(-2.0,2.0),(0.5,1.5)]),
    ("{a}*t**4 + {b}*t + {c}*cos(t)",
     ["a","b","c"], [(-2.0,2.0),(-2.0,2.0),(0.5,1.5)]),
    ("{a}*t**6 + {b}*t**4 + {c}*t**2 + {d}*sin(t)",
     ["a","b","c","d"], [(-1.5,1.5),(-1.5,1.5),(-2.0,2.0),(0.5,1.5)]),
    ("{a}*t**3 + {b}*sin(t) + {c}*t**2",
     ["a","b","c"], [(-2.0,2.0),(0.5,2.0),(-2.0,2.0)]),
    ("{a}*t**5 + {b}*t**2 + {c}*sin(2*t)",
     ["a","b","c"], [(-2.5,2.5),(-2.0,2.0),(0.5,1.5)]),
    ("{a}*t**4 + {b}*cos(t) + {c}*t**3",
     ["a","b","c"], [(-2.0,2.0),(0.5,1.5),(-2.0,2.0)]),
    ("{a}*t**3 + {b}*t**4 + {c}*cos(2*t)",
     ["a","b","c"], [(-2.0,2.0),(-1.5,1.5),(0.5,1.5)]),
]


def _sample_expr(templates):
    tmpl_str, names, ranges = random.choice(templates)
    coeffs = {}
    for n, r in zip(names, ranges):
        v = round(random.uniform(*r), 1)
        while v == 0.0:
            v = round(random.uniform(*r), 1)
        coeffs[n] = f"{v:.1f}"
    return tmpl_str.format(**coeffs)


def _sample_params():
    a     = round(random.uniform(-1.5, 0.5), 1)
    width = round(random.uniform(0.3, 1.2), 1)
    b     = round(a + width, 1)
    k     = random.choice([-1, 1])
    return {
        "g": _sample_expr(G_TEMPLATES),
        "f": _sample_expr(F_TEMPLATES),
        "a": a,
        "b": b,
        "k": k,
    }


def _build_question_text(params):
    g        = params['g']
    f        = params['f']
    a        = params['a']
    b        = params['b']
    k        = params['k']
    exp_sign = "-" if k == -1 else ""
    return (
        f"Consider the integral \\par \\begin{{equation}} "
        f"I(x) = \\int_{{{a}}}^{{{b}}} ({g}) "
        f"e^{{{exp_sign} x ({f})}} \\, dt "
        f"\\end{{equation}} "
        f"In the limit $x \\to \\infty$, the leading-order asymptotic form is "
        f"$I(x) \\approx \\frac{{A \\cdot e^{{c x}}}}{{x}}$. "
        f"What is the numerical value of the pre-exponential factor $A$ "
        f"(rounded to 2 decimals)?"
    )


def _make_wrong_options(correct, n=3, min_gap=0.03):
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
        candidate = round(candidate, 2)
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
        "domain":   "laplace_method",
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
    max_total = n * 40

    print(f"Generating {n} laplace_method questions ...")

    while len(questions) < n and attempts < max_total:
        attempts += 1
        params = _sample_params()

        try:
            result = solve_laplace_method(params)
            A      = round(float(result['A']), 2)
        except Exception as e:
            print(f"  [skip] solver error: {e}")
            continue

        if not np.isfinite(A) or abs(A) < 0.05:
            continue

        qid = f"LAPLACE_{idx:02d}"
        try:
            q, answer = _build_question(qid, params, A)
        except RuntimeError as e:
            print(f"  [skip] option generation error: {e}")
            continue

        questions.append(q)
        answers[qid] = answer
        print(f"  [{idx:02d}/{n}]  A={A:.2f}  form={result['form']}  answer={answer}")
        idx += 1

    if len(questions) < n:
        raise RuntimeError(f"Only generated {len(questions)}/{n} questions after {attempts} attempts.")

    return questions, answers
