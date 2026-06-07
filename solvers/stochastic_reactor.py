import random
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.stats import gamma as gamma_dist
from scipy.optimize import brentq


def solve_stochastic_reactor(params):
    k_fn = eval(f"lambda t: {params['k_expr']}", {"np": np})
    S_fn = eval(f"lambda t: {params['S_expr']}", {"np": np})

    C0     = float(params['C0'])
    t_end  = float(params['t_end'])
    t_lo   = float(params['t_lo'])
    t_hi   = float(params['t_hi'])
    thresh = float(params['threshold'])
    alpha  = float(params['gamma_alpha'])
    beta   = float(params['gamma_beta'])

    def rhs(t, C):
        return [-k_fn(t) * C[0] + S_fn(t)]

    sol = solve_ivp(rhs, (0.0, t_end), [C0],
                    method='DOP853', rtol=1e-12, atol=1e-14,
                    dense_output=True)
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    def C(t):
        return sol.sol(t)[0]

    def g(t):
        return C(t) - thresh

    grid  = np.linspace(t_lo, t_hi, 40001)
    gvals = g(grid)
    idx   = np.where(np.sign(gvals[:-1]) * np.sign(gvals[1:]) < 0)[0]
    roots = [brentq(g, grid[i], grid[i + 1], xtol=1e-14) for i in idx]

    breakpoints = [t_lo] + roots + [t_hi]
    valid = [
        (a, b)
        for a, b in zip(breakpoints[:-1], breakpoints[1:])
        if C(0.5 * (a + b)) > thresh
    ]

    rv = gamma_dist(a=alpha, scale=1.0 / beta)
    P  = sum(
        quad(rv.pdf, a, b, epsabs=1e-14, epsrel=1e-12, limit=200)[0]
        for a, b in valid
    )

    return {
        'P':               round(P, 6),
        'valid_intervals': valid,
        'zero_crossings':  roots,
    }


K_TEMPLATES = [
    ("{a} + {b}*np.sin(t)",          ["a", "b"],       [(0.2, 0.8), (0.1, 0.4)]),
    ("{a} + {b}*np.cos(t)",          ["a", "b"],       [(0.2, 0.8), (0.1, 0.4)]),
    ("{a} + {b}*np.exp(-{c}*t)",     ["a", "b", "c"],  [(0.2, 0.8), (0.1, 0.4), (0.05, 0.3)]),
    ("{a} + {b}*t",                  ["a", "b"],       [(0.3, 0.8), (0.02, 0.1)]),
    ("{a} / (1.0 + {b}*t)",          ["a", "b"],       [(0.5, 1.5), (0.05, 0.3)]),
    ("{a} + {b}*np.sin({c}*t)",      ["a", "b", "c"],  [(0.2, 0.8), (0.1, 0.4), (0.5, 2.0)]),
    ("{a} + {b}*np.cos({c}*t)",      ["a", "b", "c"],  [(0.2, 0.8), (0.1, 0.4), (0.5, 2.0)]),
    ("{a} + {b}*np.tanh({c}*t)",     ["a", "b", "c"],  [(0.2, 0.8), (0.1, 0.4), (0.1, 0.5)]),
]

S_TEMPLATES = [
    ("{c}*np.exp(-{d}*t)*np.cos({e}*t)",    ["c","d","e"], [(1.0,6.0),(0.02,0.2),(0.3,1.5)]),
    ("{c}*np.exp(-{d}*t)*np.sin({e}*t)",    ["c","d","e"], [(1.0,6.0),(0.02,0.2),(0.3,1.5)]),
    ("{c}*np.exp(-{d}*t)",                  ["c","d"],     [(1.0,6.0),(0.02,0.2)]),
    ("{c}*np.sin({e}*t)",                   ["c","e"],     [(1.0,5.0),(0.3,1.5)]),
    ("{c}*np.cos({e}*t)",                   ["c","e"],     [(1.0,5.0),(0.3,1.5)]),
    ("{c} / (1.0 + {d}*t)",                 ["c","d"],     [(2.0,6.0),(0.05,0.3)]),
    ("{c}*np.exp(-{d}*t) + {f}*np.cos({e}*t)",
                                            ["c","d","e","f"],
                                            [(1.0,4.0),(0.05,0.2),(0.3,1.2),(0.5,2.0)]),
    ("{c}*(1.0 - np.exp(-{d}*t))*np.cos({e}*t)",
                                            ["c","d","e"], [(1.0,5.0),(0.1,0.5),(0.3,1.5)]),
]

QUESTION_TEMPLATE = (
    "A chemical reactor's concentration C(t) evolves under the ODE "
    "dC/dt = -k(t)*C(t) + S(t), C(0) = {C0}, "
    "with k(t) = {k_expr} and S(t) = {S_expr}. "
    "A sensor measures the concentration at a random time T ~ Gamma(alpha={gamma_alpha}, beta={gamma_beta}) "
    "(rate parameterization). "
    "A reading is valid iff T in [{t_lo}, {t_hi}] and C(T) > {threshold}. "
    "Compute P = Pr(valid) to 6 decimal places."
)


def _sample_expr(templates):
    tmpl_str, names, ranges = random.choice(templates)
    coeffs = {n: round(random.uniform(*r), 4) for n, r in zip(names, ranges)}
    return tmpl_str.format(**coeffs)


def _sample_params():
    t_lo   = round(random.uniform(1.0, 4.0), 2)
    window = round(random.uniform(3.0, 6.0), 2)
    t_hi   = round(t_lo + window, 2)
    t_end  = round(t_hi + 1.0, 2)
    return {
        "k_expr":      _sample_expr(K_TEMPLATES),
        "S_expr":      _sample_expr(S_TEMPLATES),
        "C0":          round(random.uniform(0.5, 5.0), 4),
        "t_end":       t_end,
        "gamma_alpha": round(random.uniform(2.0, 6.0), 4),
        "gamma_beta":  round(random.uniform(0.4, 1.5), 4),
        "t_lo":        t_lo,
        "t_hi":        t_hi,
        "threshold":   round(random.uniform(0.3, 2.0), 4),
    }


def _make_wrong_options(correct, n=3, min_gap=0.005):
    wrong = []
    attempts = 0
    while len(wrong) < n and attempts < 2000:
        attempts += 1
        factor    = random.uniform(0.3, 0.7)
        direction = random.choice([-1, 1])
        candidate = round(max(1e-6, min(correct * (1 + direction * factor), 1.0 - 1e-6)), 6)
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
        "domain":   "stochastic_reactor",
        "question": QUESTION_TEMPLATE.format(**params),
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
    max_total = n * 30

    print(f"Generating {n} stochastic_reactor questions ...")

    while len(questions) < n and attempts < max_total:
        attempts += 1
        params = _sample_params()
        try:
            result = solve_stochastic_reactor(params)
            P = result['P']
        except Exception as e:
            print(f"  [skip] solver error: {e}")
            continue

        if P < 0.02 or P > 0.98:
            continue

        qid = f"STOCH_{idx:02d}"
        try:
            q, answer = _build_question(qid, params, P)
        except RuntimeError as e:
            print(f"  [skip] option generation error: {e}")
            continue

        questions.append(q)
        answers[qid] = answer
        print(f"  [{idx:02d}/{n}]  P={P:.6f}  answer={answer}")
        idx += 1

    if len(questions) < n:
        raise RuntimeError(f"Only generated {len(questions)}/{n} questions after {attempts} attempts.")

    return questions, answers
