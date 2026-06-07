import random
import numpy as np


def solve_dominant_balance(integral_data, epsilon, regime='auto'):
    poly = integral_data['poly']
    L    = float(integral_data['L'])
    eps  = float(epsilon)

    degrees = sorted(poly.keys())
    d_min   = degrees[0]
    d_max   = degrees[-1]
    c_min   = poly[d_min]
    c_max   = poly[d_max]

    def small_approx():
        return c_max ** (-1.0 / d_max) / eps ** (1.0 - 1.0 / d_max)

    def large_approx():
        return c_min ** (-1.0 / d_min) / eps ** (1.0 - 1.0 / d_min)

    def vlarge_approx():
        return L / eps

    if regime == 'small':
        return small_approx()
    if regime == 'large':
        return large_approx()
    if regime == 'very_large':
        return vlarge_approx()

    w_small = (eps / c_max) ** (1.0 / d_max)
    if w_small > L:
        return vlarge_approx()
    return small_approx() if eps < 1.0 else large_approx()


DEGREE_POOLS = {
    'small':      [(2, 6), (2, 8), (2, 10), (2, 12), (2, 18),
                   (3, 6), (3, 8), (4, 8),  (4, 12), (6, 10)],
    'large':      [(2, 6), (2, 8), (3, 6),  (3, 8),  (4, 8)],
    'very_large': [(2, 6), (2, 8), (3, 6),  (4, 8)],
}

REGIME_EPSILON = {
    'small':      (0.001, 0.05),
    'large':      (2.0,   10.0),
    'very_large': (10.0,  100.0),
}


def _sample_params():
    regime  = random.choice(['small', 'large', 'very_large'])
    degrees = list(random.choice(DEGREE_POOLS[regime]))
    if len(degrees) == 2 and random.random() < 0.4:
        mid = random.randint(degrees[0] + 1, degrees[1] - 1)
        if mid not in degrees:
            degrees.append(mid)
    degrees = sorted(set(degrees))
    poly    = {d: round(random.uniform(1.0, 10.0), 1) for d in degrees}
    L       = round(random.uniform(5.0, 80.0), 1)
    eps_lo, eps_hi = REGIME_EPSILON[regime]
    epsilon = round(random.uniform(eps_lo, eps_hi), 3)
    poly_str = {str(k): v for k, v in poly.items()}
    return {
        "poly":    poly_str,
        "L":       L,
        "epsilon": epsilon,
        "regime":  regime,
    }


def _build_question_text(params):
    poly_str = params['poly']
    terms = []
    for d_str in sorted(poly_str.keys(), key=lambda x: int(x)):
        d = int(d_str)
        c = poly_str[d_str]
        terms.append(f"{c} x^{{{d}}}")
    poly_latex   = " + ".join(terms)
    L            = params['L']
    epsilon      = params['epsilon']
    regime       = params['regime']
    regime_label = regime.replace("_", "-")
    return (
        f"Consider the integral $I(\\epsilon) = \\int_0^{{{L:.2f}}} "
        f"\\frac{{1}}{{\\epsilon + {poly_latex}}} dx$. "
        f"In the {regime_label}-$\\epsilon$ regime, using dominant balance "
        f"with the {'highest' if regime == 'small' else 'lowest'}-degree term of $P(x)$, "
        f"what is the numerical value of the analytical approximation $I(\\epsilon)$ "
        f"at $\\epsilon = {epsilon}$ (rounded to 2 decimals)?"
    )


def _make_wrong_options(correct, n=3, min_gap=0.03):
    wrong    = []
    attempts = 0
    while len(wrong) < n and attempts < 2000:
        attempts += 1
        factor    = random.uniform(0.3, 0.7)
        direction = random.choice([-1, 1])
        candidate = round(max(0.01, correct * (1 + direction * factor)), 2)
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
        "domain":   "dominant_balance",
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
    max_total = n * 30

    print(f"Generating {n} dominant_balance questions ...")

    while len(questions) < n and attempts < max_total:
        attempts += 1
        params = _sample_params()

        try:
            int_poly      = {int(k): v for k, v in params['poly'].items()}
            integral_data = {"poly": int_poly, "L": params["L"]}
            result        = solve_dominant_balance(integral_data, params["epsilon"], params["regime"])
            I_val         = round(float(result), 2)
        except Exception as e:
            print(f"  [skip] solver error: {e}")
            continue

        if not np.isfinite(I_val) or I_val <= 0 or I_val < 0.05:
            continue

        qid = f"DOMBAL_{idx:02d}"
        try:
            q, answer = _build_question(qid, params, I_val)
        except RuntimeError as e:
            print(f"  [skip] option generation error: {e}")
            continue

        questions.append(q)
        answers[qid] = answer
        print(f"  [{idx:02d}/{n}]  I={I_val:.2f}  regime={params['regime']}  answer={answer}")
        idx += 1

    if len(questions) < n:
        raise RuntimeError(f"Only generated {len(questions)}/{n} questions after {attempts} attempts.")

    return questions, answers
