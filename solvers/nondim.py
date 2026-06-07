import random
import numpy as np
from scipy.optimize import brentq
from sympy import Poly, sympify
from sympy.abc import x as sym_x


def nondimensionalize(terms):
    if len(terms) != 3:
        raise ValueError("Exactly 3 terms required.")

    terms_sorted = sorted(terms, key=lambda t: t[1], reverse=True)
    (a1, d1), (a2, d2), (a3, d3) = terms_sorted

    if d1 != 10:
        raise ValueError("Leading term must have degree 10.")

    if a1 < 0:
        a1, a2, a3 = -a1, -a2, -a3

    k     = d2
    scale = (abs(a3) / abs(a2)) ** (1.0 / k)
    epsilon = abs(a1) * (abs(a3) / abs(a2)) ** (d1 / k) / abs(a3)

    sign2 = int(np.sign(a2))
    sign3 = int(np.sign(a3))

    nondim_coeffs  = [epsilon, -sign2, sign3]
    nondim_degrees = [d1, d2, 0]

    orig_poly = np.zeros(d1 + 1)
    orig_poly[d1 - d1] = a1
    orig_poly[d1 - d2] = a2
    orig_poly[d1 - d3] = a3

    ndim_poly = np.zeros(d1 + 1)
    ndim_poly[0]       = epsilon
    ndim_poly[d1 - d2] = -sign2
    ndim_poly[d1]      = sign3

    def find_real_roots_brentq(coeffs, x_min=-10.0, x_max=10.0, n_grid=10000):
        f  = np.poly1d(coeffs)
        xs = np.linspace(x_min, x_max, n_grid)
        ys = f(xs)
        roots = []
        for i in range(len(xs) - 1):
            if ys[i] * ys[i + 1] < 0:
                try:
                    r = brentq(f, xs[i], xs[i + 1], xtol=1e-10)
                    roots.append(r)
                except ValueError:
                    pass
        return np.array(roots)

    ndim_roots   = find_real_roots_brentq(ndim_poly)
    mapped_roots = ndim_roots * scale
    orig_f       = np.poly1d(orig_poly)
    poly_scale   = abs(a1) * (scale ** d1)
    residuals    = (
        np.abs(orig_f(mapped_roots)) / poly_scale
        if len(mapped_roots) > 0 else np.array([])
    )
    verified = bool(len(residuals) == 0 or np.all(residuals < 1e-6))

    return {
        'epsilon':        round(epsilon, 6),
        'nondim_coeffs':  nondim_coeffs,
        'nondim_degrees': nondim_degrees,
        'scale':          scale,
        'verified':       verified,
        'residuals':      residuals.tolist(),
    }


def _parse_polynomial(poly_str):
    expr = sympify(poly_str)
    poly = Poly(expr, sym_x)
    return [(int(coef), deg[0]) for coef, deg in zip(poly.coeffs(), poly.monoms())]


def _sample_params():
    d2 = random.randint(1, 9)
    d3 = random.randint(0, d2 - 1)

    def nonzero_int(lo=-15, hi=15):
        v = 0
        while v == 0:
            v = random.randint(lo, hi)
        return v

    a1 = nonzero_int()
    a2 = nonzero_int()
    a3 = nonzero_int()

    def term_str(coeff, deg):
        if deg == 0:
            return str(coeff)
        if deg == 1:
            return f"{coeff}*x"
        return f"{coeff}*x**{deg}"

    parts    = [term_str(a1, 10), term_str(a2, d2), term_str(a3, d3)]
    poly_str = " + ".join(parts).replace("+ -", "- ")

    return {
        "polynomial": poly_str,
        "_terms":     [(a1, 10), (a2, d2), (a3, d3)],
    }


def _build_question_text(params):
    poly_str   = params['polynomial']
    poly_latex = (
        poly_str
        .replace("**", "^")
        .replace("*x", "x")
        .replace("* x", "x")
    )
    return (
        f"Nondimensionalize the polynomial P(x) = {poly_latex} into a polynomial "
        f"of the form ε·y^10 ± y^k ± 1. "
        f"Solve for ε, rounded to 2 decimal places."
    )


def _make_wrong_options(correct, n=3, min_gap=0.03):
    wrong    = []
    attempts = 0
    while len(wrong) < n and attempts < 2000:
        attempts += 1
        factor    = random.uniform(0.3, 0.7)
        direction = random.choice([-1, 1])
        candidate = abs(correct * (1 + direction * factor))
        candidate = round(max(0.01, candidate), 2)
        if abs(candidate - correct) < min_gap:
            continue
        if any(abs(candidate - w) < min_gap for w in wrong):
            continue
        wrong.append(candidate)

    if len(wrong) < n:
        raise RuntimeError(f"Could not generate {n} distinct wrong options for correct={correct}")

    n_negative   = random.randint(1, n)
    flip_indices = random.sample(range(n), n_negative)
    for i in flip_indices:
        wrong[i] = -wrong[i]

    return wrong


def _build_question(qid, params, correct):
    wrong          = _make_wrong_options(correct, n=3)
    options_values = wrong + [correct]
    random.shuffle(options_values)
    labels  = ["A", "B", "C", "D"]
    options = {lbl: val for lbl, val in zip(labels, options_values)}
    answer  = next(lbl for lbl, val in options.items() if val == correct)
    pub_params = {k: v for k, v in params.items() if not k.startswith("_")}
    question = {
        "id":       qid,
        "domain":   "nondim",
        "question": _build_question_text(params),
        "params":   pub_params,
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

    print(f"Generating {n} nondim questions ...")

    while len(questions) < n and attempts < max_total:
        attempts += 1
        params = _sample_params()

        try:
            result  = nondimensionalize(params['_terms'])
            epsilon = round(float(result['epsilon']), 2)
        except Exception as e:
            print(f"  [skip] solver error: {e}")
            continue

        if not np.isfinite(epsilon) or epsilon <= 0 or epsilon < 0.05 or epsilon > 1e6:
            continue

        qid = f"NONDIM_{idx:02d}"
        try:
            q, answer = _build_question(qid, params, epsilon)
        except RuntimeError as e:
            print(f"  [skip] option generation error: {e}")
            continue

        questions.append(q)
        answers[qid] = answer
        print(f"  [{idx:02d}/{n}]  eps={epsilon:.2f}  answer={answer}  verified={result['verified']}")
        idx += 1

    if len(questions) < n:
        raise RuntimeError(f"Only generated {len(questions)}/{n} questions after {attempts} attempts.")

    return questions, answers
