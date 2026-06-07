"""
generator.py

Usage:
    python generator.py --stoch 50 --blowup 50 --dom 50 --nondim 50 --laplace 50

Generates math benchmark MCQs across all requested domains, merges them into
a single question bank, and writes separate questions/answers JSON files.
"""

import argparse
import json
from pathlib import Path

from solvers import stochastic_reactor, ode_blowup, dominant_balance, nondim, laplace_method


DOMAIN_MODULES = {
    "stoch":   (stochastic_reactor, "STOCH"),
    "blowup":  (ode_blowup,         "BLOWUP"),
    "dom":     (dominant_balance,   "DOMBAL"),
    "nondim":  (nondim,             "NONDIM"),
    "laplace": (laplace_method,     "LAPLACE"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate math benchmark MCQs across multiple domains."
    )
    parser.add_argument("--stoch",         type=int, default=0, help="Number of stochastic_reactor questions")
    parser.add_argument("--blowup",        type=int, default=0, help="Number of ode_blowup questions")
    parser.add_argument("--dom",           type=int, default=0, help="Number of dominant_balance questions")
    parser.add_argument("--nondim",        type=int, default=0, help="Number of nondim questions")
    parser.add_argument("--laplace",       type=int, default=0, help="Number of laplace_method questions")
    parser.add_argument("--seed",          type=int, default=42, help="Global random seed (default 42)")
    parser.add_argument("--include-e",     action="store_true", default=False,
                        help="Add 'None of the Above' as option E to every question")
    parser.add_argument("--questions-out", default=None,
                        help="Output path for questions JSON (auto-named based on --include-e if omitted)")
    parser.add_argument("--answers-out",   default=None,
                        help="Output path for answers JSON (auto-named based on --include-e if omitted)")
    args = parser.parse_args()

    domain_counts = {
        "stoch":   args.stoch,
        "blowup":  args.blowup,
        "dom":     args.dom,
        "nondim":  args.nondim,
        "laplace": args.laplace,
    }

    total = sum(domain_counts.values())
    if total == 0:
        print("No questions requested. Use --stoch N, --blowup N, etc.")
        return

    all_questions = []
    all_answers   = {}
    global_idx    = 1

    for key, n in domain_counts.items():
        if n == 0:
            continue

        module, prefix = DOMAIN_MODULES[key]
        print(f"\n{'='*50}")
        questions, answers = module.generate(n=n, seed=args.seed)

        # Re-index with global continuous counter, keeping domain prefix
        id_map = {}
        for q in questions:
            new_id      = f"{prefix}_{global_idx:02d}"
            id_map[q["id"]] = new_id
            q["id"]     = new_id
            global_idx += 1

        for old_id, answer in answers.items():
            all_answers[id_map[old_id]] = answer

        all_questions.extend(questions)

    if args.include_e:
        for q in all_questions:
            q["options"]["E"] = "None of the Above"

    # Resolve output paths (auto-name if not specified)
    suffix = "_with_E" if args.include_e else ""
    q_path = Path(args.questions_out) if args.questions_out else Path(f"questions/all_questions{suffix}.json")
    a_path = Path(args.answers_out)   if args.answers_out   else Path(f"answers/all_answers{suffix}.json")
    q_path.parent.mkdir(parents=True, exist_ok=True)
    a_path.parent.mkdir(parents=True, exist_ok=True)

    q_path.write_text(json.dumps(all_questions, indent=2, ensure_ascii=False), encoding="utf-8")
    a_path.write_text(json.dumps(all_answers,   indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"Done. {len(all_questions)} questions -> {q_path}")
    print(f"Done. {len(all_answers)} answers   -> {a_path}")


if __name__ == "__main__":
    main()
