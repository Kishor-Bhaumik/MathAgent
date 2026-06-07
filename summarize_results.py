import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Summarize mathbench result JSON")
    parser.add_argument("result_file", help="Path to results JSON file")
    args = parser.parse_args()

    with open(args.result_file) as f:
        data = json.load(f)

    total = data["total"]
    correct = data["correct"]
    accuracy = correct / total * 100 if total else 0

    domain_total = defaultdict(int)
    domain_correct = defaultdict(int)
    for entry in data["details"]:
        domain = entry["domain"]
        domain_total[domain] += 1
        if entry["correct"]:
            domain_correct[domain] += 1

    width = 62
    print("=" * width)
    print(f"  {correct}/{total} correct — {accuracy:.1f}%")
    print()
    for domain in domain_total:
        label = f"  {domain}"
        score = f"{domain_correct[domain]}/{domain_total[domain]}"
        print(f"{label:<30} {score}")
    print()
    print(f"  Saved to: {args.result_file}")
    print()
    print("=" * width)


if __name__ == "__main__":
    main()
