"""Проганяє тест-кейси і записує фактичні результати для звіту.

    .venv/bin/python scripts/run_cases.py

artifacts/test_results.json – кейси з фактичним шляхом, кінцевим станом і статусом;
artifacts/coverage.json – покриття ребер і станів дерева рішень лаби 3.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zoriana_bot.cases import CASES, run_case  # noqa: E402
from zoriana_bot.tree import EDGES, ENDS, STATES  # noqa: E402

ARTIFACTS = ROOT / "artifacts"


def main() -> int:
    ARTIFACTS.mkdir(exist_ok=True)
    results = [run_case(case) for case in CASES]
    covered = {(a, b) for r in results for a, b, _ in r["trace"] if a}
    visited = {b for r in results for _, b, _ in r["trace"]}
    stamp = datetime.now().isoformat(timespec="seconds")

    (ARTIFACTS / "test_results.json").write_text(
        json.dumps({"generated": stamp, "cases": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    (ARTIFACTS / "coverage.json").write_text(json.dumps({
        "generated": stamp,
        "edges_total": len(EDGES),
        "edges_covered": len(EDGES & covered),
        "edges_missing": sorted(EDGES - covered),
        "states_total": len(STATES) + len(ENDS),
        "states_visited": sorted(visited & (set(STATES) | set(ENDS))),
        "states_missing": sorted((set(STATES) | set(ENDS)) - visited),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    failed = [r for r in results if r["problems"]]
    for r in results:
        print(f"{r['id']} {r['status']:12} {' → '.join(r['actual_path'])}")
        for p in r["problems"]:
            print("    -", p)
    print(f"\n{len(results) - len(failed)}/{len(results)} пройдено; ребер покрито {len(EDGES & covered)}/{len(EDGES)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
