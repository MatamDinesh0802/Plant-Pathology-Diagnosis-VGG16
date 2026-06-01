"""Print the metrics table from reports/metrics.json."""
from __future__ import annotations

import json

from .config import CLASS_NAMES, METRICS_PATH


def main() -> None:
    if not METRICS_PATH.exists():
        raise SystemExit("No metrics.json yet — run the Colab notebook first.")
    payload = json.loads(METRICS_PATH.read_text())
    print(f"Best model: {payload['best_model']}")
    print(f"Train: {payload.get('n_train', '—')}  Val: {payload.get('n_val', '—')}  "
          f"Test: {payload.get('n_test', '—')}")

    for name, m in payload["models"].items():
        print(f"\n=== {name} ===")
        print(f"  accuracy:    {m['accuracy']:.4f}")
        print(f"  weighted F1: {m.get('f1_weighted', '—')}")
        for cls in CLASS_NAMES:
            r = m["report"].get(cls)
            if r:
                print(f"  {cls:18s}  P {r['precision']:.3f}  R {r['recall']:.3f}  "
                      f"F1 {r['f1-score']:.3f}  support {int(r['support'])}")


if __name__ == "__main__":
    main()
