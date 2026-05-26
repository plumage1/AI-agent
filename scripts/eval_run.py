from pathlib import Path
import json
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evals.eval_runner import run_all_evals


if __name__ == "__main__":
    print(json.dumps(run_all_evals(), ensure_ascii=False, indent=2))
