from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src" / "trace_gs"))
sys.path.insert(0, str(root / "tests"))

import test_attention_semantics
import test_dem_fix
import test_trace_dna
import test_trace_rna

test_attention_semantics.main()
test_dem_fix.main()
print(json.dumps(test_trace_dna.test_synthetic(), indent=2, default=str))
test_trace_rna.main()
print("ALL_SMOKE_TESTS_PASSED")

