"""
validate_submission_local.py
Validates submission.csv against the India Runs hackathon submission rules:
  1. Exactly 4 columns: candidate_id, rank, score, reasoning
  2. Exactly 100 rows (excluding header)
  3. Ranks are integers 1-100, each appears exactly once
  4. Scores are numeric floats
  5. Scores are strictly descending (rank 1 has highest score)
  6. No blank candidate_ids or reasoning
  7. Reasoning length <= 500 chars each
  8. candidate_id values are unique (no duplicate entries)
  9. All candidate_ids are non-empty strings
"""
import csv
import sys
from pathlib import Path

CSV_PATH = Path("submission.csv")
REQUIRED_COLS = {"candidate_id", "rank", "score", "reasoning"}
MAX_REASONING_LEN = 500
EXPECTED_ROWS = 100

errors = []
warnings = []


def fail(msg):
    errors.append(f"  [ERROR] {msg}")


def warn(msg):
    warnings.append(f"  [WARN]  {msg}")


print("=" * 65)
print("SUBMISSION VALIDATOR")
print("=" * 65)

# ── 1. File exists ────────────────────────────────────────────────
if not CSV_PATH.exists():
    print(f"[FATAL] {CSV_PATH} not found!")
    sys.exit(1)

print(f"File : {CSV_PATH.absolute()}")
print(f"Size : {CSV_PATH.stat().st_size:,} bytes\n")

# ── 2. Parse CSV ──────────────────────────────────────────────────
with open(CSV_PATH, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    cols = set(reader.fieldnames or [])
    rows = list(reader)

# ── 3. Column check ───────────────────────────────────────────────
missing = REQUIRED_COLS - cols
extra   = cols - REQUIRED_COLS
if missing:
    fail(f"Missing columns: {missing}")
if extra:
    warn(f"Extra columns (ignored): {extra}")
else:
    print(f"[OK] Columns: {sorted(cols)}")

# ── 4. Row count ──────────────────────────────────────────────────
n = len(rows)
if n != EXPECTED_ROWS:
    fail(f"Expected {EXPECTED_ROWS} rows, got {n}")
else:
    print(f"[OK] Row count: {n}")

# ── 5. Per-row checks ─────────────────────────────────────────────
seen_ranks = {}
seen_cids  = {}
scores     = []
rank_score_pairs = []

for i, row in enumerate(rows, start=2):  # line 2 = first data row
    cid       = (row.get("candidate_id") or "").strip()
    rank_str  = (row.get("rank") or "").strip()
    score_str = (row.get("score") or "").strip()
    reasoning = (row.get("reasoning") or "").strip()

    # candidate_id
    if not cid:
        fail(f"Row {i}: blank candidate_id")
    if cid in seen_cids:
        fail(f"Row {i}: duplicate candidate_id '{cid}' (first seen row {seen_cids[cid]})")
    seen_cids[cid] = i

    # rank
    try:
        rank = int(rank_str)
        if rank < 1 or rank > EXPECTED_ROWS:
            fail(f"Row {i} ({cid}): rank {rank} out of range [1, {EXPECTED_ROWS}]")
        if rank in seen_ranks:
            fail(f"Row {i} ({cid}): duplicate rank {rank} (first at row {seen_ranks[rank]})")
        seen_ranks[rank] = i
    except ValueError:
        fail(f"Row {i} ({cid}): rank '{rank_str}' is not an integer")
        rank = None

    # score
    try:
        score = float(score_str)
        scores.append((rank, score, cid))
    except ValueError:
        fail(f"Row {i} ({cid}): score '{score_str}' is not a float")
        score = None

    # reasoning
    if not reasoning:
        fail(f"Row {i} ({cid}): blank reasoning")
    if len(reasoning) > MAX_REASONING_LEN:
        fail(f"Row {i} ({cid}): reasoning too long ({len(reasoning)} chars > {MAX_REASONING_LEN})")

    rank_score_pairs.append((rank, score, cid))

# ── 6. Rank completeness ──────────────────────────────────────────
expected_ranks = set(range(1, EXPECTED_ROWS + 1))
actual_ranks   = set(seen_ranks.keys())
missing_ranks  = expected_ranks - actual_ranks
if missing_ranks:
    fail(f"Missing ranks: {sorted(missing_ranks)}")
else:
    print(f"[OK] All ranks 1-{EXPECTED_ROWS} present exactly once")

# ── 7. Score descending order ─────────────────────────────────────
valid_pairs = [(r, s, c) for r, s, c in rank_score_pairs if r is not None and s is not None]
valid_pairs.sort(key=lambda x: x[0])  # sort by rank ascending

score_order_ok = True
for idx in range(len(valid_pairs) - 1):
    r1, s1, c1 = valid_pairs[idx]
    r2, s2, c2 = valid_pairs[idx + 1]
    if s1 < s2:
        # Allow ties but flag strict inversions
        fail(f"Score inversion: rank {r1} ({c1}) score={s1:.6f} < rank {r2} ({c2}) score={s2:.6f}")
        score_order_ok = False
        if idx > 2:  # don't spam all inversions
            fail(f"  ... and possibly more inversions")
            break

if score_order_ok:
    top_score = valid_pairs[0][1] if valid_pairs else 0
    bot_score = valid_pairs[-1][1] if valid_pairs else 0
    print(f"[OK] Scores strictly descending: {top_score:.6f} (rank 1) ... {bot_score:.6f} (rank 100)")

# ── 8. Reasoning stats ────────────────────────────────────────────
reasoning_lens = []
with open(CSV_PATH, encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        reasoning_lens.append(len((row.get("reasoning") or "").strip()))

if reasoning_lens:
    avg_len = sum(reasoning_lens) / len(reasoning_lens)
    print(f"[OK] Reasoning lengths: min={min(reasoning_lens)}, max={max(reasoning_lens)}, avg={avg_len:.0f} chars")
    if max(reasoning_lens) > MAX_REASONING_LEN:
        fail(f"Reasoning exceeds {MAX_REASONING_LEN} chars in {sum(1 for l in reasoning_lens if l > MAX_REASONING_LEN)} rows")

# ── 9. Unique candidate_ids ───────────────────────────────────────
if len(seen_cids) == n:
    print(f"[OK] All {n} candidate_ids are unique")

# ── Summary ───────────────────────────────────────────────────────
print()
if warnings:
    print("Warnings:")
    for w in warnings:
        print(w)
    print()

if errors:
    print(f"VALIDATION FAILED ({len(errors)} error(s)):")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("=" * 65)
    print("VALIDATION PASSED - submission.csv is ready for submission!")
    print("=" * 65)
