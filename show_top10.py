import csv

print("=" * 72)
print("TOP 10 RESULTS FROM submission.csv")
print("=" * 72)
print(f"{'Rank':<5} {'CandID':<14} {'Score':<12} Reasoning[:90]")
print("-" * 72)
with open("submission.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 10:
            break
        r = row["reasoning"][:90] + "..." if len(row["reasoning"]) > 90 else row["reasoning"]
        print(f"{row['rank']:<5} {row['candidate_id']:<14} {row['score']:<12} {r}")

print()
print("FULL REASONING for top 5:")
print("=" * 72)
with open("submission.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 5:
            break
        print(f"\n#{row['rank']} [{row['candidate_id']}] score={row['score']}")
        print(f"  {row['reasoning']}")
