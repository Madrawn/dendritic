import json
import sys

with open("results/confidence_experiments/20260129_040926_results.json", "r") as f:
    data = json.load(f)

hist = data["confidence_model_results"]["42"][0]["loss_history"]
print(f"Total entries: {len(hist)}")
for i, entry in enumerate(hist):
    step = entry.get("step", i)
    perplexity = entry.get("perplexity", None)
    eval_loss = entry.get("eval_loss", None)
    avg_eval_loss = entry.get("avg_eval_loss", None)
    if perplexity is not None and perplexity < 2.0:
        print(
            f"Step {step}: perplexity={perplexity}, eval_loss={eval_loss}, avg_eval_loss={avg_eval_loss}"
        )
        # print full entry
        # break

# also look at final entries
print("\nLast 5 entries:")
for entry in hist[-5:]:
    print(entry)

# Check standard model perplexities
std_hist = data["standard_model_results"]["42"][0]["loss_history"]
print("\nStandard model perplexities sample:")
for entry in std_hist[:5]:
    print(entry.get("perplexity", "none"), entry.get("eval_loss", "none"))
