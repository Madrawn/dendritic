import datetime
import json
import os
import time

import torch


# =====================
# Experiment Tracking
# =====================
def get_gpu_mem_usage():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.max_memory_allocated() / 1024**3  # Return in GB


class ExperimentTracker:
    def __init__(self, method_name, params):
        self.start_time = time.time()
        self.method_name = method_name
        self.params = params
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.results = {
            "experiment_id": f"{method_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
            "method": method_name,
            "params": params,
            "resources": {},
            "metrics": {},
            "training_samples": [],
        }

    def add_metric(self, name, value, step=None):
        """Add a metric, optionally at a specific step."""
        if step is not None:
            if "training_samples" not in self.results:
                self.results["training_samples"] = []
            self.results["training_samples"].append(
                {"step": step, "time_sec": time.time() - self.start_time, name: value}
            )
        else:
            self.results["metrics"][name] = value

    def finalize(self, model):
        """Save experiment results to file."""
        # Record final stats
        self.results["resources"]["total_time_min"] = (
            time.time() - self.start_time
        ) / 60
        self.results["resources"]["peak_gpu_mem_gb"] = get_gpu_mem_usage()

        # Parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.results["resources"]["total_params"] = total_params
        self.results["resources"]["trainable_params"] = trainable_params

        # Save to file
        filename = os.path.join(
            self.results_dir, f"{self.results['experiment_id']}.json"
        )
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)

        return self.results
