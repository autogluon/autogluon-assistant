import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maab.src.evaluator import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a specific metric and log results with metadata"
    )
    parser.add_argument("--pred_path", help="Path to the prediction CSV file")
    parser.add_argument("--metadata_path", help="Path to the metadata file")
    parser.add_argument("--gt_path", help="Path to the ground truth file")
    parser.add_argument("--results_path", help="Path to the overall results CSV file")
    parser.add_argument("--agent_name", help="Name of the agent or model")
    args = parser.parse_args()

    if not os.path.exists(args.gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {args.gt_path}")

    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")

    evaluate(
        pred_path=args.pred_path,
        gt_path=args.gt_path,
        results_path=args.results_path,
        metadata_path=args.metadata_path,
        agent_name=args.agent_name,
    )
