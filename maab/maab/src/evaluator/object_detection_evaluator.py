import json
import os
from typing import Dict, List

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .base_evaluator import BaseEvaluator
from .utils import load_metadata


class MapEvaluator(BaseEvaluator):
    def load_predictions(self, pred_path: str) -> List[Dict]:
        """
        Load and format predictions to COCO format
        Ensures all required fields are present including 'score'
        """
        with open(pred_path, "r") as f:
            predictions = json.load(f)

        formatted_predictions = []

        # Handle different input formats
        if isinstance(predictions, list):
            pred_list = predictions
        elif isinstance(predictions, dict):
            if "annotations" in predictions:
                pred_list = predictions["annotations"]
            else:
                # Convert dict with image_ids as keys to list
                pred_list = []
                for image_id, preds in predictions.items():
                    if isinstance(preds, list):
                        for pred in preds:
                            pred["image_id"] = int(image_id)
                            pred_list.append(pred)
        else:
            raise ValueError(f"Unsupported prediction format in {pred_path}")

        # Process each prediction and ensure required fields
        for pred in pred_list:
            if not isinstance(pred, dict):
                continue

            try:
                # Get image_id
                image_id = int(pred.get("image_id", pred.get("img_id")))

                # Get category_id
                category_id = int(pred.get("category_id", pred.get("class_id")))

                # Get bbox - ensure it's a list of 4 numbers
                bbox = pred.get("bbox", pred.get("box"))
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                bbox = [float(b) for b in bbox]

                # Get score - ensure it exists and is a float between 0 and 1
                score = pred.get("score", pred.get("confidence", None))
                if score is None:
                    # If no score found, default to 1.0
                    score = 1.0
                score = float(score)
                score = max(0.0, min(1.0, score))  # Clip between 0 and 1

                formatted_pred = {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score,
                    "area": bbox[2] * bbox[3],  # width * height
                    "iscrowd": 0,
                }
                formatted_predictions.append(formatted_pred)

            except (ValueError, TypeError, KeyError) as e:
                print(f"Warning: Skipping invalid prediction: {pred}. Error: {str(e)}")
                continue

        if not formatted_predictions:
            raise ValueError(f"No valid predictions found in {pred_path}")

        return formatted_predictions

    def calculate_metric(self, gt_path: str, pred_path: str) -> float:
        try:
            # Load ground truth COCO annotations
            cocoGt = COCO(gt_path)

            # Load and format predictions
            predictions = self.load_predictions(pred_path)

            # Save formatted predictions to temporary file
            temp_pred_file = "temp_predictions.json"
            with open(temp_pred_file, "w") as f:
                json.dump(predictions, f)

            try:
                # Load formatted predictions
                cocoDt = cocoGt.loadRes(temp_pred_file)

                # Initialize COCO evaluator
                cocoEval = COCOeval(cocoGt, cocoDt, "bbox")

                # Set evaluation parameters
                # cocoEval.params.iouThrs = np.array([0.5])  # Set IoU threshold to 0.5
                # cocoEval.params.maxDets = [1, 10, 100]  # COCO standard maxDets
                # cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2]]  # All areas
                # cocoEval.params.areaRngLbl = ['all']

                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()

                # Return mAP@0.5
                return cocoEval.stats[1]  # Use stats[1] for mAP@0.5

            finally:
                # Clean up temporary file
                if os.path.exists(temp_pred_file):
                    os.remove(temp_pred_file)

        except Exception as e:
            print(f"Raw COCO stats: {getattr(cocoEval, 'stats', 'No stats available')}")
            raise ValueError(f"Error in MAP evaluation: {str(e)}")

    @property
    def name(self) -> str:
        return "map"

    def evaluate(
        self,
        pred_path: str,
        gt_path: str,
        results_path: str,
        metadata_path: str,
        agent_name: str,
    ):
        try:
            metadata = load_metadata(metadata_path)
            score = self.calculate_metric(gt_path, pred_path)
            print(f"Evaluation Score ({self.name}): {score:.4f}")
            self.write_results(results_path, metadata, score, agent_name)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise

    @property
    def name(self) -> str:
        return "map"

    def evaluate(
        self,
        pred_path: str,
        gt_path: str,
        results_path: str,
        metadata_path: str,
        agent_name: str,
    ):
        try:
            metadata = load_metadata(metadata_path)
            score = self.calculate_metric(gt_path, pred_path)
            print(f"Evaluation Score ({self.name}): {score:.4f}")
            self.write_results(results_path, metadata, score, agent_name)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise
