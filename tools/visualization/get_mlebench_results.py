import json
import os

# File path
json_file_path = "/media/agent/mle-bench/runs/2024-09-16T16-59-30-UTC_run-group_opendevin/2024-09-19T21-55-55-GMT_grading_report.json"

# List of valid competition IDs
valid_competition_ids = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux"
]

def main():
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File not found at {json_file_path}")
        return
    
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Get all competition reports
        all_reports = data.get("competition_reports", [])
        
        # Create a dictionary to track the first report for each competition ID
        first_reports = {}
        
        # Process reports
        for report in all_reports:
            competition_id = report.get("competition_id")
            # Check if the competition ID is valid and we haven't seen it yet
            if competition_id in valid_competition_ids and competition_id not in first_reports:
                first_reports[competition_id] = report
        
        # Print the first report for each valid competition ID
        print(f"First reports for {len(first_reports)} valid competition IDs:\n")
        
        # Initialize counters for medal totals and score list
        total_gold = 0
        total_silver = 0
        total_bronze = 0
        total_above_median = 0
        score_list = []
        idx = 0

        for competition_id in valid_competition_ids:
            if competition_id in first_reports:
                idx += 1
                print(idx)
                report = first_reports[competition_id]
                print()
                print(f"Competition ID: {competition_id}")
                print(f"Score: {report.get('score')}")
                print(f"Gold threshold: {report.get('gold_threshold')}")
                print(f"Silver threshold: {report.get('silver_threshold')}")
                print(f"Bronze threshold: {report.get('bronze_threshold')}")
                print(f"Median threshold: {report.get('median_threshold')}")
                print(f"Any medal: {report.get('any_medal')}")
                print(f"Gold medal: {report.get('gold_medal')}")
                print(f"Silver medal: {report.get('silver_medal')}")
                print(f"Bronze medal: {report.get('bronze_medal')}")
                print(f"Above median: {report.get('above_median')}")
                print(f"Submission exists: {report.get('submission_exists')}")
                print(f"Valid submission: {report.get('valid_submission')}")
                print(f"Is lower better: {report.get('is_lower_better')}")
                print(f"Created at: {report.get('created_at')}")
                print(f"Submission path: {report.get('submission_path')}")
                print("-" * 80)
                
                # Update medal counters
                if report.get('gold_medal'):
                    total_gold += 1
                if report.get('silver_medal'):
                    total_silver += 1
                if report.get('bronze_medal'):
                    total_bronze += 1
                if report.get('above_median'):
                    total_above_median += 1
                    
                # Add to score list
                score_list.append((competition_id, report.get('score'), report.get('gold_medal'), report.get('silver_medal'), report.get('bronze_medal'), report.get('above_median')))
            else:
                print(f"No report found for competition ID: {competition_id}")
                print("-" * 80)
        
        # Print medal totals
        print("\nMEDAL SUMMARY")
        print("=" * 30)
        print(f"Total Gold Medals: {total_gold}")
        print(f"Total Silver Medals: {total_silver}")
        print(f"Total Bronze Medals: {total_bronze}")
        print(f"Total Above Median: {total_above_median}")
        print(f"Total Medals: {total_gold + total_silver + total_bronze}")
        
        # Print all scores in order
        print("\nALL SCORES")
        print("=" * 30)
        # Sort the score list by competition ID (alphabetically)
        score_list.sort(key=lambda x: x[0])
        
        # Create a single line with all competition IDs and scores
        score_line = " ".join([f"{item[1]}" if item[1] is not None else "-9999" for item in score_list])
        print(score_line)

        gold_line = " ".join([f"D{i+1}" for i, item in enumerate(score_list) if item[2]])
        print(gold_line)
        silver_line = " ".join([f"D{i+1}" for i, item in enumerate(score_list) if item[3]])
        print(silver_line)
        bronze_line = " ".join([f"D{i+1}" for i, item in enumerate(score_list) if item[4]])
        print(bronze_line)
        abovemedian_line = " ".join([f"{i+1}" for i, item in enumerate(score_list) if item[5]])
        print(abovemedian_line)
        
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()