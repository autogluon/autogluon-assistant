#!/usr/bin/env python3
"""
Hugging Face Top Models Fetcher
Retrieves top K trending models and most downloaded models for each task
"""
import requests
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional
import pandas as pd

class HuggingFaceModelsFetcher:
    def __init__(self):
        self.base_url = "https://huggingface.co/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HF-Models-Fetcher/1.0'
        })
    
    def get_models_by_task(self, task: str, sort_by: str = 'likes', limit: int = 10) -> List[Dict]:
        """Get models filtered by specific task"""
        url = f"{self.base_url}/models"
        params = {
            'pipeline_tag': task,
            'sort': sort_by,
            'limit': limit,
            'full': 'true'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching models for task {task}: {e}")
            return []
    
    def get_all_pipeline_tags(self) -> List[str]:
        """Get all available pipeline tags/tasks from Hugging Face"""
        return [
            # Multimodal
            'any-to-any',
            'audio-text-to-text',
            'document-question-answering',
            'visual-document-retrieval',
            'image-text-to-text',
            'video-text-to-text',
            'visual-question-answering',
            
            # Natural Language Processing
            'feature-extraction',
            'fill-mask',
            'question-answering',
            'sentence-similarity',
            'summarization',
            'table-question-answering',
            'text-classification',
            'text-generation',
            'text-ranking',
            'token-classification',
            'translation',
            'zero-shot-classification',
            
            # Computer Vision
            'depth-estimation',
            'image-classification',
            'image-feature-extraction',
            'image-segmentation',
            'image-to-image',
            'image-to-text',
            'image-to-video',
            'keypoint-detection',
            'mask-generation',
            'object-detection',
            'video-classification',
            'text-to-image',
            'text-to-video',
            'unconditional-image-generation',
            'zero-shot-image-classification',
            'zero-shot-object-detection',
            'text-to-3d',
            'image-to-3d',
            
            # Audio
            'audio-classification',
            'audio-to-audio',
            'automatic-speech-recognition',
            'text-to-speech',
            
            # Tabular
            'tabular-classification',
            'tabular-regression',
            
            # Reinforcement Learning
            'reinforcement-learning',
            
            # Additional common tasks that might use different naming conventions
            'conversational',
            'text2text-generation',
            'voice-activity-detection',
            'time-series-forecasting',
            'robotics',
            'other'
        ]
    
    def extract_model_info(self, model: Dict) -> Dict:
        """Extract relevant information from model data"""
        model_id = model.get('id', '')
        return {
            'model_id': model_id,
            'url': f"https://huggingface.co/{model_id}" if model_id else '',
            'author': model.get('author', ''),
            'likes': model.get('likes', 0),
            'downloads': model.get('downloads', 0),
            'created_at': model.get('createdAt', ''),
            'last_modified': model.get('lastModified', ''),
            'pipeline_tag': model.get('pipeline_tag', ''),
            'library_name': model.get('library_name', ''),
            'tags': model.get('tags', []),
            'model_size': model.get('safetensors', {}).get('total', 0) if model.get('safetensors') else 0
        }
    
    def get_top_models_all_tasks(self, n: int = 10, sort_by: str = 'likes', 
                                include_downloads: bool = True) -> Dict[str, List[Dict]]:
        """
        Get top N models for each task
        
        Args:
            n: Number of top models to fetch per task
            sort_by: Sort criteria ('likes', 'downloads', 'modified', 'created')
            include_downloads: Whether to also fetch top downloaded models
        
        Returns:
            Dictionary with task names as keys and list of model info as values
        """
        all_tasks_models = defaultdict(list)
        pipeline_tags = self.get_all_pipeline_tags()
        
        print(f"Fetching top {n} models for {len(pipeline_tags)} tasks...")
        
        for i, task in enumerate(pipeline_tags):
            print(f"Processing task {i+1}/{len(pipeline_tags)}: {task}")
            
            # Get top models by specified criteria (likes by default)
            models = self.get_models_by_task(task, sort_by=sort_by, limit=n)
            
            if models:
                task_models = []
                for model in models[:n]:  # Ensure we only get top N
                    model_info = self.extract_model_info(model)
                    model_info['sort_criteria'] = sort_by
                    task_models.append(model_info)
                
                all_tasks_models[task] = task_models
                print(f"  Found {len(task_models)} models for {task}")
            else:
                print(f"  No models found for {task}")
            
            # Add small delay to be respectful to the API
            time.sleep(0.1)
        
        return dict(all_tasks_models)
    
    def merge_top_models(self, liked_models: Dict[str, List[Dict]], 
                        downloaded_models: Dict[str, List[Dict]], 
                        max_per_task: int = 15) -> Dict[str, List[Dict]]:
        """
        Merge liked and downloaded models, removing duplicates and keeping top models
        
        Args:
            liked_models: Dictionary of models sorted by likes
            downloaded_models: Dictionary of models sorted by downloads
            max_per_task: Maximum number of models to keep per task
        
        Returns:
            Dictionary of merged top models per task
        """
        merged_models = defaultdict(list)
        
        # Get all unique tasks
        all_tasks = set(liked_models.keys()) | set(downloaded_models.keys())
        
        for task in all_tasks:
            liked_list = liked_models.get(task, [])
            downloaded_list = downloaded_models.get(task, [])
            
            # Create a dictionary to track unique models by model_id
            unique_models = {}
            
            # Add liked models first
            for model in liked_list:
                model_id = model['model_id']
                if model_id not in unique_models:
                    model_copy = model.copy()
                    model_copy['source'] = 'liked'
                    unique_models[model_id] = model_copy
            
            # Add downloaded models, updating existing entries
            for model in downloaded_list:
                model_id = model['model_id']
                if model_id in unique_models:
                    # Model exists in both lists, mark as both
                    unique_models[model_id]['source'] = 'both'
                else:
                    # New model from downloads
                    model_copy = model.copy()
                    model_copy['source'] = 'downloaded'
                    unique_models[model_id] = model_copy
            
            # Convert back to list and sort by a composite score
            models_list = list(unique_models.values())
            
            # Sort by composite score: prioritize models that appear in both lists,
            # then by likes + normalized downloads
            def composite_score(model):
                base_score = model.get('likes', 0) + (model.get('downloads', 0) / 1000)  # Normalize downloads
                if model['source'] == 'both':
                    base_score *= 1.5  # Bonus for appearing in both lists
                return base_score
            
            models_list.sort(key=composite_score, reverse=True)
            
            # Keep only top models per task
            merged_models[task] = models_list[:max_per_task]
            
            print(f"Task {task}: {len(models_list)} unique models merged, keeping top {len(merged_models[task])}")
        
        return dict(merged_models)
    
    def save_to_json(self, data: Dict, filename: str = 'top_hf_models.json'):
        """Save results to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving to JSON: {e}")
    
    def save_to_csv(self, data: Dict, filename: str = 'top_hf_models.csv'):
        """Save results to CSV file"""
        try:
            # Flatten the data for CSV format
            rows = []
            for task, models in data.items():
                for model in models:
                    row = model.copy()
                    row['task'] = task
                    rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    
    def print_summary(self, data: Dict, title: str = "TOP MODELS BY TASK"):
        """Print a summary of the results"""
        print("\n" + "="*60)
        print(title)
        print("="*60)
        
        total_models = sum(len(models) for models in data.values())
        print(f"Total tasks processed: {len(data)}")
        print(f"Total models found: {total_models}")
        
        print(f"\nTop 3 most liked models overall:")
        all_models = []
        for task, models in data.items():
            for model in models:
                model_copy = model.copy()
                model_copy['task'] = task
                all_models.append(model_copy)
        
        # Sort by likes
        top_overall = sorted(all_models, key=lambda x: x.get('likes', 0), reverse=True)[:3]
        for i, model in enumerate(top_overall, 1):
            source_info = f" ({model.get('source', 'unknown')})" if 'source' in model else ""
            print(f"{i}. {model['model_id']} ({model['task']}) - {model['likes']} likes{source_info}")
        
        print(f"\nTasks with most models available:")
        task_counts = [(task, len(models)) for task, models in data.items()]
        task_counts.sort(key=lambda x: x[1], reverse=True)
        for task, count in task_counts[:5]:
            print(f"  {task}: {count} models")
        
        # Show source distribution if available
        if all_models and 'source' in all_models[0]:
            source_counts = defaultdict(int)
            for model in all_models:
                source_counts[model.get('source', 'unknown')] += 1
            
            print(f"\nModel source distribution:")
            for source, count in source_counts.items():
                print(f"  {source}: {count} models")

def main():
    """Main function to demonstrate usage"""
    fetcher = HuggingFaceModelsFetcher()
    
    # Get top models for each task
    k = 3
    print(f"Fetching top {k} most liked models for each task...")
    top_liked_models = fetcher.get_top_models_all_tasks(n=k, sort_by='likes')
    
    print(f"\nFetching top {k} most downloaded models for each task...")
    top_downloaded_models = fetcher.get_top_models_all_tasks(n=k, sort_by='downloads')
    
    # Merge the results
    print(f"\nMerging liked and downloaded models...")
    merged_top_models = fetcher.merge_top_models(
        top_liked_models, 
        top_downloaded_models, 
        max_per_task=2*k,
    )
    
    # Print summaries
    fetcher.print_summary(top_liked_models, "TOP LIKED MODELS BY TASK")
    fetcher.print_summary(top_downloaded_models, "TOP DOWNLOADED MODELS BY TASK")
    fetcher.print_summary(merged_top_models, "MERGED TOP MODELS BY TASK")
    
    # Save results
    # fetcher.save_to_json(merged_top_models, f'top_merged_models_by_task.json')
    fetcher.save_to_csv(merged_top_models, f'top_{k}_merged_models_by_task.csv')
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print("Files saved:")
    #print("- top_merged_models_by_task.json")
    print(f"- top_{k}_merged_models_by_task.csv")

if __name__ == "__main__":
    main()
