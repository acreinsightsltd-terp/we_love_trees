from src.config import Config
from src.pipeline import CanopyCoverPipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Canopy Cover Analysis')
    parser.add_argument(
        '--stage',
        type=str,
        choices=['preprocess', 'indices', 'model', 'validate'],
        default='all',
        help='Decide which stage of the analysis to run'
    )
    args = parser.parse_args()
    config = Config()
    pipeline = CanopyCoverPipeline(config)
    
    if args.stage == 'all':
        pipeline.run_full_pipeline()
    elif args.stage == 'preprocess':
        pipeline.run_preprocessing()
    elif args.stage == 'indices':
        pipeline.run_indices()
    elif args.stage == 'model':
        pipeline.run_model()
    elif args.stage == 'validate':
        pipeline.run_validate()