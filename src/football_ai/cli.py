import argparse
import os
import sys
from football_ai.config.loader import load_config, merge_config_overrides
from football_ai.pipelines.detection_pipeline import DetectionPipeline
from football_ai.pipelines.tracking_pipeline import TrackingPipeline
from football_ai.pipelines.classification_pipeline import ClassificationPipeline
from football_ai.pipelines.field_mapping_pipeline import FieldMappingPipeline
from football_ai.pipelines.analytics_pipeline import AnalyticsPipeline
from football_ai.pipelines.commentary_pipeline import CommentaryPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Football AI Video Commentary System CLI")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to the system configuration yaml"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        help="Path to the input video (overrides config)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Directory to store system outputs (overrides config)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda", "mps"],
        help="Target compute device (overrides config)"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["detect", "track", "classify", "map", "analytics", "commentary", "full"],
        default="detect",
        help="System execution pipeline mode"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Process only the first N frames (for debugging/smoke test)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)
        
    try:
        config = load_config(args.config)
        config = merge_config_overrides(config, {
            "device": args.device,
            "input_path": args.input,
            "output_dir": args.output
        })
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)
        
    print("================================================")
    print(f"   FOOTBALL AI SYSTEM - MODE: {args.mode.upper()}")
    print("================================================")
    print(f"Config File:  {args.config}")
    print(f"Input Video:  {config.video.input_path}")
    print(f"Output Dir:   {config.video.output_dir}")
    print(f"Device:       {config.device}")
    print("------------------------------------------------")

    try:
        if args.mode == "detect":
            pipeline = DetectionPipeline(config)
            pipeline.run(max_frames=args.max_frames)
        elif args.mode == "track":
            pipeline = TrackingPipeline(config)
            pipeline.run(max_frames=args.max_frames)
        elif args.mode == "classify":
            pipeline = ClassificationPipeline(config)
            pipeline.run(max_frames=args.max_frames)
        elif args.mode == "map":
            pipeline = FieldMappingPipeline(config)
            pipeline.run(max_frames=args.max_frames)
        elif args.mode == "analytics":
            pipeline = AnalyticsPipeline(config)
            pipeline.run(max_frames=args.max_frames)
        elif args.mode == "commentary":
            pipeline = CommentaryPipeline(config)
            pipeline.run()
        elif args.mode == "full":
            # Step A: Analytics
            analytics = AnalyticsPipeline(config)
            analytics.run(max_frames=args.max_frames)
            
            # Step B: Optional Commentary
            if config.commentary.enabled:
                commentary = CommentaryPipeline(config)
                commentary.run()
        else:
            print(f"Mode '{args.mode}' skeleton complete. Pipeline TODO.")
    except FileNotFoundError as fnf:
        print(f"\n[!] ERROR: Resource missing!")
        print(fnf)
        print("\nDid you run standard download script?")
        print("CMD: python scripts/download_assets.py\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
