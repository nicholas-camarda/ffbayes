#!/usr/bin/env python3
"""
run_pipeline.py - Master Fantasy Football Analytics Pipeline
Runs the complete pipeline in the proper sequence.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add scripts/utils to path for progress monitoring
sys.path.append(str(Path.cwd() / 'scripts' / 'utils'))
try:
    from progress_monitor import ProgressMonitor
except ImportError:
    ProgressMonitor = None


def run_step(step_name, script_path, description):
    """Run a pipeline step and report results."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    print(f"📋 {description}")
    print(f"🚀 Running: {script_path}")
    
    # Check if script exists
    if not Path(script_path).exists():
        print(f"❌ Script not found: {script_path}")
        return False, 0.0
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, check=True)
        
        elapsed_time = time.time() - start_time
        print(f"✅ {step_name} completed successfully in {elapsed_time:.1f} seconds")
        
        # Show output (last 1000 chars for better context)
        if result.stdout:
            print("\n📤 Output:")
            print(result.stdout[-1000:])  # Last 1000 chars
        
        return True, elapsed_time
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {step_name} failed after {elapsed_time:.1f} seconds")
        print(f"Error code: {e.returncode}")
        
        # Show both stdout and stderr for better debugging
        if e.stdout:
            print("\n📤 Standard output:")
            print(e.stdout[-500:])
        
        if e.stderr:
            print("\n🚨 Error details:")
            print(e.stderr[-500:])
        
        return False, elapsed_time
    
    except FileNotFoundError:
        elapsed_time = time.time() - start_time
        print(f"❌ {step_name} failed: Python interpreter not found")
        return False, elapsed_time
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {step_name} failed with unexpected error: {e}")
        return False, elapsed_time

def main():
    """Run the complete fantasy football analytics pipeline."""
    print("🏈 FANTASY FOOTBALL ANALYTICS PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start = time.time()
    
    # Define pipeline steps
    pipeline_steps = [
        {
            "name": "Data Collection",
            "script": "scripts/data_pipeline/01_collect_data.py",
            "description": "Collect raw NFL data from multiple sources"
        },
        {
            "name": "Data Validation", 
            "script": "scripts/data_pipeline/02_validate_data.py",
            "description": "Validate data quality and completeness"
        },
        {
            "name": "Monte Carlo Simulation",
            "script": "scripts/analysis/montecarlo-historical-ff.py", 
            "description": "Generate team-level projections using historical data"
        },
        {
            "name": "Bayesian Predictions",
            "script": "scripts/analysis/bayesian-hierarchical-ff-modern.py",
            "description": "Generate player-level predictions with uncertainty using PyMC4"
        }
    ]
    
    # Track results
    results = []
    total_time = 0
    
    # Use enhanced progress monitoring if available
    if ProgressMonitor:
        monitor = ProgressMonitor("Pipeline Execution")
        monitor.start_timer()
        
        with monitor.monitor(len(pipeline_steps), "Pipeline Steps"):
            # Run each step
            for i, step in enumerate(pipeline_steps, 1):
                print(f"\n🔄 Step {i}/{len(pipeline_steps)}: {step['name']}")
                
                success, elapsed_time = run_step(
                    step['name'], 
                    step['script'], 
                    step['description']
                )
                
                results.append({
                    'step': step['name'],
                    'success': success,
                    'time': elapsed_time
                })
                
                total_time += elapsed_time
                
                if not success:
                    print(f"\n🚨 Pipeline failed at step {i}: {step['name']}")
                    print("🔄 Please fix the issue and re-run the pipeline")
                    break
    else:
        # Fallback to basic progress tracking
        # Run each step
        for i, step in enumerate(pipeline_steps, 1):
            print(f"\n🔄 Step {i}/{len(pipeline_steps)}: {step['name']}")
            
            success, elapsed_time = run_step(
                step['name'], 
                step['script'], 
                step['description']
            )
            
            results.append({
                'step': step['name'],
                'success': success,
                'time': elapsed_time
            })
            
            total_time += elapsed_time
            
            if not success:
                print(f"\n🚨 Pipeline failed at step {i}: {step['name']}")
                print("🔄 Please fix the issue and re-run the pipeline")
                break
    
    # Pipeline summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    
    successful_steps = sum(1 for r in results if r['success'])
    total_steps = len(results)
    
    print(f"📊 Steps completed: {successful_steps}/{total_steps}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"🎯 Success rate: {(successful_steps/total_steps)*100:.1f}%")
    
    # Enhanced status reporting
    if successful_steps == total_steps:
        print("\n🎉 Pipeline completed successfully!")
        print("📈 Ready for fantasy football analysis!")
    elif successful_steps == 0:
        print("\n❌ Pipeline failed completely. Check all steps.")
        print("🔧 Review error messages and fix issues before re-running.")
    else:
        print(f"\n⚠️  Pipeline partially completed ({successful_steps}/{total_steps} steps)")
        print("🔧 Check failed steps and re-run as needed")
    
    print(f"\n🏁 Pipeline finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return exit code for automation
    return 0 if successful_steps == total_steps else 1

if __name__ == "__main__":
    main()

