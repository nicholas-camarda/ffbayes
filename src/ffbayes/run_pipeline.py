#!/usr/bin/env python3
"""
run_pipeline.py - Master Fantasy Football Analytics Pipeline
Runs the complete pipeline in the proper sequence with enhanced orchestration.
"""

import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from ffbayes.utils.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator
except Exception:
    EnhancedPipelineOrchestrator = None

# Global timeout configuration (fallback)
STEP_TIMEOUT = 300  # 5 minutes per step


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Pipeline step timed out")


def create_required_directories():
    """Create all required directories for the pipeline."""
    print("📁 Creating required directories...")
    print()
    
    required_dirs = [
        # Data directories
        ("datasets/season_datasets", "Raw NFL season data files"),
        ("datasets/combined_datasets", "Combined and processed datasets"), 
        ("misc-datasets", "Additional datasets and projections"),
        ("snake_draft_datasets", "Draft strategy and VOR calculations"),
        
        # Results directories
        ("results/montecarlo_results", "Monte Carlo simulation outputs"), 
        ("results/bayesian-hierarchical-results", "Bayesian model results and traces"),
        
        # Output directories
        ("plots", "Generated charts and visualizations"),
        ("my_ff_teams", "Your fantasy team configurations")
    ]
    
    created_count = 0
    for dir_path, description in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dir_path}")
            print(f"   📝 Purpose: {description}")
            created_count += 1
        else:
            print(f"📁 Exists: {dir_path}")
            print(f"   📝 Purpose: {description}")
    
    print()
    if created_count > 0:
        print(f"✅ Created {created_count} new directories")
    else:
        print("✅ All required directories already exist")
    
    print()


def run_step(step_name, script_path, description):
    """Run a pipeline step and report results with timeout."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    print(f"📋 {description}")
    print(f"🚀 Running: {script_path}")
    print(f"⏱️  Timeout: {STEP_TIMEOUT} seconds")
    print()
    
    # Check if script exists
    if not Path(script_path).exists():
        print(f"❌ Script not found: {script_path}")
        return False, 0.0
    
    start_time = time.time()
    
    try:
        # Set timeout for this step
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(STEP_TIMEOUT)
        
        print("🔄 Starting execution...")
        print("-" * 40)
        
        # Run the script with real-time output instead of capturing
        result = subprocess.run([
            sys.executable, script_path
        ], timeout=STEP_TIMEOUT)
        
        # Clear the alarm
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print("-" * 40)
            print(f"✅ {step_name} completed successfully in {elapsed_time:.1f} seconds")
        else:
            print("-" * 40)
            print(f"❌ {step_name} failed with exit code {result.returncode}")
        
        return result.returncode == 0, elapsed_time
        
    except subprocess.CalledProcessError as e:
        # Clear the alarm
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        print(f"❌ {step_name} failed with exit code {e.returncode}")
        
        if e.stdout:
            print("\n📤 Output:")
            print(e.stdout[-500:])
        
        if e.stderr:
            print("\n🚨 Error details:")
            print(e.stderr[-500:])
        
        return False, elapsed_time
    
    except TimeoutError:
        # Clear the alarm
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        print(f"⏰ {step_name} timed out after {STEP_TIMEOUT} seconds")
        return False, elapsed_time
    
    except FileNotFoundError:
        # Clear the alarm
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        print(f"❌ {step_name} failed: Python interpreter not found")
        return False, elapsed_time
    
    except Exception as e:
        # Clear the alarm
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        print(f"❌ {step_name} failed with unexpected error: {e}")
        return False, elapsed_time

def main():
    """Run the complete fantasy football analytics pipeline with enhanced orchestration."""
    print("🏈 FANTASY FOOTBALL ANALYTICS PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create required directories first
    create_required_directories()
    
    # Check if enhanced orchestrator is available
    if EnhancedPipelineOrchestrator is not None:
        print("🚀 Using Enhanced Pipeline Orchestrator")
        print("📊 Features: Dependency management, sequential execution with internal multiprocessing, error recovery")
        print()
        
        try:
            # Use enhanced orchestrator
            orchestrator = EnhancedPipelineOrchestrator()
            
            # Execute pipeline with enhanced features
            success = orchestrator.execute_pipeline()
            
            # Get comprehensive summary
            summary = orchestrator.get_execution_summary()
            
            # Display enhanced summary
            print("\n" + "="*80)
            print("ENHANCED PIPELINE ORCHESTRATION - EXECUTION SUMMARY")
            print("="*80)
            
            print(f"Pipeline Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
            print(f"Total Steps: {summary['pipeline_info']['total_steps']}")
            print(f"Completed Steps: {summary['performance_metrics']['completed_steps']}")
            print(f"Failed Steps: {summary['performance_metrics']['failed_steps']}")
            print(f"Total Execution Time: {summary['pipeline_info']['total_time']:.1f}s")
            print(f"Parallel Efficiency: {summary['performance_metrics']['parallel_efficiency']:.2f}")
            
            if summary['error_recovery_state']['total_retries'] > 0:
                print(f"Total Retries: {summary['error_recovery_state']['total_retries']}")
            
            print("\nStep Results:")
            for result in summary['step_results']:
                status_icon = "✅" if result['success'] else "❌"
                print(f"  {status_icon} {result['step_name']}: {result['execution_time']:.1f}s")
                if result['retry_attempts'] > 0:
                    print(f"    Retries: {result['retry_attempts']}")
                if result['error_message']:
                    print(f"    Error: {result['error_message']}")
            
            print("="*80)
            
            return 0 if success else 1
            
        except Exception as e:
            print(f"❌ Enhanced orchestrator failed: {e}")
            print("🔄 Falling back to basic pipeline execution...")
            print()
    
    # Fallback to basic pipeline execution
    print("🚀 Using Basic Pipeline Execution (Fallback)")
    print("📊 Features: Sequential execution, basic error handling")
    print()
    
    pipeline_start = time.time()
    
    # Define basic pipeline steps (fallback)
    pipeline_steps = [
        {
            "name": "Data Collection",
            "script": "src/ffbayes/data_pipeline/collect_data.py",
            "description": "Collect raw NFL data from multiple sources"
        },
        {
            "name": "Data Validation", 
            "script": "src/ffbayes/data_pipeline/validate_data.py",
            "description": "Validate data quality and completeness"
        },
        {
            "name": "Monte Carlo Simulation",
            "script": "src/ffbayes/analysis/montecarlo_historical_ff.py", 
            "description": "Generate team-level projections using historical data"
        },
        {
            "name": "Bayesian Predictions",
            "script": "src/ffbayes/analysis/bayesian_hierarchical_ff_modern.py",
            "description": "Generate player-level predictions with uncertainty using PyMC4"
        }
    ]
    
    # Track results
    results = []
    total_time = 0
    
    print("🔄 Starting basic pipeline execution...")
    print("📁 All required directories have been created")
    print()
    
    # Run each step with clear visibility
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
        
        print(f"✅ Step {i} completed. Moving to next step...")
        print()
    
    # Pipeline summary
    print(f"\n{'='*60}")
    print("BASIC PIPELINE SUMMARY")
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

