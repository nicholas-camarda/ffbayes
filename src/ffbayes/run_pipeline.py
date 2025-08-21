#!/usr/bin/env python3
"""
run_pipeline.py - Master Fantasy Football Analytics Pipeline
Runs the complete pipeline in the proper sequence with enhanced orchestration.
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from ffbayes.utils.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator
except Exception:
    EnhancedPipelineOrchestrator = None

# Global timeout configuration (fallback)
STEP_TIMEOUT = 300  # 5 minutes per step

# Simple logging (optional global log file)
LOG_FILE_HANDLE: Optional[object] = None


def log_write(message: str) -> None:
    """Write message to stdout and the log file if configured."""
    print(message)
    if LOG_FILE_HANDLE is not None:
        try:
            LOG_FILE_HANDLE.write(message + "\n")
            LOG_FILE_HANDLE.flush()
        except Exception:
            pass


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
        ("results/team_aggregation", "Team aggregation results and analysis"),
        ("results/draft_strategy", "Draft strategy outputs and configurations"),
        ("results/draft_strategy_comparison", "Draft strategy comparison reports"),
        ("results/model_comparison", "Model comparison and evaluation results"),
        
        # Output directories - organized subfolders
        ("plots/team_aggregation", "Team aggregation visualizations and charts"),
        ("plots/monte_carlo", "Monte Carlo simulation visualizations"),
        ("plots/draft_strategy_comparison", "Draft strategy comparison charts"),
        ("plots/bayesian_model", "Bayesian model visualizations and diagnostics"),
        ("plots/test_runs", "Test run outputs and debugging visualizations"),
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
    log_write(f"\n{'='*60}")
    log_write(f"STEP: {step_name}")
    log_write(f"{'='*60}")
    log_write(f"📋 {description}")
    log_write(f"🚀 Running: {script_path}")
    log_write(f"⏱️  Timeout: {STEP_TIMEOUT} seconds")
    log_write("")
    
    # Check if script exists (for file paths) or is a valid module (for python -m commands)
    if script_path.startswith("python -m "):
        # This is a module execution command; separate module and args
        parts = shlex.split(script_path)
        # parts looks like ["python", "-m", "module", "--flag", "value", ...]
        module_name = parts[2] if len(parts) >= 3 else ""
        try:
            __import__(module_name)
            log_write(f"✅ Module {module_name} can be imported")
        except ImportError as e:
            log_write(f"❌ Module {module_name} cannot be imported: {e}")
            return False, 0.0
    elif not Path(script_path).exists():
        log_write(f"❌ Script not found: {script_path}")
        return False, 0.0
    
    start_time = time.time()
    
    try:
        # Set timeout for this step
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(STEP_TIMEOUT)
        
        log_write("🔄 Starting execution...")
        log_write("-" * 40)
        
        # Run the script with real-time output streaming to console and log
        if script_path.startswith("python -m "):
            parts = shlex.split(script_path)
            cmd = [sys.executable, "-m"] + parts[2:]
        else:
            cmd = [sys.executable, script_path]

        # Pass through environment variables (including QUICK_TEST)
        env = os.environ.copy()

        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, env=env) as proc:
            try:
                for line in proc.stdout:
                    log_write(line.rstrip("\n"))
                proc.wait(timeout=STEP_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise TimeoutError("Pipeline step timed out")

        class _Result:
            def __init__(self, returncode):
                self.returncode = returncode
        result = _Result(proc.returncode)
        
        # Clear the alarm
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            log_write("-" * 40)
            log_write(f"✅ {step_name} completed successfully in {elapsed_time:.1f} seconds")
        else:
            log_write("-" * 40)
            log_write(f"❌ {step_name} failed with exit code {result.returncode}")
        
        return result.returncode == 0, elapsed_time
        
    except subprocess.CalledProcessError as e:
        # Clear the alarm
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        log_write(f"❌ {step_name} failed with exit code {e.returncode}")
        
        if e.stdout:
            log_write("\n📤 Output:")
            log_write(e.stdout[-500:])
        
        if e.stderr:
            log_write("\n🚨 Error details:")
            log_write(e.stderr[-500:])
        
        return False, elapsed_time
    
    except TimeoutError:
        # Clear the alarm
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        log_write(f"⏰ {step_name} timed out after {STEP_TIMEOUT} seconds")
        return False, elapsed_time
    
    except FileNotFoundError:
        # Clear the alarm
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        log_write(f"❌ {step_name} failed: Python interpreter not found")
        return False, elapsed_time
    
    except Exception as e:
        # Clear the alarm
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        log_write(f"❌ {step_name} failed with unexpected error: {e}")
        return False, elapsed_time

def main():
    """Run the complete fantasy football analytics pipeline with enhanced orchestration."""
    print("🏈 FANTASY FOOTBALL ANALYTICS PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CLI to control phases in fallback mode
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phase", choices=["draft", "validate", "full"], default="full",
                        help="Which phase to run: draft (Phase A), validate (Phase B), or full")
    parser.add_argument("--team-file", type=str, help="Path to TSV team file for Monte Carlo validation")
    known_args, _ = parser.parse_known_args()

    # Initialize logging to file named with phase and timestamp
    try:
        logs_dir = Path('logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        start_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_path = logs_dir / f"pipeline-{known_args.phase}-{start_ts}.log"
        global LOG_FILE_HANDLE
        LOG_FILE_HANDLE = log_path.open('a', encoding='utf-8')
        log_write("🏈 FANTASY FOOTBALL ANALYTICS PIPELINE")
        log_write("=" * 60)
        log_write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception:
        pass
    
    # Create required directories first
    create_required_directories()
    
    # Check if enhanced orchestrator is available
    if EnhancedPipelineOrchestrator is not None and known_args.phase == "full":
        log_write("🚀 Using Enhanced Pipeline Orchestrator")
        log_write("📊 Features: Dependency management, sequential execution with internal multiprocessing, error recovery")
        log_write("")
        
        try:
            # Use enhanced orchestrator
            orchestrator = EnhancedPipelineOrchestrator()
            
            # Execute pipeline with enhanced features
            success = orchestrator.execute_pipeline()
            
            # Get comprehensive summary
            summary = orchestrator.get_execution_summary()
            
            # Display enhanced summary
            log_write("\n" + "="*80)
            log_write("ENHANCED PIPELINE ORCHESTRATION - EXECUTION SUMMARY")
            log_write("="*80)
            
            log_write(f"Pipeline Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
            log_write(f"Total Steps: {summary['pipeline_info']['total_steps']}")
            log_write(f"Completed Steps: {summary['performance_metrics']['completed_steps']}")
            log_write(f"Failed Steps: {summary['performance_metrics']['failed_steps']}")
            log_write(f"Total Execution Time: {summary['pipeline_info']['total_time']:.1f}s")
            log_write(f"Parallel Efficiency: {summary['performance_metrics']['parallel_efficiency']:.2f}")
            
            if summary['error_recovery_state']['total_retries'] > 0:
                log_write(f"Total Retries: {summary['error_recovery_state']['total_retries']}")
            
            log_write("\nStep Results:")
            for result in summary['step_results']:
                status_icon = "✅" if result['success'] else "❌"
                log_write(f"  {status_icon} {result['step_name']}: {result['execution_time']:.1f}s")
                if result['retry_attempts'] > 0:
                    log_write(f"    Retries: {result['retry_attempts']}")
                if result['error_message']:
                    log_write(f"    Error: {result['error_message']}")
            
            log_write("="*80)
            
            return 0 if success else 1
            
        except Exception as e:
            log_write(f"❌ Enhanced orchestrator failed: {e}")
            log_write("🔄 Falling back to basic pipeline execution...")
            log_write("")
    
    # Fallback to basic pipeline execution
    log_write("🚀 Using Basic Pipeline Execution (Fallback)")
    log_write("📊 Features: Sequential execution, basic error handling")
    log_write("")
    
    pipeline_start = time.time()
    
    # Define basic pipeline steps (fallback) - CORRECTED ORDER - USING PYTHON MODULE EXECUTION
    all_steps = [
        {
            "name": "Data Collection",
            "script": "python -m ffbayes.data_pipeline.collect_data",
            "description": "Collect raw NFL data from multiple sources"
        },
        {
            "name": "Data Validation", 
            "script": "python -m ffbayes.data_pipeline.validate_data",
            "description": "Validate data quality and completeness"
        },
        {
            "name": "Data Preprocessing",
            "script": "python -m ffbayes.data_pipeline.preprocess_analysis_data",
            "description": "Preprocess data for analysis"
        },
        {
            "name": "Bayesian Analysis",
            "script": "python -m ffbayes.analysis.bayesian_hierarchical_ff_unified",
            "description": "Generate player-level predictions with uncertainty using PyMC4"
        },
        {
            "name": "Draft Strategy Generation",
            "script": "python -m ffbayes.draft_strategy.bayesian_draft_strategy --draft-position 3 --league-size 10 --risk-tolerance medium",
            "description": "Generate optimal draft strategies using Bayesian predictions"
        },
        {
            "name": "Monte Carlo Validation",
            "script": "python -m ffbayes.analysis.montecarlo_historical_ff",
            "description": "Validate drafted teams using Monte Carlo simulation (adversarial evaluation)"
        }
    ]
    
    # Phase selection
    phase = known_args.phase
    if phase == "draft":
        pipeline_steps = all_steps[:5]
    elif phase == "validate":
        pipeline_steps = [all_steps[0], all_steps[1], all_steps[2], all_steps[5]]
        # Append team-file if provided
        if known_args.team_file:
            pipeline_steps[-1]["script"] += f" --team-file {known_args.team_file}"
    else:
        # full
        pipeline_steps = all_steps
        if known_args.team_file:
            pipeline_steps[-1]["script"] += f" --team-file {known_args.team_file}"
    
    # Track results
    results = []
    total_time = 0
    
    log_write(f"Selected phase: {phase}")
    if phase == "validate" and not known_args.team_file:
        log_write("⚠️  --team-file not provided; Monte Carlo will require it and may fail.")
    
    # Run each step with clear visibility
    for i, step in enumerate(pipeline_steps, 1):
        log_write(f"\n🔄 Step {i}/{len(pipeline_steps)}: {step['name']}")
        
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
            log_write(f"\n🚨 Pipeline failed at step {i}: {step['name']}")
            log_write("🔄 Please fix the issue and re-run the pipeline")
            break
        
        log_write(f"✅ Step {i} completed. Moving to next step...")
        log_write("")
    
    # Pipeline summary
    log_write(f"\n{'='*60}")
    log_write("BASIC PIPELINE SUMMARY")
    log_write(f"{'='*60}")
    
    successful_steps = sum(1 for r in results if r['success'])
    total_steps = len(results)
    
    log_write(f"📊 Steps completed: {successful_steps}/{total_steps}")
    log_write(f"⏱️  Total time: {total_time:.1f} seconds")
    log_write(f"🎯 Success rate: {(successful_steps/total_steps)*100:.1f}%")
    
    # Enhanced status reporting
    if successful_steps == total_steps:
        log_write("\n🎉 Pipeline completed successfully!")
        log_write("📈 Ready for fantasy football analysis!")
    elif successful_steps == 0:
        log_write("\n❌ Pipeline failed completely. Check all steps.")
        log_write("🔧 Review error messages and fix issues before re-running.")
    else:
        log_write(f"\n⚠️  Pipeline partially completed ({successful_steps}/{total_steps} steps)")
        log_write("🔧 Check failed steps and re-run as needed")
    
    log_write(f"\n🏁 Pipeline finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Close log file handle if open
    try:
        if LOG_FILE_HANDLE is not None:
            LOG_FILE_HANDLE.close()
    except Exception:
        pass
    
    # Return exit code for automation
    return 0 if successful_steps == total_steps else 1

if __name__ == "__main__":
    main()

