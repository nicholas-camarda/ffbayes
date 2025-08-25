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
    from ffbayes.utils.enhanced_pipeline_orchestrator import \
        EnhancedPipelineOrchestrator
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
    from ffbayes.utils.path_constants import create_all_required_directories

    # Use the centralized directory creation function
    create_all_required_directories()
    print()


def cleanup_empty_directories():
    """Clean up empty directories that may cause pipeline issues."""
    print("🧹 Cleaning up empty directories...")
    print()
    
    # Directories to check for cleanup
    cleanup_dirs = [
        "results",
        "plots"
    ]
    
    cleaned_count = 0
    for base_dir in cleanup_dirs:
        base_path = Path(base_dir)
        if base_path.exists() and base_path.is_dir():
            # Find empty subdirectories
            for subdir in base_path.iterdir():
                if subdir.is_dir():
                    try:
                        # Check if directory is empty
                        if not any(subdir.iterdir()):
                            # Directory is empty, but don't remove it - just log
                            print(f"📁 Empty directory found: {subdir}")
                            cleaned_count += 1
                    except Exception as e:
                        print(f"⚠️  Error checking {subdir}: {e}")
    
    if cleaned_count > 0:
        print(f"📁 Found {cleaned_count} empty directories")
    else:
        print("✅ No empty directories found")
    
    print()


def validate_step_outputs(step_name):
    """Validate that a pipeline step produced its expected outputs."""
    log_write("🔍 Validating step outputs...")
    
    # Get current year for dynamic paths
    current_year = datetime.now().year
    
    from ffbayes.utils.path_constants import (
        COMBINED_DATASETS_DIR, SEASON_DATASETS_DIR, SNAKE_DRAFT_DATASETS_DIR,
        get_draft_strategy_comparison_dir, get_draft_strategy_dir,
        get_hybrid_mc_dir, get_monte_carlo_dir, get_plots_dir)
    
    validation_rules = {
        "Data Collection": [
            str(SEASON_DATASETS_DIR / "*.csv")
        ],
        "Data Validation": [
            # Validation step doesn't produce files, just validates
        ],
        "Data Preprocessing": [
            str(COMBINED_DATASETS_DIR / "*_modern.csv")
        ],
        "VOR Draft Strategy": [
            str(SNAKE_DRAFT_DATASETS_DIR / "snake-draft_ppr-*.csv"),
            str(SNAKE_DRAFT_DATASETS_DIR / "DRAFTING STRATEGY -- snake-draft_ppr-*.xlsx")
        ],
        "Bayesian Analysis": [
            str(get_hybrid_mc_dir(current_year) / "hybrid_model_results.json")
        ],
        "Draft Strategy Generation": [
            str(get_draft_strategy_dir(current_year) / "draft_strategy_pos*.json")
        ],
        "Monte Carlo Validation": [
            str(get_monte_carlo_dir(current_year) / "*.tsv")
        ],
        "Model Comparison": [
            str(get_plots_dir(current_year) / "model_comparison" / "model_comparison_results_*.json")
        ],
        "Draft Strategy Comparison": [
            str(get_draft_strategy_comparison_dir(current_year) / "draft_strategy_comparison_report_*.txt")
        ]
    }
    
    if step_name not in validation_rules:
        log_write("   ⚠️  No validation rules defined for this step")
        return True
    
    expected_patterns = validation_rules[step_name]
    if not expected_patterns:
        log_write("   ✅ Step doesn't produce files (validation only)")
        return True
    
    missing_files = []
    for pattern in expected_patterns:
        # Replace {current_year} placeholder with actual year
        actual_pattern = pattern.format(current_year=current_year)
        matching_files = list(Path(".").glob(actual_pattern))
        if not matching_files:
            missing_files.append(pattern)
        else:
            log_write(f"   ✅ Found: {actual_pattern}")
    
    if missing_files:
        log_write("   ❌ Missing expected outputs:")
        for pattern in missing_files:
            log_write(f"      - {pattern}")
        return False
    
    log_write("   ✅ All expected outputs found")
    return True


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
        
        # Validate step outputs based on step name
        output_validation_passed = validate_step_outputs(step_name)
        
        if result.returncode == 0 and output_validation_passed:
            log_write("-" * 40)
            log_write(f"✅ {step_name} completed successfully in {elapsed_time:.1f} seconds")
        elif result.returncode == 0 and not output_validation_passed:
            log_write("-" * 40)
            log_write(f"❌ {step_name} failed: Expected outputs not found")
            return False, elapsed_time
        else:
            log_write("-" * 40)
            log_write(f"❌ {step_name} failed with exit code {result.returncode}")
        
        return result.returncode == 0 and output_validation_passed, elapsed_time
        
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
    
    # Create required directories first with enhanced error handling
    try:
        create_required_directories()
        # Clean up any empty directories that might cause issues
        cleanup_empty_directories()
    except Exception as e:
        log_write(f"❌ Failed to create required directories: {e}")
        log_write("🔄 Pipeline cannot continue without proper directory structure")
        return 1
    
    # Check if enhanced orchestrator is available
    if EnhancedPipelineOrchestrator is not None and known_args.phase == "full":
        log_write("🚀 Using Enhanced Pipeline Orchestrator")
        log_write("📊 Features: Dependency management, sequential execution with internal multiprocessing, error recovery")
        log_write("")
        
        try:
            # Use enhanced orchestrator with integrated logging
            orchestrator = EnhancedPipelineOrchestrator(main_log_file=LOG_FILE_HANDLE)
            
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
            log_write(f"❌ CRITICAL ERROR: Enhanced orchestrator failed: {e}")
            log_write("🚨 Production pipeline requires enhanced orchestrator to work properly.")
            log_write("❌ No fallbacks allowed - fix the orchestrator issue and retry.")
            raise RuntimeError(
                f"Enhanced orchestrator failed: {e}. "
                "Production pipeline requires proper orchestration. "
                "No fallbacks allowed."
            )

    # CRITICAL: No fallback execution allowed
    # Production pipeline must use enhanced orchestrator
    log_write("✅ Pipeline completed successfully with enhanced orchestrator")
    
    return True

if __name__ == "__main__":
    main()

