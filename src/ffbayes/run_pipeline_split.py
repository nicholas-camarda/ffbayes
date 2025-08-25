#!/usr/bin/env python3
"""
Split pipeline runner for fantasy football analysis.
Runs either pre-draft or post-draft pipeline based on user selection.
Only creates directories when steps actually run.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict


class SplitPipelineRunner:
    """Pipeline runner that dynamically creates directories and runs split pipelines."""
    
    def __init__(self, pipeline_type: str = "pre_draft"):
        """
        Initialize pipeline runner.
        
        Args:
            pipeline_type: Either "pre_draft" or "post_draft"
        """
        self.pipeline_type = pipeline_type
        self.current_year = datetime.now().year
        from ffbayes.utils.path_constants import (get_post_draft_config_file,
                                                  get_pre_draft_config_file)
        if pipeline_type == "pre_draft":
            self.config_file = str(get_pre_draft_config_file())
        else:
            self.config_file = str(get_post_draft_config_file())
        self.config = self.load_config()
        self.created_dirs = set()  # Track which directories we actually create

        # Initialize simple file logging
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_path = logs_dir / f"pipeline-{self.pipeline_type}-{timestamp}.log"
        self.log_file = self.log_path.open('a', encoding='utf-8')
        self._log_header()
    
    def _log(self, message: str = ""):
        try:
            self.log_file.write(message + "\n")
            self.log_file.flush()
        except Exception:
            pass
    
    def _log_header(self):
        self._log("🏈 FANTASY FOOTBALL ANALYTICS PIPELINE (Split Runner)")
        self._log("=" * 80)
        self._log(f"Pipeline Type: {self.pipeline_type}")
        self._log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 80)

    def load_config(self) -> Dict:
        """Load pipeline configuration."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Pipeline config not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def create_directory_if_needed(self, dir_path: str, description: str = ""):
        """Create directory only when needed, with tracking."""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            self.created_dirs.add(dir_path)
            print(f"📁 Created: {dir_path} {description}")
            self._log(f"📁 Created: {dir_path} {description}")
    
    def create_output_directories(self, step_name: str):
        """Create output directories for a specific step."""
        # All directories are now created centrally in path_constants.py
        # This function is kept for compatibility but no longer creates directories
        pass
    
    def run_step(self, step: Dict) -> bool:
        """Run a single pipeline step."""
        step_name = step['name']
        script = step['script']
        description = step.get('description', '')
        
        print(f"\n{'='*60}")
        print(f"🚀 Running: {step_name}")
        print(f"📝 {description}")
        print(f"🔧 Script: {script}")
        print(f"{'='*60}")
        
        self._log(f"\n{'='*60}")
        self._log(f"STEP: {step_name}")
        self._log(f"{description}")
        self._log(f"SCRIPT: {script}")
        
        # Create directories for this step BEFORE running it
        self.create_output_directories(step_name)
        
        # Build command
        cmd = [sys.executable, "-m", script]
        if step.get('args'):
            cmd.extend(step['args'].split())
        
        try:
            # Run the step
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=step.get('timeout', 300))
            end_time = time.time()
            
            # Print output
            if result.stdout:
                print("📤 Output:")
                print(result.stdout)
                self._log("\n📤 Output:")
                self._log(result.stdout.rstrip("\n"))
            
            if result.stderr:
                print("⚠️  Errors/Warnings:")
                print(result.stderr)
                self._log("\n⚠️  Errors/Warnings:")
                self._log(result.stderr.rstrip("\n"))
            
            # Check result
            if result.returncode == 0:
                duration = end_time - start_time
                print(f"✅ {step_name} completed successfully in {duration:.1f}s")
                self._log(f"✅ {step_name} completed successfully in {duration:.1f}s")
                
                # Post-processing: Organize files automatically
                self.organize_step_outputs(step_name)
                
                return True
            else:
                print(f"❌ {step_name} failed with return code {result.returncode}")
                self._log(f"❌ {step_name} failed with return code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {step_name} timed out after {step.get('timeout', 300)}s")
            self._log(f"⏰ {step_name} timed out after {step.get('timeout', 300)}s")
            return False
        except Exception as e:
            print(f"💥 {step_name} failed with error: {e}")
            self._log(f"💥 {step_name} failed with error: {e}")
            return False
    
    def check_dependencies(self, step: Dict, completed_steps: set) -> bool:
        """Check if all dependencies for a step are satisfied."""
        dependencies = step.get('dependencies', [])
        missing_deps = [dep for dep in dependencies if dep not in completed_steps]
        
        if missing_deps:
            print(f"⏳ {step['name']} waiting for dependencies: {missing_deps}")
            self._log(f"⏳ {step['name']} waiting for dependencies: {missing_deps}")
            return False
        
        return True
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
        print(f"\n{'='*80}")
        print(f"🎯 Running {self.pipeline_type.upper().replace('_', ' ')} Pipeline")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        self._log(f"\n{'='*80}")
        self._log(f"🎯 Running {self.pipeline_type.upper().replace('_', ' ')} Pipeline")
        self._log(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"{'='*80}")
        
        steps = self.config['steps']
        completed_steps = set()
        failed_steps = []
        
        # Track execution order
        execution_order = []
        
        # Run steps in dependency order
        while len(completed_steps) < len(steps):
            progress_made = False
            
            for step in steps:
                if step['name'] in completed_steps:
                    continue
                
                if self.check_dependencies(step, completed_steps):
                    # Run the step
                    success = self.run_step(step)
                    
                    if success:
                        completed_steps.add(step['name'])
                        execution_order.append(step['name'])
                        progress_made = True
                        print(f"📊 Progress: {len(completed_steps)}/{len(steps)} steps completed")
                        self._log(f"📊 Progress: {len(completed_steps)}/{len(steps)} steps completed")
                    else:
                        failed_steps.append(step['name'])
                        if step.get('critical', True):
                            print(f"💥 Critical step {step['name']} failed - stopping pipeline")
                            self._log(f"💥 Critical step {step['name']} failed - stopping pipeline")
                            break
                        else:
                            print(f"⚠️  Non-critical step {step['name']} failed - continuing")
                            self._log(f"⚠️  Non-critical step {step['name']} failed - continuing")
                            completed_steps.add(step['name'])  # Mark as "completed" even if failed
                            progress_made = True
            
            if not progress_made:
                print("🔄 No progress made - dependency cycle detected")
                self._log("🔄 No progress made - dependency cycle detected")
                break
        
        # Summary
        print(f"\n{'='*80}")
        print("🎉 PIPELINE EXECUTION COMPLETE")
        print(f"{'='*80}")
        
        self._log(f"\n{'='*80}")
        self._log("🎉 PIPELINE EXECUTION COMPLETE")
        self._log(f"{'='*80}")
        
        print(f"✅ Completed: {len(completed_steps)}/{len(steps)} steps")
        self._log(f"✅ Completed: {len(completed_steps)}/{len(steps)} steps")
        if failed_steps:
            # De-duplicate failed steps for cleaner summary
            unique_failed = list(dict.fromkeys(failed_steps))
            print(f"❌ Failed: {len(unique_failed)} steps")
            self._log(f"❌ Failed: {len(unique_failed)} steps")
            for step in unique_failed:
                print(f"   - {step}")
                self._log(f"   - {step}")
        
        print(f"\n📁 Directories created: {len(self.created_dirs)}")
        self._log(f"\n📁 Directories created: {len(self.created_dirs)}")
        for dir_path in sorted(self.created_dirs):
            print(f"   - {dir_path}")
            self._log(f"   - {dir_path}")
        
        print("\n🔄 Execution order:")
        self._log("\n🔄 Execution order:")
        for i, step in enumerate(execution_order, 1):
            print(f"   {i:2d}. {step}")
            self._log(f"   {i:2d}. {step}")
        
        success_rate = len(completed_steps) / len(steps) * 100
        print(f"\n📊 Success Rate: {success_rate:.1f}%")
        self._log(f"\n📊 Success Rate: {success_rate:.1f}%")
        
        if failed_steps:
            print("\n💥 Pipeline completed with errors!")
            self._log("\n💥 Pipeline completed with errors!")
        else:
            print("\n🎉 Pipeline completed successfully!")
            self._log("\n🎉 Pipeline completed successfully!")
            
            # Manage visualizations after successful pipeline completion
            try:
                from ffbayes.utils.visualization_manager import \
                    manage_visualizations
                print("\n🖼️  Managing visualizations...")
                self._log("\n🖼️  Managing visualizations...")
                viz_results = manage_visualizations(self.current_year)
                print(f"✅ Visualization management complete: {len(viz_results['copied_files'])} files copied to docs/images/")
                self._log(f"✅ Visualization management complete: {len(viz_results['copied_files'])} files copied to docs/images/")
            except Exception as e:
                print(f"⚠️  Visualization management failed: {e}")
                self._log(f"⚠️  Visualization management failed: {e}")
        
        try:
            self.log_file.close()
        except Exception:
            pass
        
        if failed_steps:
            return False
        return True
    
    def organize_step_outputs(self, step_name: str):
        """Automatically organize outputs after each step completes."""
        if step_name == 'vor_draft_strategy':
            self.organize_vor_outputs()
    
    def organize_vor_outputs(self):
        """Organize VOR strategy outputs automatically."""
        current_year = datetime.now().year
        
        # Source: Raw VOR data and processed strategy in datasets
        from ffbayes.utils.path_constants import SNAKE_DRAFT_DATASETS_DIR
        source_dir = str(SNAKE_DRAFT_DATASETS_DIR)
        
        # Destination: Organized results structure
        from ffbayes.utils.path_constants import get_vor_strategy_dir
        dest_dir = str(get_vor_strategy_dir(current_year))
        
        # Files to copy to organized structure
        from ffbayes.utils.vor_filename_generator import (
            get_vor_csv_filename, get_vor_excel_filename)
        
        files_to_copy = [
            get_vor_excel_filename(current_year),
            get_vor_csv_filename(current_year)
        ]
        
        print("   📁 Organizing VOR outputs...")
        self._log("   📁 Organizing VOR outputs...")
        
        for filename in files_to_copy:
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            if os.path.exists(source_path):
                # Copy to organized structure
                import shutil
                shutil.copy2(source_path, dest_path)
                print(f"      ✅ Organized: {filename}")
                self._log(f"      ✅ Organized: {filename}")
            else:
                print(f"      ⚠️  Not found: {filename}")
                self._log(f"      ⚠️  Not found: {filename}")
        
        print(f"      📊 Raw VOR data remains in: {source_dir}")
        print(f"      📁 Strategy copied to: {dest_dir}")
        self._log(f"      📊 Raw VOR data remains in: {source_dir}")
        self._log(f"      📁 Strategy copied to: {dest_dir}")


def main():
    """Main function to run split pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline_split.py <pipeline_type>")
        print("  pipeline_type: 'pre_draft' or 'post_draft'")
        sys.exit(1)
    
    pipeline_type = sys.argv[1].lower()
    
    if pipeline_type not in ['pre_draft', 'post_draft']:
        print("❌ Invalid pipeline type. Use 'pre_draft' or 'post_draft'")
        sys.exit(1)
    
    try:
        runner = SplitPipelineRunner(pipeline_type)
        success = runner.run_pipeline()
        
        if success:
            sys.exit(0)
        else:
            print("\n💥 Pipeline completed with errors!")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Pipeline runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
