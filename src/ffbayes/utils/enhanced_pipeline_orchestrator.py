#!/usr/bin/env python3
"""
Enhanced Pipeline Orchestrator for Fantasy Football Analytics.

This module provides advanced pipeline orchestration with:
- Dependency management and stage sequencing
- Sequential execution with internal multiprocessing (each step can use multiple cores)
- Comprehensive error recovery and graceful degradation
- Progress monitoring and performance tracking
- Configuration management and validation

Note: Parallel execution refers to using multiple CPU cores within individual tasks,
not running multiple heavy tasks simultaneously (which would overwhelm most computers).
"""

import importlib
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StepStatus(Enum):
	"""Pipeline step execution status."""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	SKIPPED = "skipped"
	TIMEOUT = "timeout"


class PipelineError(Exception):
	"""Custom exception for pipeline errors."""
	pass


@dataclass
class PipelineStep:
	"""Represents a single pipeline step."""
	name: str
	script: str
	args: str = ""
	description: str
	dependencies: List[str]
	timeout: int
	retry_count: int
	critical: bool
	parallel_group: str
	status: StepStatus = StepStatus.PENDING
	start_time: Optional[float] = None
	end_time: Optional[float] = None
	execution_time: Optional[float] = None
	retry_attempts: int = 0
	error_message: Optional[str] = None
	exit_code: Optional[int] = None


@dataclass
class PipelineResult:
	"""Result of pipeline execution."""
	step_name: str
	success: bool
	execution_time: float
	retry_attempts: int
	error_message: Optional[str] = None
	exit_code: Optional[int] = None
	output: Optional[str] = None
	stderr: Optional[str] = None


class EnhancedPipelineOrchestrator:
	"""Enhanced pipeline orchestrator with advanced features."""
	
	def __init__(self, config_file: str = "config/pipeline_config.json"):
		"""Initialize the enhanced pipeline orchestrator.
		
		Args:
			config_file: Path to pipeline configuration file
		"""
		self.config_file = config_file
		self.config = self._load_configuration()
		self.steps = self._create_pipeline_steps()
		self.execution_order = self._calculate_execution_order()
		self.results: List[PipelineResult] = []
		self.pipeline_start_time: Optional[float] = None
		self.pipeline_end_time: Optional[float] = None
		
		# Performance tracking
		self.performance_metrics = {
			'total_steps': len(self.steps),
			'completed_steps': 0,
			'failed_steps': 0,
			'total_execution_time': 0.0,
			'parallel_efficiency': 0.0
		}
		
		# Error recovery state
		self.error_recovery_state = {
			'total_retries': 0,
			'critical_failures': 0,
			'graceful_degradation_active': False
		}
	
	def _load_configuration(self) -> Dict[str, Any]:
		"""Load pipeline configuration from file.
		
		Returns:
			Pipeline configuration dictionary
			
		Raises:
			PipelineError: If configuration file cannot be loaded or is invalid
		"""
		try:
			if not os.path.exists(self.config_file):
				raise PipelineError(f"Configuration file not found: {self.config_file}")
			
			with open(self.config_file, 'r') as f:
				config = json.load(f)
			
			# Validate configuration structure
			self._validate_configuration(config)
			
			logger.info(f"Configuration loaded from {self.config_file}")
			return config
			
		except json.JSONDecodeError as e:
			raise PipelineError(f"Invalid JSON in configuration file: {e}")
		except Exception as e:
			raise PipelineError(f"Failed to load configuration: {e}")
	
	def _validate_configuration(self, config: Dict[str, Any]) -> None:
		"""Validate pipeline configuration structure and values.
		
		Args:
			config: Configuration dictionary to validate
			
		Raises:
			PipelineError: If configuration is invalid
		"""
		required_keys = ['pipeline_steps', 'global_config', 'parallel_groups']
		for key in required_keys:
			if key not in config:
				raise PipelineError(f"Missing required configuration key: {key}")
		
		# Validate pipeline steps
		if not config['pipeline_steps']:
			raise PipelineError("Pipeline must have at least one step")
		
		step_names = set()
		for step in config['pipeline_steps']:
			required_step_keys = ['name', 'script', 'description', 'dependencies', 
								'timeout', 'retry_count', 'critical', 'parallel_group']
			
			for key in required_step_keys:
				if key not in step:
					raise PipelineError(f"Step '{step.get('name', 'unknown')}' missing required key: {key}")
			
			# Check for duplicate step names
			if step['name'] in step_names:
				raise PipelineError(f"Duplicate step name: {step['name']}")
			step_names.add(step['name'])
			
			# Validate timeout and retry values
			if step['timeout'] <= 0:
				raise PipelineError(f"Step '{step['name']}' must have positive timeout")
			if step['retry_count'] < 0:
				raise PipelineError(f"Step '{step['name']}' must have non-negative retry count")
		
		# Validate global configuration
		global_config = config['global_config']
		if global_config['max_parallel_steps'] <= 0:
			raise PipelineError("max_parallel_steps must be positive")
		if global_config['pipeline_timeout'] <= 0:
			raise PipelineError("pipeline_timeout must be positive")
	
	def _create_pipeline_steps(self) -> List[PipelineStep]:
		"""Create PipelineStep objects from configuration.
		
		Returns:
			List of PipelineStep objects
		"""
		steps = []
		for step_config in self.config['pipeline_steps']:
			step = PipelineStep(
				name=step_config['name'],
				script=step_config['script'],
				args=step_config.get('args', ""),
				description=step_config['description'],
				dependencies=step_config['dependencies'],
				timeout=step_config['timeout'],
				retry_count=step_config['retry_count'],
				critical=step_config['critical'],
				parallel_group=step_config['parallel_group']
			)
			steps.append(step)
		return steps
	
	def _calculate_execution_order(self) -> List[str]:
		"""Calculate the order in which steps should be executed based on dependencies.
		
		Returns:
			List of step names in execution order
			
		Raises:
			PipelineError: If circular dependencies are detected
		"""
		# Create dependency graph
		dependency_graph = {step.name: step.dependencies for step in self.steps}
		
		# Check for circular dependencies
		if self._has_circular_dependencies(dependency_graph):
			raise PipelineError("Circular dependencies detected in pipeline configuration")
		
		# Topological sort
		execution_order = []
		visited = set()
		temp_visited = set()
		
		def visit(step_name: str) -> None:
			if step_name in temp_visited:
				raise PipelineError(f"Circular dependency detected involving step: {step_name}")
			if step_name in visited:
				return
			
			temp_visited.add(step_name)
			
			# Visit dependencies first
			for dependency in dependency_graph.get(step_name, []):
				visit(dependency)
			
			temp_visited.remove(step_name)
			visited.add(step_name)
			execution_order.append(step_name)
		
		# Visit all steps
		for step_name in dependency_graph:
			if step_name not in visited:
				visit(step_name)
		
		return execution_order
	
	def _has_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> bool:
		"""Check for circular dependencies in the dependency graph.
		
		Args:
			dependency_graph: Dictionary mapping step names to their dependencies
			
		Returns:
			True if circular dependencies exist, False otherwise
		"""
		visited = set()
		rec_stack = set()
		
		def has_circle(node: str) -> bool:
			visited.add(node)
			rec_stack.add(node)
			
			for neighbor in dependency_graph.get(node, []):
				if neighbor not in visited:
					if has_circle(neighbor):
						return True
				elif neighbor in rec_stack:
					return True
			
			rec_stack.remove(node)
			return False
		
		for node in dependency_graph:
			if node not in visited:
				if has_circle(node):
					return True
		
		return False
	
	def _get_ready_steps(self, completed_steps: set) -> List[PipelineStep]:
		"""Get steps that are ready to execute (all dependencies completed).
		
		Args:
			completed_steps: Set of completed step names
			
		Returns:
			List of PipelineStep objects ready for execution
		"""
		ready_steps = []
		for step in self.steps:
			if (step.status == StepStatus.PENDING and 
				step.name not in completed_steps and
				all(dep in completed_steps for dep in step.dependencies)):
				ready_steps.append(step)
		return ready_steps
	
	def _can_execute_parallel(self, step: PipelineStep, running_steps: List[PipelineStep]) -> bool:
		"""Check if a step can be executed in parallel with currently running steps.
		
		Args:
			step: Step to check
			running_steps: Currently running steps
			
		Returns:
			True if step can run in parallel, False otherwise
		"""
		# Check parallel group limits
		group = self.config['parallel_groups'].get(step.parallel_group, {})
		max_concurrent = group.get('max_concurrent', 1)
		
		# Count steps in the same group that are currently running
		running_in_group = sum(1 for s in running_steps if s.parallel_group == step.parallel_group)
		
		return running_in_group < max_concurrent
	
	def _execute_step(self, step: PipelineStep) -> PipelineResult:
		"""Execute a single pipeline step.
		
		Args:
			step: PipelineStep to execute
			
		Returns:
			PipelineResult containing execution results
		"""
		step.status = StepStatus.RUNNING
		step.start_time = time.time()
		
		logger.info(f"Executing step: {step.name}")
		
		# Support Python module execution (preferred)
		# If the script string looks like a Python module path, try running with -m
		module_name = None
		if '.' in step.script and os.path.sep not in step.script:
			try:
				importlib.import_module(step.script)
				module_name = step.script
			except Exception:
				module_name = None
		
		try:
			# Execute the script or module
			if module_name:
				cmd = [sys.executable, "-m", module_name] + (shlex.split(step.args) if step.args else [])
			else:
				# Fall back to file path execution if it exists
				if not os.path.exists(step.script):
					error_msg = f"Script not found: {step.script}"
					step.status = StepStatus.FAILED
					step.error_message = error_msg
					step.end_time = time.time()
					step.execution_time = step.end_time - step.start_time
					return PipelineResult(
						step_name=step.name,
						success=False,
						execution_time=step.execution_time,
						retry_attempts=step.retry_attempts,
						error_message=error_msg,
						exit_code=-1
					)
				cmd = [sys.executable, step.script]
			
			result = subprocess.run(
				cmd,
				capture_output=True,
				text=True,
				timeout=step.timeout
			)
			
			step.end_time = time.time()
			step.execution_time = step.end_time - step.start_time
			step.exit_code = result.returncode
			
			if result.returncode == 0:
				step.status = StepStatus.COMPLETED
				logger.info(f"Step {step.name} completed successfully in {step.execution_time:.1f}s")
				return PipelineResult(
					step_name=step.name,
					success=True,
					execution_time=step.execution_time,
					retry_attempts=step.retry_attempts,
					exit_code=result.returncode,
					output=result.stdout
				)
			else:
				step.status = StepStatus.FAILED
				error_msg = f"Step failed with exit code {result.returncode}"
				step.error_message = error_msg
				logger.error(f"Step {step.name} failed: {error_msg}")
				return PipelineResult(
					step_name=step.name,
					success=False,
					execution_time=step.execution_time,
					retry_attempts=step.retry_attempts,
					error_message=error_msg,
					exit_code=result.returncode,
					stderr=result.stderr
				)
		except subprocess.TimeoutExpired:
			step.status = StepStatus.TIMEOUT
			step.end_time = time.time()
			step.execution_time = step.end_time - step.start_time
			error_msg = f"Step timed out after {step.timeout} seconds"
			step.error_message = error_msg
			logger.error(f"Step {step.name} timed out")
			return PipelineResult(
				step_name=step.name,
				success=False,
				execution_time=step.execution_time,
				retry_attempts=step.retry_attempts,
				error_message=error_msg,
				exit_code=-1
			)
		except Exception as e:
			step.status = StepStatus.FAILED
			step.end_time = time.time()
			step.execution_time = step.end_time - step.start_time
			error_msg = f"Unexpected error: {str(e)}"
			step.error_message = error_msg
			logger.error(f"Step {step.name} failed with unexpected error: {e}")
			return PipelineResult(
				step_name=step.name,
				success=False,
				execution_time=step.execution_time,
				retry_attempts=step.retry_attempts,
				error_message=error_msg,
				exit_code=-1
			)
		
		try:
			# Execute the script
			result = subprocess.run(
				[sys.executable, step.script],
				capture_output=True,
				text=True,
				timeout=step.timeout
			)
			
			step.end_time = time.time()
			step.execution_time = step.end_time - step.start_time
			step.exit_code = result.returncode
			
			if result.returncode == 0:
				step.status = StepStatus.COMPLETED
				logger.info(f"Step {step.name} completed successfully in {step.execution_time:.1f}s")
				
				return PipelineResult(
					step_name=step.name,
					success=True,
					execution_time=step.execution_time,
					retry_attempts=step.retry_attempts,
					exit_code=result.returncode,
					output=result.stdout
				)
			else:
				step.status = StepStatus.FAILED
				error_msg = f"Step failed with exit code {result.returncode}"
				step.error_message = error_msg
				logger.error(f"Step {step.name} failed: {error_msg}")
				
				return PipelineResult(
					step_name=step.name,
					success=False,
					execution_time=step.execution_time,
					retry_attempts=step.retry_attempts,
					error_message=error_msg,
					exit_code=result.returncode,
					stderr=result.stderr
				)
				
		except subprocess.TimeoutExpired:
			step.status = StepStatus.TIMEOUT
			step.end_time = time.time()
			step.execution_time = step.end_time - step.start_time
			error_msg = f"Step timed out after {step.timeout} seconds"
			step.error_message = error_msg
			logger.error(f"Step {step.name} timed out")
			
			return PipelineResult(
				step_name=step.name,
				success=False,
				execution_time=step.execution_time,
				retry_attempts=step.retry_attempts,
				error_message=error_msg,
				exit_code=-1
			)
			
		except Exception as e:
			step.status = StepStatus.FAILED
			step.end_time = time.time()
			step.execution_time = step.end_time - step.start_time
			error_msg = f"Unexpected error: {str(e)}"
			step.error_message = error_msg
			logger.error(f"Step {step.name} failed with unexpected error: {e}")
			
			return PipelineResult(
				step_name=step.name,
				success=False,
				execution_time=step.execution_time,
				retry_attempts=step.retry_attempts,
				error_message=error_msg,
				exit_code=-1
			)
	
	def _should_retry_step(self, step: PipelineStep, result: PipelineResult) -> bool:
		"""Determine if a step should be retried.
		
		Args:
			step: PipelineStep that failed
			result: PipelineResult from the failed execution
			
		Returns:
			True if step should be retried, False otherwise
		"""
		# Check if we've exceeded retry attempts
		if step.retry_attempts >= step.retry_count:
			return False
		
		# Check if we've exceeded total retries
		if self.error_recovery_state['total_retries'] >= self.config['error_handling']['max_total_retries']:
			return False
		
		# Don't retry timeout errors (they're usually not transient)
		if result.error_message and "timed out" in result.error_message:
			return False
		
		return True
	
	def _retry_step(self, step: PipelineStep) -> PipelineResult:
		"""Retry a failed pipeline step.
		
		Args:
			step: PipelineStep to retry
			
		Returns:
			PipelineResult from retry attempt
		"""
		step.retry_attempts += 1
		step.status = StepStatus.PENDING
		step.start_time = None
		step.end_time = None
		step.execution_time = None
		step.error_message = None
		step.exit_code = None
		
		self.error_recovery_state['total_retries'] += 1
		
		# Calculate retry delay with exponential backoff
		retry_delay = self.config['global_config']['retry_delay']
		if self.config['error_handling']['retry_strategies']['exponential_backoff']:
			backoff_multiplier = min(step.retry_attempts, 
									self.config['error_handling']['retry_strategies']['max_backoff_multiplier'])
			retry_delay *= (2 ** (backoff_multiplier - 1))
		
		# Add jitter if enabled
		if self.config['error_handling']['retry_strategies']['jitter']:
			import random
			jitter = random.uniform(0.8, 1.2)
			retry_delay *= jitter
		
		logger.info(f"Retrying step {step.name} in {retry_delay:.1f}s (attempt {step.retry_attempts})")
		time.sleep(retry_delay)
		
		return self._execute_step(step)
	
	def _handle_step_failure(self, step: PipelineStep, result: PipelineResult) -> bool:
		"""Handle step failure and determine next action.
		
		Args:
			step: PipelineStep that failed
			result: PipelineResult from the failed execution
			
		Returns:
			True if pipeline should continue, False if it should stop
		"""
		if step.critical:
			self.error_recovery_state['critical_failures'] += 1
			
			# Check if we should stop the pipeline
			if self.config['error_handling']['critical_failure_action'] == 'stop_pipeline':
				logger.error(f"Critical step {step.name} failed. Stopping pipeline.")
				return False
		
		# Try to retry the step
		if self._should_retry_step(step, result):
			retry_result = self._retry_step(step)
			if retry_result.success:
				logger.info(f"Step {step.name} succeeded on retry")
				self.results.append(retry_result)
				return True
		
		# Step failed after retries
		if step.critical:
			logger.error(f"Critical step {step.name} failed after retries. Pipeline cannot continue.")
			return False
		
		# Non-critical step failed - check graceful degradation
		if self.config['error_handling']['non_critical_failure_action'] == 'continue_with_warnings':
			logger.warning(f"Non-critical step {step.name} failed. Continuing with warnings.")
			step.status = StepStatus.SKIPPED
			return True
		
		return False
	
	def _monitor_resources(self) -> Dict[str, float]:
		"""Monitor system resource usage.
		
		Returns:
			Dictionary containing resource usage percentages
		"""
		try:
			cpu_percent = psutil.cpu_percent(interval=1)
			memory = psutil.virtual_memory()
			disk = psutil.disk_usage('/')
			
			return {
				'cpu_percent': cpu_percent,
				'memory_percent': memory.percent,
				'disk_percent': disk.percent
			}
		except Exception as e:
			logger.warning(f"Failed to monitor resources: {e}")
			return {'cpu_percent': 0.0, 'memory_percent': 0.0, 'disk_percent': 0.0}
	
	def _check_resource_alerts(self, resource_usage: Dict[str, float]) -> None:
		"""Check resource usage against alert thresholds.
		
		Args:
			resource_usage: Current resource usage percentages
		"""
		thresholds = self.config['monitoring']['alert_thresholds']
		
		if resource_usage['cpu_percent'] > thresholds['cpu_usage_warning'] * 100:
			logger.warning(f"High CPU usage: {resource_usage['cpu_percent']:.1f}%")
		
		if resource_usage['memory_percent'] > thresholds['memory_usage_warning'] * 100:
			logger.warning(f"High memory usage: {resource_usage['memory_percent']:.1f}%")
	
	def execute_pipeline(self) -> bool:
		"""Execute the complete pipeline with enhanced orchestration.
		
		Returns:
			True if pipeline completed successfully, False otherwise
		"""
		logger.info("Starting enhanced pipeline execution")
		self.pipeline_start_time = time.time()
		
		# Check pipeline timeout
		pipeline_timeout = self.config['global_config']['pipeline_timeout']
		
		completed_steps = set()
		running_steps = []
		
		try:
			while len(completed_steps) < len(self.steps):
				# Check pipeline timeout
				if time.time() - self.pipeline_start_time > pipeline_timeout:
					logger.error(f"Pipeline timed out after {pipeline_timeout} seconds")
					return False
				
				# Get steps ready for execution
				ready_steps = self._get_ready_steps(completed_steps)
				
				# Start new steps if possible
				for step in ready_steps:
					if self._can_execute_parallel(step, running_steps):
						# Execute step in thread
						thread = threading.Thread(
							target=self._execute_step_thread,
							args=(step, completed_steps, running_steps)
						)
						thread.daemon = True
						thread.start()
						running_steps.append(step)
				
				# Check for completed, failed, and skipped steps
				completed_steps.update([step.name for step in self.steps if step.status in 
									[StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.TIMEOUT, StepStatus.SKIPPED]])
				
				# Remove completed steps from running list
				running_steps = [step for step in running_steps if step.status not in 
							   [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.TIMEOUT, StepStatus.SKIPPED]]
				
				# Check if all steps are done (including failed ones)
				if len(completed_steps) >= len(self.steps):
					break
				
				# Monitor resources
				if self.config['monitoring']['resource_usage_tracking']:
					resource_usage = self._monitor_resources()
					self._check_resource_alerts(resource_usage)
				
				# Progress update
				if self.config['monitoring']['progress_update_interval'] > 0:
					if int(time.time() - self.pipeline_start_time) % self.config['monitoring']['progress_update_interval'] == 0:
						self._print_progress(completed_steps)
				
				time.sleep(1)  # Small delay to prevent busy waiting
			
			# All steps completed
			self.pipeline_end_time = time.time()
			self._finalize_pipeline()
			
			# Check if any critical steps failed
			critical_failures = any(step.status == StepStatus.FAILED and step.critical for step in self.steps)
			if critical_failures:
				logger.error("Critical steps failed - pipeline execution unsuccessful")
				return False
			
			return True
			
		except KeyboardInterrupt:
			logger.info("Pipeline interrupted by user")
			return False
		except Exception as e:
			logger.error(f"Pipeline execution failed: {e}")
			return False
	
	def _execute_step_thread(self, step: PipelineStep, completed_steps: set, running_steps: List[PipelineStep]) -> None:
		"""Execute a step in a separate thread.
		
		Args:
			step: PipelineStep to execute
			completed_steps: Set of completed step names
			running_steps: List of currently running steps
		"""
		try:
			result = self._execute_step(step)
			
			if result.success:
				self.results.append(result)
				completed_steps.add(step.name)
				self.performance_metrics['completed_steps'] += 1
			else:
				# Handle step failure
				should_continue = self._handle_step_failure(step, result)
				if not should_continue:
					# Critical failure - stop pipeline
					return
				
				self.performance_metrics['failed_steps'] += 1
				
				if step.status == StepStatus.SKIPPED:
					completed_steps.add(step.name)
			
		except Exception as e:
			logger.error(f"Error executing step {step.name}: {e}")
			step.status = StepStatus.FAILED
			step.error_message = str(e)
	
	def _print_progress(self, completed_steps: set) -> None:
		"""Print pipeline progress update.
		
		Args:
			completed_steps: Set of completed step names
		"""
		total_steps = len(self.steps)
		completed_count = len(completed_steps)
		progress_percent = (completed_count / total_steps) * 100
		
		elapsed_time = time.time() - self.pipeline_start_time
		if completed_count > 0:
			estimated_total = elapsed_time * total_steps / completed_count
			eta = estimated_total - elapsed_time
			eta_str = f"ETA: {eta:.1f}s"
		else:
			eta_str = "ETA: unknown"
		
		logger.info(f"Pipeline Progress: {completed_count}/{total_steps} steps completed ({progress_percent:.1f}%) - {eta_str}")
	
	def _finalize_pipeline(self) -> None:
		"""Finalize pipeline execution and calculate metrics."""
		self.pipeline_end_time = time.time()
		total_time = self.pipeline_end_time - self.pipeline_start_time
		
		# Calculate performance metrics
		self.performance_metrics['total_execution_time'] = total_time
		
		# Calculate parallel efficiency
		if self.performance_metrics['completed_steps'] > 0:
			theoretical_sequential_time = sum(result.execution_time for result in self.results)
			self.performance_metrics['parallel_efficiency'] = theoretical_sequential_time / total_time
		
		# Log final summary
		logger.info("Pipeline execution completed")
		logger.info(f"Total execution time: {total_time:.1f}s")
		logger.info(f"Steps completed: {self.performance_metrics['completed_steps']}")
		logger.info(f"Steps failed: {self.performance_metrics['failed_steps']}")
		logger.info(f"Parallel efficiency: {self.performance_metrics['parallel_efficiency']:.2f}")
	
	def get_execution_summary(self) -> Dict[str, Any]:
		"""Get comprehensive execution summary.
		
		Returns:
			Dictionary containing execution summary and metrics
		"""
		return {
			'pipeline_info': {
				'total_steps': len(self.steps),
				'execution_order': self.execution_order,
				'start_time': datetime.fromtimestamp(self.pipeline_start_time).isoformat() if self.pipeline_start_time else None,
				'end_time': datetime.fromtimestamp(self.pipeline_end_time).isoformat() if self.pipeline_end_time else None,
				'total_time': self.performance_metrics['total_execution_time']
			},
			'performance_metrics': self.performance_metrics,
			'error_recovery_state': self.error_recovery_state,
			'step_results': [
				{
					'step_name': result.step_name,
					'success': result.success,
					'execution_time': result.execution_time,
					'retry_attempts': result.retry_attempts,
					'error_message': result.error_message,
					'exit_code': result.exit_code
				}
				for result in self.results
			],
			'step_statuses': [
				{
					'name': step.name,
					'status': step.status.value,
					'execution_time': step.execution_time,
					'retry_attempts': step.retry_attempts,
					'error_message': step.error_message
				}
				for step in self.steps
			]
		}


def main():
	"""Main function to run the enhanced pipeline orchestrator."""
	try:
		# Initialize orchestrator
		orchestrator = EnhancedPipelineOrchestrator()
		
		# Execute pipeline
		success = orchestrator.execute_pipeline()
		
		# Get and display summary
		summary = orchestrator.get_execution_summary()
		
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
				print(f"	Retries: {result['retry_attempts']}")
			if result['error_message']:
				print(f"	Error: {result['error_message']}")
		
		print("="*80)
		
		return 0 if success else 1
		
	except Exception as e:
		logger.error(f"Pipeline orchestration failed: {e}")
		return 1


if __name__ == "__main__":
	sys.exit(main())
