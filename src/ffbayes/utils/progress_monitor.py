#!/usr/bin/env python3
"""
progress_monitor.py - Progress Monitoring Utilities
Provides consistent progress monitoring across all fantasy football scripts.
"""

import time
from contextlib import contextmanager
from typing import Callable

from alive_progress import alive_bar


class ProgressMonitor:
	"""Centralized progress monitoring for fantasy football analytics."""
	
	def __init__(self, title: str = "Processing", bar_style: str = "smooth"):
		self.title = title
		self.bar_style = bar_style
		self.start_time = None
	
	@contextmanager
	def monitor(self, total: int, description: str = None):
		"""Context manager for progress monitoring."""
		if description:
			title = f"{self.title}: {description}"
		else:
			title = self.title
			
		with alive_bar(total, title=title, bar=self.bar_style) as bar:
			yield bar
	
	def start_timer(self):
		"""Start timing an operation."""
		self.start_time = time.time()
		return self.start_time
	
	def elapsed_time(self) -> float:
		"""Get elapsed time since start_timer was called."""
		if self.start_time is None:
			return 0.0
		return time.time() - self.start_time
	
	def format_time(self, seconds: float) -> str:
		"""Format time in human-readable format."""
		if seconds < 60:
			return f"{seconds:.1f}s"
		elif seconds < 3600:
			minutes = seconds / 60
			return f"{minutes:.1f}m"
		else:
			hours = seconds / 3600
			return f"{hours:.1f}h"


def monitor_operation(title: str, operation: Callable, *args, **kwargs):
	"""Decorator-style function monitoring."""
	monitor = ProgressMonitor(title)
	start_time = monitor.start_timer()
	
	try:
		result = operation(*args, **kwargs)
		elapsed = monitor.elapsed_time()
		print(f"✅ {title} completed in {monitor.format_time(elapsed)}")
		return result
	except Exception as e:
		elapsed = monitor.elapsed_time()
		print(f"❌ {title} failed after {monitor.format_time(elapsed)}")
		raise e


def create_progress_bar(total: int, title: str, description: str = None):
	"""Create a progress bar with consistent styling."""
	if description:
		full_title = f"{title}: {description}"
	else:
		full_title = title
	
	return alive_bar(total, title=full_title, bar="smooth")


def monitor_data_processing(title: str, data_length: int):
	"""Create a progress monitor specifically for data processing."""
	return ProgressMonitor(title).monitor(data_length, "Processing Data")


def monitor_model_training(title: str, iterations: int):
	"""Create a progress monitor specifically for model training."""
	return ProgressMonitor(title).monitor(iterations, "Training Model")


def monitor_file_operations(title: str, file_count: int):
	"""Create a progress monitor specifically for file operations."""
	return ProgressMonitor(title).monitor(file_count, "Processing Files")

# Example usage functions

def example_usage():
	"""Show how to use the progress monitoring utilities."""
	
	# Method 1: Using ProgressMonitor class
	monitor = ProgressMonitor("Data Collection")
	monitor.start_timer()
	
	with monitor.monitor(5, "Collecting Years"):
		for i in range(5):
			time.sleep(0.1)  # Simulate work
			print(f"Processed year {2015 + i}")
	
	print(f"Total time: {monitor.format_time(monitor.elapsed_time())}")
	
	# Method 2: Using context manager directly
	with create_progress_bar(10, "Feature Engineering", "Creating Features"):
		for i in range(10):
			time.sleep(0.05)  # Simulate work
	
	# Method 3: Using decorator-style monitoring
	def sample_operation():
		time.sleep(0.2)  # Simulate work
		return "Operation completed"
	
	result = monitor_operation("Sample Operation", sample_operation)
	print(f"Result: {result}")

if __name__ == "__main__":
	example_usage()
