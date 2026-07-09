"""
Memory Profiler for Multi-Resolution Models

Profiles GPU/MPS memory usage and benchmark performance against baseline
single-resolution models. Critical for determining optimal batch sizes and
identifying memory bottlenecks.

Key Metrics:
- Peak memory usage
- Memory per resolution level
- Throughput (samples/sec)
- Comparison vs baseline
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import psutil
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class MemoryProfile:
    """Container for memory profiling results."""
    model_name: str
    batch_size: int
    resolution: int or List[int]
    peak_memory_mb: float
    avg_memory_mb: float
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    throughput_samples_per_sec: float
    num_parameters: int
    device: str


class MemoryProfiler:
    """
    Memory and performance profiler for multi-resolution models.

    Supports both MPS (Apple Silicon) and CUDA devices with fallback to CPU.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: Target device (auto-detected if None)
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.device_type = str(self.device).split(':')[0]
        print(f"MemoryProfiler initialized on device: {self.device} ({self.device_type})")

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB (device-specific)."""
        if self.device_type == 'cuda':
            return torch.cuda.memory_allocated() / 1024**2
        elif self.device_type == 'mps':
            # MPS doesn't have direct memory query, use system memory as proxy
            return psutil.Process(os.getpid()).memory_info().rss / 1024**2
        else:  # CPU
            return psutil.Process(os.getpid()).memory_info().rss / 1024**2

    def _reset_memory_stats(self):
        """Reset memory statistics (device-specific)."""
        if self.device_type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        elif self.device_type == 'mps':
            torch.mps.empty_cache()

    def profile_model(
        self,
        model: nn.Module,
        input_data: Dict[int, torch.Tensor] or torch.Tensor,
        target: torch.Tensor,
        criterion: nn.Module,
        model_name: str = "model",
        num_iterations: int = 10,
        warmup_iterations: int = 3
    ) -> MemoryProfile:
        """
        Profile model memory and performance.

        Args:
            model: Model to profile
            input_data: Input tensor or dict of tensors (multi-res)
            target: Target tensor for loss calculation
            criterion: Loss function
            model_name: Name for identification
            num_iterations: Number of profiling iterations
            warmup_iterations: Warmup iterations (not counted)

        Returns:
            MemoryProfile with profiling results
        """
        model = model.to(self.device)
        model.train()

        # Move data to device
        if isinstance(input_data, dict):
            input_data = {k: v.to(self.device) for k, v in input_data.items()}
            resolution = list(input_data.keys())
            batch_size = list(input_data.values())[0].shape[0]
        else:
            input_data = input_data.to(self.device)
            resolution = input_data.shape[2]  # Assuming square images
            batch_size = input_data.shape[0]

        target = target.to(self.device)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Reset memory stats
        self._reset_memory_stats()

        # Warmup
        print(f"\nWarming up for {warmup_iterations} iterations...")
        for _ in range(warmup_iterations):
            output = model(input_data)
            if isinstance(output, dict):
                output = output['segmentation']
            loss = criterion(output, target)
            loss.backward()
            model.zero_grad()

        # Actual profiling
        print(f"Profiling for {num_iterations} iterations...")
        self._reset_memory_stats()

        forward_times = []
        backward_times = []
        memory_samples = []

        for i in range(num_iterations):
            # Measure memory before
            mem_before = self._get_memory_usage_mb()

            # Forward pass
            torch.cuda.synchronize() if self.device_type == 'cuda' else None
            start_time = time.perf_counter()

            output = model(input_data)
            if isinstance(output, dict):
                output = output['segmentation']

            torch.cuda.synchronize() if self.device_type == 'cuda' else None
            forward_time = (time.perf_counter() - start_time) * 1000  # ms

            # Backward pass
            loss = criterion(output, target)

            torch.cuda.synchronize() if self.device_type == 'cuda' else None
            start_time = time.perf_counter()

            loss.backward()

            torch.cuda.synchronize() if self.device_type == 'cuda' else None
            backward_time = (time.perf_counter() - start_time) * 1000  # ms

            # Measure memory after
            mem_after = self._get_memory_usage_mb()

            forward_times.append(forward_time)
            backward_times.append(backward_time)
            memory_samples.append(mem_after - mem_before)

            model.zero_grad()

        # Get peak memory
        peak_memory = max(memory_samples) if memory_samples else self._get_memory_usage_mb()

        # Calculate statistics
        avg_forward_time = np.mean(forward_times)
        avg_backward_time = np.mean(backward_times)
        total_time = avg_forward_time + avg_backward_time
        throughput = (batch_size * 1000) / total_time  # samples/sec

        profile = MemoryProfile(
            model_name=model_name,
            batch_size=batch_size,
            resolution=resolution,
            peak_memory_mb=peak_memory,
            avg_memory_mb=np.mean(memory_samples) if memory_samples else 0,
            forward_time_ms=avg_forward_time,
            backward_time_ms=avg_backward_time,
            total_time_ms=total_time,
            throughput_samples_per_sec=throughput,
            num_parameters=num_params,
            device=str(self.device)
        )

        return profile

    def compare_models(
        self,
        baseline_model: nn.Module,
        multi_res_model: nn.Module,
        baseline_input: torch.Tensor,
        multi_res_input: Dict[int, torch.Tensor],
        target: torch.Tensor,
        criterion: nn.Module,
        batch_size: int = 2
    ) -> Dict[str, MemoryProfile]:
        """
        Compare baseline single-resolution vs multi-resolution model.

        Args:
            baseline_model: Single-resolution baseline
            multi_res_model: Multi-resolution model
            baseline_input: Single tensor input for baseline
            multi_res_input: Dict of multi-res inputs
            target: Ground truth target
            criterion: Loss function
            batch_size: Batch size for comparison

        Returns:
            Dictionary with profiles for both models
        """
        print("="*70)
        print("MEMORY PROFILING: Baseline vs Multi-Resolution")
        print("="*70)

        # Profile baseline
        print("\n[1/2] Profiling BASELINE model...")
        baseline_profile = self.profile_model(
            model=baseline_model,
            input_data=baseline_input,
            target=target,
            criterion=criterion,
            model_name="Baseline UNet++",
            num_iterations=10
        )

        # Profile multi-res
        print("\n[2/2] Profiling MULTI-RESOLUTION model...")
        multi_res_profile = self.profile_model(
            model=multi_res_model,
            input_data=multi_res_input,
            target=target,
            criterion=criterion,
            model_name="Multi-Res UNet++",
            num_iterations=10
        )

        # Print comparison
        self.print_comparison(baseline_profile, multi_res_profile)

        return {
            'baseline': baseline_profile,
            'multi_res': multi_res_profile
        }

    def print_comparison(self, baseline: MemoryProfile, multi_res: MemoryProfile):
        """Print formatted comparison of two profiles."""
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)

        print(f"\n{'Metric':<30} {'Baseline':<20} {'Multi-Res':<20} {'Ratio':<10}")
        print("-"*80)

        # Parameters
        print(f"{'Parameters':<30} {baseline.num_parameters:>19,} {multi_res.num_parameters:>19,} "
              f"{multi_res.num_parameters/baseline.num_parameters:>9.2f}x")

        # Memory
        print(f"{'Peak Memory (MB)':<30} {baseline.peak_memory_mb:>19.1f} {multi_res.peak_memory_mb:>19.1f} "
              f"{multi_res.peak_memory_mb/baseline.peak_memory_mb:>9.2f}x")

        # Time
        print(f"{'Forward Time (ms)':<30} {baseline.forward_time_ms:>19.1f} {multi_res.forward_time_ms:>19.1f} "
              f"{multi_res.forward_time_ms/baseline.forward_time_ms:>9.2f}x")

        print(f"{'Backward Time (ms)':<30} {baseline.backward_time_ms:>19.1f} {multi_res.backward_time_ms:>19.1f} "
              f"{multi_res.backward_time_ms/baseline.backward_time_ms:>9.2f}x")

        print(f"{'Total Time (ms)':<30} {baseline.total_time_ms:>19.1f} {multi_res.total_time_ms:>19.1f} "
              f"{multi_res.total_time_ms/baseline.total_time_ms:>9.2f}x")

        # Throughput
        print(f"{'Throughput (samples/s)':<30} {baseline.throughput_samples_per_sec:>19.1f} "
              f"{multi_res.throughput_samples_per_sec:>19.1f} "
              f"{multi_res.throughput_samples_per_sec/baseline.throughput_samples_per_sec:>9.2f}x")

        print("-"*80)

        # Summary
        memory_overhead = (multi_res.peak_memory_mb / baseline.peak_memory_mb - 1) * 100
        speed_impact = (multi_res.total_time_ms / baseline.total_time_ms - 1) * 100

        print(f"\nSummary:")
        print(f"  Memory overhead: {memory_overhead:+.1f}%")
        print(f"  Speed impact: {speed_impact:+.1f}%")
        print(f"  Parameter increase: {(multi_res.num_parameters/baseline.num_parameters - 1)*100:.1f}%")

    def save_profile(self, profile: MemoryProfile, filepath: str):
        """Save profile to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(profile.__dict__, f, indent=2)
        print(f"\nProfile saved to: {filepath}")


# Testing
if __name__ == '__main__':
    import segmentation_models_pytorch as smp
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

    from multi_resolution.multi_res_unetpp import MultiResUNetPlusPlus

    print("Testing Memory Profiler...")

    # Setup
    batch_size = 2
    n_classes = 4
    resolution = 1024

    # Create dummy data
    baseline_input = torch.randn(batch_size, 3, resolution, resolution)
    multi_res_input = {
        1024: torch.randn(batch_size, 3, 1024, 1024),
        512: torch.randn(batch_size, 3, 512, 512),
        256: torch.randn(batch_size, 3, 256, 256)
    }
    target = torch.randint(0, 2, (batch_size, n_classes, resolution, resolution)).float()

    # Create models
    print("\nCreating models...")
    baseline_model = smp.UnetPlusPlus(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=n_classes
    )

    multi_res_model = MultiResUNetPlusPlus(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        resolutions=[1024, 512, 256],
        pyramid_channels=64,
        fusion_type='concat',
        n_classes=n_classes
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Profile
    profiler = MemoryProfiler()

    profiles = profiler.compare_models(
        baseline_model=baseline_model,
        multi_res_model=multi_res_model,
        baseline_input=baseline_input,
        multi_res_input=multi_res_input,
        target=target,
        criterion=criterion,
        batch_size=batch_size
    )

    # Save profiles
    os.makedirs('profiling_results', exist_ok=True)
    profiler.save_profile(profiles['baseline'], 'profiling_results/baseline_profile.json')
    profiler.save_profile(profiles['multi_res'], 'profiling_results/multi_res_profile.json')

    print("\n✓ Memory Profiler test PASSED")
