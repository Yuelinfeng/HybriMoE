# Prompt Stream Universality Report

This report aggregates live HybriMoE traces by run, workload phase, and prompt category.

## Universality Criteria

- high cache regime: cache_size >= threshold and prefetch_size > 0
- phenomenon: cache hit >= 80%, issued assignment coverage <= 5%, redundant candidates >= 90%
- dynamic drift: route drift >= 60% for shifted workloads

## Summary

- group_count: 3
- high_cache_prefetch_group_count: 2
- high_hit_low_utility_group_count: 2
- high_hit_low_utility_fraction: 1.0000
- dynamic_group_count: 0
- dynamic_high_route_drift_count: 0
- dynamic_high_route_drift_fraction: 0.0000
- mean_high_cache_hit: 0.9056
- mean_high_cache_issued_coverage: 0.0061
- mean_high_cache_redundant: 0.9698

## Run-Level Metrics

| run | workload | cache | prefetch | hit | issued coverage | redundant | route drift | eviction damage | prefetch wait late |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mechanism_timing_cache16_prefetch8 | unknown | 16 | 8 | 51.52% | 2.58% | 83.48% | 79.00% | 69.35% | 100.00% |
| mechanism_timing_cache48_prefetch8 | unknown | 48 | 8 | 87.54% | 0.72% | 96.12% | 79.44% | 35.19% | 100.00% |
| mechanism_timing_cache56_prefetch8 | unknown | 56 | 8 | 93.59% | 0.49% | 97.85% | 78.56% | 27.39% | 100.00% |