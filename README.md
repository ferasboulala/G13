# G13
M1 GPU architecture instrospection. Determining cache sizes and memory latencies across the hierarchy.

# Local Memory Size
Local memory or threadgroup memory sizes can be found here: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf

# Latencies
For M1 Max:

| | L1 | L2 | Local Memory | Global memory |
| :--- | :----: | :----: | :----: | ---: |
| Size | 8 KB | 512 KB | 32 KB | - |
| Line | 32 B | 32 B | - | - |
| Associativity | 128 | - | - | - |
| Latency | ~60 ns | ~115 ns | 65 ns | < 400 ns |

These numbers were obtained by pointer chasing in a single thread, in a single core.

# Sources
1. http://igoro.com/archive/gallery-of-processor-cache-effects/
2. Fedor Pikus' _The Art of Writing Efficient Programs: An advanced programmer's guide to efficient hardware utilization and compiler optimizations using C++ examples_