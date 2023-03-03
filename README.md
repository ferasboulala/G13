# G13
M1 GPU architecture instrospection. Determining cache sizes, shared memory and global memory bandwidths and latencies.

# Local Memory
Local memory or threadgroup memory sizes can be found here: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf

# Cache
For M1 Max:

| | L1 | L2|
| :--- | :----: | ---: |
| Size | 8 KB | 512 KB |
| Line | 32 B | 32 B |
| Associativity | 128 | - |
| Latency | ~60 ns | ~115 ns |


# Sources
1. http://igoro.com/archive/gallery-of-processor-cache-effects/
2. Fedor Pikus' _The Art of Writing Efficient Programs: An advanced programmer's guide to efficient hardware utilization and compiler optimizations using C++ examples_