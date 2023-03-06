# G13
M1 GPU architecture instrospection. Determining cache sizes and memory latencies across the hierarchy.

# Shared Memory Size
Shared memory or threadgroup memory sizes can be found here: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf

# Latencies
For M1 Max:

| | L1 | L2 | Shared Memory | Global memory |
| :--- | :----: | :----: | :----: | ---: |
| Size | 8 KB | 512 KB | 32 KB | - |
| Line | 32 B | 32 B | - | - |
| Associativity | 128 | - | - | - |
| Latency | ~47 ns | ~100 ns | ~70 ns | < 350 ns |

These numbers were obtained by pointer chasing in a single thread, in a single core.

# Reproducibility
Refer to this link for requirements: https://larsgeb.github.io/2022/04/20/m1-gpu.html . Essentially, you will need to use a different version of clang (not the one that ships with your machine). This is achieved by installing `llvm`. Then, follow these steps:

1. `$ ./runbuild`
2. Execute one of the binaries.

# Sources
1. http://igoro.com/archive/gallery-of-processor-cache-effects/
2. Fedor Pikus' _The Art of Writing Efficient Programs: An advanced programmer's guide to efficient hardware utilization and compiler optimizations using C++ examples_