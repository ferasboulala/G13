cmake -B build
cmake --build build -j$(sysctl -n hw.physicalcpu)
