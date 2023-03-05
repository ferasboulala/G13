#include <numeric>
#include <random>
#include <vector>

namespace g13 {
// Generates a vector of indices that are unique and that form a single cycle of length size. This
// algorithm is called Satollo's algorithm.
template <typename T>
std::vector<T> satolloRandomIndices(uint64_t size) {
    assert(size >= 2);
    std::random_device               randomDevice;
    std::mt19937                     generator(randomDevice());
    std::uniform_int_distribution<T> distribution;
    std::vector<T>                   indices(size);

    std::iota(indices.begin(), indices.end(), 0);
    for (uint64_t i = size - 1; i > 1; --i) {
        std::swap(indices[i], indices[distribution(generator) % i]);
    }
    std::swap(indices[0], indices[1]);

    return indices;
}
}  // namespace g13