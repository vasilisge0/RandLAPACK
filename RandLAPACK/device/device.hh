#pragma once

namespace RandLAPACK {

enum class Device : std::size_t { CPU = 0, OMP = 1, CUDA = 2, Count };

} // namespace RandLAPACK