#pragma once
#include "device.hh"
#include "memory.hh"
#include "precision.hh"
#include "storage_cpu_kernels.hh"
#include "type_macros.hh"

namespace RandLAPACK {

template <Device device, typename value_t>
void initialize_strided_storage(dim<2> size, DenseStorage& source) {
    StridedStorage& S = std::get<StridedStorage>(source);
    size_t non_lead_dim = (S.layout_ == Layout::ROW_MAJOR) ? size[0] : size[1];
    value_t* t;
    malloc<device, value_t>(S.lead_dim_ * non_lead_dim, &t);
    S.values_ = t;
}  //  initialize_strided_storage

INSTANTIATE_ALL_DEVICES_AND_TYPES(initialize_strided_storage,
                                  (dim<2>, DenseStorage&))

template <Device device, typename value_t>
void free_strided_storage(DenseStorage& source) {
    StridedStorage& S = std::get<StridedStorage>(source);
    value_t* v = std::get<value_t*>(S.values_);
    if (v != nullptr) free<device, value_t>(v);
}  //  free_strided_storage

INSTANTIATE_ALL_DEVICES_AND_TYPES(free_strided_storage, (DenseStorage&))

template <Device device_source, Device device_target, typename value_in_t,
          typename value_out_t>
void copy_strided_storage(dim<2> size, DenseStorage& source,
                          DenseStorage& target) {
    StridedStorage& S_source = std::get<StridedStorage>(source);
    StridedStorage& S_target = std::get<StridedStorage>(target);
    if constexpr (device_source == RandLAPACK::Device::CPU &&
                  device_target == RandLAPACK::Device::CPU) {
        RandLAPACK::cpu::strided_copy_kernel(
            size, std::get<value_in_t*>(S_source.values_), S_source.lead_dim_,
            std::get<value_out_t*>(S_target.values_), S_target.lead_dim_);
    } else {
        throw std::invalid_argument(
            "Copying between the specified devices is not supported.");
    }
}  //  copy_strided_storage

INSTANTIATE_ALL_DEVICE_PAIRS_FP_PAIRS(copy_strided_storage,
                                      (dim<2>, DenseStorage&, DenseStorage&))

}  // namespace RandLAPACK