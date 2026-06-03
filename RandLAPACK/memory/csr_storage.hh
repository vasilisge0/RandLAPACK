#pragma once

#include "precision.hh"
#include "device.hh"
#include "dimensions.hh"

namespace RandLAPACK {

template <Device device, typename value_t, typename index_t>
void initialize_csr_storage(dim<2> size, SparseStorage& source) {
    CsrStorage& S = std::get<CsrStorage>(source);

    value_t* t;
    malloc<device, value_t>(S.nnz_, &t);
    S.values_ = t;

    index_t* rp;
    malloc<device, index_t>(size[0] + 1, &rp);
    S.row_ptrs_ = rp;

    index_t* ci;
    malloc<device, index_t>(S.nnz_, &ci);
    S.col_idxs_ = ci;
} //  initialize_csr_storage

INSTANTIATE_ALL_DEVICES_TYPES_INDEX(initialize_csr_storage, (dim<2>, SparseStorage&))

template <Device device, typename value_t, typename index_t>
void free_csr_storage(SparseStorage& source) {
    CsrStorage& S = std::get<CsrStorage>(source);

    value_t* v = std::get<value_t*>(S.values_);
    if (v != nullptr)
        free<device, value_t>(v);

    index_t* rp = std::get<index_t*>(S.row_ptrs_);
    if (rp != nullptr)
        free<device, index_t>(rp);

    index_t* ci = std::get<index_t*>(S.col_idxs_);
    if (ci != nullptr)
        free<device, index_t>(ci);
} //  free_csr_storage

INSTANTIATE_ALL_DEVICES_TYPES_INDEX(free_csr_storage, (SparseStorage&))

}   // namespace RandLAPACK