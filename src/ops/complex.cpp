// src/ops/complex.cpp
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/elementwise.h"

namespace ins {

    bool is_complex(const Array& x) {
        return x.dtype() == DType::C32 || x.dtype() == DType::C64;
    }

    bool has_complex_shape(const Array& x) {
        int ndim = x.shape().ndim();
        return ndim >= 1 && x.shape().dim(ndim - 1) == 2;
    }

    Array to_complex(const Array& real) {
        Array real_exp = unsqueeze(real, -1);
        Array zeros_arr = zeros(real_exp.shape(), real_exp.dtype(), real_exp.place());
        Array complex_storage = concat({ real_exp, zeros_arr }, -1);
        Array result = as_complex(complex_storage);

        return result;
    }

    Array to_complex(const Array& real, const Array& imag) {
        Array real_exp = unsqueeze(real, -1);
        Array imag_exp = unsqueeze(imag, -1);

        Array complex_storage = concat({ real_exp, imag_exp }, -1);
        Array result = as_complex(complex_storage);

        return result;
    }

    Array as_complex(const Array& x) {
        INS_CHECK(x.defined(), "as_complex: input is undefined");
        INS_CHECK(has_complex_shape(x), "as_complex: input must have last dimension = 2");
        INS_CHECK(x.dtype() == DType::F32 || x.dtype() == DType::F64,
            "as_complex: input must be float32 or float64, but got input dtype: ", dtype_name(x.dtype()));

        // Remove last dimension
        std::vector<int64_t> new_dims = x.shape().dims();
        new_dims.pop_back();
        Shape new_shape(new_dims);

        DType new_dtype = (x.dtype() == DType::F32) ? DType::C32 : DType::C64;

        // Zero-copy view
        return x.view(new_shape, new_dtype);
    }

    Array as_real(const Array& x) {
        INS_CHECK(x.defined(), "as_real: input is undefined");
        INS_CHECK(is_complex(x), "as_real: input must be complex64 or complex128");

        // Add last dimension of size 2
        std::vector<int64_t> new_dims = x.shape().dims();
        new_dims.push_back(2);
        Shape new_shape(new_dims);

        DType new_dtype = (x.dtype() == DType::C32) ? DType::F32 : DType::F64;

        // Zero-copy view
        return x.view(new_shape, new_dtype);
    }

    Array real(const Array& z) {
        INS_CHECK(z.defined(), "real: input is undefined");
        bool is_cmplx = is_complex(z);
        INS_CHECK(is_cmplx, "real: input must be complex, but got dtype: ", dtype_name(z.dtype()));
        Array real_view = as_real(z);
        int last_dim = real_view.shape().ndim() - 1;
        Array real_part = real_view.slice(last_dim, 0, 1);
        Array result = squeeze(real_part, -1);

        return result;
    }
    Array imag(const Array& z) {
        INS_CHECK(z.defined(), "imag: input is undefined");

        bool is_cmplx = is_complex(z);
        INS_CHECK(is_cmplx, "imag: input must be complex, but got dtype: ", dtype_name(z.dtype()));
        Array real_view = as_real(z);
        int last_dim = real_view.shape().ndim() - 1;
        Array imag_part = real_view.slice(last_dim, 1, 2);
        Array result = squeeze(imag_part, -1);
        return result;
    }

} // namespace ins