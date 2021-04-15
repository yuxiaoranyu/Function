/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "log_matrix_determinant.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "Eigen/LU"

namespace {
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 2;
const char* kLogMatrixDeterminant = "LogMatrixDeterminant"
#define LOG_MATRIX_DETERMINANT_COMPUTE_CASE(DTYPE, TYPE, CTX)        \
  case (DTYPE): {                                                    \
    uint32_t result = LogMatrixDeterminantCompute<TYPE>(CTX);        \
    if (result != KERNEL_STATUS_ok) {                                \
      \
    KERNEL_LOG_ERROR("LogMatrixDeterminant kernel compute failed."); \
      return result;                                                 \
    \
}  // namespace                                                      \
break;                                                               \
}
}
namespace aicpu {
uint32_t LogMatrixDeterminantCpuKernel::Compute(CpuKernelContext& ctx) {
  KERNEL_HANDLE_ERROR(NormalChect(ctx, kInputNum, kOutputNum),
                      "LogMatrixDeterminant check input and output number failed.");
  KERNEL_HANDLE_ERROR(LogMatrixDeterminantCheck(ctx), "LogMatrixDeterminant check params.");
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    LOG_MATRIX_DETERMINANT_COMPUTE_CASE(DT_FLOAT, float, ctx)
    LOG_MATRIX_DETERMINANT_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    LOG_MATRIX_DETERMINANT_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    LOG_MATRIX_DETERMINANT_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("LogMatrixDeterminant kernel data type [%u] not support.", data_type);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t LogMatrixDeterminantCpuKernel::LogMatrixDeterminantCheck(CpuKernelContext& ctx) {
  auto input_0 = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_0, KERNEL_STATUS_PARAM_INVALID, "Get input failed.")
  auto output_0 = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_0, KERNEL_STATUS_PARAM_INVALID, "Get output[0] failed.")
  auto output_1 = ctx.Output(1);
  KERNEL_CHECK_NULLPTR(output_1, KERNEL_STATUS_PARAM_INVALID, "Get output[1] data failed.")
  KERNEL_CHECK_NULLPTR(input_0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input tensor shape failed.")
  std::vector<int64_t> shape_x = input_0->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_x.size() > 1), KERNEL_STATUS_PARAM_INVALID, "Input must be as lease rank 2,get [%d].",
                     shape_x.size())
  size_t shape_size = shape_x.size();
  KERNEL_CHECK_FALSE((shape_x[shape_size - 2] == shape_x[shape_size - 1]), KERNEL_STATUS_PARAM_INVALID,
                     "Dimensions must be equal,but are [%d] and [%d].", shape_x[shape_size - 2],
                     shape_x[shape_size - 1])
  KERNEL_LOG_INFO(
      "LogMatrixDeterminantCpuKernel[%s],input:size[%llu];"
      "output1:size[%llu],output2:size[%llu].",
      ctx.GetOptype().c_str(), ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize(), ctx.Output(1)->GetDataSize());
  return KERNEL_STATUS_OK
}
template <typename T>
uint32_t LogMatrixDeterminantCpuKernel::LogMatrixDeterminantCompute(CpuKernelContext &ctx){
    auto input_x=reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto output_sign=reinterpret_cast<T*>(ctx.Output(0)->GetData());
    auto output_y=reinterpret_cast<T*>(ctx.Output(1)->GetData());

    std::vector<int64_t> shape_x=ctx.Input(0)->GetTensorShape()->GetDimSizes();
    size_t shape_size=shape_x.size();
    int64_t m=shape_x[shape_size-1];
    size_t matrix_num=ctx.Input(0)->NumElements()/(m*m);
    typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::RowMajor>MartixXd;
    using RealT =typename Eigen::NumTraits<T>::Real;
    Eigen::PartialPivLU<MartixXd>lu;
    RealT log_abs_det=0;
    T sign=1;
    for (size_t i =0;i<martix_num;i++){
        log_abs_det=0
        sign
    }
}
}
