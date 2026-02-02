# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements MXFP8 quantization for efficient tensor storage and computation."""

import torch

from ..qtensor.base_qtensor import BaseQuantizedTensor
from ..utils import reduce_block_amax, reduce_block_padding

__all__ = ["MXFP8QTensor"]


class MXFP8QTensor(BaseQuantizedTensor):
    """Implements the MXFP8 quantization on tensors for more efficient storage or computation.

    MXFP8 uses:
    - FP8 E4M3 format for elements
    - E8M0 format for shared scales (power-of-2 only, stored as biased uint8 exponent)
    - Block size of 32 elements along the last dimension

    Attributes:
        quantized_data (torch.Tensor): The quantized data stored as float8_e4m3fn tensor.
    """

    E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    BLOCK_SIZE = 32
    SCALE_DTYPE = torch.uint8  # E8M0 format stores biased exponent as uint8

    @classmethod
    def _compute_e8m0_exponent(cls, amax: torch.Tensor) -> torch.Tensor:
        """Compute E8M0 exponent from per-block amax values.

        Args:
            amax: Per-block absolute max values.

        Returns:
            torch.Tensor: Float tensor of E8M0 exponents (unbiased, range [-127, 127]).
        """
        # Compute E8M0 scale: scale = 2^ceil(log2(amax / E4M3_max))
        descale = amax.float() / cls.E4M3_MAX

        # Handle zero/inf/nan cases
        min_value = torch.tensor(-127.0, device=descale.device)
        log2_descale = torch.where(
            descale > 0,
            torch.log2(descale),
            min_value,
        )

        e8m0_exponent = torch.ceil(log2_descale)

        # Clamp exponent to valid E8M0 range
        return torch.clamp(e8m0_exponent, min=-127, max=127)

    @classmethod
    def get_weights_scaling_factor(cls, weight: torch.Tensor) -> torch.Tensor:
        """Returns E8M0 scale (uint8 biased exponent) for weight tensor.

        Args:
            weight: The weight tensor to compute scale for. Must be at least 2D.
                Supports 2D (out_dim, in_dim) and 3D MoE (num_experts, out_dim, in_dim).

        Returns:
            torch.Tensor: E8M0 scale as uint8 tensor with shape [..., out_dim, in_dim // 32].
                For 2D input: (out_dim, in_dim // 32)
                For 3D MoE input: (num_experts, out_dim, in_dim // 32)
        """
        assert weight.dim() >= 2, f"Weight must be at least 2D, got {weight.dim()}D"

        in_dim = weight.shape[-1]

        assert in_dim % cls.BLOCK_SIZE == 0, (
            f"Weight inner dimension ({in_dim}) must be divisible by MXFP8 block size ({cls.BLOCK_SIZE})"
        )

        # Compute amax per block (reduce_block_amax handles N-dimensional tensors)
        amax = reduce_block_amax(weight, block_sizes={-1: cls.BLOCK_SIZE})

        # Compute E8M0 exponent and convert to biased uint8 (bias = 127)
        e8m0_exponent = cls._compute_e8m0_exponent(amax)
        return (e8m0_exponent + 127).to(cls.SCALE_DTYPE)

    @classmethod
    def get_weights_scaling_factor_from_quantizer(
        cls,
        weight: torch.Tensor,
        weight_quantizer,
    ) -> torch.Tensor:
        """Returns E8M0 scale from quantizer or computes from weight.

        This method handles extracting the scale from a weight quantizer,
        with proper format conversion and shape correction.

        Args:
            weight: The weight tensor. Can be 2D (out_dim, in_dim) or
                3D for MoE (num_experts, out_dim, in_dim).
            weight_quantizer: The weight quantizer with block_sizes and optional _scale.

        Returns:
            torch.Tensor: E8M0 scale as uint8 tensor with shape [..., out_dim, in_dim // 32].
        """
        assert hasattr(weight_quantizer, "block_sizes"), (
            "weight_quantizer must have 'block_sizes' attribute"
        )
        assert weight_quantizer.block_sizes[-1] == cls.BLOCK_SIZE, (
            f"MXFP8 requires block size {cls.BLOCK_SIZE}, got {weight_quantizer.block_sizes[-1]}"
        )
        assert weight.dim() >= 2, f"Weight must be at least 2D, got {weight.dim()}D"

        in_dim = weight.shape[-1]
        # Expected scale shape: all dims except last, with last dim reduced by block size
        # For 2D: (out_dim, in_dim // 32)
        # For 3D MoE: (num_experts, out_dim, in_dim // 32)
        expected_shape = (*weight.shape[:-1], in_dim // cls.BLOCK_SIZE)

        # Use FlashInfer's mxfp8_quantize for CUTLASS-compatible scales
        # This ensures the scale matches the FP8 values produced by quantize_with_scale
        if weight.is_cuda and in_dim % cls.BLOCK_SIZE == 0:
            try:
                from flashinfer import mxfp8_quantize

                # Ensure input is in supported dtype and contiguous
                quant_input = weight.contiguous()
                if quant_input.dtype not in (torch.float16, torch.bfloat16):
                    quant_input = quant_input.to(torch.bfloat16)

                # FlashInfer expects 2D input [M, K], flatten if needed
                if quant_input.dim() > 2:
                    # Flatten all dimensions except the last one
                    quant_input = quant_input.view(-1, in_dim)

                # FlashInfer's mxfp8_quantize returns FP8 values and E8M0 scales
                _, scale_1d = mxfp8_quantize(
                    quant_input,
                    is_sf_swizzled_layout=False,
                )
                # Synchronize to catch any CUDA errors immediately
                torch.cuda.synchronize()
                # Reshape scale to match expected shape
                scale = scale_1d.view(expected_shape)
                return scale
            except ImportError:
                pass  # Fall through to PyTorch path
            except Exception:
                pass  # FlashInfer failed, fall through to PyTorch path

        # Fallback: use quantizer scale or compute from weight
        if hasattr(weight_quantizer, "_scale") and weight_quantizer._scale is not None:
            scale = weight_quantizer._scale

            assert scale.dtype == cls.SCALE_DTYPE, (
                f"MXFP8 scale must be {cls.SCALE_DTYPE} (E8M0 format), got {scale.dtype}"
            )
            assert scale.shape == expected_shape, (
                f"Scale shape {scale.shape} does not match expected shape {expected_shape}"
            )
            return scale

        # No scale in quantizer, compute from weight
        return cls.get_weights_scaling_factor(weight)

    @classmethod
    def quantize_with_scale(
        cls,
        weight: torch.Tensor,
        weights_scaling_factor: torch.Tensor,
        use_flashinfer: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight tensor using FlashInfer for CUTLASS-compatible FP8 values.

        This method is used during export to produce MXFP8 quantized weights.
        When FlashInfer is available, it returns BOTH the FP8 values AND the
        scale that FlashInfer computed, ensuring they are always consistent.

        Args:
            weight: The weight tensor to quantize. Must be at least 1D.
            weights_scaling_factor: E8M0 scale as uint8 biased exponent (bias = 127).
                This is the pre-computed scale which may be replaced by FlashInfer's
                scale when use_flashinfer=True.
                Shape should be [..., out_dim, in_dim // 32] for 2D+ tensors,
                or [in_dim // 32] for 1D tensors.
            use_flashinfer: If True and FlashInfer is available, use FlashInfer's
                mxfp8_quantize GPU kernel for CUTLASS-compatible FP8 values.
                Default: True.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (quantized_weight, scale)
                - quantized_weight: FP8 E4M3 values with same shape as input
                - scale: E8M0 scale (uint8) - may differ from input if FlashInfer is used
        """
        assert weights_scaling_factor.dtype == cls.SCALE_DTYPE, (
            f"weights_scaling_factor must be {cls.SCALE_DTYPE} (E8M0 format), "
            f"got {weights_scaling_factor.dtype}"
        )

        in_dim = weight.shape[-1]
        num_blocks = in_dim // cls.BLOCK_SIZE

        assert in_dim % cls.BLOCK_SIZE == 0, (
            f"Weight inner dimension ({in_dim}) must be divisible by MXFP8 block size ({cls.BLOCK_SIZE})"
        )

        # Use FlashInfer's GPU kernel for CUTLASS-compatible FP8 values
        # PyTorch's .to(float8_e4m3fn) produces different bit patterns than the
        # TRT-LLM/FlashInfer GPU kernel, causing accuracy issues with CUTLASS GEMM.
        #
        # IMPORTANT: We return BOTH the FP8 values AND the scale from FlashInfer
        # to ensure they are always consistent. The pre-computed scale may differ
        # slightly from FlashInfer's scale, causing accuracy issues.
        if use_flashinfer and weight.is_cuda:
            try:
                from flashinfer import mxfp8_quantize

                # Ensure input is in supported dtype and contiguous
                quant_input = weight.contiguous()
                original_shape = quant_input.shape
                if quant_input.dtype not in (torch.float16, torch.bfloat16):
                    quant_input = quant_input.to(torch.bfloat16)

                # FlashInfer expects 2D input [M, K], flatten if needed
                if quant_input.dim() > 2:
                    # Flatten all dimensions except the last one
                    quant_input = quant_input.view(-1, in_dim)

                # FlashInfer's mxfp8_quantize returns BOTH FP8 values AND E8M0 scales
                # We use BOTH to ensure they are always consistent
                quantized_weight, scale_1d = mxfp8_quantize(
                    quant_input,
                    is_sf_swizzled_layout=False,  # Linear layout for compatibility
                )
                # Synchronize to catch any CUDA errors immediately
                torch.cuda.synchronize()

                # Reshape back to original shape
                quantized_weight = quantized_weight.view(original_shape)

                # Reshape scale to match expected shape: [..., out_dim, in_dim // 32]
                expected_scale_shape = (*original_shape[:-1], num_blocks)
                scale = scale_1d.view(expected_scale_shape)

                return quantized_weight, scale
            except ImportError:
                pass  # Fall through to PyTorch path
            except Exception:
                pass  # FlashInfer failed, fall through to PyTorch path

        # Fallback: PyTorch implementation (may not be CUTLASS-compatible)
        # Convert E8M0 biased exponent to scale factor: scale = 2^(127 - exponent)
        scale_factor = torch.exp2(127 - weights_scaling_factor.float())

        weight_reshaped = weight.view(*weight.shape[:-1], num_blocks, cls.BLOCK_SIZE)
        scale_factor_expanded = scale_factor.unsqueeze(-1)
        scaled_weight = weight_reshaped * scale_factor_expanded
        scaled_weight = torch.clamp(scaled_weight, min=-cls.E4M3_MAX, max=cls.E4M3_MAX)
        quantized_weight = scaled_weight.to(torch.float8_e4m3fn)

        # Return both FP8 values and the original scale (unchanged for PyTorch path)
        return quantized_weight.view(weight.shape), weights_scaling_factor

    @classmethod
    def quantize_swizzled(
        cls,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight tensor with SWIZZLED scales for FlashInfer mm_mxfp8 CUTLASS kernel.

        This method is specifically designed for vLLM/FlashInfer compatibility.
        The mm_mxfp8 CUTLASS kernel requires scales in F8_128x4 swizzled layout.

        Args:
            weight: The weight tensor to quantize. Must be 2D [out_features, in_features].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (quantized_weight, swizzled_scale)
                - quantized_weight: FP8 E4M3 values with same shape as input
                - swizzled_scale: 1D uint8 tensor in F8_128x4 swizzled layout
                  for direct use with FlashInfer mm_mxfp8

        Raises:
            ImportError: If FlashInfer is not available.
            ValueError: If weight is not 2D or dimensions are invalid.
        """
        if weight.dim() != 2:
            raise ValueError(f"quantize_swizzled requires 2D weight, got {weight.dim()}D")

        _, in_features = weight.shape
        if in_features % cls.BLOCK_SIZE != 0:
            raise ValueError(
                f"Weight K dimension ({in_features}) must be divisible by "
                f"MXFP8 block size ({cls.BLOCK_SIZE})"
            )

        try:
            from flashinfer import mxfp8_quantize
        except ImportError:
            raise ImportError(
                "FlashInfer is required for quantize_swizzled. Install with: pip install flashinfer"
            )

        # Ensure weight is in supported dtype and contiguous
        quant_input = weight.contiguous()
        if quant_input.dtype not in (torch.float16, torch.bfloat16):
            quant_input = quant_input.to(torch.bfloat16)

        # Use FlashInfer's mxfp8_quantize with SWIZZLED layout
        # This produces scales in F8_128x4 layout required by mm_mxfp8 CUTLASS kernel
        quantized_weight, swizzled_scale = mxfp8_quantize(
            quant_input,
            is_sf_swizzled_layout=True,  # SWIZZLED for CUTLASS mm_mxfp8!
        )
        torch.cuda.synchronize()

        return quantized_weight, swizzled_scale

    @classmethod
    def quantize(
        cls,
        input: torch.Tensor,
        weights_scaling_factor: torch.Tensor | None = None,
        use_flashinfer: bool = True,
    ) -> tuple:
        """Convert a tensor to MXFP8 quantized format.

        Args:
            input (torch.Tensor): The input tensor to be quantized.
            weights_scaling_factor (torch.Tensor | None): Optional pre-computed E8M0 scale
                as uint8 biased exponent. If None, the scale will be computed from the input.
                Shape should be [..., in_dim // 32] matching input dimensions.
            use_flashinfer (bool): If True and FlashInfer is available, use FlashInfer's
                mxfp8_quantize GPU kernel for CUTLASS-compatible FP8 values. This produces
                FP8 bit patterns that work correctly with FlashInfer's mm_mxfp8 CUTLASS kernel.
                Recommended for inference with vLLM/FlashInfer. Default: False.

        Returns:
            tuple: (MXFP8QTensor, weights_scaling_factor) where weights_scaling_factor is
                E8M0 scale as uint8 biased exponent.
        """
        original_shape = input.shape
        original_dtype = input.dtype

        input = reduce_block_padding(input, block_sizes={-1: cls.BLOCK_SIZE})

        # Use FlashInfer's GPU kernel for CUTLASS-compatible FP8 values
        if use_flashinfer and input.is_cuda and weights_scaling_factor is None:
            try:
                from flashinfer import mxfp8_quantize

                # Ensure input is in supported dtype
                quant_input = input
                if input.dtype not in (torch.float16, torch.bfloat16):
                    quant_input = input.to(torch.bfloat16)

                # FlashInfer's mxfp8_quantize returns FP8 values and E8M0 scales
                quantized_data, scale_1d = mxfp8_quantize(
                    quant_input,
                    is_sf_swizzled_layout=False,  # Linear layout for compatibility
                )

                # Reshape scale from 1D to match input dimensions
                scale_shape = (*input.shape[:-1], input.shape[-1] // cls.BLOCK_SIZE)
                weights_scaling_factor = scale_1d.view(scale_shape)

                # Crop back to original shape
                quantized_data = quantized_data[..., : original_shape[-1]]

                return cls(original_shape, original_dtype, quantized_data), weights_scaling_factor
            except ImportError:
                raise ImportError("Flashinfer is required!")

        # Fallback: PyTorch implementation
        if weights_scaling_factor is None:
            input_amax = reduce_block_amax(input, block_sizes={-1: cls.BLOCK_SIZE})
            e8m0_exponent = cls._compute_e8m0_exponent(input_amax)
            weights_scaling_factor = (e8m0_exponent + 127).to(cls.SCALE_DTYPE)

        quantized_data, updated_scale = cls.quantize_with_scale(input, weights_scaling_factor)

        # Crop back to original shape
        quantized_data = quantized_data[..., : original_shape[-1]]

        return cls(original_shape, original_dtype, quantized_data), updated_scale

    def dequantize(self, dtype: torch.dtype = None, **kwargs) -> torch.Tensor:
        """Dequantize MXFP8 tensor back to the target dtype.

        Args:
            dtype (torch.dtype | None): Target dtype for dequantization. Defaults to original dtype.
            **kwargs: Must contain 'scale' (E8M0 biased uint8).

        Returns:
            torch.Tensor: Dequantized tensor in the target dtype.
        """
        assert "scale" in kwargs, "dequantize requires 'scale' in kwargs"

        e8m0_scale = kwargs["scale"]

        if dtype is None:
            dtype = self.metadata["dtype"]

        original_shape = self.metadata["shape"]
        quantized_data = self._quantized_data.float()
        quantized_data = reduce_block_padding(quantized_data, block_sizes={-1: self.BLOCK_SIZE})

        num_blocks = quantized_data.shape[-1] // self.BLOCK_SIZE
        quantized_blocked = quantized_data.view(
            *quantized_data.shape[:-1], num_blocks, self.BLOCK_SIZE
        )

        # Convert E8M0 biased exponent back to scale factor: descale = 2^(exponent - 127)
        descale = torch.exp2(e8m0_scale.float() - 127)

        dequantized = quantized_blocked * descale.unsqueeze(-1)

        # Reshape and crop back to original shape
        dequantized = dequantized.view(*quantized_data.shape[:-1], quantized_data.shape[-1])
        dequantized = dequantized[..., : original_shape[-1]]

        return dequantized.to(dtype)
