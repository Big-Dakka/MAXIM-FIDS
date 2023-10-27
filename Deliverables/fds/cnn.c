/*******************************************************************************
* Copyright (C) Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// fds
// Created using ./ai8xize.py --test-dir sdk/Examples/MAX78000/CNN --overwrite --prefix fds --checkpoint-file trained/ai85-fds-qat8-q.pth.tar --config-file networks/fds.yaml --sample-input tests/sample_fds.npy --softmax --fifo --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint --verbose

// DO NOT EDIT - regenerate this file instead!

// Configuring 9 layers:
// Layer 0: 3x64x64 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 16x64x64 output
// Layer 1: 16x64x64 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 20x64x64 output
// Layer 2: 20x64x64 (HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 20x32x32 output
// Layer 3: 20x32x32 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 20x32x32 output
// Layer 4: 20x32x32 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 44x32x32 output
// Layer 5: 44x32x32 (HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 48x16x16 output
// Layer 6: 48x16x16 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 48x16x16 output
// Layer 7: 48x16x16 (HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 96x8x8 output
// Layer 8: 96x8x8 (HWC data), no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, no activation, 10x8x8 output

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "gcfr_regs.h"
#include "cnn.h"
#include "weights.h"

void CNN_ISR(void)
{
  // Acknowledge interrupt to all groups
  *((volatile uint32_t *) 0x50100000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50500000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50900000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50d00000) &= ~((1<<12) | 1);

  CNN_COMPLETE; // Signal that processing is complete
#ifdef CNN_INFERENCE_TIMER
  cnn_time = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);
#else
  cnn_time = 1;
#endif
}

int cnn_continue(void)
{
  cnn_time = 0;

  *((volatile uint32_t *) 0x50100000) |= 1; // Re-enable group 0

  return CNN_OK;
}

int cnn_stop(void)
{
  *((volatile uint32_t *) 0x50100000) &= ~1; // Disable group 0

  return CNN_OK;
}

void memcpy32(uint32_t *dst, const uint32_t *src, int n)
{
  while (n-- > 0) {
    *dst++ = *src++;
  }
}

// Kernels:
static const uint32_t kernels_0[] = KERNELS_0;
static const uint32_t kernels_1[] = KERNELS_1;
static const uint32_t kernels_2[] = KERNELS_2;
static const uint32_t kernels_3[] = KERNELS_3;
static const uint32_t kernels_4[] = KERNELS_4;
static const uint32_t kernels_5[] = KERNELS_5;
static const uint32_t kernels_6[] = KERNELS_6;
static const uint32_t kernels_7[] = KERNELS_7;
static const uint32_t kernels_8[] = KERNELS_8;
static const uint32_t kernels_9[] = KERNELS_9;
static const uint32_t kernels_10[] = KERNELS_10;
static const uint32_t kernels_11[] = KERNELS_11;
static const uint32_t kernels_12[] = KERNELS_12;
static const uint32_t kernels_13[] = KERNELS_13;
static const uint32_t kernels_14[] = KERNELS_14;
static const uint32_t kernels_15[] = KERNELS_15;
static const uint32_t kernels_16[] = KERNELS_16;
static const uint32_t kernels_17[] = KERNELS_17;
static const uint32_t kernels_18[] = KERNELS_18;
static const uint32_t kernels_19[] = KERNELS_19;
static const uint32_t kernels_20[] = KERNELS_20;
static const uint32_t kernels_21[] = KERNELS_21;
static const uint32_t kernels_22[] = KERNELS_22;
static const uint32_t kernels_23[] = KERNELS_23;
static const uint32_t kernels_24[] = KERNELS_24;
static const uint32_t kernels_25[] = KERNELS_25;
static const uint32_t kernels_26[] = KERNELS_26;
static const uint32_t kernels_27[] = KERNELS_27;
static const uint32_t kernels_28[] = KERNELS_28;
static const uint32_t kernels_29[] = KERNELS_29;
static const uint32_t kernels_30[] = KERNELS_30;
static const uint32_t kernels_31[] = KERNELS_31;
static const uint32_t kernels_32[] = KERNELS_32;
static const uint32_t kernels_33[] = KERNELS_33;
static const uint32_t kernels_34[] = KERNELS_34;
static const uint32_t kernels_35[] = KERNELS_35;
static const uint32_t kernels_36[] = KERNELS_36;
static const uint32_t kernels_37[] = KERNELS_37;
static const uint32_t kernels_38[] = KERNELS_38;
static const uint32_t kernels_39[] = KERNELS_39;
static const uint32_t kernels_40[] = KERNELS_40;
static const uint32_t kernels_41[] = KERNELS_41;
static const uint32_t kernels_42[] = KERNELS_42;
static const uint32_t kernels_43[] = KERNELS_43;
static const uint32_t kernels_44[] = KERNELS_44;
static const uint32_t kernels_45[] = KERNELS_45;
static const uint32_t kernels_46[] = KERNELS_46;
static const uint32_t kernels_47[] = KERNELS_47;
static const uint32_t kernels_48[] = KERNELS_48;
static const uint32_t kernels_49[] = KERNELS_49;
static const uint32_t kernels_50[] = KERNELS_50;
static const uint32_t kernels_51[] = KERNELS_51;
static const uint32_t kernels_52[] = KERNELS_52;
static const uint32_t kernels_53[] = KERNELS_53;
static const uint32_t kernels_54[] = KERNELS_54;
static const uint32_t kernels_55[] = KERNELS_55;
static const uint32_t kernels_56[] = KERNELS_56;
static const uint32_t kernels_57[] = KERNELS_57;
static const uint32_t kernels_58[] = KERNELS_58;
static const uint32_t kernels_59[] = KERNELS_59;

int cnn_load_weights(void)
{
  *((volatile uint8_t *) 0x50180001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50180000, kernels_0, 583);
  *((volatile uint8_t *) 0x50184001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50184000, kernels_1, 583);
  *((volatile uint8_t *) 0x50188001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50188000, kernels_2, 583);
  *((volatile uint8_t *) 0x5018c051) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5018c000, kernels_3, 538);
  *((volatile uint8_t *) 0x50190001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50190000, kernels_4, 583);
  *((volatile uint8_t *) 0x50194001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50194000, kernels_5, 583);
  *((volatile uint8_t *) 0x50198001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50198000, kernels_6, 583);
  *((volatile uint8_t *) 0x5019c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5019c000, kernels_7, 583);
  *((volatile uint8_t *) 0x501a0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501a0000, kernels_8, 583);
  *((volatile uint8_t *) 0x501a4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501a4000, kernels_9, 583);
  *((volatile uint8_t *) 0x501a8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501a8000, kernels_10, 583);
  *((volatile uint8_t *) 0x501ac001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501ac000, kernels_11, 583);
  *((volatile uint8_t *) 0x501b0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501b0000, kernels_12, 583);
  *((volatile uint8_t *) 0x501b4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501b4000, kernels_13, 583);
  *((volatile uint8_t *) 0x501b8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501b8000, kernels_14, 583);
  *((volatile uint8_t *) 0x501bc001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501bc000, kernels_15, 583);
  *((volatile uint8_t *) 0x50580001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50580000, kernels_16, 583);
  *((volatile uint8_t *) 0x50584001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50584000, kernels_17, 583);
  *((volatile uint8_t *) 0x50588001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50588000, kernels_18, 583);
  *((volatile uint8_t *) 0x5058c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5058c000, kernels_19, 583);
  *((volatile uint8_t *) 0x50590001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50590000, kernels_20, 583);
  *((volatile uint8_t *) 0x50594001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50594000, kernels_21, 583);
  *((volatile uint8_t *) 0x50598001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50598000, kernels_22, 583);
  *((volatile uint8_t *) 0x5059c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5059c000, kernels_23, 583);
  *((volatile uint8_t *) 0x505a0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505a0000, kernels_24, 583);
  *((volatile uint8_t *) 0x505a4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505a4000, kernels_25, 583);
  *((volatile uint8_t *) 0x505a8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505a8000, kernels_26, 583);
  *((volatile uint8_t *) 0x505ac001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505ac000, kernels_27, 583);
  *((volatile uint8_t *) 0x505b0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505b0000, kernels_28, 583);
  *((volatile uint8_t *) 0x505b4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505b4000, kernels_29, 583);
  *((volatile uint8_t *) 0x505b8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505b8000, kernels_30, 583);
  *((volatile uint8_t *) 0x505bc001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505bc000, kernels_31, 583);
  *((volatile uint8_t *) 0x50980001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50980000, kernels_32, 583);
  *((volatile uint8_t *) 0x50984001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50984000, kernels_33, 583);
  *((volatile uint8_t *) 0x50988001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50988000, kernels_34, 583);
  *((volatile uint8_t *) 0x5098c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5098c000, kernels_35, 583);
  *((volatile uint8_t *) 0x50990001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50990000, kernels_36, 583);
  *((volatile uint8_t *) 0x50994001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50994000, kernels_37, 583);
  *((volatile uint8_t *) 0x50998001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50998000, kernels_38, 583);
  *((volatile uint8_t *) 0x5099c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5099c000, kernels_39, 583);
  *((volatile uint8_t *) 0x509a0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509a0000, kernels_40, 583);
  *((volatile uint8_t *) 0x509a4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509a4000, kernels_41, 583);
  *((volatile uint8_t *) 0x509a8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509a8000, kernels_42, 583);
  *((volatile uint8_t *) 0x509ac001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509ac000, kernels_43, 583);
  *((volatile uint8_t *) 0x509b0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509b0000, kernels_44, 583);
  *((volatile uint8_t *) 0x509b4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509b4000, kernels_45, 583);
  *((volatile uint8_t *) 0x509b8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509b8000, kernels_46, 583);
  *((volatile uint8_t *) 0x509bc001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509bc000, kernels_47, 583);
  *((volatile uint8_t *) 0x50d80001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d80000, kernels_48, 45);
  *((volatile uint8_t *) 0x50d84001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d84000, kernels_49, 45);
  *((volatile uint8_t *) 0x50d88001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d88000, kernels_50, 45);
  *((volatile uint8_t *) 0x50d8c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d8c000, kernels_51, 45);
  *((volatile uint8_t *) 0x50d90001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d90000, kernels_52, 45);
  *((volatile uint8_t *) 0x50d94001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d94000, kernels_53, 45);
  *((volatile uint8_t *) 0x50d98001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d98000, kernels_54, 45);
  *((volatile uint8_t *) 0x50d9c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d9c000, kernels_55, 45);
  *((volatile uint8_t *) 0x50da0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50da0000, kernels_56, 45);
  *((volatile uint8_t *) 0x50da4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50da4000, kernels_57, 45);
  *((volatile uint8_t *) 0x50da8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50da8000, kernels_58, 45);
  *((volatile uint8_t *) 0x50dac001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50dac000, kernels_59, 45);

  return CNN_OK;
}

static const uint8_t bias_0[] = BIAS_0;
static const uint8_t bias_1[] = BIAS_1;
static const uint8_t bias_2[] = BIAS_2;
static const uint8_t bias_3[] = BIAS_3;

static void memcpy_8to32(uint32_t *dst, const uint8_t *src, int n)
{
  while (n-- > 0) {
    *dst++ = *src++;
  }
}

int cnn_load_bias(void)
{
  memcpy_8to32((uint32_t *) 0x50108000, bias_0, sizeof(uint8_t) * 96);
  memcpy_8to32((uint32_t *) 0x50508000, bias_1, sizeof(uint8_t) * 78);
  memcpy_8to32((uint32_t *) 0x50908000, bias_2, sizeof(uint8_t) * 68);
  memcpy_8to32((uint32_t *) 0x50d08000, bias_3, sizeof(uint8_t) * 80);

  return CNN_OK;
}

int cnn_init(void)
{
  *((volatile uint32_t *) 0x50001000) = 0x00000000; // AON control
  *((volatile uint32_t *) 0x50100000) = 0x00108008; // Stop SM
  *((volatile uint32_t *) 0x50100004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50100008) = 0x00000008; // Layer count
  *((volatile uint32_t *) 0x50500000) = 0x00108008; // Stop SM
  *((volatile uint32_t *) 0x50500004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50500008) = 0x00000008; // Layer count
  *((volatile uint32_t *) 0x50900000) = 0x00108008; // Stop SM
  *((volatile uint32_t *) 0x50900004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50900008) = 0x00000008; // Layer count
  *((volatile uint32_t *) 0x50d00000) = 0x00108008; // Stop SM
  *((volatile uint32_t *) 0x50d00004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50d00008) = 0x00000008; // Layer count

  return CNN_OK;
}

int cnn_configure(void)
{
  // Layer 0 group 0
  *((volatile uint32_t *) 0x50100010) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50100090) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50100310) = 0x00002400; // SRAM write ptr
  *((volatile uint32_t *) 0x50100410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100590) = 0x00008b20; // Layer control
  *((volatile uint32_t *) 0x50100a10) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50100610) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50100690) = 0x0000003f; // TRAM ptr max
  *((volatile uint32_t *) 0x50100710) = 0x00070007; // Mask and processor enables
  *((volatile uint32_t *) 0x50100810) = 0x00001000; // Stream processing start
  *((volatile uint32_t *) 0x50100990) = 0x00001000; // Input frame size

  // Layer 0 group 1
  *((volatile uint32_t *) 0x50500010) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50500090) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50500310) = 0x00002400; // SRAM write ptr
  *((volatile uint32_t *) 0x50500410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a10) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50500610) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50500690) = 0x0000003f; // TRAM ptr max
  *((volatile uint32_t *) 0x50500810) = 0x00001000; // Stream processing start
  *((volatile uint32_t *) 0x50500990) = 0x00001000; // Input frame size

  // Layer 0 group 2
  *((volatile uint32_t *) 0x50900010) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50900090) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50900310) = 0x00002400; // SRAM write ptr
  *((volatile uint32_t *) 0x50900410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a10) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50900610) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50900690) = 0x0000003f; // TRAM ptr max
  *((volatile uint32_t *) 0x50900810) = 0x00001000; // Stream processing start
  *((volatile uint32_t *) 0x50900990) = 0x00001000; // Input frame size

  // Layer 0 group 3
  *((volatile uint32_t *) 0x50d00010) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50d00090) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50d00310) = 0x00002400; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a10) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50d00610) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50d00690) = 0x0000003f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d00790) = 0x00001040; // Post processing register
  *((volatile uint32_t *) 0x50d00810) = 0x00001000; // Stream processing start
  *((volatile uint32_t *) 0x50d00990) = 0x00001000; // Input frame size

  // Layer 1 group 0
  *((volatile uint32_t *) 0x50100014) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50100094) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50100314) = 0x0000a800; // SRAM write ptr
  *((volatile uint32_t *) 0x50100414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100514) = 0x00000400; // SRAM read ptr
  *((volatile uint32_t *) 0x50100594) = 0x0000ab20; // Layer control
  *((volatile uint32_t *) 0x50100a14) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50100614) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50100694) = 0x0000003f; // TRAM ptr max
  *((volatile uint32_t *) 0x50100794) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100714) = 0xfff0fff0; // Mask and processor enables

  // Layer 1 group 1
  *((volatile uint32_t *) 0x50500014) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50500094) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50500314) = 0x0000a800; // SRAM write ptr
  *((volatile uint32_t *) 0x50500414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500514) = 0x00000400; // SRAM read ptr
  *((volatile uint32_t *) 0x50500594) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a14) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50500614) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50500694) = 0x0000003f; // TRAM ptr max
  *((volatile uint32_t *) 0x50500794) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500714) = 0x000f000f; // Mask and processor enables

  // Layer 1 group 2
  *((volatile uint32_t *) 0x50900014) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50900094) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50900314) = 0x0000a800; // SRAM write ptr
  *((volatile uint32_t *) 0x50900414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900514) = 0x00000400; // SRAM read ptr
  *((volatile uint32_t *) 0x50900594) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a14) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50900614) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50900694) = 0x0000003f; // TRAM ptr max
  *((volatile uint32_t *) 0x50900794) = 0x00022000; // Post processing register

  // Layer 1 group 3
  *((volatile uint32_t *) 0x50d00014) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50d00094) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50d00314) = 0x0000a800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00514) = 0x00000400; // SRAM read ptr
  *((volatile uint32_t *) 0x50d00594) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a14) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50d00614) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50d00694) = 0x0000003f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d00794) = 0x0002302c; // Post processing register

  // Layer 2 group 0
  *((volatile uint32_t *) 0x50100018) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50100098) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50100198) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50100218) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50100298) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50100318) = 0x00014800; // SRAM write ptr
  *((volatile uint32_t *) 0x50100418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100518) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50100598) = 0x00006ba0; // Layer control
  *((volatile uint32_t *) 0x50100a18) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50100618) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50100698) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50100798) = 0x00022000; // Post processing register

  // Layer 2 group 1
  *((volatile uint32_t *) 0x50500018) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50500098) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50500198) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50500218) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50500298) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50500318) = 0x00014800; // SRAM write ptr
  *((volatile uint32_t *) 0x50500418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500518) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50500598) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a18) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50500618) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50500698) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50500798) = 0x00023030; // Post processing register
  *((volatile uint32_t *) 0x50500718) = 0xfff0fff0; // Mask and processor enables

  // Layer 2 group 2
  *((volatile uint32_t *) 0x50900018) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50900098) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50900198) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50900218) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50900298) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50900318) = 0x00014800; // SRAM write ptr
  *((volatile uint32_t *) 0x50900418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900518) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50900598) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a18) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50900618) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50900698) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50900798) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900718) = 0x00ff00ff; // Mask and processor enables

  // Layer 2 group 3
  *((volatile uint32_t *) 0x50d00018) = 0x00010041; // Rows
  *((volatile uint32_t *) 0x50d00098) = 0x00010041; // Columns
  *((volatile uint32_t *) 0x50d00198) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d00218) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d00298) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d00318) = 0x00014800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00518) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d00598) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a18) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50d00618) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50d00698) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d00798) = 0x00022000; // Post processing register

  // Layer 3 group 0
  *((volatile uint32_t *) 0x5010001c) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x5010009c) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x5010041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5010051c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x5010059c) = 0x0000cb20; // Layer control
  *((volatile uint32_t *) 0x50100a1c) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x5010061c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x5010069c) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x5010079c) = 0x00022000; // Post processing register

  // Layer 3 group 1
  *((volatile uint32_t *) 0x5050001c) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x5050009c) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x5050041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5050051c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x5050059c) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a1c) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x5050061c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x5050069c) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x5050079c) = 0x00022000; // Post processing register

  // Layer 3 group 2
  *((volatile uint32_t *) 0x5090001c) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x5090009c) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x5090041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5090051c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x5090059c) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a1c) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x5090061c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x5090069c) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x5090079c) = 0x00023030; // Post processing register
  *((volatile uint32_t *) 0x5090071c) = 0xff00ff00; // Mask and processor enables

  // Layer 3 group 3
  *((volatile uint32_t *) 0x50d0001c) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50d0009c) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50d0041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d0051c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d0059c) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a1c) = 0x00009800; // Layer control 2
  *((volatile uint32_t *) 0x50d0061c) = 0x00000098; // Mask offset and count
  *((volatile uint32_t *) 0x50d0069c) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d0079c) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50d0071c) = 0x0fff0fff; // Mask and processor enables

  // Layer 4 group 0
  *((volatile uint32_t *) 0x50100020) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x501000a0) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50100320) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50100420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005a0) = 0x0000ab20; // Layer control
  *((volatile uint32_t *) 0x50100a20) = 0x00015800; // Layer control 2
  *((volatile uint32_t *) 0x50100620) = 0x00a001f8; // Mask offset and count
  *((volatile uint32_t *) 0x501006a0) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x501007a0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100720) = 0xffffffff; // Mask and processor enables

  // Layer 4 group 1
  *((volatile uint32_t *) 0x50500020) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x505000a0) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50500320) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50500420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005a0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a20) = 0x00015800; // Layer control 2
  *((volatile uint32_t *) 0x50500620) = 0x00a001f8; // Mask offset and count
  *((volatile uint32_t *) 0x505006a0) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x505007a0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500720) = 0x000f000f; // Mask and processor enables

  // Layer 4 group 2
  *((volatile uint32_t *) 0x50900020) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x509000a0) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50900320) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50900420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005a0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a20) = 0x00015800; // Layer control 2
  *((volatile uint32_t *) 0x50900620) = 0x00a001f8; // Mask offset and count
  *((volatile uint32_t *) 0x509006a0) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x509007a0) = 0x00022000; // Post processing register

  // Layer 4 group 3
  *((volatile uint32_t *) 0x50d00020) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50d000a0) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50d00320) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005a0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a20) = 0x00015800; // Layer control 2
  *((volatile uint32_t *) 0x50d00620) = 0x00a001f8; // Mask offset and count
  *((volatile uint32_t *) 0x50d006a0) = 0x0000001f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007a0) = 0x00023000; // Post processing register

  // Layer 5 group 0
  *((volatile uint32_t *) 0x50100024) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x501000a4) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x501001a4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50100224) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x501002a4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50100424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100524) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x501005a4) = 0x00006ba0; // Layer control
  *((volatile uint32_t *) 0x50100a24) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50100624) = 0x02000378; // Mask offset and count
  *((volatile uint32_t *) 0x501006a4) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x501007a4) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100724) = 0xffffffff; // Mask and processor enables

  // Layer 5 group 1
  *((volatile uint32_t *) 0x50500024) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x505000a4) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x505001a4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50500224) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x505002a4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50500424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500524) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x505005a4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a24) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50500624) = 0x02000378; // Mask offset and count
  *((volatile uint32_t *) 0x505006a4) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x505007a4) = 0x00023000; // Post processing register
  *((volatile uint32_t *) 0x50500724) = 0xffffffff; // Mask and processor enables

  // Layer 5 group 2
  *((volatile uint32_t *) 0x50900024) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x509000a4) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x509001a4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50900224) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x509002a4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50900424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900524) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x509005a4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a24) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50900624) = 0x02000378; // Mask offset and count
  *((volatile uint32_t *) 0x509006a4) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x509007a4) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900724) = 0x0fff0fff; // Mask and processor enables

  // Layer 5 group 3
  *((volatile uint32_t *) 0x50d00024) = 0x00010021; // Rows
  *((volatile uint32_t *) 0x50d000a4) = 0x00010021; // Columns
  *((volatile uint32_t *) 0x50d001a4) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d00224) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d002a4) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d00424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00524) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005a4) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a24) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50d00624) = 0x02000378; // Mask offset and count
  *((volatile uint32_t *) 0x50d006a4) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007a4) = 0x00022000; // Post processing register

  // Layer 6 group 0
  *((volatile uint32_t *) 0x50100028) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x501000a8) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x50100328) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50100428) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005a8) = 0x00006b20; // Layer control
  *((volatile uint32_t *) 0x50100a28) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50100628) = 0x038004f8; // Mask offset and count
  *((volatile uint32_t *) 0x501006a8) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x501007a8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100728) = 0xffffffff; // Mask and processor enables

  // Layer 6 group 1
  *((volatile uint32_t *) 0x50500028) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x505000a8) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x50500328) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50500428) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005a8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a28) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50500628) = 0x038004f8; // Mask offset and count
  *((volatile uint32_t *) 0x505006a8) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x505007a8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500728) = 0xffffffff; // Mask and processor enables

  // Layer 6 group 2
  *((volatile uint32_t *) 0x50900028) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x509000a8) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x50900328) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50900428) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005a8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a28) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50900628) = 0x038004f8; // Mask offset and count
  *((volatile uint32_t *) 0x509006a8) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x509007a8) = 0x00023000; // Post processing register
  *((volatile uint32_t *) 0x50900728) = 0xffffffff; // Mask and processor enables

  // Layer 6 group 3
  *((volatile uint32_t *) 0x50d00028) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x50d000a8) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x50d00328) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00428) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005a8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a28) = 0x00017800; // Layer control 2
  *((volatile uint32_t *) 0x50d00628) = 0x038004f8; // Mask offset and count
  *((volatile uint32_t *) 0x50d006a8) = 0x0000000f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007a8) = 0x00022000; // Post processing register

  // Layer 7 group 0
  *((volatile uint32_t *) 0x5010002c) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x501000ac) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x501001ac) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x5010022c) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x501002ac) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x5010042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501004ac) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x5010052c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x501005ac) = 0x00006ba0; // Layer control
  *((volatile uint32_t *) 0x50100a2c) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x5010062c) = 0x050007f8; // Mask offset and count
  *((volatile uint32_t *) 0x501006ac) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x501007ac) = 0x00023000; // Post processing register
  *((volatile uint32_t *) 0x5010072c) = 0xffffffff; // Mask and processor enables

  // Layer 7 group 1
  *((volatile uint32_t *) 0x5050002c) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x505000ac) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x505001ac) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x5050022c) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x505002ac) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x5050042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505004ac) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x5050052c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x505005ac) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a2c) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x5050062c) = 0x050007f8; // Mask offset and count
  *((volatile uint32_t *) 0x505006ac) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x505007ac) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x5050072c) = 0xffffffff; // Mask and processor enables

  // Layer 7 group 2
  *((volatile uint32_t *) 0x5090002c) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x509000ac) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x509001ac) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x5090022c) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x509002ac) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x5090042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509004ac) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x5090052c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x509005ac) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a2c) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x5090062c) = 0x050007f8; // Mask offset and count
  *((volatile uint32_t *) 0x509006ac) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x509007ac) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x5090072c) = 0xffffffff; // Mask and processor enables

  // Layer 7 group 3
  *((volatile uint32_t *) 0x50d0002c) = 0x00010011; // Rows
  *((volatile uint32_t *) 0x50d000ac) = 0x00010011; // Columns
  *((volatile uint32_t *) 0x50d001ac) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d0022c) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d002ac) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d0042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d004ac) = 0x00000001; // Write ptr multi-pass channel offs
  *((volatile uint32_t *) 0x50d0052c) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005ac) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a2c) = 0x00017810; // Layer control 2
  *((volatile uint32_t *) 0x50d0062c) = 0x050007f8; // Mask offset and count
  *((volatile uint32_t *) 0x50d006ac) = 0x00000007; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007ac) = 0x00022000; // Post processing register

  // Layer 8 group 0
  *((volatile uint32_t *) 0x50100030) = 0x00000007; // Rows
  *((volatile uint32_t *) 0x501000b0) = 0x00000007; // Columns
  *((volatile uint32_t *) 0x50100330) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x501003b0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005b0) = 0x00016920; // Layer control
  *((volatile uint32_t *) 0x50100a30) = 0x00004801; // Layer control 2
  *((volatile uint32_t *) 0x50100630) = 0x48004898; // Mask offset and count
  *((volatile uint32_t *) 0x50100130) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50100730) = 0xffffffff; // Mask and processor enables

  // Layer 8 group 1
  *((volatile uint32_t *) 0x50500030) = 0x00000007; // Rows
  *((volatile uint32_t *) 0x505000b0) = 0x00000007; // Columns
  *((volatile uint32_t *) 0x50500330) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x505003b0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005b0) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x50500a30) = 0x00004801; // Layer control 2
  *((volatile uint32_t *) 0x50500630) = 0x48004898; // Mask offset and count
  *((volatile uint32_t *) 0x50500130) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007b0) = 0x00001044; // Post processing register
  *((volatile uint32_t *) 0x50500730) = 0xffffffff; // Mask and processor enables

  // Layer 8 group 2
  *((volatile uint32_t *) 0x50900030) = 0x00000007; // Rows
  *((volatile uint32_t *) 0x509000b0) = 0x00000007; // Columns
  *((volatile uint32_t *) 0x50900330) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x509003b0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005b0) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x50900a30) = 0x00004801; // Layer control 2
  *((volatile uint32_t *) 0x50900630) = 0x48004898; // Mask offset and count
  *((volatile uint32_t *) 0x50900130) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50900730) = 0xffffffff; // Mask and processor enables

  // Layer 8 group 3
  *((volatile uint32_t *) 0x50d00030) = 0x00000007; // Rows
  *((volatile uint32_t *) 0x50d000b0) = 0x00000007; // Columns
  *((volatile uint32_t *) 0x50d00330) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d003b0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005b0) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x50d00a30) = 0x00004801; // Layer control 2
  *((volatile uint32_t *) 0x50d00630) = 0x48004898; // Mask offset and count
  *((volatile uint32_t *) 0x50d00130) = 0x00000100; // 1D


  *((volatile uint32_t *) 0x50000000) = 0x00001908; // FIFO control

  return CNN_OK;
}

int cnn_start(void)
{
  cnn_time = 0;

  *((volatile uint32_t *) 0x50100000) = 0x00108808; // Enable group 0
  *((volatile uint32_t *) 0x50500000) = 0x00108809; // Enable group 1
  *((volatile uint32_t *) 0x50900000) = 0x00108809; // Enable group 2
  *((volatile uint32_t *) 0x50d00000) = 0x00108809; // Enable group 3

#ifdef CNN_INFERENCE_TIMER
  MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
#endif

  CNN_START; // Allow capture of processing time
  *((volatile uint32_t *) 0x50100000) = 0x00108809; // Master enable group 0

  return CNN_OK;
}

// Custom unload for this network: 32-bit data, shape: [10, 8, 8]
int cnn_unload(uint32_t *out_buf)
{
  volatile uint32_t *addr;
  int i;

  addr = (volatile uint32_t *) 0x50402000;
  for (i = 0; i < 32; i++) {
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
  }
  addr = (volatile uint32_t *) 0x5040a000;
  for (i = 0; i < 32; i++) {
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
  }
  addr = (volatile uint32_t *) 0x50412000;
  for (i = 0; i < 16; i++) {
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
  }

  return CNN_OK;
}

int cnn_enable(uint32_t clock_source, uint32_t clock_divider)
{
  // Reset all domains, restore power to CNN
  MXC_GCFR->reg3 = 0xf; // Reset
  MXC_GCFR->reg1 = 0xf; // Mask memory
  MXC_GCFR->reg0 = 0xf; // Power
  MXC_GCFR->reg2 = 0x0; // Iso
  MXC_GCFR->reg3 = 0x0; // Reset

  MXC_GCR->pclkdiv = (MXC_GCR->pclkdiv & ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | MXC_F_GCR_PCLKDIV_CNNCLKSEL))
                     | clock_divider | clock_source;
  MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN); // Enable CNN clock

  NVIC_SetVector(CNN_IRQn, CNN_ISR); // Set CNN complete vector

  return CNN_OK;
}

int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin)
{
  mxc_gpio_cfg_t gpio_out;
  gpio_out.port = port;
  gpio_out.mask = pin;
  gpio_out.pad = MXC_GPIO_PAD_NONE;
  gpio_out.func = MXC_GPIO_FUNC_OUT;
  MXC_GPIO_Config(&gpio_out);
  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);

  return CNN_OK;
}

int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin)
{
  mxc_gpio_cfg_t gpio_out;
  gpio_out.port = port;
  gpio_out.mask = pin;
  gpio_out.pad = MXC_GPIO_PAD_NONE;
  gpio_out.func = MXC_GPIO_FUNC_OUT;
  MXC_GPIO_Config(&gpio_out);
  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);

  return CNN_OK;
}

int cnn_disable(void)
{
  // Disable CNN clock
  MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN);

  // Disable power to CNN
  MXC_GCFR->reg3 = 0xf; // Reset
  MXC_GCFR->reg2 = 0xf; // Iso
  MXC_GCFR->reg0 = 0x0; // Power
  MXC_GCFR->reg1 = 0x0; // Mask memory
  MXC_GCFR->reg3 = 0x0; // Reset

  return CNN_OK;
}

