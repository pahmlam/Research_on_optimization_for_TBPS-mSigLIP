# System Information: Qualcomm RB3 Gen2 Development Kit
> **Generated:** 2026-01-25

---

## 1. Hardware Information

### 1.1 System Overview
| Property | Value |
| :--- | :--- |
| **Hostname** | `qc-rb3g2` |
| **Platform** | Qualcomm Robotics RB3 Gen2 |
| **SoC** | Qualcomm QCS6490 |
| **Uptime** | 6 days |

### 1.2 CPU Specification
| Property | Value |
| :--- | :--- |
| **Architecture** | aarch64 (ARM64) |
| **CPU(s)** | 8 |
| **CPU op-modes** | 32-bit, 64-bit |
| **Byte Order** | Little Endian |
| **Vendor ID** | ARM |

**CPU Clusters (big.LITTLE)**
| Cluster | Model | Cores | Max Freq | Min Freq |
| :--- | :--- | :--- | :--- | :--- |
| **Performance** | Cortex-A78 | 4 (cores 4-7) | 2707 MHz | 691 MHz |
| **Efficiency** | Cortex-A55 | 4 (cores 0-3) | 1958 MHz | 300 MHz |

**CPU Features & Acceleration**
* **Flags:** `fp` `asimd` `evtstrm` `aes` `pmull` `sha1` `sha2` `crc32` `atomics` `fphp` `asimdhp` `cpuid` `asimdrdm` `lrcpc` `dcpop` `asimddp`
* **NEON/ASIMD:** Advanced SIMD for ML acceleration
* **AES/SHA:** Hardware cryptography
* **FP16:** Half-precision floating point support
* **CRC32:** Hardware CRC acceleration

### 1.3 Memory
| Property | Value |
| :--- | :--- |
| **Total RAM** | 5.2 GiB |
| **Available** | ~4.0 GiB |
| **Swap** | 0 B (disabled) |

### 1.4 GPU
| Property | Value |
| :--- | :--- |
| **Model** | Adreno 643 |
| **DRI Device** | `/dev/dri/card0` |
| **Render Device** | `/dev/dri/renderD128` |

### 1.5 DSP/NPU
| Component | Status |
| :--- | :--- |
| **Hexagon DSP** | Hexagon 770 |
| **ADSP RPC** | Not accessible |
| **NPU** | Qualcomm AI Engine |

### 1.6 Storage
| Filesystem | Size | Used | Avail | Use% | Mounted on |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/dev/root` | 58G | 12G | 44G | 22% | `/` |
| `tmpfs` | 2.6G | 0 | 2.6G | 0% | `/dev/shm` |

---

## 2. Software Information

### 2.1 Operating System
| Property | Value |
| :--- | :--- |
| **OS** | Ubuntu 24.04.3 LTS |
| **Codename** | Noble Numbat |
| **Kernel** | Linux aarch64 |

### 2.2 Python Environment
| Component | Version |
| :--- | :--- |
| **Python** | 3.13 |
| **pip** | 25.3 |

> *Note: Conda Base environment active*

### 2.3 Deep Learning Frameworks 
*(Environment: `~/sigm/venv`)*

| Framework | Version |
| :--- | :--- |
| **PyTorch** | 2.10.0+cpu |
| **torchvision** | 0.25.0+cpu |
| **ONNX Runtime** | 1.23.2 |
| **timm** | 1.0.24 |
| **NumPy** | 2.3.5 |
| **Pandas** | 3.0.0 |

### 2.4 Qualcomm Packages
| Package | Version | Description |
| :--- | :--- | :--- |
| `qcom-fastcv-binaries` | 1.8.0 | FastCV computer vision |
| `qcom-fastrpc` | 1.0.0-12.3 | FastRPC for DSP |
| `qcom-camx-*` | 1.0.0 | Camera stack |
| `qcom-sensors-*` | 1.0.0 | Sensor APIs |
| `qcom-iot-defaults` | 1.19 | IoT board config |

---

## 3. AI/ML Capabilities

### 3.1 Available Accelerators
| Accelerator | Status | Notes |
| :--- | :--- | :--- |
| **CPU (NEON)** | ✅ Active | 8 cores with SIMD |
| **GPU (Adreno)** | ⚠️ Limited | OpenCL available |
| **DSP (Hexagon)** | ❌ Requires SDK | SNPE/QNN needed |
| **NPU** | ❌ Requires SDK | QNN needed |

### 3.2 SNPE Runtime Status
* **Library:** `/usr/lib/libhta_hexagon_runtime_snpe.so` (FOUND)
* **CLI Tools:** NOT IN PATH
* **SDK:** NOT INSTALLED

### 3.3 Recommended SDK
**Qualcomm AI Engine Direct SDK**
* **Download:** [Developer Portal Link](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
* **Enables:** DSP acceleration, NPU acceleration, INT8 quantization
* **Expected Speedup:** 10-50x over CPU

---

## 4. Network Configuration
| Property | Value |
| :--- | :--- |
| **Web Console** | `https://qcrb3g2-ansalab.j12tee.qzz.io` |
| **Management** | Cockpit |
| **Last Login** | Jan 24, 05:43 PM |

---

## 5. Security Status

### 5.1 CPU Vulnerabilities
| Vulnerability | Status |
| :--- | :--- |
| Gather data sampling | Not affected |
| Itlb multihit | Not affected |
| L1tf | Not affected |
| Mds | Not affected |
| Meltdown | Not affected |
| Mmio stale data | Not affected |
| Retbleed | Not affected |
| Spec rstack overflow | Not affected |
| Spectre v1/v2 | Not affected |

### 5.2 Updates Available
> **18 updates available** (17 security fixes)
> **Packages:** `avahi`, `glib`, `libxml2`, `python3-pyasn1`, `docker-compose-plugin`

---

## 6. Performance Profile

| Setting | Value |
| :--- | :--- |
| **Current Profile** | None (default) |
| **CPU Governor** | schedutil |
| **Scaling** | 51-65% |

### Recommended for ML Benchmarks
Để tối ưu hóa hiệu suất khi chạy các bài kiểm tra ML, hãy thiết lập CPU governor sang chế độ `performance`:

```bash
# Set performance governor
sudo cpupower frequency-set -g performance
```
*Hoặc cấu hình qua Cockpit: **Configuration > Performance profile***

---
*System information collected from `qc-rb3g2` Qualcomm RB3 Gen2 Development Kit.*