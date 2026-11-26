# M5Stack CoreS3 + YOLO26n 安全帽（PPE）端侧监测与企业微信告警系统

基于 **M5Stack CoreS3**（ESP32-S3 + 8MB Quad PSRAM + 16MB QIO Flash + GC0308 摄像头 + 2.0" 触摸屏 + AW88298 1W 扬声器）、**Ultralytics YOLO26n** 与 **Construction-PPE 数据集** 实现的纯端侧安全帽佩戴监测与双通道告警系统。

## 系统特性

- **纯端侧离线推理（ESP-IDF v5.5 + ESP-DL）**：无需任何 PC/服务器中转，YOLO26n 512×512（INT8/INT16 混合精度 + Step-Interpolated INT16 LUT）以零拷贝方式存放于 CoreS3 16MB Flash `.rodata` 并由 ESP32-S3 本地执行。
- **3 类精简重映射数据集**：自动从 Ultralytics `construction-ppe`（原 12 类）提取并重映射为 `0: helmet`、`1: no_helmet`、`2: person`，把端侧模型容量聚焦在安全帽检测任务上。
- **双规则 + 两级防误报状态机**：
  - **直接规则**：检测到 `no_helmet` 即计入违规；
  - **兜底规则**：检测到完整 `person` 且头部区域无 `helmet` 匹配时触发兜底告警（自动过滤画面顶部边缘被截断的人体，防止误报）；
  - **第 1 帧（预警）**：屏幕橙色边框 + 横幅提示 + 扬声器短促“滴”提示音；
  - **连续 ≥ 2 帧（确认告警）**：屏幕红色闪烁告警框 + 播报内置中文语音（*“警告，未佩戴安全帽，请立即佩戴”*）+ **企业微信群机器人** HTTPS 推送（Markdown 告警卡片 + 带红绿检测框的 320×240 JPEG 现场抓拍图，默认 30s 冷却去重）。
- **双核解耦与触摸交互**：
  - **Core 0**：GC0308 实时采集、320×240 LCD 流畅预览与检测框叠加、触摸控制（底部一键 `[MUTE]` 静音 / `[PAUSE]` 暂停检测）、AW88298 音频播报、Wi-Fi + SNTP 校时 + 企微异步 HTTPS 推送；
  - **Core 1（+ ESP-DL 双核协同）**：执行 YOLO26n `320×320` 端侧推理（CoreS3 实测约 **3.9 秒/帧**），不卡死屏幕预览。

---

## 目录结构

```text
PPE-yolo/
├── ml/                                  # Python 训练、量化与语音资产管线（uv 管理）
│   ├── pyproject.toml                   # 锁定依赖（ultralytics==8.4.7, esp-ppq==1.2.4 等）
│   ├── prepare_dataset.py               # 下载 construction-ppe 并重映射为 3 类数据集
│   ├── train_yolo26n.py                 # Apple MPS / CUDA 训练 YOLO26n (512×512)
│   ├── quantize_espdl.py                # ONNX 导出 + PTQ/TQT/INT16 LUT + 图手术导出 .espdl
│   ├── generate_audio.py                # 生成 16kHz 16-bit 单声道中文语音 PCM 资产
│   └── vendor/                          # Espressif esp-dl 官方量化补丁与 esp_ppq_lut 扩展
└── firmware/                            # M5Stack CoreS3 ESP-IDF 固件工程
    ├── idf_env.sh                       # ESP-IDF v5.5.1（ESP32-S3 精简工具链）环境加载脚本
    ├── partitions.csv                   # 16MB Flash 分区表（factory 10MB 零拷贝承载 .espdl）
    ├── sdkconfig.defaults               # CoreS3 硬件配置（16MB QIO Flash / 8MB Quad PSRAM）
    ├── components/yolo26_ppe/           # 支持 INT8/INT16 混合输出与反 Letterbox 映射的解码器
    ├── tests/test_violation_engine.cpp  # 宿主机告警状态机单元测试
    └── main/
        ├── Kconfig.projbuild            # menuconfig 参数（Wi-Fi / 企微 Webhook / 阈值 / 冷却）
        ├── assets/alert_zh.pcm          # 嵌入 Flash 的中文语音告警音频
        ├── models/yolo26n_ppe_512_s8_s3.espdl
        ├── include/ & src/              # 摄像头、UI、音频、企微推送、违规引擎与双核主循环
```

---

## 快速上手

### 1. 模型训练与 ESP-DL 量化（`ml/`）

```bash
cd ml

# 1) 安装依赖（国内可选加 --default-index https://pypi.tuna.tsinghua.edu.cn/simple）
uv sync --python 3.11

# 2) 下载并生成 3 类 PPE 数据集（helmet / no_helmet / person）
uv run python prepare_dataset.py

# 3) 训练 YOLO26n (512×512)，自动启用 Apple M1/M2/M3 MPS 或 CUDA
uv run python train_yolo26n.py --epochs 50 --imgsz 512 --batch 16

# 4) 使用 esp-ppq 混合精度 PTQ + TQT + INT16 LUT 导出 .espdl 并自动复制到固件目录
uv run python quantize_espdl.py --imgsz 512 --platform s3
```

> **提示**：实机部署后，若希望进一步提升 GC0308 摄像头在真实光照下的 INT8 精度，可将 CoreS3 实拍图片放入自定义目录，通过 `--calib-dir <目录>` 重新执行量化校准。

### 2. 配置 Wi-Fi 与企业微信群机器人（`menuconfig`）

本项目采用纯编译期配置，敏感配置（`firmware/sdkconfig`）已在 `.gitignore` 中排除，不会提交到 Git。

```bash
# 在仓库根目录加载 ESP-IDF v5.5.1 (ESP32-S3) 工具链环境
# （若本机已安装完整 ESP-IDF 全部工具，也可改用 source ~/esp/esp-idf/export.sh）
source firmware/idf_env.sh
cd firmware

idf.py set-target esp32s3
idf.py menuconfig
```

进入 **`PPE Helmet Monitor (CoreS3 + YOLO26)`** 菜单填写：
- **`Wi-Fi SSID` / `Wi-Fi Password`**：2.4GHz Wi-Fi 凭证（留空则纯离线运行，仅本地声光报警）
- **`WeCom Group Robot Webhook URL`**：企业微信群机器人完整地址，例如 `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxxxxx`
- **`Device Identifier`**：告警卡片中显示的设备名称（默认 `CoreS3-PPE-01`）
- 按需微调置信度阈值（默认 `40%`）、连续帧数（默认 `2`）、企微冷却（默认 `30s`）、扬声器音量（默认 `85%`）

### 3. 编译、烧录与串口监控

将 M5Stack CoreS3 通过 USB-C 连接电脑后执行：

```bash
source firmware/idf_env.sh
cd firmware

idf.py build
idf.py -p /dev/cu.usbmodem* flash monitor
```

### 4. 自定义中文告警语音（可选）

```bash
python3 ml/generate_audio.py --text "警告，未佩戴安全帽，请立即佩戴"
```

---

## 实测指标（Apple M1 Pro 训练 + ESP-DL S3 硬件仿真验证）

### 数据集（`construction-ppe` 3 类重映射）

- `train`：800 张图像（8 张背景图）｜标注框：`helmet = 1282`、`no_helmet = 634`、`person = 1962`
- `val`：200 张图像（954 个实例）｜标注框：`helmet = 294`、`no_helmet = 179`、`person = 481`

### 模型精度（固定 512×512 验证集，200 图 / 954 实例）

| 阶段 / 模型 | 体积 | Precision | Recall | mAP50 | mAP50-95 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| **FP32 (`yolo26n_ppe.pt`)** | 5.1 MB | 0.795 | 0.729 | **0.7905** | **0.5809** | 50 epochs (Apple M1 Pro MPS)；融合后 2.38M 参数 / 5.2 GFLOPs |
| ├─ `helmet` (294) | — | 0.802 | 0.783 | 0.837 | 0.618 | — |
| ├─ `no_helmet` (179) | — | 0.726 | 0.520 | 0.609 | 0.353 | 小目标样本相对偏少 |
| └─ `person` (481) | — | 0.856 | 0.884 | 0.926 | 0.772 | 支撑“完整 `person` 头部无 `helmet`”兜底规则 |
| **ESP-DL S3 512×512 混合量化 (`.espdl`)** | 2.86 MB | 0.829 | 0.701 | **0.7830** | **0.5669** | 主干 INT8 + 48 层 INT16 + INT16 Swish LUT（CoreS3 单核约 `10.8s/帧`） |
| **ESP-DL S3 320×320 混合量化 (`.espdl`，当前默认)** | **2.87 MB** | 0.794 | 0.634 | **0.6850** | **0.4831** | 与 GC0308 `320×240` 1:1 匹配；双核 + PSRAM 权重实测 **`3.9s/帧`** |

- 图手术后最终离散输出 6 张量：`one2one_p{3,4,5}_{box,cls}`（已自动复制到 `firmware/main/models/yolo26n_ppe_512_s8_s3.espdl`）。

### CoreS3 固件编译与测试结果

- **工具链与组件**：ESP-IDF `v5.5.1` + `espressif/m5stack_core_s3@4.1.0` + `espressif/esp-dl@3.3.15` + `espressif/esp_video@2.0.1` + `espressif/esp_new_jpeg@1.0.2`
- **固件镜像体积**：
  - `cores3_ppe_yolo26.bin`：**`4.82 MB`**（`0x4d2400`，占 10 MB `factory` 分区的 **48%**，剩余 **52%**）
  - `bootloader.bin`：**`22 KB`**（`0x57e0`，剩余 39%）；`partition-table.bin`：`3.0 KB`
  - Flash 占用：`.text` `1.59 MB`（`1,670,756 B`），`.rodata` `3.13 MB`（`3,285,292 B`，含零拷贝 16 字节对齐嵌入的 `2.86 MB` `.espdl` 模型与 `110 KB` 中文语音 PCM）
- **内部 RAM（DIRAM）**：已用 **`116.0 KB / 341.8 KB`（33.95%）**，剩余 **`225.7 KB`**；模型参数采用 `param_copy = false` 直接从 16MB QIO Flash `.rodata` 零拷贝映射，不占用 8MB Quad PSRAM
- **宿主机单元测试**：`ViolationEngine`（`firmware/tests/test_violation_engine.cpp`）**8/8 全部通过**（零编译警告；覆盖戴盔放行、顶部截断放行、直接/兜底双规则、两级消抖及语音/企微独立冷却）
- **M5Stack CoreS3 真机全链路实测**：
  - 摄像头采集：GC0308 `320×240 RGB565_BE @ 20fps` + LCD 双缓冲 Ping-Pong 预览（无雪花/无撕裂），触摸 `[MUTE]`/`[PAUSE]` 响应正常
  - 端侧推理耗时（320×320 + `RUNTIME_MODE_MULTI_CORE` + PSRAM 权重驻留 `param_copy=true`）：`Pre ≈ 50 ms` | **`Inf ≈ 3.89~3.98 s`** | `Post ≈ 1 ms`（较 512×512 单核 `10.8s` 提速 **2.8×**；单帧直发模式下告警响应从 ~22s 缩短至 **~4s**）
  - 联网与告警内存余量（Wi-Fi STA + SNTP + TLS + 相机 + LCD 双缓冲 + YOLO26n 并发运行）：Free DRAM **`28~37 KB`**、Free PSRAM **`3,362 KB`**
  - 企业微信群机器人 HTTPS 推送：TLS 证书链校验通过，Markdown 告警卡片返回 `HTTP 200 {"errcode":0,"errmsg":"ok"}`，并自动完成 320×240 带框现场抓拍图（JPEG `~9.6 KB` + Base64/MD5）上传
