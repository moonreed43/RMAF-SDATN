# 数据来源与格式

## 中国站点

- [PVOD ScienceDB](https://www.scidb.cn/en/detail?dataSetId=f8f3d7af144f441795c5781497e56b62) 为作者工具所引 DOI 的官方落地页。作者工具引用 V4，当前详情页公开元数据标示 5.0.0。下载脚本从 ScienceDB 公开 V4 清单取回 7,983,824 字节的 ZIP 原档，校验大小和所需成员后解压。页面变动时，在官方详情选择 V4 并下载 Data Files。期刊 [Solar Energy](https://doi.org/10.1016/j.solener.2021.09.050) 的附件也是作者说明的入口。
- 作者数据论文的表 1 规定 CSV `power` 为 MW、`metadata.csv` 的 `Capacity` 为 kW，因此容量除以 1,000。作者工具按 UTC 解析原始观测。主实验将 H1、H2、H4、H6、H7、H8 映射到后缀为 01、02、04、06、07、08 的发布站点。数据文件按 2018-07 到 2019-06 的北京时间范围裁切。
- [国家电网竞赛的 Scientific Data 数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9492786/) 与 [Figshare V4](https://doi.org/10.6084/m9.figshare.17304221.v4) 给出全部采集站点和原始 RAR。官方 Figshare 文件 ID 35215012。脚本从论文链接的 [作者仓库](https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis) 固定提交 `efe508dfe76b33fe437e2e6d3274534b50cc2f37` 取 `data_original/solar_stations` 中 G5 和 G7 的 XLSX。两份原始文件已与提交的 Git blob 比对。列为 `Time(year-month-day h:m:s)` 与 `Power (MW)`，按北京时间保留 2019-07 到 2020-06。容量分别为 110 MW 和 30 MW。

## 美国 PVDAQ

[OEDI 4568](https://data.openei.org/submissions/4568) 在公开 S3 存储桶下发布 2023 Solar Data Prize 归档。旧 [PVDAQ v3 API](https://developer.nlr.gov/docs/solar/pvdaq-v3/) 已停用。下载器使用匿名 S3 HTTP inventory 和对象下载，将大小与 inventory 比对，支持在 S3 接受 HTTP Range 时从 `.part` 续传。无 API 密钥。直接使用发布文件时保留原文件名。默认读取记录的 DC 额定容量，电表功率转换到 kW 后归一化。设备可能具有不同的容量口径。若掌握论文作者使用的容量，以带来源的覆盖项替换。

| 系统 | 归档内功率资料 | 默认处理 |
|---|---|---|
| 2105 | `2105_meter_data.csv` | 发布元数据标注的 AC kW 电表为 15 分钟。实查该列日内均值很平，建议在与论文对照前确认作者用的测点。系统文件额定容量为 110 kW DC，OEDI 通用说明列 237 kW。可通过带来源的配置覆盖电表与容量。 |
| 2107 | `2107_meter_15m_data.csv` | 直接使用收入计量电表读数；`PST8PDT`，DC 893 kW。 |
| 9068 | `9068_ac_power_data.csv` | 取总 AC 电表列，避开相别或重复的逆变器值。将负向净输入裁为零发电并计数。2021–2023 实查该通道最低为 −46 kW，约一半读数为负，直接删除会造成系统性夜间缺口。 |
| 7333 | `7333_5_min_OEDI` 年度 AC 文件 | 2021–2023 三份 CSV 已全部下载。以 UTC 读取总站 MW 列，并转换到 kW，按系统元数据 257,600 kW DC 归一化。原始夜间小幅负向净输入按零发电裁剪并计数。该通道在早期年份比文件网格稀疏，保留 54,620 个完整 15 分钟观测。2021 四季和 2022 前两季不满足默认连续窗口覆盖率。 |
| 9069 | `9069_meter_data.csv` | 2021–2023 采用收入电表 2 的总 AC 功率。实查该期间仅电表 2 有读数，电表 1 仅在 2016–2019 有值。以公开元数据的数值时区码配置固定偏移。 |

## 用户提供的数据

可以直接放原始 CSV、XLSX 或 Parquet，并通过 `data/metadata/capacities.yaml` 覆盖路径、容量和时间列。没有气象变量要求。以下为配置字段说明，容量值应替换为同一实际站点公开额定容量。

```yaml
H7:
  files: [data/raw/pvod/station07.csv]
  timestamp_column: date_time
  power_column: power
  input_timezone: UTC
  timezone: Asia/Shanghai
  capacity:
    value: 20.0  # 示意字段。实际值使用本站额定容量。
    unit: MW
    source: "填写站点容量记录出处"
```

使用自备电表序列时，也可在 `configs/data.yaml` 的 `data.stations` 注册站点信息，将源/目标或周期设置为实验所需。校验清洗后生成的 `data/processed/*.json` 以确认实际容量、日期范围、数据缺口与 SHA-256 文件摘要。
