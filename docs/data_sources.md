# Data Sources and Formats

## Chinese Stations

- [PVOD ScienceDB](https://www.scidb.cn/en/detail?dataSetId=f8f3d7af144f441795c5781497e56b62) is the official landing page for the DOI cited by the authors' tools. The tools reference V4, while the public metadata on the current details page indicates version 5.0.0. The download script retrieves the original 7,983,824-byte ZIP archive from ScienceDB's public V4 file listing, verifies its size and required members, and extracts it. If the page changes, select V4 on the official details page and download Data Files. The supplementary materials for the [Solar Energy](https://doi.org/10.1016/j.solener.2021.09.050) article also provide access to the authors' instructions.
- Table 1 of the authors' data paper specifies that the CSV `power` column is in MW and `Capacity` in `metadata.csv` is in kW, so capacity is divided by 1,000. The authors' tools parse the original observations as UTC. The main experiment maps H1, H2, H4, H6, H7, and H8 to the published stations with suffixes 01, 02, 04, 06, 07, and 08, respectively. Data files are restricted to July 2018 through June 2019 in Beijing time.
- The [Scientific Data paper on the State Grid Competition](https://pmc.ncbi.nlm.nih.gov/articles/PMC9492786/) and [Figshare V4](https://doi.org/10.6084/m9.figshare.17304221.v4) provide all collection stations and the original RAR archive. The official Figshare file ID is 35215012. The script retrieves the G5 and G7 XLSX files from `data_original/solar_stations` at the pinned commit `efe508dfe76b33fe437e2e6d3274534b50cc2f37` in the [authors' repository](https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis) linked by the paper. Both original files have been checked against the commit's Git blobs. The columns are `Time(year-month-day h:m:s)` and `Power (MW)`. Records from July 2019 through June 2020 are retained in Beijing time. The capacities are 110 MW and 30 MW, respectively.

## U.S. PVDAQ

[OEDI 4568](https://data.openei.org/submissions/4568) publishes the 2023 Solar Data Prize archive in a public S3 bucket. The legacy [PVDAQ v3 API](https://developer.nlr.gov/docs/solar/pvdaq-v3/) has been discontinued. The downloader uses anonymous S3 HTTP file listings and object downloads, checks file sizes against the listings, and supports resuming from `.part` files when S3 accepts HTTP Range requests. No API key is required. Preserve the original filenames when using the published files directly. By default, the recorded DC rated capacity is used, and meter power is converted to kW before normalization. Equipment may use different capacity definitions. If the capacity used by the paper's authors is known, replace the default with an override that includes its source.

| System | Power Data in the Archive | Default Processing |
|---|---|---|
| 2105 | `2105_meter_data.csv` | The published metadata specifies a 15-minute AC kW meter. Inspection shows a nearly flat intraday average profile for this column; confirm the measurement channel used by the authors before comparing against the paper. The system file reports a rated capacity of 110 kW DC, while the general OEDI description lists 237 kW. The meter and capacity can be overridden through configuration with documented sources. |
| 2107 | `2107_meter_15m_data.csv` | Use the revenue meter readings directly; `PST8PDT`, 893 kW DC. |
| 9068 | `9068_ac_power_data.csv` | Use the total AC meter column, avoiding individual phases or duplicate inverter values. Clip negative net import readings to zero generation and count them. Inspection of this channel for 2021–2023 found a minimum of −46 kW and approximately half of its readings to be negative; removing them directly would create systematic nighttime gaps. |
| 7333 | Annual AC files in `7333_5_min_OEDI` | All three CSV files for 2021–2023 have been downloaded. Read the station-wide MW column in UTC, convert it to kW, and normalize by the system metadata capacity of 257,600 kW DC. Small negative nighttime net import readings are clipped to zero generation and counted. In earlier years, this channel is more sparsely reported than the file's time grid; 54,620 complete 15-minute observations are retained. All four seasons of 2021 and the first two seasons of 2022 fail to meet the default contiguous-window coverage requirement. |
| 9069 | `9069_meter_data.csv` | Use the total AC power from revenue meter 2 for 2021–2023. Inspection found readings only from meter 2 during this period; meter 1 contains values only for 2016–2019. Configure a fixed offset using the numeric time zone code in the public metadata. |

## User-Provided Data

Original CSV, XLSX, or Parquet files can be supplied directly, with paths, capacity, and timestamp columns overridden in `data/metadata/capacities.yaml`. Meteorological variables are not required. The following example describes the configuration fields; replace the capacity value with the published rated capacity of the actual station.

```yaml
H7:
  files: [data/raw/pvod/station07.csv]
  timestamp_column: date_time
  power_column: power
  input_timezone: UTC
  timezone: Asia/Shanghai
  capacity:
    value: 20.0  # Example field. Use this station's actual rated capacity.
    unit: MW
    source: "Enter the source of the station capacity record"
```

When using your own meter sequences, you can also register station information under `data.stations` in `configs/data.yaml` and configure the sources, target, or periods as required by the experiment. Check the `data/processed/*.json` files generated after cleaning to confirm the actual capacity, date range, data gaps, and SHA-256 file hashes.
