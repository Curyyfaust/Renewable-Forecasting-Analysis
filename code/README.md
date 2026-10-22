# 论文核心代码与数值结果

本文件夹是原研究目录的精简副本。保留了研究代码、已训练模型和必要的数值结果，未包含原始数据。脚本仍使用原有相对目录，运行时请先进入本文件夹。

## 文件结构

```text
code/
  model.py, exp.py, main_train.py       四类预测模型和训练
  dataloader/                          风电与气象样本对齐
  forecasting_2020.py                  2020 年预测生成
  PyPSA-china/main_dispatch.py         本研究两阶段省级调度
  00_intro.py ... 06_sen.py             正文图表、归因、干预及储能分析
  supp/01_*.py ... 06_*.py             补充分析
  supp/supp_figure/*.csv               补充图对应的数值结果
  figure/*.csv                        正文图对应的辅助数值结果
  data/ssp15_controlled_states.csv     小型派生情景配置，非原始观测数据
  results/                            主要数值结果、模型权重及运行清单
  requirements.txt                    当前环境中的依赖版本
  FILE_MANIFEST.json                  文件清单、大小和 SHA-256
```

## 结果对应关系

| 研究环节 | 主要结果 |
|---|---|
| 预测训练 | `results/*_best.pt`、`*_history.csv`、`training_summary.json` |
| 2020 年预测与调度 | `wind_forecast_2020_all_models.csv`、`pypsa_dispatch_2020_all_models.csv`、`pypsa_two_stage_2020/` |
| 时空分析与输电 | `01_time_metrics.csv`、`02_spatial_metrics.csv`、`03_transfer_flows.csv` |
| SHAP | `04_shap_*.csv` 与分析清单 |
| 分量干预 | `05_counterfactual_*.csv`、`05_sampled_days.csv` |
| 2022 年动态校准 | `05_dynamic_linear_2022/`，保留合并的年度/每日结果、预测指标、增益、归一化参数和 2021 年模型 |
| 年度收益不确定性 | `05_dynamic_*reductions.csv`、`05_dynamic_reduction_uncertainty.csv` |
| 储能与容量敏感性 | `06_sen/`，保留标准命名的小时输出、年度指标、代表日、响应曲线和情景清单 |

结果 CSV 中的 forecast、observed 或 raw 字段是结果定义的一部分；本包未单独复制原始观测时序文件。`04_shap_values_cost.csv` 等兼容命名被保留，因为补充分析脚本仍读取它们。

## 完整计算所需的外部数据（本包不含）

请自行取得并按原目录放置以下输入。没有这些数据时，可以查看随附结果，不能从头完成训练或调度。项目不自动下载或生成这些输入。

```text
dataset/provincial_hourly_2018_2023/
  provincial_hourly_wind_2018_2023.csv
  provincial_hourly_solar_2018_2023.csv
  provincial_hourly_load_2018_2023.csv
dataset/weather_forecast_24h/
  weatherbench2_hres_china_2018_2022.npz
dataset/pypsa_china_thu_existing/
  onwind_capacity.csv, solar_capacity.csv, hydroelectricity_capacity.csv
  nuclear_capacity.csv, coal_capacity.csv, chp_coal_capacity.csv
  ocgt_capacity.csv, chp_gas_capacity.csv
dataset/carbon_emission_factors/carrier_operational_co2_factors.csv
dataset/boundaries/python_chinamap/cnmap.nc
dataset/boundaries/python_chinamap/cnhimap.nc
PyPSA-china/resources/data/grids/edges_current.csv
```

气象文件需要 `origin`、`provinces`、`features`、`forecast` 四组字段；功率文件以 `timestamp_beijing` 和省名列对齐。各输入格式以数据加载与调度脚本为准。电网线路、装机、排放因子和地图均视为外部输入，未混入精简包。

## 完整研究的运行顺序

先配置 Python 和 `requirements.txt` 中的依赖，再补齐上述外部输入。以下命令在本文件夹中运行：

```bash
python main_train.py --epochs 50
python forecasting_2020.py
python PyPSA-china/main_dispatch.py
python 00_intro.py
python 01_time.py
python 02_spatial.py
python 03_transfer.py
python 04_shap.py
python 05_dynamic_linear_2022.py --stage all
python 05_val.py
python 06_sen.py --days-per-season 6
```

最后运行 `supp/` 下编号 01--06 的各补充脚本。部分脚本仍会读取外部数据，即使其所需的数值结果已经随附；请勿将本包理解为无数据即可端到端复现的程序。

无需原始数据、仅从现有数值结果重绘的示例包括：

```bash
python supp/02_supp.py
python supp/04_shap_supp.py
python supp/05_validation_supp.py
python supp/06_typical_day_dispatch.py
```

## 精简范围

未复制原始 `dataset/`、电网原始线路、地图文件、完整上游 PyPSA-China 仓库、求解缓存、重复的逐模型导出、2022 年大体积系数追踪和逐行审计、预测副本、已生成图像及论文排版文件。原文献数据提取脚本依赖本机备份，也未列入最小运行集；所需派生情景配置已保留。

模型权重按已有研究输出原样保留。本文件夹的复制和语法检查不代表重新训练、重新求解或全流程复现。PyPSA-China 子目录仅包含本研究使用的调度入口和原仓库许可证文本，不是完整上游发行版。
