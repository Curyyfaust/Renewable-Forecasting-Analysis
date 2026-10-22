# Renewable-Forecasting-Analysis

Research code and processed results supporting **Spatiotemporal forecast errors shape the operational value of renewable energy prediction in China**.

The workflow links four provincial wind-power forecasting architectures (CNN, LSTM, Transformer and DNN) to a two-stage, transmission-constrained representation of China's 31-province power system. It includes spatiotemporal diagnostics, SHAP surrogate attribution, controlled error interventions, independent-year dynamic calibration, and storage sensitivities.

## Contents

All research files are under [`code/`](code/). The package includes core and supplementary analysis scripts, trained forecasting checkpoints, processed numerical results and a file manifest. Original generation and weather datasets, network inputs, map boundaries, solver caches and generated figure images are excluded.

See [`code/README.md`](code/README.md) for the full file map, input requirements and execution order. Recorded environment versions are in [`code/requirements.txt`](code/requirements.txt).

## Running the workflow

```bash
cd code
pip install -r requirements.txt
```

Provide the external inputs at the paths documented in `code/README.md` before running training or dispatch. The main sequence is:

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

Numbered scripts in `code/supp/` provide supplementary analyses. Some plots can be regenerated from the included results alone; other analyses require the excluded external inputs. This release records existing research outputs and has not been independently rerun in a clean environment.

## Authors

Chenghan Li, Wanjin Feng, Mingchen Li, Yuxuan Liang, Yong Li, Guan Wang and Yinliang Xu.

## License and citation

The original research code is distributed under the [MIT License](LICENSE). Third-party material retains its original notices and licenses in `code/PyPSA-china/LICENSES/`. See [CITATION.cff](CITATION.cff) for the software citation metadata. No DOI has been assigned to this repository yet.
