---

# Forecasting Grid System Imbalance: Case Study in Belgium

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31010/)
[![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![types - mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy)

### Main branch for thesis R&D

**Author: Tomás Urdiales**

MSc Sustainable Energy Systems Engineering (SELECT, EIT InnoEnergy). TU Eindhoven.

Work in collaboration with Elia Group.

___

$2H \frac{dw}{dt} = P_m(t) - P_e(t) \quad$ ; $\quad H = \frac{\frac{1}{2}Jw^2}{S}$

**Abstract:**
European transmission system operators (TSOs) face the need to adapt to a rapidly evolving energy ecosystem as renewable energy and electrification gain traction, prompting the adoption of data-driven decision-making and advanced forecasting models. An essential aspect of this effort is understanding the nature of system imbalance, which directly reflects the net mismatch between electricity supply and demand, offering crucial insights into the network’s condition. This study employs empirical analysis of the Belgian grid to ascertain and interpret system imbalance characteristics for optimal short-term predictions. Feature engineering, rigorous cross-validation, and custom linear and non-linear machine learning modeling techniques are combined to establish a comprehensive methodology. Results reveal rapid growth in Belgian system imbalance volumes over the past three years, with increasing extreme imbalance events. Relevant covariates are identified, including day-ahead and intra- day cross-border nominations, key autoregressive features, and variables related to wind power, load, net regulation volumes, and ambient temperature. Collectively, the methodology developed reliably achieves a reduction in prediction error of 10-11% with respect to the Belgian TSO’s current forecasting model, bringing the cross-validated prediction error down to just over 100MW on average. Emphasising interpretable linear models and non-sensitive, readily available data, this study provides a solid foundation for future expansion as the European grid continues to facilitate the energy transition in the coming years.

---

Notes:

- The "data" folder contains all data used for this study in .parquet files. It is already cleaned and datetime indexed, so it can be experimented with directly.
- This repository contains files for a Codespaces setup that may be ignored when running locally.
- Original large data files are only available on local
- Comitting from terminal shows proper logs for the pre-commit hooks
