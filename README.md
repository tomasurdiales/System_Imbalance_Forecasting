---

# Forecasting Grid System Imbalance: Case Study in Belgium

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31010/)

**Author: Tomás Urdiales**

MSc Sustainable Energy Systems Engineering (SELECT, EIT InnoEnergy). TU Eindhoven.

Work in collaboration with Elia Group.

---

$2H \frac{dw}{dt} = P_m(t) - P_e(t) \quad$ ; $\quad H = \frac{\frac{1}{2}Jw^2}{S}$

The principal research goal of this work is to describe
the short-term forecastability of quarter-hourly grid system
imbalance time-series signals, and to establish what are the
most effective methodologies and covariates to do so. This is
done by means of an empirical case study with the Belgian
electricity grid.

Additionally, initial stages of data analysis aim to ascertain
trends and patterns in system imbalance, and analyse them
in relation to the recent changes in the European energy
ecosystem.

Correspondence: tomas.urdiales@gmail.com

---

Notes:

- The "data" folder contains all data used for this study in .parquet files. It is already cleaned and datetime indexed, so it can be experimented with directly.
- This repository contains files for a Codespaces setup that may be ignored when running locally.
