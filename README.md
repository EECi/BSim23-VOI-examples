# Too much of a good thing? - The role of Value of Information Analysis in rationalising information gathering for building energy decision making

This repository supports the conference paper submission 'The role of Value of Information Analysis in rationalising information gathering for building energy decision making', submitted to 'The 18th International IBPSA Conference and Exhibition - [Building Simulation 2023](https://bs2023.org/index)'. It provides the code used to perform the Value of Information (VoI) computations for the example problems presented in the submission.

## Technical Requirements

Use of this codebase requires Python 3.9 or later.

```
conda create --name myenv python>=3.9
pip install -r requirements.txt
```

## Codebase Structure

This repository contains:

- Three scripts for performing the EVPI calculations for each example problem:
    1. `building_ventilation.py`
    2. `ASHP_maintenance.py`
    3. `GSHP_design.py`
- `evpi.py`, a generic implementation of the EVPI calculation for a general one-stage decision problem with perfect measurements.
- The `data` directory, containing input data and cached utility evaluations for the GHSP design example.
- The `plots` directory, containing code to produce the influence diagram figures for each example problem.
- A helpful caching function wrapper in `utils.caching`