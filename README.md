# Architectures 2023
This repository contains the code and dataset used to generate the figures and statistic for the paper *[Updated Catalog of Kepler Planet Candidates: Focus on Accuracy and Orbital Periods](https://ui.adsabs.harvard.edu/abs/2023arXiv231100238L/abstract)*.

## Installing
This repository uses [the python dependency manager, uv](https://docs.astral.sh/uv/). Once installed, simply clone this repository and run:

```uv sync```

## Running
Run

```uv run main.py```

to execute the code to generate the figures. Analysis-related code is in the `architectures_2023` directory with the file name representing the type of analysis it is related to e.g. `period.py` for period-related analysis.
