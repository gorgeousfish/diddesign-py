"""Built-in example datasets for diddesign."""

import pandas as pd
from importlib.resources import files


def load_malesky2014() -> pd.DataFrame:
    """Load Malesky et al. (2014) Vietnam communes repeated cross-section data.

    This dataset contains observations from Vietnamese communes across three time
    periods (2006, 2008, 2010), used to study the effects of recentralization on
    public services. Suitable for RCS (repeated cross-section) design.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including: id_district, year, treatment,
        post_treat, pro4, tapwater, agrext, lnarea, lnpopden, city, reg8.

    Examples
    --------
    >>> from diddesign.data import load_malesky2014
    >>> data = load_malesky2014()
    >>> from diddesign import did
    >>> result = did(data, outcome="pro4", treatment="treatment",
    ...             time="year", post="post_treat", data_type="rcs",
    ...             id_cluster="id_district", n_boot=20)
    """
    path = files("diddesign.data").joinpath("malesky2014.csv")
    return pd.read_csv(path)


def load_paglayan2019() -> pd.DataFrame:
    """Load Paglayan (2019) US states teacher collective bargaining panel data.

    This dataset contains state-level panel data on teacher salaries and
    expenditures from 1959-2000, used to study the effects of collective
    bargaining laws. Suitable for staggered adoption (SA) design.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including: state, year, treatment,
        pupil_expenditure, teacher_salary.

    Examples
    --------
    >>> from diddesign.data import load_paglayan2019
    >>> data = load_paglayan2019()
    >>> from diddesign import did
    >>> result = did(data, outcome="pupil_expenditure", treatment="treatment",
    ...             time="year", unit_id="state", design="sa",
    ...             thres=1, n_boot=20)
    """
    path = files("diddesign.data").joinpath("paglayan2019.csv")
    return pd.read_csv(path)


def data(name: str) -> pd.DataFrame:
    """Load a built-in example dataset by name.

    Parameters
    ----------
    name : str
        Dataset name. Available: "malesky2014", "paglayan2019".

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If name is not a recognized dataset.

    Examples
    --------
    >>> from diddesign.data import data
    >>> df = data("malesky2014")
    """
    datasets = {
        "malesky2014": load_malesky2014,
        "paglayan2019": load_paglayan2019,
    }
    if name not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {available}"
        )
    return datasets[name]()


__all__ = ["data", "load_malesky2014", "load_paglayan2019"]
