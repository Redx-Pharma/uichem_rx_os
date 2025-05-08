# Module .app_methods

Module for dealing with multi objective problems. The main focus is to use Pareto front methods to identify the most promising candidates and rank.

??? example "View Source"
        #!/usr/bin.env python3

        # -*- coding: utf-8 -*-

        """

        Module for dealing with multi objective problems. The main focus is to use Pareto front methods to identify the most promising candidates and rank.

        """

        import logging

        import os

        from typing import List, Optional, Union

        import numpy as np

        import pandas as pd

        from paretoset import paretorank

        from shapely.geometry import Polygon

        from uimols import helpers

        log = logging.getLogger(__name__)

        par_eff_col_name = "pareto_efficent_set"

        par_rank_col_name = "pareto_rank"

        random_seed = 1155775



        def get_pareto_ranking(

            data_df: pd.DataFrame,

            minmax: List[str],

            ignore_duplicates: bool = True,

            objective_columns: Union[List[str], str] = "all",

            _debug: bool = False,

            _debug_filename: str = "tmp_pareto_eff.csv",

            _verbose: bool = True,

        ) -> pd.DataFrame:

            """

            Function to derive the paraeto ranking set based on N objective column vectors and min or max optimization criteria

            Args:

                data_df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more)

                minmax (List[str]): list of optimal directions for the objective columns. Must be the same number of values as the number of objective columns

                ignore_duplicates (bool, optional): How to deal with duplicate rows, if True it keeps the first and ignores all others. Defaults to True.

                objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".

                _debug (bool): Whether to save the dataframe that is used for the pareto analysis or not. Default False.

                _debug_filename (str): The debug file name to save to. Default is "tmp_pareto_eff.csv".

                _verbose (bool): Whether to log the optimal direction for each column or not

            Returns:

                pd.DataFrame: The same as the input data_df with a integer column appended called `pareto_ranks`.

            """

            # Get the objetcive function columns if they are not explicitly defined

            if objective_columns == "all":

                objective_columns = data_df.columns.values.tolist()

            # make sure the expected number of objetcive columns and minmax are there

            if helpers.check_lengths_same_two_lists(objective_columns, minmax) is False:

                raise RuntimeError(

                    f"The number of entries in the objetcive columns ({len(objective_columns)}) and minmax ({len(minmax)}) are different. These must be the same."

                )

            if objective_columns != "all":

                df = helpers.get_pd_column_subset(data_df, cols_to_keep=objective_columns)

            else:

                df = data_df.copy()

            total_n_data_points = len(df.index)

            log.info(

                f"Total Number of data points before removing any missing data {total_n_data_points}"

            )

            # df and data_df have the same index here

            mask = df.isna().any(axis=1).to_numpy().astype(int)

            optcol = np.zeros(len(df.index))

            # The index changes after this operation

            df = df.dropna(axis=0, how="any").copy()

            log.info(

                f"After removing any missing data there are {len(df.index)} data points a difference of {total_n_data_points - len(df.index)}"

            )

            if _debug is True:

                df.to_csv(_debug_filename, index=False)

            if _verbose is True:

                log.info(

                    f"{os.linesep}"

                    + f"{os.linesep}".join(

                        [f"{sense}: {nam}" for nam, sense in zip(df.columns, minmax)]

                    )

                )

            ranks = paretorank(df, sense=minmax, distinct=ignore_duplicates, use_numba=True)

            df[par_rank_col_name] = ranks

            log.info(

                f"The Pareto efficent set contain {len(np.where(ranks == 1)[0])} data points out of {len(df.index)}."

            )

            # merged_df = data_df.merge(df, how='left', suffixes=('', '_updated'))

            for ith in data_df.index:

                if mask[ith] == 1:

                    log.debug(

                        f"ith: {ith} mask[{ith}]: {mask[ith]} optcol[{ith}]: {optcol[ith]}"

                    )

                    log.debug("not updating")

                    optcol[ith] = None

                elif mask[ith] == 0:

                    log.debug(

                        f"ith: {ith} mask[{ith}]: {mask[ith]} df2.loc[{ith}, 'pareto_rank']: {df.loc[ith, 'pareto_rank']} optcol[{ith}]: {optcol[ith]}"

                    )

                    log.debug("updating")

                    optcol[ith] = int(df.loc[ith, "pareto_rank"])

            data_df[par_rank_col_name] = optcol

            return data_df



        def calculate_non_overlapping_area(points1, points2):

            """

            Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

            Args:

                points1 (list of tuples): List of (x, y) coordinates defining the first polygon.

                points2 (list of tuples): List of (x, y) coordinates defining the second polygon.

            Returns:

                float: Non-overlapping area between the two polygons.

            """

            # Create polygons from the points

            polygon1 = Polygon(points1)

            polygon2 = Polygon(points2)

            # Calculate the intersection area

            intersection_area = polygon1.intersection(polygon2).area

            log.debug(f"Intersection area: {intersection_area}")

            # Calculate the total area of both polygons

            total_area = polygon1.area + polygon2.area

            log.debug(

                f"Total area: {total_area} polygon1.area: {polygon1.area} polygon2.area: {polygon2.area}"

            )

            # Calculate the non-overlapping area

            non_overlapping_area = total_area - (2 * intersection_area)

            return non_overlapping_area



        def calculate_difference_area(points1, points2, reference=1):

            """

            Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

            Args:

                points1 (list of tuples): List of (x, y) coordinates defining the first polygon assumes this one is a reference.

                points2 (list of tuples): List of (x, y) coordinates defining the second polygon assumes this one is a test.

                reference (int): 1 if polygon1 is the reference, 2 if polygon2 is the reference.

            Returns:

                float: Non-overlapping area of polygon2 with respect to polygon1 if reference == 1. If reference == 2 then non-overlapping area of polygon1 with respect to polygon2.

            """

            # Create polygons from the points

            polygon1 = Polygon(points1)

            polygon2 = Polygon(points2)

            # Calculate the difference area between the test polygon and the reference polygon

            if reference == 1:

                intersection_area = polygon2.difference(polygon1).area

            else:

                intersection_area = polygon1.difference(polygon2).area

            return intersection_area



        def calculate_overlapping_area(points1, points2):

            """

            Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

            Args:

                points1 (list of tuples): List of (x, y) coordinates defining the first polygon.

                points2 (list of tuples): List of (x, y) coordinates defining the second polygon.

            Returns:

                float: Non-overlapping area between the two polygons.

            """

            # Create polygons from the points

            polygon1 = Polygon(points1)

            polygon2 = Polygon(points2)

            # Calculate the intersection area

            intersection_area = polygon1.intersection(polygon2).area

            return intersection_area



        def calculate_area(points1):

            """

            Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

            Args:

                points1 (list of tuples): List of (x, y) coordinates defining the first polygon.

            Returns:

                float: Non-overlapping area between the two polygons.

            """

            # Create polygons from the points

            polygon1 = Polygon(points1)

            # Calculate the area

            polygon_area = polygon1.area

            return polygon_area

## Variables

```python3
log
```

```python3
par_eff_col_name
```

```python3
par_rank_col_name
```

```python3
random_seed
```

## Functions


### calculate_area

```python3
def calculate_area(
    points1
)
```

Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| points1 | list of tuples | List of (x, y) coordinates defining the first polygon. | None |

**Returns:**

| Type | Description |
|---|---|
| float | Non-overlapping area between the two polygons. |

??? example "View Source"
        def calculate_area(points1):

            """

            Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

            Args:

                points1 (list of tuples): List of (x, y) coordinates defining the first polygon.

            Returns:

                float: Non-overlapping area between the two polygons.

            """

            # Create polygons from the points

            polygon1 = Polygon(points1)

            # Calculate the area

            polygon_area = polygon1.area

            return polygon_area


### calculate_difference_area

```python3
def calculate_difference_area(
    points1,
    points2,
    reference=1
)
```

Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| points1 | list of tuples | List of (x, y) coordinates defining the first polygon assumes this one is a reference. | None |
| points2 | list of tuples | List of (x, y) coordinates defining the second polygon assumes this one is a test. | None |
| reference | int | 1 if polygon1 is the reference, 2 if polygon2 is the reference. | None |

**Returns:**

| Type | Description |
|---|---|
| float | Non-overlapping area of polygon2 with respect to polygon1 if reference == 1. If reference == 2 then non-overlapping area of polygon1 with respect to polygon2. |

??? example "View Source"
        def calculate_difference_area(points1, points2, reference=1):

            """

            Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

            Args:

                points1 (list of tuples): List of (x, y) coordinates defining the first polygon assumes this one is a reference.

                points2 (list of tuples): List of (x, y) coordinates defining the second polygon assumes this one is a test.

                reference (int): 1 if polygon1 is the reference, 2 if polygon2 is the reference.

            Returns:

                float: Non-overlapping area of polygon2 with respect to polygon1 if reference == 1. If reference == 2 then non-overlapping area of polygon1 with respect to polygon2.

            """

            # Create polygons from the points

            polygon1 = Polygon(points1)

            polygon2 = Polygon(points2)

            # Calculate the difference area between the test polygon and the reference polygon

            if reference == 1:

                intersection_area = polygon2.difference(polygon1).area

            else:

                intersection_area = polygon1.difference(polygon2).area

            return intersection_area


### calculate_non_overlapping_area

```python3
def calculate_non_overlapping_area(
    points1,
    points2
)
```

Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| points1 | list of tuples | List of (x, y) coordinates defining the first polygon. | None |
| points2 | list of tuples | List of (x, y) coordinates defining the second polygon. | None |

**Returns:**

| Type | Description |
|---|---|
| float | Non-overlapping area between the two polygons. |

??? example "View Source"
        def calculate_non_overlapping_area(points1, points2):

            """

            Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

            Args:

                points1 (list of tuples): List of (x, y) coordinates defining the first polygon.

                points2 (list of tuples): List of (x, y) coordinates defining the second polygon.

            Returns:

                float: Non-overlapping area between the two polygons.

            """

            # Create polygons from the points

            polygon1 = Polygon(points1)

            polygon2 = Polygon(points2)

            # Calculate the intersection area

            intersection_area = polygon1.intersection(polygon2).area

            log.debug(f"Intersection area: {intersection_area}")

            # Calculate the total area of both polygons

            total_area = polygon1.area + polygon2.area

            log.debug(

                f"Total area: {total_area} polygon1.area: {polygon1.area} polygon2.area: {polygon2.area}"

            )

            # Calculate the non-overlapping area

            non_overlapping_area = total_area - (2 * intersection_area)

            return non_overlapping_area


### calculate_overlapping_area

```python3
def calculate_overlapping_area(
    points1,
    points2
)
```

Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| points1 | list of tuples | List of (x, y) coordinates defining the first polygon. | None |
| points2 | list of tuples | List of (x, y) coordinates defining the second polygon. | None |

**Returns:**

| Type | Description |
|---|---|
| float | Non-overlapping area between the two polygons. |

??? example "View Source"
        def calculate_overlapping_area(points1, points2):

            """

            Calculate the non-overlapping area between two sets of points defining arbitrary polygons in 2D.

            Args:

                points1 (list of tuples): List of (x, y) coordinates defining the first polygon.

                points2 (list of tuples): List of (x, y) coordinates defining the second polygon.

            Returns:

                float: Non-overlapping area between the two polygons.

            """

            # Create polygons from the points

            polygon1 = Polygon(points1)

            polygon2 = Polygon(points2)

            # Calculate the intersection area

            intersection_area = polygon1.intersection(polygon2).area

            return intersection_area


### get_pareto_ranking

```python3
def get_pareto_ranking(
    data_df: pandas.core.frame.DataFrame,
    minmax: List[str],
    ignore_duplicates: bool = True,
    objective_columns: Union[List[str], str] = 'all',
    _debug: bool = False,
    _debug_filename: str = 'tmp_pareto_eff.csv',
    _verbose: bool = True
) -> pandas.core.frame.DataFrame
```

Function to derive the paraeto ranking set based on N objective column vectors and min or max optimization criteria

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| data_df | pd.DataFrame | Data frame that conatins at least the objective columns (can contain more) | None |
| minmax | List[str] | list of optimal directions for the objective columns. Must be the same number of values as the number of objective columns | None |
| ignore_duplicates | bool | How to deal with duplicate rows, if True it keeps the first and ignores all others. Defaults to True. | True |
| objective_columns | Union[List[str], str] | The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all". | "all" |
| _debug | bool | Whether to save the dataframe that is used for the pareto analysis or not. Default False. | None |
| _debug_filename | str | The debug file name to save to. Default is "tmp_pareto_eff.csv". | None |
| _verbose | bool | Whether to log the optimal direction for each column or not | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | The same as the input data_df with a integer column appended called `pareto_ranks`. |

??? example "View Source"
        def get_pareto_ranking(

            data_df: pd.DataFrame,

            minmax: List[str],

            ignore_duplicates: bool = True,

            objective_columns: Union[List[str], str] = "all",

            _debug: bool = False,

            _debug_filename: str = "tmp_pareto_eff.csv",

            _verbose: bool = True,

        ) -> pd.DataFrame:

            """

            Function to derive the paraeto ranking set based on N objective column vectors and min or max optimization criteria

            Args:

                data_df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more)

                minmax (List[str]): list of optimal directions for the objective columns. Must be the same number of values as the number of objective columns

                ignore_duplicates (bool, optional): How to deal with duplicate rows, if True it keeps the first and ignores all others. Defaults to True.

                objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".

                _debug (bool): Whether to save the dataframe that is used for the pareto analysis or not. Default False.

                _debug_filename (str): The debug file name to save to. Default is "tmp_pareto_eff.csv".

                _verbose (bool): Whether to log the optimal direction for each column or not

            Returns:

                pd.DataFrame: The same as the input data_df with a integer column appended called `pareto_ranks`.

            """

            # Get the objetcive function columns if they are not explicitly defined

            if objective_columns == "all":

                objective_columns = data_df.columns.values.tolist()

            # make sure the expected number of objetcive columns and minmax are there

            if helpers.check_lengths_same_two_lists(objective_columns, minmax) is False:

                raise RuntimeError(

                    f"The number of entries in the objetcive columns ({len(objective_columns)}) and minmax ({len(minmax)}) are different. These must be the same."

                )

            if objective_columns != "all":

                df = helpers.get_pd_column_subset(data_df, cols_to_keep=objective_columns)

            else:

                df = data_df.copy()

            total_n_data_points = len(df.index)

            log.info(

                f"Total Number of data points before removing any missing data {total_n_data_points}"

            )

            # df and data_df have the same index here

            mask = df.isna().any(axis=1).to_numpy().astype(int)

            optcol = np.zeros(len(df.index))

            # The index changes after this operation

            df = df.dropna(axis=0, how="any").copy()

            log.info(

                f"After removing any missing data there are {len(df.index)} data points a difference of {total_n_data_points - len(df.index)}"

            )

            if _debug is True:

                df.to_csv(_debug_filename, index=False)

            if _verbose is True:

                log.info(

                    f"{os.linesep}"

                    + f"{os.linesep}".join(

                        [f"{sense}: {nam}" for nam, sense in zip(df.columns, minmax)]

                    )

                )

            ranks = paretorank(df, sense=minmax, distinct=ignore_duplicates, use_numba=True)

            df[par_rank_col_name] = ranks

            log.info(

                f"The Pareto efficent set contain {len(np.where(ranks == 1)[0])} data points out of {len(df.index)}."

            )

            # merged_df = data_df.merge(df, how='left', suffixes=('', '_updated'))

            for ith in data_df.index:

                if mask[ith] == 1:

                    log.debug(

                        f"ith: {ith} mask[{ith}]: {mask[ith]} optcol[{ith}]: {optcol[ith]}"

                    )

                    log.debug("not updating")

                    optcol[ith] = None

                elif mask[ith] == 0:

                    log.debug(

                        f"ith: {ith} mask[{ith}]: {mask[ith]} df2.loc[{ith}, 'pareto_rank']: {df.loc[ith, 'pareto_rank']} optcol[{ith}]: {optcol[ith]}"

                    )

                    log.debug("updating")

                    optcol[ith] = int(df.loc[ith, "pareto_rank"])

            data_df[par_rank_col_name] = optcol

            return data_df
