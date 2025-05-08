#!/usr/bin/env python3

"""
Module of tests for the app methods
"""

import logging

import pandas as pd
import pytest
from uimols import app_methods

logging.basicConfig(format="%(levelname)-9s : %(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)


@pytest.fixture
def pandas_dataframe() -> pd.DataFrame:
    """
    pandas_dataframe fixture holding the default input dataframe

    Returns:
        pd.DataFrame - pandas data frame
    """
    test_file = pd.DataFrame(
        [
            ["c1ccccc1", "benzene", 0.5, 1.0, 0.25, 1.2],
            ["C1CCCC1C(N)C", "1-cyclopentylethanamine", 0.9, 0.1, 1.2, 0.9],
            ["C1CCCC1C(=O)C", "1-cyclopentylethanone", 0.75, 0.05, 1.2, 0.9],
            ["C1CCCC1C(O)C", "1-cyclopentylethanol", 0.95, 0.12, 1.2, 0.9],
            ["C1CCCCC1C(N)C", "1-cyclohexylethanamine", 0.95, 0.15, 1.22, 0.95],
            ["C1CCCCC1C(=O)C", "1-cyclohexylethanone", 0.79, 0.02, 1.24, 0.97],
            ["C1CCCCC1C(O)C", "1-cyclohexylethanol", 1.1, 1.2, 1.4, 0.95],
            ["NCc1ccccc1", "benzylamine", 1.2, 0.02, 2.2, 0.75],
            ["C", "methane", -1.2, 0.01, 0.02, -10.0],
            ["CC", "ethane", -1.0, 0.2, 0.07, -10.2],
            ["CCC", "propane", -1.0, -0.4, 0.1, -10.7],
            ["CCCC", "butane", 0.75, 0.25, 0.4, 1.4],
            ["CCCCC", "pentane", -0.7, -0.9, 0.2, -11.0],
        ],
        columns=["smiles", "names", "bind_target_0", "bind_target_1", "tox", "sol"],
    )
    return test_file


@pytest.fixture
def pandas_dataframe_swapped_bind1_bind2_order() -> pd.DataFrame:
    """
    pandas_dataframe fixture holding the default input dataframe

    Returns:
        pd.DataFrame - pandas data frame
    """
    test_file = pd.DataFrame(
        [
            ["c1ccccc1", "benzene", 1.0, 0.5, 0.25, 1.2],
            ["C1CCCC1C(N)C", "1-cyclopentylethanamine", 0.1, 0.9, 1.2, 0.9],
            ["C1CCCC1C(=O)C", "1-cyclopentylethanone", 0.05, 0.75, 1.2, 0.9],
            ["C1CCCC1C(O)C", "1-cyclopentylethanol", 0.12, 0.95, 1.2, 0.9],
            ["C1CCCCC1C(N)C", "1-cyclohexylethanamine", 0.15, 0.95, 1.22, 0.95],
            ["C1CCCCC1C(=O)C", "1-cyclohexylethanone", 0.02, 0.79, 1.24, 0.97],
            ["C1CCCCC1C(O)C", "1-cyclohexylethanol", 1.2, 1.1, 1.4, 0.95],
            ["NCc1ccccc1", "benzylamine", 0.02, 1.2, 2.2, 0.75],
            ["C", "methane", 0.01, -1.2, 0.02, -10.0],
            ["CC", "ethane", 0.2, -1.0, 0.07, -10.2],
            ["CCC", "propane", -0.4, -1.0, 0.1, -10.7],
            ["CCCC", "butane", 0.25, 0.75, 0.4, 1.4],
            ["CCCCC", "pentane", -0.9, -0.7, 0.2, -11.0],
        ],
        columns=["smiles", "names", "bind_target_1", "bind_target_0", "tox", "sol"],
    )
    return test_file


def test_get_pareto_ranking(pandas_dataframe: pd.DataFrame):
    """
    Function to test finding a Pareto ranking.

    Args:
        pandas_dataframe (pd.DataFrame): pandas dataframe to find the efficent solutions of

    Raises:
        aerr: rasied if there are unexpected molecules found on the Pareto front or missing molecules in the Pareto front
    """

    efficent_set_df = app_methods.get_pareto_ranking(
        pandas_dataframe,
        minmax=["max", "min"],
        objective_columns=["bind_target_0", "bind_target_1"],
    )

    expected_ranking = [5, 2, 3, 2, 3, 2, 2, 1, 3, 4, 2, 4, 1]
    found_ranking = efficent_set_df["pareto_rank"].tolist()

    try:
        assert all(
            ent == expected_ranking[ith] for ith, ent in enumerate(found_ranking)
        )
        assert len(found_ranking) == len(expected_ranking)
    except AssertionError as aerr:
        log.critical(f"Expected: {expected_ranking} but found {found_ranking}")
        raise aerr


def test_get_pareto_ranking_with_swapped_column_order_input_is_the_same(
    pandas_dataframe_swapped_bind1_bind2_order: pd.DataFrame,
):
    """
    Function to test finding a Pareto ranking finds the same ranking regardless of column order of the input data frame

    Args:
        pandas_dataframe (pd.DataFrame): pandas dataframe to find the efficent solutions of

    Raises:
        aerr: rasied if there are unexpected molecules found on the Pareto front or missing molecules in the Pareto front
    """

    efficent_set_df = app_methods.get_pareto_ranking(
        pandas_dataframe_swapped_bind1_bind2_order,
        minmax=["max", "min"],
        objective_columns=["bind_target_0", "bind_target_1"],
    )

    expected_ranking = [5, 2, 3, 2, 3, 2, 2, 1, 3, 4, 2, 4, 1]
    found_ranking = efficent_set_df["pareto_rank"].tolist()

    try:
        assert all(
            ent == expected_ranking[ith] for ith, ent in enumerate(found_ranking)
        )
        assert len(found_ranking) == len(expected_ranking)
    except AssertionError as aerr:
        log.critical(f"Expected: {expected_ranking} but found {found_ranking}")
        raise aerr


def test_get_pareto_ranking_with_swapped_min_max_leads_to_different_ranking(
    pandas_dataframe: pd.DataFrame,
):
    """
    Function to test finding a Pareto ranking finds a different ranking set if min and max are changed

    Args:
        pandas_dataframe (pd.DataFrame): pandas dataframe to find the efficent solutions of

    Raises:
        aerr: rasied if there are unexpected molecules found on the Pareto front or missing molecules in the Pareto front
    """

    efficent_set_df = app_methods.get_pareto_ranking(
        pandas_dataframe,
        minmax=["min", "max"],
        objective_columns=["bind_target_0", "bind_target_1"],
    )

    expected_ranking = [5, 2, 3, 2, 3, 2, 2, 1, 3, 4, 2, 4, 1]
    found_ranking = efficent_set_df["pareto_rank"].tolist()

    try:
        assert any(
            ent != expected_ranking[ith] for ith, ent in enumerate(found_ranking)
        )
        assert len(found_ranking) == len(expected_ranking)
    except AssertionError as aerr:
        log.critical(f"Expected: {expected_ranking} but found {found_ranking}")
        raise aerr


def test_calculate_non_overlapping_area():
    """
    Function to test the calculation of the non-overlapping area between two polygons
    There are two variation on overlapping squares in this case. The first one is a square
    inside another square and the second one is a square overlaps in one corer with another square.

    Test 1: Square inside another square
    2 |-----------------z
      |             B   |
    1 |--------x        |
      |    A   |        |
    0 |-------------------------
      0        1        2
    area_A = 1^2 = 1
    area_B = 2^2 = 4
    area_intersection = 1^2 = 1
    area_non_overlapping = 4 + 1 - (2*1) = 3

    Test 1 a using fractional units
    area_A = 0.5^2 = 0.25
    area_B = 1^2 = 1
    area_intersection = 0.5^2 = 0.25
    area_non_overlapping = 1 + 0.25 - (2*0.25) = 0.75

    Test 2: Square overlaps in one corner with another square
    6 |    z---------z
      |    |     B   |
    4 |---------x    |
      |    |    |    |
    2 |    z----|----|
      |  A      |
    0 |---------------
      0    2    4    6

    area_A = 4^2 = 16
    area_B = 4^2 = 16
    area_intersection = 2^2 = 4
    area_non_overlapping = 16 + 16 - (2*4) = 24
    """

    points1 = [(0, 0), (0, 1), (1, 1), (1, 0)]
    points2 = [(0, 0), (0, 2), (2, 2), (2, 0)]

    non_overlapping_area = app_methods.calculate_non_overlapping_area(points1, points2)
    log.error(f"Non-overlapping area: {non_overlapping_area}")
    assert non_overlapping_area == 3.0

    # Same test as above geomterically but using fractional values
    points1 = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)]
    points2 = [(0, 0), (0, 1.0), (1.0, 1.0), (1.0, 0)]

    non_overlapping_area = app_methods.calculate_non_overlapping_area(points1, points2)
    log.error(f"Non-overlapping area: {non_overlapping_area}")
    assert non_overlapping_area == 0.75

    points1_arbitary = [(0, 0), (4, 0), (4, 4), (0, 4)]
    points2_arbitary = [(2, 2), (6, 2), (6, 6), (2, 6)]
    non_overlapping_area = app_methods.calculate_non_overlapping_area(
        points1_arbitary, points2_arbitary
    )
    log.error(f"Non-overlapping area: {non_overlapping_area}")
    assert non_overlapping_area == 24.0


def test_calculate_overlapping_area():
    """
    Function to test the calculation of the non-overlapping area between two polygons
    There are two variation on overlapping squares in this case. The first one is a square
    inside another square and the second one is a square overlaps in one corer with another square.

    Test 1: Square inside another square
    2 |-----------------z
      |             B   |
    1 |--------x        |
      |    A   |        |
    0 |-------------------------
      0        1        2
    area_A = 1^2 = 1
    area_B = 2^2 = 4
    area_intersection = 1^2 = 1

    Test 1 a using fractional units
    area_A = 0.5^2 = 0.25
    area_B = 1^2 = 1
    area_intersection = 0.5^2 = 0.25

    Test 2: Square overlaps in one corner with another square
    6 |    z---------z
      |    |     B   |
    4 |---------x    |
      |    |    |    |
    2 |    z----|----|
      |  A      |
    0 |---------------
      0    2    4    6

    area_A = 4^2 = 16
    area_B = 4^2 = 16
    area_intersection = 2^2 = 4
    """

    points1 = [(0, 0), (0, 1), (1, 1), (1, 0)]
    points2 = [(0, 0), (0, 2), (2, 2), (2, 0)]

    overlapping_area = app_methods.calculate_overlapping_area(points1, points2)
    log.error(f"Overlapping area: {overlapping_area}")
    assert overlapping_area == 1.0

    # Same test as above geomterically but using fractional values
    points1 = [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)]
    points2 = [(0, 0), (0, 1.0), (1.0, 1.0), (1.0, 0)]

    overlapping_area = app_methods.calculate_overlapping_area(points1, points2)
    log.error(f"Overlapping area: {overlapping_area}")
    assert overlapping_area == 0.25

    points1_arbitary = [(0, 0), (4, 0), (4, 4), (0, 4)]
    points2_arbitary = [(2, 2), (6, 2), (6, 6), (2, 6)]
    overlapping_area = app_methods.calculate_overlapping_area(
        points1_arbitary, points2_arbitary
    )
    log.error(f"Overlapping area: {overlapping_area}")
    assert overlapping_area == 4.0


def test_calculate_difference_area():
    """
    Function to test the calculation of the difference area between two polygons. Note this is not a symmetric function.
    """

    # square = sg.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    shifted_square = [(1, 1), (4, 1), (4, 4), (1, 4)]
    rectangle = [(0, 0), (2, 0), (2, 3), (0, 3)]
    # triangle = sg.polygon.Polygon([(0, 0), (0.5, 0.5), (0.75, 0.1)])

    assert (
        app_methods.calculate_difference_area(shifted_square, rectangle, reference=1)
        == 4.0
    )


def test_calculate_area():
    """
    Function to test the calculation of the non-overlapping area between two polygons
    There are two variation on overlapping squares in this case. The first one is a square
    inside another square and the second one is a square overlaps in one corer with another square.

    Test 1: Square square for qsaures A and B
    2 |-----------------z
      |             B   |
    1 |--------x        |
      |    A   |        |
    0 |-------------------------
      0        1        2
    area_A = 1^2 = 1
    area_B = 2^2 = 4

    """

    points1 = [(0, 0), (0, 1), (1, 1), (1, 0)]
    points2 = [(0, 0), (0, 2), (2, 2), (2, 0)]

    area_A = app_methods.calculate_area(points1)
    area_B = app_methods.calculate_area(points2)

    log.error(f"Area A: {area_A}")
    log.error(f"Area B: {area_B}")

    assert area_A == 1.0
    assert area_B == 4.0
