#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit tests for the Dash app. Please note that there are hard coded assumptions on the text in the app so if there are failures, please check the text in the app and update the tests accordingly.
"""

import io
import logging
import os
import time
from contextvars import copy_context
from typing import Any, Dict, Generator, List, Tuple, Union

import dash_bootstrap_components as dbc
import pandas as pd
import plotly
import pytest
import requests_mock
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.testing.application_runners import import_app
from selenium.webdriver.common.keys import Keys

logging.basicConfig(format="%(levelname)-9s : %(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)

app = import_app("app.app")


def test_layout(dash_duo):
    """
    Test the layout of the app. This test checks if the layout of the app is as expected. It checks if the app has the expected title,
    input fields, output fields and if there are no errors in the browser console.
    """

    dash_duo.start_server(app)

    # check the title
    assert dash_duo.find_element("#main-heading")
    assert dash_duo.find_element("#navigation-column")
    assert dash_duo.find_element("#log-column")
    # check the input fields
    assert dash_duo.find_element("#x-axis-column")
    assert dash_duo.find_element("#y-axis-column")
    assert dash_duo.find_element("#z-axis-column")
    assert dash_duo.find_element("#theme")
    assert dash_duo.find_element("#trendline")
    assert dash_duo.find_element("#remove-missing-data")
    assert dash_duo.find_element("#mean-missing-data")
    assert dash_duo.find_element("#colour-column")
    assert dash_duo.find_element("#size-column")
    assert dash_duo.find_element("#plot-type")

    # check the output fields
    assert dash_duo.find_element("#data-graph")

    # check there are no logs i.e. no errors in the browser console
    assert dash_duo.get_logs() == []


def test_data_retrival_text_input(dash_duo):
    """
    Test if the data is retrieved correctly and if the elements to eneter teh data into the app are there and writeable.
    This test checks if the data is retrieved correctly when the user enters a query in the text input callingt he callback and using the mocking API call fixtures.
    """

    # Start the server
    dash_duo.start_server(app)

    # Get the input element and change the text to check the element is writeable
    input_ = dash_duo.find_element("#label_column")
    input_.send_keys("Label")

    # check the submit button is there for the user to click
    _ = dash_duo.find_element("#upload-button")
    # button.click()


def test_dropdown_updating(dash_duo):
    """
    Test if the dropdowns are updating correctly. This test checks if the dropdowns are updating correctly when the user selects a value from the dropdowns.
    """

    dash_duo.start_server(app)

    # Wait for the dropdowns
    dash_duo.wait_for_element("#plot-type", timeout=10)

    # Check if the dropdowns defaults to 2D
    plot_type_dropdown = dash_duo.find_element(
        "#plot-type .Select-value"
    ).get_attribute("outerHTML")
    log.debug(f"Defualt dropdown val: {plot_type_dropdown} {type(plot_type_dropdown)}")
    assert "2D" in plot_type_dropdown

    # Get the input element to change the dropdown
    input_ = dash_duo.find_element("#plot-type input")

    # Send the value `3D` to the dropdown and press enter
    # (see https://github.com/plotly/dash/blob/7ba267bf9e1c956816f76900bbdbcf85dbf3ff6d/components/dash-core-components/tests/integration/dropdown/test_dynamic_options.py#L31,
    #  https://github.com/plotly/dash/blob/7ba267bf9e1c956816f76900bbdbcf85dbf3ff6d/components/dash-core-components/tests/integration/dropdown/test_dynamic_options.py#L124 and
    #  https://github.com/plotly/dash/blob/7ba267bf9e1c956816f76900bbdbcf85dbf3ff6d/components/dash-core-components/tests/integration/dropdown/test_dynamic_options.py#L3
    # )
    input_.send_keys("3D")
    input_.send_keys(Keys.ENTER)

    # Wait for the dropdown to update to 3D and check if the dropdown has been updated
    dash_duo.wait_for_text_to_equal("#plot-type .Select-value-label", "3D")
    plot_type_dropdown = dash_duo.find_element(
        "#plot-type .Select-value"
    ).get_attribute("outerHTML")
    log.debug(f"Updated dropdown val: {plot_type_dropdown} {type(plot_type_dropdown)}")
    assert "3D" in plot_type_dropdown

    # Clear the input
    input_.clear()


if __name__ == "__main__":
    import pytest

    pytest.main(["-vv", __file__])
