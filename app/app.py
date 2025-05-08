#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
App as a user interface
"""

import base64
import logging
import os
import re
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Hashable, List, Tuple, Union

import dash
import dash_auth
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import uimols
from dash import Input, Output, State, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
from pandas.api.types import is_numeric_dtype
from rdkit import Chem
from rdkit.Chem import Draw
from uimols import app_methods, helpers

library_version = uimols.__version__


class DashLoggingHandler(logging.Handler):
    def __init__(self, log_store):
        super().__init__()
        self.log_store = log_store

    def emit(self, record):
        log_entry = self.format(record)
        self.log_store.append(log_entry)


log_store = []
log_handler = DashLoggingHandler(log_store)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
logging.getLogger().addHandler(log_handler)
logging.getLogger().setLevel(logging.INFO)

pareto_col = "pareto_rank"
PAGE_SIZE = 10

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

if os.environ.get("DASH_ENV") == "production":
    import user_pass

    VALID_USERNAME_PASSWORD_PAIRS = user_pass.user_pass

    auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)

where_to_find_information = """
## FAQ
1. __There is an error what do I do?__ If there is an error with the app, please check the log messages at the top right of the page for more information. You can also check the console for any errors.
2. __How do I get the data I want?__ Upload a data file as a CSV file using UTF-8 encoding. The file should contain the data you want to plot. The app will automatically detect the columns in the file and load the data into the app. If there a SMILES or inchi column present in the data, the app will automatically generate images of the molecules and display them in the plot.
3. __How do I plot the data?__ Once the data has been loaded into the app, you can select the columns you want to plot on the X and Y axes for 2D plots or X, Y and Z axes for 3D plots. The plot will be updated automatically as you choose columns and options. If the SMILES or inchi column is present in the data, the plot will display the molecule images when you hover over scatter points. You can also plot a radar plot over multiple properties to compare molecules.
4. __How do I download the data?__ You can download the data as a CSV file by clicking the 'Download Data CSV' button. This will download the data that has been loaded into the app as a CSV file.
5. __How do I upload my own data?__ You can upload your own data by clicking the 'Upload' button and selecting a CSV file.
6. __How do I run a Pareto analysis?__ You can run a Pareto analysis by selecting the objective columns from the dropdown menu and entering the optimal directions for the objective columns as comma separated lists of 'min' and 'max' in teh smae order as the objective columns. This will rank the molecules into groups with 1 being the best group providing the optimal trade off between the objective for the given set of molecules.
7. __How do I download a plot?__ You can download the graph as an interactive HTML file by clicking the 'Download graph as interactive HTML' button. This will download the graph as an interactive HTML file which can be opened in a web browser. Alternatively the plot top right hand menu allows you download the plot as a static png image file.
"""

plotting_instructions = """
The following provides guidance on how to plot the data once it has been loaded into the app.

### Scatter Plot
* Select the column you want to plot on the X and Y axes for 2D plots or X, Y and Z axes for 3D plots from the dropdown menus. These menus are populated with the column names from the data that has been uploaded.
* The plot will be updated automatically as you choose columns and options
* If the SMILES or InChI column is present in the data, the plot will display the molecule images when you hover over scatter points automatically. If the SMILES column is not present it will not.
* You can select the colour and size of the points on the scatter plot by selecting the column names from the dropdown menus. These will be used to colour and size the points on the plot respectively allowing a pseudo 5D plot.
* You can select the plot type from the dropdown menu. This will allow you to choose between a 2D and 3D plot. Note that choosing a Z axis column will not automatically change the plot to 3D you must switch 2D and 3D using the dropdown.
* You can select the theme of the plot from the radio button menu. This will allow you to choose between a number of different themes and styles.
* You can add an OLS trendline to the 2D plot by selecting 'Yes' from the radio button menu. This will add a trendline to the plot.
* You can remove missing data from the plot by selecting 'Yes' from the radio button menu. This will remove any missing data from the plot.
* You can fill missing data with the mean values over each column by selecting 'Yes' from the radio button menu. This will fill any missing data with the mean value of the column.
* You can include the hover data pointers without images in the HTML file download by selecting 'Yes' from the radio button menu. This may increase the file size of the HTML file. Note this will not include molecule image only the underying data values.
* you can also download the graph as an interactive HTML file by clicking the 'Download graph as interactive HTML' button. This will download the graph as an interactive HTML file which can be opened in a web browser. Alternatively the plot top right hand menu allows you download the plot as a static png image file.

### Radar Plot
* Select the objective columns from the dropdown menu; note that in principle as many as you want but it will become confusing with too many. These are the columns that you want to compare in the radar plot. Also note that the columns must have a value for each row being compared otherwis an error will reported
* Optionally enter the optimal directions for the objective columns in the text area. This should be a comma separated list of 'min' or 'max' to define the optimal direction for the objective columns note the order should match the order in the objective columns choice. If you enter values here, a Pareto analysis will be run and a column added to the data called 'pareto_rank'. This is a group rank with 1 being the best group. It suggests that molecules in the group rank 1 provide the optimal trade off between all of your objectives. If these are selected an ideal trace with dashed lines will also be plotted.
* Select the ID column from the dropdown menu. This is the column that is unique to each row in the data. This is used to identify the rows in the radar plot. Once chosen the Row ID and reference row ID dropdowns will be populated with the unique values from the column.
    * Note that the reference row ID can be set to:
    * None to show no reference
    * mean or median for column averages of each objective
    * pareto_mean_best or pareto_mean_worst for averages over the pareto rank best and worst sets
    * a specific row ID
* Select at least one row ID from the dropdown menu. This is the unique value from the ID column that you want to plot.
* Select the reference row ID from the dropdown menu. This is the unique value from the ID column that you want to use as a reference for the radar plot.
* Select the whether to normalize the data or not and use the raw data from the radio button menu. This will scale the data to be between 0 and 1 using min max scaling.
* Select the theme of the plot from the radio button menu. This will allow you to choose between a number of different themes and styles.

"""

capabilities_and_limits = """
This section details some of the capabilities and limits of the app.

## Capabilities
* The app front end allows the data to be downloaded as a CSV file and plotted in 2D or 3D or as a radar plot
* The app allows the user to upload their own data. You can attempt to merge another data file with the previously uploaded file. If you do this it is strongly advised that merged data is carefully checked by downloading the data as a csv file.
* The app allows the user to perform a Pareto analysis on the data and plot the results. This analysis is based on the objective columns provided by the user and will rank molecules into groups with 1 being the best group providing the optimal trade off between the objective for the given set of molecules.
* Note that the radar plot provides a table of metrics. These metrics provide a percentage overlap and difference between a test molecule and the reference or ideal case. The overlap the ratio of the area of the test molecule that is covered by the reference or ideal case over the reference or ideal area expressed as a percentage. The difference is the ratio ofthe area of the test molecule that is not covered by the reference or ideal case over the reference or ideal area expressed as a percentage. This means, the overlap and difference are calculated as a percentage of the area of the reference or ideal case.

## Limits
* Large amounts of data will be slow to run
* The app is limited to the capabilities scatter and radar plots
* Data merging is not perfect and may require some manual intervention to ensure the data is merged correctly
8 Metrics are given as a guide be require careful interpretation and consideration as these may not be right for all applications including your own
"""

instructions_markdown = f"""
#### Instructions
App version {library_version}


1. Please upload your data as a CSV file using UTF-8 encoding. The file should contain the data you want to plot. If there a SMILES or InChI column present in the data, the app will automatically generate images of the molecules and display them in the plot.
2. Please enter a column name from your data that is unique for each row as the label column. Ensure you are happy and able to upload the data to app host.
3. Optionally, you can enter a column name to merge on if uploading additional data. It is strongly recommended to check the merged data by downloading the data as a CSV file.
4. Press upload when ready.
"""

make_new_col_markdown = """
You can make a new column from the existing data using certain mathmatical operations defined in the next few steps. Pick two columns from the dropdown menus and select the operation you want to perform. The result will be added to the end of the dataframe as a new column (scroll all the way to the right in the preview).
"""

plotting_markdown = """
The next section allows for visuzaliation of the data that has been loaded into the app. The data can be plotted in 2D or 3D scatter plots or as a radar plot. The scatter plot is useful or identifying trends in the data. The radar plot is useful for comparing multiple objectives for a set of molecules.
"""

plotting_scatter_markdown = """
1. After the data has been loaded, you can select the columns you want to plot on the X and Y axes for 2D plots or X, Y and Z axes for 3D plots.
2. The plot will be updated automatically.
3. If the SMILES column is present in the data, the plot will display the molecule images.
"""

plotting_radar_markdown = """
The radar plot will show the objectives as a series of axes with the values for each molecule plotted as a trace on the axes. The radar plot can be used to compare the objectives for a set of molecules.
If a reference and or optimal directions are provided then a table will be generated quantifying the degree of overlap and the difference in area between the reference and or ideal and the selected molecule. These are presented as percentages of the reference or ideal areas.
"""


# Load the logo image
p = Path(os.path.dirname(os.path.abspath(__file__)))
logging.info(f"Path to app.py: {p}")

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Br(),
                        html.Br(),
                        html.A(
                            html.H1(
                                "Data and Analysis Portal",
                                style={"textAlign": "center"},
                            ),
                            id="main-heading",
                        ),
                        html.Br(),
                        html.Br(),
                    ],
                    width="auto",
                ),
            ],
            align="center",
            justify="center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Br(),
                        html.Br(),
                        html.H3(
                            "Navigation",
                            style={"text-align": "center"},
                            id="navigation-column",
                        ),
                        html.Div(
                            [
                                html.A(
                                    html.Button(
                                        "Upload",
                                        id="query-heading-btn",
                                        n_clicks=0,
                                        style={
                                            "backgroundColor": "blue",
                                            "color": "white",
                                            "width": "100%",
                                            "border": "1.5px black solid",
                                            "height": "50px",
                                            "text-align": "center",
                                            "marginLeft": "0px",
                                            "marginTop": 10,
                                        },
                                    ),
                                    href="#upload-heading",
                                ),
                                html.Br(),
                                html.A(
                                    html.Button(
                                        "Plotting",
                                        id="plotying-heading-btn",
                                        n_clicks=0,
                                        style={
                                            "backgroundColor": "blue",
                                            "color": "white",
                                            "width": "100%",
                                            "border": "1.5px black solid",
                                            "height": "50px",
                                            "text-align": "center",
                                            "marginLeft": "0px",
                                            "marginTop": 10,
                                        },
                                    ),
                                    href="#plotting-heading",
                                ),
                                html.Br(),
                                html.A(
                                    html.Button(
                                        "Scatter Plot",
                                        id="scatter-plot-heading-btn",
                                        n_clicks=0,
                                        style={
                                            "backgroundColor": "blue",
                                            "color": "white",
                                            "width": "100%",
                                            "border": "1.5px black solid",
                                            "height": "50px",
                                            "text-align": "center",
                                            "marginLeft": "0px",
                                            "marginTop": 10,
                                        },
                                    ),
                                    href="#scatter-plot-heading",
                                ),
                                html.Br(),
                                html.A(
                                    html.Button(
                                        "Radar Plot",
                                        id="radar-plot-heading-btn",
                                        n_clicks=0,
                                        style={
                                            "backgroundColor": "blue",
                                            "color": "white",
                                            "width": "100%",
                                            "border": "1.5px black solid",
                                            "height": "50px",
                                            "text-align": "center",
                                            "marginLeft": "0px",
                                            "marginTop": 10,
                                        },
                                    ),
                                    href="#radar-plot-heading",
                                ),
                                html.Br(),
                                html.A(
                                    html.Button(
                                        "Detailed Instructions",
                                        id="instructions-heading-btn",
                                        n_clicks=0,
                                        style={
                                            "backgroundColor": "blue",
                                            "color": "white",
                                            "width": "100%",
                                            "border": "1.5px black solid",
                                            "height": "50px",
                                            "text-align": "center",
                                            "marginLeft": "0px",
                                            "marginTop": 10,
                                        },
                                    ),
                                    href="#instructions-heading",
                                ),
                                html.Br(),
                            ]
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        dcc.Store(id="stored-dataframe"),
                        dcc.Store(id="hover-data"),
                        html.A(html.H2("Data CSV Upload"), id="upload-heading"),
                        dcc.Markdown(children=instructions_markdown),
                        html.Br(),
                        html.H6(
                            "Pick csv file to upload (utf-8 encoded)",
                            style={"textAlign": "center"},
                        ),
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(
                                [
                                    "Drag and drop or ",
                                    html.A(
                                        "click here to select a CSV File",
                                        style={"color": "blue"},
                                    ),
                                ]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px",
                            },
                            multiple=False,
                        ),
                        html.H6(
                            "Label Column (required - this should contain unique values per row)",
                            style={"textAlign": "center"},
                        ),
                        dbc.Input(
                            id="label_column",
                            type="text",
                            placeholder="Enter a column name to act as the label column",
                            style={"margin": "10px", "width": "100%", "height": "75px"},
                        ),
                        html.H6(
                            "Merge Column (optional)", style={"textAlign": "center"}
                        ),
                        dbc.Textarea(
                            id="merge-column",
                            placeholder="Enter column header to merge on if uploading additional data (optional), it is strongly recommended to check the merged data by downloading the data as a CSV file",
                            style={"margin": "10px", "width": "100%", "height": "75px"},
                        ),
                        dbc.Button(
                            "Upload",
                            id="upload-button",
                            color="info",
                            style={"margin": "10px", "width": "100%"},
                            className="me-1",
                        ),
                        dcc.Loading(
                            html.Div(id="loading-user-data-output"),
                            id="loading-user-data-component",
                        ),
                        html.Div(id="output-data-upload"),
                        dbc.Button(
                            "Download Data CSV",
                            id="btn_download",
                            color="success",
                            className="me-1",
                            style={"margin": "10px", "width": "100%"},
                        ),
                        dcc.Download(id="download-dataframe-csv"),
                        html.Br(),
                        dbc.Alert(id="output", color="info", is_open=False),
                        html.Br(),
                    ],
                    width=5,
                ),
                dbc.Col(
                    [
                        html.Br(),
                        html.H3(
                            "Log Messages",
                            style={"text-align": "center"},
                            id="log-column",
                        ),
                        dcc.Textarea(
                            id="log-output",
                            style={"width": "100%", "height": "900px"},
                            readOnly=True,
                        ),
                    ]
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            "Pareto Front Analysis",
                            style={"text-align": "center"},
                            id="mp_opt_heading",
                        ),
                        dcc.Dropdown(
                            id="mp-objective-columns-dropdown",
                            placeholder="Select objective columns (more than one)",
                            multi=True,
                        ),
                        dbc.Label("Multi-parameter optimization directions"),
                        dbc.Textarea(
                            id="mp-post-query-directions",
                            placeholder="Enter a comma separared list of min or max to define the optimal direction for the objective columns in the previous box (note the order is the same) [default: None]",
                            style={"width": "100%", "height": "110px"},
                        ),
                        dbc.Button(
                            "Run Pareto Analysis",
                            id="run-pareto",
                            color="primary",
                            style={"margin-top": "10px", "width": "100%"},
                            className="me-1",
                        ),
                        dcc.Loading(
                            html.Div(id="running-pareto"), id="running-pareto-component"
                        ),
                        dbc.Alert(id="output-pareto", color="info", is_open=False),
                        html.Br(),
                        html.Br(),
                        html.H3(
                            "Data Upload",
                            style={"text-align": "center"},
                            id="upload_data",
                        ),
                    ],
                    width={"size": 10, "offset": 2},
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.A(
                            html.H4("Data Table In Preview"),
                            id="data-table-preview-heading",
                        ),
                        dash_table.DataTable(
                            id="data-table-preview",
                            columns=[
                                {"name": "Column 1", "id": "Column 1"},
                                {"name": "Column 2", "id": "Column 2"},
                            ],
                            data=[
                                {
                                    "Column 1": "No Data loaded yet",
                                    "Column 2": "No Data loaded yet",
                                }
                            ],
                            style_table={"width": "100%", "overflowX": "auto"},
                            fixed_columns={"headers": True},
                            style_cell={
                                "textAlign": "center",
                                "padding": "5px",
                                "font-family": "sans-serif",
                                "fontSize": "14px",
                                "height": "auto",
                                "minwWidth": "200px",
                                "width": "200px",
                                "maxWidth": "400px",
                                "whiteSpace": "normal",
                            },
                            style_header={"padding": "5px", "fontWeight": "bold"},
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "rgb(220, 220, 220)",
                                }
                            ],
                            page_current=0,
                            page_size=PAGE_SIZE,
                            page_action="custom",
                            sort_action="custom",
                            sort_mode="single",
                            sort_by=[],
                        ),
                        html.Br(),
                        html.H5("Make a new column from operation between two columns"),
                        dcc.Markdown(children=make_new_col_markdown),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="column-dropdown-1",
                                    placeholder="Select first column",
                                ),
                                html.Br(),
                                dcc.Dropdown(
                                    id="operation-dropdown",
                                    options=[
                                        {"label": "Add", "value": "add"},
                                        {"label": "Subtract", "value": "subtract"},
                                        {"label": "Multiply", "value": "multiply"},
                                        {"label": "Divide", "value": "divide"},
                                    ],
                                    placeholder="Select operation",
                                ),
                                html.Br(),
                                dcc.Dropdown(
                                    id="column-dropdown-2",
                                    placeholder="Select second column",
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Perform Operation",
                                    id="perform-operation-button",
                                    n_clicks=0,
                                    color="primary",
                                    style={"margin-top": "10px", "width": "100%"},
                                    className="me-1",
                                ),
                                html.Div(id="operation-result"),
                                html.Br(),
                            ]
                        ),
                    ],
                    width={"size": 10, "offset": 2},
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.A(html.H1("Plotting"), id="plotting-heading"),
                        dcc.Markdown(children=plotting_markdown),
                        #### SCATTER PLOT ####
                        html.A(html.H2("Scatter Plot"), id="scatter-plot-heading"),
                        dcc.Markdown(children=plotting_scatter_markdown),
                        dbc.Label("Select X-axis column"),
                        dcc.Dropdown(
                            id="x-axis-column",
                            placeholder="Select X-axis column",
                        ),
                        dcc.Clipboard(id="x-axis-copy", style={"fontSize": 20}),
                        dbc.Label("Custom X-axis Name"),
                        dbc.Input(
                            id="x-axis-name",
                            type="text",
                            placeholder="Enter custom X-axis name",
                        ),
                        dbc.Label("X-axis Scale"),
                        dcc.RadioItems(
                            id="x-axis-scale",
                            options=[
                                {"label": "Linear", "value": "linear"},
                                {"label": "Logarithmic", "value": "log"},
                            ],
                            value="linear",
                            inline=True,
                            inputStyle={"margin-right": "5px", "margin-left": "20px"},
                        ),
                        dbc.Label("Select Y-axis column"),
                        dcc.Dropdown(
                            id="y-axis-column", placeholder="Select Y-axis column"
                        ),
                        dcc.Clipboard(id="y-axis-copy", style={"fontSize": 20}),
                        dbc.Label("Custom Y-axis Name"),
                        dbc.Input(
                            id="y-axis-name",
                            type="text",
                            placeholder="Enter custom Y-axis name",
                        ),
                        dbc.Label("Y-axis Scale"),
                        dcc.RadioItems(
                            id="y-axis-scale",
                            options=[
                                {"label": "Linear", "value": "linear"},
                                {"label": "Logarithmic", "value": "log"},
                            ],
                            value="linear",
                            inline=True,
                            inputStyle={"margin-right": "5px", "margin-left": "20px"},
                        ),
                        dbc.Label("Select Z-axis column"),
                        dcc.Dropdown(
                            id="z-axis-column", placeholder="Select Z-axis column"
                        ),
                        dcc.Clipboard(id="z-axis-copy", style={"fontSize": 20}),
                        dbc.Label("Custom Z-axis Name"),
                        dbc.Input(
                            id="z-axis-name",
                            type="text",
                            placeholder="Enter custom Z-axis name",
                        ),
                        dbc.Label("Z-axis Scale"),
                        dcc.RadioItems(
                            id="z-axis-scale",
                            options=[
                                {"label": "Linear", "value": "linear"},
                                {"label": "Logarithmic", "value": "log"},
                            ],
                            value="linear",
                            inline=True,
                            inputStyle={"margin-right": "5px", "margin-left": "20px"},
                        ),
                        dbc.Label("Select colour column"),
                        dcc.Dropdown(
                            id="colour-column", placeholder="Select colour column"
                        ),
                        dcc.Clipboard(id="colour-copy", style={"fontSize": 20}),
                        dbc.Label("Custom Colour Name"),
                        dbc.Input(
                            id="colour-column-name",
                            type="text",
                            placeholder="Enter custom colour column name",
                        ),
                        dbc.Label("Select size column"),
                        dcc.Dropdown(
                            id="size-column", placeholder="Select size column"
                        ),
                        dcc.Clipboard(id="size-copy", style={"fontSize": 20}),
                        dbc.Label("Select Plot Type"),
                        dcc.Dropdown(
                            id="plot-type",
                            options=[
                                {"label": "2D", "value": "2D"},
                                {"label": "3D", "value": "3D"},
                            ],
                            value="2D",
                        ),
                        dbc.Label("Select Theme"),
                        dcc.RadioItems(
                            id="theme",
                            options=[
                                {"label": "Plotly", "value": "plotly"},
                                {"label": "Plotly White", "value": "plotly_white"},
                                {"label": "Plotly Dark", "value": "plotly_dark"},
                                {"label": "GGPlot2", "value": "ggplot2"},
                                {"label": "Seaborn", "value": "seaborn"},
                                {"label": "Simple White", "value": "simple_white"},
                                {"label": "None", "value": "none"},
                            ],
                            value="plotly",
                            inline=True,
                            inputStyle={"margin-right": "5px", "margin-left": "20px"},
                        ),
                        dbc.Label("Add OLS trendline to 2D plot"),
                        dcc.RadioItems(
                            id="trendline",
                            options=[
                                {"label": "Yes", "value": "ols"},
                                {"label": "No", "value": "none"},
                            ],
                            value="none",
                            inline=True,
                            inputStyle={"margin-right": "5px", "margin-left": "20px"},
                        ),
                        dbc.Label("Remove missing data"),
                        dcc.RadioItems(
                            id="remove-missing-data",
                            options=[
                                {"label": "Yes", "value": True},
                                {"label": "No", "value": False},
                            ],
                            value=False,
                            inline=True,
                            inputStyle={"margin-right": "5px", "margin-left": "20px"},
                        ),
                        dbc.Label(
                            "Fill missing data with mean values over each column"
                        ),
                        dcc.RadioItems(
                            id="mean-missing-data",
                            options=[
                                {"label": "Yes", "value": True},
                                {"label": "No", "value": False},
                            ],
                            value=False,
                            inline=True,
                            inputStyle={"margin-right": "5px", "margin-left": "20px"},
                        ),
                        dbc.Label(
                            "Include the hover data pointers without images in the HTML file download (may increase file size)?"
                        ),
                        dcc.RadioItems(
                            id="include-data-pointer-in-html",
                            options=[
                                {"label": "Yes", "value": True},
                                {"label": "No", "value": False},
                            ],
                            value=False,
                            inline=True,
                            inputStyle={"margin-right": "5px", "margin-left": "20px"},
                        ),
                        dcc.Graph(
                            id="data-graph",
                            responsive=True,
                            clear_on_unhover=True,
                            style={"height": "800px"},
                        ),
                        dcc.Tooltip(
                            id="data-graph-tooltip",
                            style={"max-width": "none"},
                            zindex=1000,
                        ),
                        dbc.Button(
                            "Download scatter graph as interactive HTML",
                            id="download-html-btn",
                            color="success",
                            className="me-1",
                            style={"margin-top": "10px", "width": "50%"},
                        ),
                        dcc.Download(id="download-html"),
                        #### BACK TO THE TOP ####
                        html.A(
                            html.Button(
                                "Back to top",
                                id="plotting-btt-btn",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "blue",
                                    "color": "white",
                                    "width": "100%",
                                    "border": "1.5px black solid",
                                    "height": "50px",
                                    "text-align": "center",
                                    "marginLeft": "0px",
                                    "marginTop": 10,
                                },
                            ),
                            href="#main-heading",
                        ),
                        #### RADAR PLOT ####
                        dcc.Store(id="hover-data-radar"),
                        dcc.Store(id="radar-data-table-store"),
                        html.Br(),
                        html.Br(),
                        html.A(html.H2("Radar Plot"), id="radar-plot-heading"),
                        dcc.Markdown(children=plotting_radar_markdown),
                        html.Br(),
                        dcc.Dropdown(
                            id="objective-columns",
                            placeholder="Select objective columns (more than one)",
                            multi=True,
                        ),
                        html.Br(),
                        dbc.Textarea(
                            id="directions",
                            placeholder="Optionally, enter directions (comma-separated, min or max)",
                            style={"width": "100%"},
                            value=None,
                            persistence=False,
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id="id-column",
                            placeholder="Select an ID column i.e. one that is unique to each row",
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id="row-id",
                            placeholder="Enter row ID(s) for radar plot",
                            multi=True,
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id="reference-row-id",
                            placeholder="Enter reference row ID or aggregation method for the radar plot",
                            value="None",
                        ),
                        html.Br(),
                        dbc.Label("Scale the data between 0 and 1?"),
                        dcc.RadioItems(
                            id="min-max-scale-radar-data",
                            options=[
                                {"label": "Yes", "value": True},
                                {"label": "No", "value": False},
                            ],
                            value=True,
                            inline=True,
                            inputStyle={"margin-right": "5px", "margin-left": "20px"},
                        ),
                        html.Br(),
                        dbc.Label("Select Theme"),
                        dcc.RadioItems(
                            id="theme-radar",
                            options=[
                                {"label": "Plotly", "value": "plotly"},
                                {"label": "Plotly White", "value": "plotly_white"},
                                {"label": "Plotly Dark", "value": "plotly_dark"},
                                {"label": "GGPlot2", "value": "ggplot2"},
                                {"label": "Seaborn", "value": "seaborn"},
                                {"label": "Simple White", "value": "simple_white"},
                                {"label": "None", "value": "none"},
                            ],
                            value="plotly",
                            labelStyle={
                                "display": "inline-block",
                                "margin-right": "10px",
                            },
                        ),
                        html.Br(),
                        dbc.Button(
                            "Plot interactive radar graph",
                            id="run-analysis-button",
                            n_clicks=0,
                            color="primary",
                            className="me-1",
                            style={"margin-top": "10px", "width": "50%"},
                        ),
                        html.Br(),
                        dbc.Alert(
                            id="radar-plotting-output", color="info", is_open=False
                        ),
                        html.Br(),
                        dcc.Graph(
                            id="radar-plot",
                            responsive=True,
                            clear_on_unhover=True,
                            style={"height": "800px"},
                        ),
                        dcc.Tooltip(
                            id="radar-graph-tooltip",
                            style={"max-width": "none"},
                            zindex=1000,
                        ),
                        html.Br(),
                        dash_table.DataTable(
                            id="radar-data-table",
                            style_table={"width": "100%", "overflowX": "auto"},
                            style_cell={
                                "textAlign": "center",
                                "padding": "5px",
                                "font-family": "sans-serif",
                                "fontSize": "14px",
                            },
                            style_header={"padding": "5px", "fontWeight": "bold"},
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "rgb(220, 220, 220)",
                                }
                            ],
                        ),
                        html.Br(),
                        dbc.Button(
                            "Download radar graph as interactive HTML",
                            id="download-html-radar-btn",
                            color="success",
                            className="me-1",
                            style={"margin-top": "10px", "width": "50%"},
                        ),
                        dcc.Download(id="download-html-radar"),
                        html.Br(),
                        dbc.Button(
                            "Download table as CSV",
                            id="download-table-radar-btn",
                            color="success",
                            className="me-1",
                            style={"margin-top": "10px", "width": "50%"},
                        ),
                        dcc.Download(id="download-table-radar"),
                        html.Br(),
                        #### BACK TO THE TOP ####
                        html.A(
                            html.Button(
                                "Back to top",
                                id="plotting-totop-btn",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "blue",
                                    "color": "white",
                                    "width": "100%",
                                    "border": "1.5px black solid",
                                    "height": "50px",
                                    "text-align": "center",
                                    "marginLeft": "0px",
                                    "marginTop": 10,
                                },
                            ),
                            href="#main-heading",
                        ),
                    ],
                    width={"size": 10, "offset": 2},
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.A(html.H1("Instructions"), id="instructions-heading"),
                        dcc.Markdown(children=where_to_find_information),
                        html.A(
                            html.H2("Plotting Instructions"),
                            id="plotting-instructions-heading",
                        ),
                        dcc.Markdown(children=plotting_instructions),
                        html.A(
                            html.H1("Capabilities and Limits"),
                            id="capabilities-heading",
                        ),
                        dcc.Markdown(children=capabilities_and_limits),
                        html.A(
                            html.Button(
                                "Back to top",
                                id="instructions-btt-btn",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "blue",
                                    "color": "white",
                                    "width": "100%",
                                    "border": "1.5px black solid",
                                    "height": "50px",
                                    "text-align": "center",
                                    "marginLeft": "0px",
                                    "marginTop": 10,
                                },
                            ),
                            href="#main-heading",
                        ),
                    ],
                    width={"size": 10, "offset": 2},
                ),
            ]
        ),
    ]
)


##### Data Download functions #####
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_download", "n_clicks"),
    State("stored-dataframe", "data"),
    prevent_initial_call=True,
)
def download_dataframe(n_clicks: int, data: Union[List, dict]) -> Any:
    """
    Function to download the dataframe as a CSV file
    Args:
        n_clicks: int: Number of clicks on the download button
        data: dict: raw data to make dataframes from
    Returns:
        Any: Data
    """
    df = pd.DataFrame(data)
    if "image" in df.columns:
        df = df.drop(columns=["image"])
    now = datetime.now()
    string_date = now.strftime("%d-%m-%Y_%H-%M-%S")
    logging.info("Data downloaded successfully")
    return dcc.send_data_frame(df.to_csv, f"app_data_{string_date}.csv", index=False)


##### Pareto optimization functions #####
@app.callback(
    Output("stored-dataframe", "data", allow_duplicate=True),
    Output("output-pareto", "children"),
    Output("output-pareto", "is_open"),
    Output("output-pareto", "color"),
    Output("running-pareto", "children"),
    Output("x-axis-column", "options", allow_duplicate=True),
    Output("y-axis-column", "options", allow_duplicate=True),
    Output("z-axis-column", "options", allow_duplicate=True),
    Output("colour-column", "options", allow_duplicate=True),
    Output("size-column", "options", allow_duplicate=True),
    Output("id-column", "options", allow_duplicate=True),
    Output("mp-objective-columns-dropdown", "options", allow_duplicate=True),
    Output("objective-columns", "options", allow_duplicate=True),
    Output("log-output", "value", allow_duplicate=True),
    Input("run-pareto", "n_clicks"),
    State("mp-objective-columns-dropdown", "value"),
    State("mp-post-query-directions", "value"),
    State("stored-dataframe", "data"),
    prevent_initial_call=True,
)
def run_pareto_analysis(
    n_clicks, obj_cols, obj_directions, data_df
) -> Tuple[List[dict], str]:
    """
    Function to run the pareto analysis
    Args:
        n_clicks: int: Number of clicks on the run pareto button
        obj_cols: str: Objective columns
        obj_directions: str: Objective directions
        data_df: dict: Data dataframe
    Returns:
        dict: Data dataframe
        str: Running pareto message
    """
    if n_clicks is None:
        raise PreventUpdate

    if not isinstance(data_df, pd.DataFrame):
        data_df = pd.DataFrame(data_df)

    if not isinstance(obj_directions, list):
        obj_directions = [ent.strip() for ent in obj_directions.split(",")]
    else:
        obj_directions = [ent.strip() for ent in obj_directions]

    logging.info(
        f"Objective columns are: {', '.join(obj_cols)} and directions are: {', '.join(obj_directions)}"
    )

    if len(obj_cols) != len(obj_directions):
        options = [
            {"label": col, "value": col} for col in data_df.columns if col != "image"
        ]
        numeric_options = [
            {"label": col, "value": col}
            for col in data_df.select_dtypes(include=np.number).columns
            if col != "image"
        ]
        return (
            data_df.to_dict("records"),
            "Number of objective columns and directions do not match",
            True,
            "danger",
            "",
            options,
            options,
            options,
            options,
            options,
            options,
            numeric_options,
            numeric_options,
            "\n".join(log_store),
        )

    if obj_cols is None or obj_directions is None:
        options = [
            {"label": col, "value": col} for col in data_df.columns if col != "image"
        ]
        numeric_options = [
            {"label": col, "value": col}
            for col in data_df.select_dtypes(include=np.number).columns
            if col != "image"
        ]
        return (
            data_df.to_dict("records"),
            "No objective columns or directions provided",
            True,
            "warning",
            "",
            options,
            options,
            options,
            options,
            options,
            options,
            numeric_options,
            numeric_options,
            "\n".join(log_store),
        )

    data_df = app_methods.get_pareto_ranking(
        data_df=pd.DataFrame(data_df),
        objective_columns=obj_cols,
        minmax=obj_directions,
    )

    options = [
        {"label": col, "value": col} for col in data_df.columns if col != "image"
    ]
    numeric_options = [
        {"label": col, "value": col}
        for col in data_df.select_dtypes(include=np.number).columns
        if col != "image"
    ]

    return (
        data_df.to_dict("records"),
        "Pareto analysis run successfully",
        True,
        "success",
        "",
        options,
        options,
        options,
        options,
        options,
        options,
        numeric_options,
        numeric_options,
        "\n".join(log_store),
    )


##### Data upload functions #####
@app.callback(
    Output("output-data-upload", "children"),
    Output("stored-dataframe", "data", allow_duplicate=True),
    Output("x-axis-column", "options", allow_duplicate=True),
    Output("y-axis-column", "options", allow_duplicate=True),
    Output("z-axis-column", "options", allow_duplicate=True),
    Output("colour-column", "options", allow_duplicate=True),
    Output("size-column", "options", allow_duplicate=True),
    Output("id-column", "options", allow_duplicate=True),
    Output("objective-columns", "options", allow_duplicate=True),
    Output("mp-objective-columns-dropdown", "options", allow_duplicate=True),
    Output("log-output", "value", allow_duplicate=True),
    Output("loading-user-data-output", "children"),
    Input("upload-button", "n_clicks"),
    State("label_column", "value"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("merge-column", "value"),
    State("stored-dataframe", "data"),
    prevent_initial_call=True,
)
def upload_data(
    n_clicks: int, label_column: str, contents, filename, merge_column, stored_dataframe
) -> tuple[str, list[dict]]:
    """
    Function to upload data
    Args:
        n_clicks: int: Number of clicks on the submit button
        contents: str: Contents
        filename: str: Filename
        merge_column: str: Column to merge on
        stored_dataframe: dict: Stored dataframe
    Returns:
        str: Output message
        list: list of dicts dataframe representation
        list: X-axis column options
        list: Y-axis column options
        list: Z-axis column options
        list: Colour column options
        list: Size column options
        list: Multi-parameter optimization objective columns dropdown options
        str: Log output
        str: Loading output
    """
    if contents is None:
        return "No file uploaded presently"

    logging.info(
        f"Uploading the users file {filename} and attempting merging on column {merge_column} if given"
    )
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    uploaded_df = pd.read_csv(StringIO(decoded.decode("utf-8")))
    logging.info(f"Uploaded data shape: {uploaded_df.shape}")

    if stored_dataframe is not None:
        stored_dataframe = pd.DataFrame(stored_dataframe)
    else:
        stored_dataframe = pd.DataFrame()

    logging.info(f"Merge column: {merge_column} of type {type(merge_column)}")
    if merge_column is not None:
        logging.info("Merging dataframes")
        if (
            merge_column in stored_dataframe.columns
            and merge_column in uploaded_df.columns
        ):
            stored_dataframe = pd.merge(
                stored_dataframe,
                uploaded_df,
                on=merge_column,
                how="left",
                suffixes=("_from_database", "_uploaded"),
            )

            smiles_col = find_column_name_from_re(
                stored_dataframe.columns.tolist(), rex="SMILES"
            )
            inchi_col = find_column_name_from_re(
                stored_dataframe.columns.tolist(), rex="inchi"
            )
            if smiles_col is not None and "image" not in stored_dataframe.columns:
                logging.info("Generating molecule images")
                start = datetime.now()
                stored_dataframe["image"] = stored_dataframe[smiles_col].apply(
                    smiles_to_image, axis=1
                )
                end = datetime.now()
                tdelta = end - start
                logging.info(
                    f"Finished generating molecule images (took {tdelta.seconds}s)"
                )
                logging.info("Molecule images generated successfully")
            elif inchi_col is not None and "image" not in stored_dataframe.columns:
                logging.info("Generating molecule images")
                start = datetime.now()
                stored_dataframe["image"] = stored_dataframe[inchi_col].apply(
                    inchi_to_image
                )
                end = datetime.now()
                tdelta = end - start
                logging.info(
                    f"Finished generating molecule images (took {tdelta.seconds}s)"
                )
                logging.info("Molecule images generated successfully")
            else:
                if "image" in stored_dataframe.columns:
                    logging.info("Molecule images already present in the data")
                else:
                    logging.info(
                        "SMILES column not found in the data therefore no images will be displayed in plots."
                    )

            logging.info(
                f"Uploaded data merged successfully. Stored dataframe now has {len(stored_dataframe)} rows and {len(stored_dataframe.columns)} columns."
            )

            if (
                label_column is not None
                and label_column != ""
                and label_column in uploaded_df.columns
            ):
                stored_dataframe.insert(
                    0, "name_id", [str(ent) for ent in uploaded_df[label_column].values]
                )
                logging.info(f"Adding name_id column from {label_column}")
            else:
                stored_dataframe.insert(
                    0, "name_id", [str(ent) for ent in stored_dataframe.index.values]
                )
                logging.info("No name_id column found in the data, using index instead")

            options = [
                {"label": col, "value": col}
                for col in stored_dataframe.columns
                if col != "image"
            ]

            numeric_options = [
                {"label": col, "value": col}
                for col in stored_dataframe.select_dtypes(include=np.number).columns
                if col != "image"
            ]

            return (
                f"Uploaded data merged successfully. Stored dataframe now has {len(stored_dataframe)} rows and {len(stored_dataframe.columns)} columns.",
                stored_dataframe.to_dict("records"),
                options,
                options,
                options,
                options,
                options,
                options,
                numeric_options,
                numeric_options,
                "\n".join(log_store),
                "",
            )
        else:
            logging.error(
                f"Merge error cannot find merge column {merge_column} in both dataframes"
            )
            options = [
                {"label": col, "value": col}
                for col in stored_dataframe.columns
                if col != "image"
            ]

            numeric_options = [
                {"label": col, "value": col}
                for col in stored_dataframe.select_dtypes(include=np.number).columns
                if col != "image"
            ]
            # No update so no need to repeat the attempt to make images
            return (
                f'Column "{merge_column}" not found in both dataframes. Database data has not been updated.',
                stored_dataframe.to_dict("records"),
                options,
                options,
                options,
                options,
                options,
                options,
                numeric_options,
                numeric_options,
                "\n".join(log_store),
                "",
            )
    else:
        stored_dataframe = uploaded_df
        smiles_col = find_column_name_from_re(
            stored_dataframe.columns.tolist(), rex="SMILES"
        )
        inchi_col = find_column_name_from_re(
            stored_dataframe.columns.tolist(), rex="inchi"
        )
        if smiles_col is not None and "image" not in stored_dataframe.columns:
            logging.info(
                f"Generating molecule images using column {smiles_col} for smiles input"
            )
            start = datetime.now()
            stored_dataframe["image"] = stored_dataframe[smiles_col].apply(
                smiles_to_image
            )
            end = datetime.now()
            tdelta = end - start
            logging.info(
                f"Finished generating molecule images (took {tdelta.seconds}s)"
            )
        elif inchi_col is not None and "image" not in stored_dataframe.columns:
            logging.info(
                f"Generating molecule images using column {inchi_col} for inchi input"
            )
            start = datetime.now()
            stored_dataframe["image"] = stored_dataframe[inchi_col].apply(
                inchi_to_image
            )
            end = datetime.now()
            tdelta = end - start
            logging.info(
                f"Finished generating molecule images (took {tdelta.seconds}s)"
            )
        else:
            logging.info(
                "SMILES column not found in the data therefore no images will be displayed in plots."
            )

        if (
            label_column is not None
            and label_column != ""
            and label_column in uploaded_df.columns
        ):
            stored_dataframe.insert(
                0, "name_id", [str(ent) for ent in uploaded_df[label_column].values]
            )
            logging.info(f"Adding name_id column from {label_column}")
        else:
            stored_dataframe.insert(
                0, "name_id", [str(ent) for ent in stored_dataframe.index.values]
            )
            logging.info("No name_id column found in the data, using index instead")

        options = [
            {"label": col, "value": col}
            for col in stored_dataframe.columns
            if col != "image"
        ]
        numeric_options = [
            {"label": col, "value": col}
            for col in stored_dataframe.select_dtypes(include=np.number).columns
            if col != "image"
        ]
        return (
            f"Uploaded data has been saved. Stored dataframe has {len(stored_dataframe)} rows and {len(stored_dataframe.columns)} columns.",
            stored_dataframe.to_dict("records"),
            options,
            options,
            options,
            options,
            options,
            options,
            numeric_options,
            numeric_options,
            "\n".join(log_store),
            "",
        )


@app.callback(
    Output("upload-data", "children"),
    Input("upload-data", "contents"),
)
def update_upload_button(contents) -> Any:
    """
    Function to update the upload button with the contents provided.
    Args:
        contents: The contents of the upload button
    Returns:
        str: The contents of the upload
    """
    if contents is not None:
        return html.Div(
            [
                "Upload ready ",
                html.A(
                    "click here to select a different CSV File", style={"color": "blue"}
                ),
            ]
        )
    else:
        return html.Div(
            [
                "Drag and drop or ",
                html.A("click here to select a CSV File", style={"color": "blue"}),
            ]
        )


##### Data table functions #####
@app.callback(
    Output("operation-result", "children"),
    Output("stored-dataframe", "data", allow_duplicate=True),
    Output("x-axis-column", "options", allow_duplicate=True),
    Output("y-axis-column", "options", allow_duplicate=True),
    Output("z-axis-column", "options", allow_duplicate=True),
    Output("colour-column", "options", allow_duplicate=True),
    Output("size-column", "options", allow_duplicate=True),
    Output("id-column", "options", allow_duplicate=True),
    Output("mp-objective-columns-dropdown", "options", allow_duplicate=True),
    Output("objective-columns", "options", allow_duplicate=True),
    Input("perform-operation-button", "n_clicks"),
    State("stored-dataframe", "data"),
    State("column-dropdown-1", "value"),
    State("column-dropdown-2", "value"),
    State("operation-dropdown", "value"),
    prevent_initial_call=True,
)
def perform_operation(
    n_clicks, data, col1, col2, operation
) -> Tuple[
    str,
    List[dict],
    List[dict],
    List[dict],
    List[dict],
    List[dict],
    List[dict],
    List[dict],
    List[dict],
]:
    """
    Function to perform the operation between two columns and add the result as a new column to the dataframe.
    Args:
        n_clicks: The number of times the button has been clicked
        data: The data to be used to perform the operation
        col1: The first column to be used in the operation
        col2: The second column to be used in the operation
        operation: The operation to be performed
    Returns:
        operation_result: The result of the operation
        data: The data with the new column added
        options: The options to be displayed in the dropdowns
        numeric_options: The numeric options to be displayed in the dropdowns
    """
    if n_clicks == 0 or not data or not col1 or not col2 or not operation:
        raise PreventUpdate

    df = pd.DataFrame(data)
    try:
        if operation == "add":
            df[f"{col1} + {col2}"] = df[col1] + df[col2]
        elif operation == "subtract":
            df[f"{col1} - {col2}"] = df[col1] - df[col2]
        elif operation == "multiply":
            df[f"{col1} * {col2}"] = df[col1] * df[col2]
        elif operation == "divide":
            df[f"{col1} / {col2}"] = df[col1] / df[col2]
    except KeyError as kerr:
        options = [
            {"label": col, "value": col}
            for col in pd.DataFrame(data).columns
            if col != "image"
        ]
        numeric_options = [
            {"label": col, "value": col}
            for col in pd.DataFrame(data).select_dtypes(include=np.number).columns
            if col != "image"
        ]
        return (
            f"Error: {kerr}",
            data,
            options,
            options,
            options,
            options,
            options,
            options,
            numeric_options,
            numeric_options,
        )

    options = [{"label": col, "value": col} for col in df.columns if col != "image"]
    numeric_options = [
        {"label": col, "value": col}
        for col in df.select_dtypes(include=np.number).columns
        if col != "image"
    ]

    return (
        "Successfully performed the operation",
        df.to_dict("records"),
        options,
        options,
        options,
        options,
        options,
        options,
        numeric_options,
        numeric_options,
    )


# take as input stored-dataframe and return the data and columns for the table preview id data-table-preview
@app.callback(
    Output("data-table-preview", "data"),
    Output("data-table-preview", "columns"),
    Output("column-dropdown-1", "options"),
    Output("column-dropdown-2", "options"),
    Input("stored-dataframe", "data"),
    Input("data-table-preview", "page_current"),
    Input("data-table-preview", "page_size"),
    Input("data-table-preview", "sort_by"),
)
def update_table_preview(
    data, current_page, page_size, sort_by
) -> Tuple[List[dict[Hashable, Any]] | List[dict[str, str]]]:
    """
    Function to update the data table preview with the data provided. This will update the data and columns of the table
    with the data provided.
    Args:
        data: The data to be displayed in the table
    Returns:
        data: The data to be displayed in the table
        columns: The columns to be displayed in the table
    """
    if data is None:
        raise PreventUpdate

    df = pd.DataFrame(data)

    if len(sort_by) > 0:
        dff = df.sort_values(
            sort_by[0]["column_id"],
            ascending=sort_by[0]["direction"] == "asc",
            inplace=False,
        )
    else:
        # No sort is applied
        dff = df

    columns = [dict(name=i, id=i) for i in dff.columns]
    options = [
        {"label": col, "value": col}
        for col in dff.columns
        if is_numeric_dtype(dff[col])
    ]
    return (
        dff.iloc[current_page * page_size : (current_page + 1) * page_size].to_dict(
            "records"
        ),
        columns,
        options,
        options,
    )


##### Scatter plot functions #####


@app.callback(
    Output("data-graph", "figure"),
    Output("hover-data", "data"),
    Input("x-axis-column", "value"),
    Input("y-axis-column", "value"),
    Input("z-axis-column", "value"),
    Input("plot-type", "value"),
    Input("colour-column", "value"),
    Input("size-column", "value"),
    Input("x-axis-scale", "value"),
    Input("y-axis-scale", "value"),
    Input("z-axis-scale", "value"),
    Input("theme", "value"),
    Input("trendline", "value"),
    Input("remove-missing-data", "value"),
    Input("mean-missing-data", "value"),
    Input("include-data-pointer-in-html", "value"),
    Input("x-axis-name", "value"),
    Input("y-axis-name", "value"),
    Input("z-axis-name", "value"),
    Input("colour-column-name", "value"),
    Input("stored-dataframe", "data"),
)
def update_scatter_graph(
    x_column: str,
    y_column: str,
    z_column: str,
    plot_type: str,
    colour_col: str,
    size_col: str,
    x_scale: str,
    y_scale: str,
    z_scale: str,
    theme: str,
    trendline: str,
    remove_missing_data: bool,
    fill_missing_data: bool,
    include_data_pointer_in_html: bool,
    x_axis_name: str,
    y_axis_name: str,
    z_axis_name: str,
    colour_column_name: str,
    data: List[dict],
) -> Tuple[dict, List[dict]]:
    """
    Function to update the scatter graph based on the user input
    Args:
        x_column: str: X-axis column
        y_column: str: Y-axis column
        z_column: str: Z-axis column
        plot_type: str: Plot type
        colour_col: str: Colour column
        size_col: str: Size column
        x_scale: str: X-axis scale
        y_scale: str: Y-axis scale
        z_scale: str: Z-axis scale
        theme: str: Theme
        trendline: str: Trendline
        remove_missing_data: bool: Remove missing data
        fill_missing_data: bool: Fill missing data
        include_data_pointer_in_html: bool: Include data pointer in HTML
        x_axis_name: str: X-axis name
        y_axis_name: str: Y-axis name
        z_axis_name: str: Z-axis name
        colour_column_name: str: Colour column name
        data: List[dict]: Data
    Returns:
        dict: Figure
        List[dict]: Hover data
    """
    if x_column is None or y_column is None or data is None:
        raise PreventUpdate

    df = pd.DataFrame(data)

    if plot_type == "3D" and z_column is None:
        raise PreventUpdate

    logging.debug(f"Plotting columns: {x_column}, {y_column}, {z_column}")
    logging.debug(f"Plot type: {plot_type}")
    logging.debug(f"Color column: {colour_col}")
    logging.debug(f"Size column: {size_col}")
    logging.debug(f"Data shape: {df.shape}")
    logging.debug(
        f"x axis name: {x_axis_name} y axis name: {y_axis_name} z axis name: {z_axis_name}"
    )
    # logging.debug(f"df columns: {df.columns}")

    hover_data = {"name_id": True}

    # Get the data for present in the hover records
    data_cols = []
    for inp in [x_column, y_column, z_column, colour_col, size_col]:
        if inp is not None:
            data_cols.append(inp)

    # Get the images if present
    if "image" in df.columns:
        hover_data_records = df[["name_id", "image"] + data_cols].to_dict("records")
        df = df.drop(columns=["image"]).copy()
    else:
        hover_data_records = df[["name_id"] + data_cols].to_dict("records")

    # Remove missing data if required this will drop any rows with missing data in the columns being plotted including colour and size
    if remove_missing_data is True:
        data_cols = []
        for col in [x_column, y_column, z_column, colour_col, size_col]:
            if col is not None:
                data_cols.append(col)
        df = df.dropna(axis=0, subset=data_cols, how="any")

    # Get the colour and size columns if they are present and fill missing data with the mean of the column if asked to do so
    #  NOTE: missing data will be removed prior to this if the remove_missing_data is set to True hence this will have no effect
    if colour_col is not None and fill_missing_data is True:
        if df[colour_col].isnull().values.any():
            df[colour_col] = df[colour_col].fillna(df[colour_col].mean())

    if size_col is not None:
        if df[size_col].isnull().values.any():
            sizes = df[size_col].copy().fillna(df[size_col].mean())
        else:
            sizes = df[size_col].copy()
    else:
        sizes = None

    if plot_type == "3D":
        # "3D plot selected"
        x_label = x_axis_name if x_axis_name else x_column
        y_label = y_axis_name if y_axis_name else y_column
        z_label = z_axis_name if z_axis_name else z_column

        title = preprocess_label(
            f"{x_label} vs {y_label} vs {z_label}", every_n_spaces=12
        )
        x_label = preprocess_label(x_label)
        y_label = preprocess_label(y_label)
        z_label = preprocess_label(z_label)

        fig = px.scatter_3d(
            df,
            x=x_column,
            y=y_column,
            z=z_column,
            color=colour_col,
            size=sizes,
            hover_data=hover_data,
            title=f"{x_column} vs {y_column} vs {z_column}",
            labels={x_column: x_label, y_column: y_label, z_column: z_label},
            color_continuous_scale=[(0, "#e2112a"), (1, "#3979CB")],
            color_discrete_sequence=[
                "#e2112a",
                "#BE4873",
                "#AF6A98",
                "#A28FBE",
                "#99BBE9",
                "#7BACE1",
                "#6399D5",
                "#4F8DD5",
                "#3979CB",
            ],
            template=theme,
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(type=x_scale),
                yaxis=dict(type=y_scale),
                zaxis=dict(type=z_scale),
            ),
            width=700,
            margin=dict(r=20, b=10, l=10, t=10),
        )

    else:
        # "2D plot selected"

        # "preprocessing the lables and title"
        x_label = x_axis_name if x_axis_name else x_column
        y_label = y_axis_name if y_axis_name else y_column
        title = preprocess_label(f"{x_label} vs {y_label}", every_n_spaces=7)
        x_label = preprocess_label(x_label)
        y_label = preprocess_label(y_label)

        # plot
        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            color=colour_col,
            size=sizes,
            hover_data=hover_data,
            title=title,
            labels={x_column: x_label, y_column: y_label},
            color_continuous_scale=[(0, "#e2112a"), (1, "#3979CB")],
            color_discrete_sequence=[
                "#e2112a",
                "#BE4873",
                "#AF6A98",
                "#A28FBE",
                "#99BBE9",
                "#7BACE1",
                "#6399D5",
                "#4F8DD5",
                "#3979CB",
            ],
            trendline=trendline if trendline == "ols" else None,
            template=theme,
        )

        fig.update_layout(
            xaxis=dict(type=x_scale),
            yaxis=dict(type=y_scale),
        )

        # "Updating the axes"
        fig.update_xaxes(title_font_size=17, automargin=True, tickfont_size=15)
        fig.update_yaxes(title_font_size=17, automargin=True, tickfont_size=15)

    # "Updating the plot of graph layout"
    fig.update_layout(
        height=800,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        title=dict(
            text=title,
            font=dict(size=20),
            automargin=True,
            yref="container",
            y=0.9,
            x=0.5,
            xanchor="center",
            yanchor="top",
        ),
        font=dict(family="Arial"),
        title_font_size=18,
    )

    if include_data_pointer_in_html is False:
        # "Updating the hover data to remove the hover info"
        fig.update_traces(hoverinfo="none", hovertemplate=None)

    # "updating the markers"
    fig.update_traces(
        marker=dict(line=dict(width=1.0, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    if colour_col is not None:
        colour_label = colour_column_name if colour_column_name else colour_col
        # Update colour
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=preprocess_label(colour_label, every_n_spaces=4),
                tickfont_size=12,
                title_font_size=15,
            )
        )

    if "image" in df.columns:
        # Update the hover data to include the image
        hover_template = (
            "<b>%{text}</b><br><br>" + "X: %{x}<br>" + "Y: %{y}<br><extra></extra>"
        )
        fig.update_traces(hovertemplate=hover_template)

    return fig, hover_data_records


@app.callback(
    Output("download-html", "data"),
    Input("download-html-btn", "n_clicks"),
    State("data-graph", "figure"),
    prevent_initial_call=True,
)
def download_html_scatter(
    n_clicks: int,
    figure: dict,
) -> dict:
    """
    Function to download the graph as an interactive HTML file
    Args:
        n_clicks: int: Number of clicks on the download button
        figure: dict: Figure
    Returns:
        dict: Data
    """
    html_str = pio.to_html(figure, full_html=False)
    now = datetime.now()
    string_date = now.strftime("%d-%m-%Y_%H-%M-%S")
    return dict(content=html_str, filename=f"plotly_scatter_graph_{string_date}.html")


@app.callback(
    Output("data-graph-tooltip", "show"),
    Output("data-graph-tooltip", "bbox"),
    Output("data-graph-tooltip", "children"),
    Input("data-graph", "hoverData"),
    State("hover-data", "data"),
)
def update_tooltip(
    hoverData: dict,
    hover_data: List[dict],
) -> Tuple[bool, dict, dict]:
    """
    Function to update the tooltip
    Args:
        hoverData: dict: Hover data
        hover_data: List[dict]: Hover data
    Returns:
        bool: Show
        dict: Bbox
        dict: Children
    """

    # "Updating the tooltip and remove when not hovering"
    if hoverData is None or hover_data is None:
        return False, no_update, no_update

    point_index = hoverData["points"][0]["pointNumber"]
    bbox = hoverData["points"][0]["bbox"]
    name_id = hover_data[point_index]["name_id"]
    try:
        image = hover_data[point_index]["image"]
    except KeyError as kerr:
        logging.error(f"Error: not column image! Key error is: {kerr}")
        image = None

    data_str = []
    for inps in hover_data[point_index].keys():
        if inps not in ["name_id", "image"]:
            name = preprocess_label(inps, every_n_spaces=7, linebreak=os.linesep)
            data_str.append(f"{name}: {hover_data[point_index][inps]}")
            data_str.append(html.Br())

    tooltip_content = html.Div(
        [
            html.H5(f"Name ID: {name_id}"),
            html.P(data_str, style={"whiteSpace": "pre-wrap"}),
            html.Img(src=image, style={"width": "100%"})
            if image
            else "No image available",
        ]
    )

    return True, bbox, tooltip_content


##### Radar plot functions #####


@app.callback(
    Output("radar-plot", "figure"),
    Output("log-output", "value", allow_duplicate=True),
    Output("radar-data-table-store", "data"),
    Output("radar-plotting-output", "children"),
    Output("radar-plotting-output", "is_open"),
    Output("radar-plotting-output", "color"),
    Input("run-analysis-button", "n_clicks"),
    State("stored-dataframe", "data"),
    State("objective-columns", "value"),
    State("directions", "value"),
    State("id-column", "value"),
    State("row-id", "value"),
    State("reference-row-id", "value"),
    Input("min-max-scale-radar-data", "value"),
    Input("theme-radar", "value"),
    prevent_initial_call=True,
)
def update_radar_plot(
    n_clicks,
    stored_data,
    objective_columns,
    directions,
    id_column,
    row_id,
    reference_row_id,
    scale,
    theme,
) -> Any | str:
    """
    Function to plot a radar plot of key properties provided. This function will plot the data in a radar plot
    and highlight the row_id molecule in the plot. If multiple row_ids are provided, they will all be plotted.
    ut also provided multi-objective optimization directions and objectives to plot the pareto front.
    Args:
        n_clicks: The number of times the button has been clicked
        stored_data: The data to be plotted
        objective_columns: The columns to be used as objectives
        directions: The directions to be used for the objectives
        id_column: The column to be used as the unique identifier
        row_id: The row_id to be plotted
        reference_row_id: The reference row_id to be plotted
        scale: Whether to scale the data
        theme: The theme to be used for the plot
    Returns:

    """
    if n_clicks == 0:
        raise PreventUpdate

    df = pd.DataFrame(stored_data)

    df.columns = [
        preprocess_label(" ".join(col.strip().split("_")), every_n_spaces=2)
        for col in df.columns
    ]

    objectives = [
        preprocess_label(" ".join(col.strip().split("_")), every_n_spaces=2)
        for col in objective_columns
    ]
    id_column = preprocess_label(" ".join(id_column.strip().split("_")))
    image_column = "image<br>"

    # account for the case where the user has not selected any objective directions by not running Pareto analysis
    if directions is not None:
        directions = [dir.strip().lower() for dir in directions.split(",")]
        # Run Pareto analysis
        try:
            pareto_df = app_methods.get_pareto_ranking(
                data_df=df,
                objective_columns=objectives,
                minmax=directions,
            )
            logging.debug(f"Pareto analysis complete")
        except RuntimeError as rerr:
            logging.error(f"Error in Pareto analysis: {rerr}")
            return (
                go.Figure(),
                "\n".join(log_store),
                pd.DataFrame([{}]).to_dict("records"),
                f"Error in Pareto analysis please check: {rerr}",
                True,
                "danger",
            )
    else:
        logging.warning(f"Directions not provided not performing Pareto analysis")
        pareto_df = df.copy()

    logging.debug(f"Objectives: {objectives}, Directions: {directions}")
    logging.debug(f"Pareto df columns: {pareto_df.columns}")
    # drop rows which are missing data in the objective and id columns warn if this removes all data or error if it removes a molecule being analyszed
    pareto_df = pareto_df.dropna(subset=objectives + [id_column], axis=0, how="any")

    # Data validate 1: Check if the data is empty 2: Check if the row_id is present in the data 3: Check if the row_id has missing data
    if len(pareto_df.index) == 0:
        logging.warning(f"No data found for the selected columns")
        return (
            go.Figure(),
            "\n".join(log_store),
            pd.DataFrame([{}]).to_dict("records"),
            "No data found for the selected columns",
            True,
            "danger",
        )
    elif any(ent not in pareto_df[id_column].unique() for ent in row_id):
        logging.error(
            f"Row ID not found in data. Likely you have asked to analyze a molecule which has not got data for all objectives selected"
        )
        return (
            go.Figure(),
            "\n".join(log_store),
            pd.DataFrame([{}]).to_dict("records"),
            "Row ID not found in data. Likely you have asked to analyze a molecule which has not got data for all objectives selected",
            True,
            "danger",
        )

    for rid in row_id:
        tmp_row = pareto_df[pareto_df[id_column] == rid].copy()
        if tmp_row[objectives].isnull().any().any():
            logging.error(f"Row ID {rid} has missing data for the objectives selected")
            return (
                go.Figure(),
                "\n".join(log_store),
                pd.DataFrame([{}]).to_dict("records"),
                f"Row ID {rid} has missing data for the objectives selected",
                True,
                "danger",
            )

    # scale the data to min-max or use unscaled
    if scale is True:
        logging.info(f"Scaling data to min-max")
        id_col = pareto_df[id_column].copy().values
        if image_column in pareto_df.columns:
            image_col = pareto_df[image_column].copy().values
        else:
            image_col = None

        pareto_column_name = find_column_name_from_re(
            pareto_df.columns.tolist(), rex=pareto_col
        )
        if pareto_column_name is not None:
            pareto_column = pareto_df[pareto_column_name].copy().values
        else:
            pareto_column = None

        pareto_df = helpers.pandas_df_min_max_scale(pareto_df[objectives].copy())
        pareto_df[id_column] = id_col
        if image_column is not None:
            pareto_df[image_column] = image_col
        if pareto_column is not None:
            pareto_df[pareto_column_name] = pareto_column
        scale_range = [0, 1]
        logging.debug(f"Pareto data after scaling applied: {pareto_df}")
    else:
        # This is to match the operations in the if clause previously
        keep_cols = objectives + [id_column]
        if image_column in pareto_df.columns:
            keep_cols.append(image_column)

        pareto_column_name = find_column_name_from_re(
            pareto_df.columns.tolist(), rex=pareto_col
        )

        if pareto_column_name is not None:
            keep_cols.append(pareto_column_name)
        pareto_df = pareto_df[keep_cols].copy()
        scale_range = [
            min(pareto_df[objectives].min().min(), 0),
            max(pareto_df[objectives].max().max(), 1),
        ]

    logging.debug(f"Pareto df columns: {pareto_df.columns}")
    logging.info(f"Pareto df shape: {pareto_df.shape}")

    # Sort the columns so that the max and mins are clustered
    max_indexes = [
        ith for ith, d in enumerate(directions) if d.strip().lower() == "max"
    ]
    min_indexes = [
        ith for ith, d in enumerate(directions) if d.strip().lower() == "min"
    ]
    logging.debug(f"Max indexes: {max_indexes}, Min indexes: {min_indexes}")
    max_objectives = sorted([objectives[ith] for ith in max_indexes])
    min_objectives = sorted([objectives[ith] for ith in min_indexes])
    logging.debug(f"Max objectives: {max_objectives}, Min objectives: {min_objectives}")
    objectives = max_objectives + min_objectives
    directions = ["max" for _ in max_objectives] + ["min" for _ in min_objectives]

    # Create radar plot
    fig = go.Figure()

    if len(row_id) == 1:
        logging.info("Plotting a single row")
        row_data = pareto_df[pareto_df[id_column] == row_id[0]].iloc[0].copy()
        raw_row_data = df[df[id_column] == row_id[0]].iloc[0].copy()
        r = row_data[objectives].tolist() + row_data[objectives].tolist()[:1]
        theta = objectives + objectives[:1]
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                fill="toself",
                name=f"{row_id[0]}",
                hovertemplate="Molecule: "
                + row_data[id_column]
                + "".join(
                    [
                        f"<br><b>{prop.replace('<br>', ' ')}</b>: {float(raw_row_data[prop]):.2f}<br>"
                        for prop in objectives
                    ]
                ),
            )
        )

    elif len(row_id) > 1:
        logging.info("Plotting multiple rows")
        for ridx in row_id:
            row_data = pareto_df[pareto_df[id_column] == ridx].iloc[0].copy()
            raw_row_data = df[df[id_column] == ridx].iloc[0].copy()
            r = row_data[objectives].tolist() + row_data[objectives].tolist()[:1]
            theta = objectives + objectives[:1]
            fig.add_trace(
                go.Scatterpolar(
                    r=r,
                    theta=theta,
                    fill="toself",
                    name=f"{ridx}",
                    hovertemplate="Molecule: "
                    + ridx
                    + "".join(
                        [
                            f"<br><b>{prop.replace('<br>', ' ')}</b>: {float(raw_row_data[prop]):.2f}<br>"
                            for prop in objectives
                        ]
                    ),
                )
            )

    if directions is not None:
        logging.info("Plotting ideal trace")
        ideal_trace = []
        for dir in directions:
            if dir == "max":
                ideal_trace.append(max(pareto_df[objectives].max().max(), 1.0))
            elif dir == "min":
                ideal_trace.append(0.001)

        fig.add_trace(
            go.Scatterpolar(
                r=ideal_trace + ideal_trace[:1],
                theta=objectives + objectives[:1],
                fill="none",
                mode="lines",
                line=dict(color="grey", width=2.0, dash="dash"),
                name=f"Ideal",
                hovertemplate="Ideal based on optimal directions:<br>"
                + "".join(
                    [
                        f"<br><b>{prop.replace('<br>', ' ')}</b>: {float(ideal_trace[ith]):.2f}<br>"
                        for ith, prop in enumerate(objectives)
                    ]
                ),
            )
        )

    if reference_row_id == "None":
        logging.debug(f"No reference row selected")
    else:
        logging.info(f"Plotting reference row")
        if reference_row_id == "mean":
            logging.debug(f"Calculating mean row")
            reference_data = pareto_df[objectives].mean()
            raw_reference_data = df[objectives].mean()
        elif reference_row_id == "median":
            logging.debug(f"Calculating median row")
            reference_data = pareto_df[objectives].median()
            raw_reference_data = df[objectives].median()
        elif reference_row_id == "mean_pareto_best":
            logging.debug(f"Calculating pareto best row")

            pareto_column_name = find_column_name_from_re(
                pareto_df.columns.tolist(), rex=pareto_col
            )
            if pareto_column_name is not None:
                logging.error(
                    f"Pareto column not found in data you must pre-calculated pareto ranking or provide the optimal directions as a comma separed list"
                )
                return (
                    go.Figure(),
                    "\n".join(log_store),
                    pd.DataFrame([{}]).to_dict("records"),
                    "Pareto column not found in data you must pre-calculated pareto ranking or provide the optimal directions as a comma separed list",
                    True,
                    "danger",
                )
            reference_pareto_best = pareto_df[pareto_df[pareto_column_name] == 1].copy()
            reference_data = reference_pareto_best[objectives].mean()
            raw_reference_data = df[
                df[id_column].isin(reference_pareto_best[id_column])
            ][objectives].mean()
        elif reference_row_id == "mean_pareto_worst":
            logging.debug(f"Calculating pareto worst row")
            if pareto_column_name is not None:
                logging.error(
                    f"Pareto column not found in data you must pre-calculated pareto ranking or provide the optimal directions as a comma separed list"
                )
                return (
                    go.Figure(),
                    "\n".join(log_store),
                    pd.DataFrame([{}]).to_dict("records"),
                    "Pareto column not found in data you must pre-calculated pareto ranking or provide the optimal directions as a comma separed list",
                    True,
                    "danger",
                )
            worst_group_index = sorted(pareto_df[pareto_column_name].unique())[-1]
            logging.debug(f"Worst Pareto group index: {worst_group_index}")
            reference_pareto_worst = pareto_df[
                pareto_df[pareto_column_name] == worst_group_index
            ].copy()
            reference_data = reference_pareto_worst[objectives].mean()
            raw_reference_data = df[
                df[id_column].isin(reference_pareto_worst[id_column])
            ][objectives].mean()
        else:
            logging.debug(f"Reference row selected")
            reference_data = (
                pareto_df[pareto_df[id_column] == reference_row_id].iloc[0].copy()
            )
            raw_reference_data = df[df[id_column] == reference_row_id].iloc[0].copy()
            if reference_data.empty:
                logging.error(
                    f"Reference row does not appear to contain all objective data values. Please check or select a different reference."
                )
                return (
                    go.Figure(),
                    "\n".join(log_store),
                    pd.DataFrame([{}]).to_dict("records"),
                    "Reference row does not appear to contain all objective data values. Please check or select a different reference.",
                    True,
                    "danger",
                )

        fig.add_trace(
            go.Scatterpolar(
                r=reference_data[objectives].to_list()
                + reference_data[objectives].to_list()[:1],
                theta=objectives + objectives[:1],
                fill="toself",
                name=f"Reference {reference_row_id}",
                marker=dict(color="mediumaquamarine"),
                hovertemplate="Reference: "
                + reference_row_id
                + "".join(
                    [
                        f"<br><b>{prop.replace('<br>', ' ')}</b>: {float(raw_reference_data[prop]):.2f}<br>"
                        for prop in objectives
                    ]
                ),
            )
        )

    fig.update_coloraxes(autocolorscale=True)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=scale_range,
                tickfont=dict(size=14),
            ),
            angularaxis=dict(
                tickfont=dict(size=14),
            ),
        ),
        showlegend=True,
        colorscale=dict(
            sequential=[(0, "#e2112a"), (1, "#3979CB")],
            diverging=[
                (0, "#e2112a"),
                (0.1, "#BE4873"),
                (0.2, "#AF6A98"),
                (0.3, "#A28FBE"),
                (0.4, "#99BBE9"),
                (0.5, "#99BBE9"),
                (0.6, "#7BACE1"),
                (0.7, "#6399D5"),
                (0.8, "#4F8DD5"),
                (0.9, "#4080CE"),
                (1, "#3979CB"),
            ],
        ),
        colorway=[
            "#e2112a",
            "#BE4873",
            "#AF6A98",
            "#A28FBE",
            "#99BBE9",
            "#99BBE9",
            "#7BACE1",
            "#6399D5",
            "#4F8DD5",
            "#4080CE",
            "#3979CB",
        ],
    )

    fig.update_layout(
        template=theme,
    )

    # hover_template = "<b>%{r}</b>: %{theta}<br>Molecule: {text}"
    # fig.update_traces(hovertemplate=hover_template)

    logging.info(f"Radar plot created")

    # account for the possible error of trying to calculate an area of a line i.e. only two points
    # TODO: should this be allowed and just add two more points with a small difference to make a rectangle representing a line?
    if len(objectives) <= 2:
        logging.warning(
            f"Only two objectives selected. Radar plot may not be the most informative and areas cannot be calculated"
        )
        return (
            fig,
            "\n".join(log_store),
            pd.DataFrame([{}]).to_dict("records"),
            "Only two objectives selected. Radar plot may not be the most informative and areas cannot be calculated",
            True,
            "warning",
        )

    # Extract Cartesian coordinates
    cartesian_coords = []
    cartesian_coords_test_traces = {}
    cartesian_coords_reference_trace = {}
    cartesian_coords_ideal_trace = {}
    delta_theta = 360.0 / len(objectives)
    for ith, trace in enumerate(fig.data):
        r = trace["r"][:-1]
        theta = [360.0 - (delta_theta * jth) for jth in range(len(r))]
        name = trace["name"]
        logging.debug(f"Trace {ith} {name}: Theta: {theta} r: {r}")
        theta_rad = np.deg2rad([float(t) for t in theta])
        x = r * np.cos(theta_rad) if r != 0.0 else 1e-10
        y = r * np.sin(theta_rad) if r != 0.0 else 1e-10
        cartesian_coords.append(
            {"name": trace["name"], "x": x.tolist(), "y": y.tolist()}
        )
        if "Reference" in name:
            cartesian_coords_reference_trace = {"x": x.tolist(), "y": y.tolist()}
        elif "Ideal" in name:
            cartesian_coords_ideal_trace = {"x": x.tolist(), "y": y.tolist()}
        else:
            cartesian_coords_test_traces[name] = {"x": x.tolist(), "y": y.tolist()}
        logging.debug(
            f"Trace {ith}: x: {x} y: {y} area: {app_methods.calculate_area([(xp, yp) for xp, yp in zip(x, y)])}"
        )

    # Calculate the intersection area and divergence area (i.e. the non-overlapping area) of the reference and test traces if given with each test trace using the app_methods functions
    # save to a dictionary as a float value
    if len(cartesian_coords_reference_trace) > 0:
        logging.info(
            f"Calculating intersection and divergence areas against the reference trace"
        )
        reference_table_data = True
        intersection_areas_ref = {}
        non_intersection_areas_ref = {}
        area_reference_only_area = 1e-15
        overlap_fraction_reference = {}
        overlap_percentage_reference = {}
        non_overlap_fraction_reference = {}
        non_overlap_percentage_reference = {}

        area_reference_only_area = app_methods.calculate_area(
            [
                (rx, ry)
                for rx, ry in zip(
                    cartesian_coords_reference_trace["x"],
                    cartesian_coords_reference_trace["y"],
                )
            ]
        )

        for test_trace in cartesian_coords_test_traces:
            intersection_area = app_methods.calculate_overlapping_area(
                [
                    (rx, ry)
                    for rx, ry in zip(
                        cartesian_coords_reference_trace["x"],
                        cartesian_coords_reference_trace["y"],
                    )
                ],
                [
                    (tx, ty)
                    for tx, ty in zip(
                        cartesian_coords_test_traces[test_trace]["x"],
                        cartesian_coords_test_traces[test_trace]["y"],
                    )
                ],
            )
            divergence_area = app_methods.calculate_difference_area(
                [
                    (rx, ry)
                    for rx, ry in zip(
                        cartesian_coords_reference_trace["x"],
                        cartesian_coords_reference_trace["y"],
                    )
                ],
                [
                    (tx, ty)
                    for tx, ty in zip(
                        cartesian_coords_test_traces[test_trace]["x"],
                        cartesian_coords_test_traces[test_trace]["y"],
                    )
                ],
            )
            intersection_areas_ref[test_trace] = intersection_area
            non_intersection_areas_ref[test_trace] = divergence_area
            overlap_fraction_reference[test_trace] = (
                intersection_area / area_reference_only_area
            )
            overlap_percentage_reference[test_trace] = (
                overlap_fraction_reference[test_trace] * 100.0
            )
            non_overlap_fraction_reference[test_trace] = (
                divergence_area / area_reference_only_area
            )
            non_overlap_percentage_reference[test_trace] = (
                non_overlap_fraction_reference[test_trace] * 100.0
            )

    else:
        reference_table_data = False

    if len(cartesian_coords_ideal_trace) > 0:
        logging.info(
            f"Calculating intersection and divergence areas against the ideal trace"
        )
        ideal_table_data = True
        intersection_areas_idl = {}
        non_intersection_areas_idl = {}
        area_ideal_only_area = 1e-15
        overlap_fraction_ideal = {}
        overlap_percentage_ideal = {}
        non_overlap_fraction_ideal = {}
        non_overlap_percentage_ideal = {}

        area_ideal_only_area = app_methods.calculate_area(
            [
                (rx, ry)
                for rx, ry in zip(
                    cartesian_coords_ideal_trace["x"],
                    cartesian_coords_ideal_trace["y"],
                )
            ]
        )

        for test_trace in cartesian_coords_test_traces:
            intersection_area = app_methods.calculate_overlapping_area(
                [
                    (rx, ry)
                    for rx, ry in zip(
                        cartesian_coords_ideal_trace["x"],
                        cartesian_coords_ideal_trace["y"],
                    )
                ],
                [
                    (tx, ty)
                    for tx, ty in zip(
                        cartesian_coords_test_traces[test_trace]["x"],
                        cartesian_coords_test_traces[test_trace]["y"],
                    )
                ],
            )
            divergence_area = app_methods.calculate_difference_area(
                [
                    (rx, ry)
                    for rx, ry in zip(
                        cartesian_coords_ideal_trace["x"],
                        cartesian_coords_ideal_trace["y"],
                    )
                ],
                [
                    (tx, ty)
                    for tx, ty in zip(
                        cartesian_coords_test_traces[test_trace]["x"],
                        cartesian_coords_test_traces[test_trace]["y"],
                    )
                ],
            )
            intersection_areas_idl[test_trace] = intersection_area
            non_intersection_areas_idl[test_trace] = divergence_area
            overlap_fraction_ideal[test_trace] = (
                intersection_area / area_ideal_only_area
            )
            overlap_percentage_ideal[test_trace] = (
                overlap_fraction_ideal[test_trace] * 100.0
            )
            non_overlap_fraction_ideal[test_trace] = (
                divergence_area / area_ideal_only_area
            )
            non_overlap_percentage_ideal[test_trace] = (
                non_overlap_fraction_ideal[test_trace] * 100.0
            )

    else:
        ideal_table_data = False

    # Prepare data for the table
    if reference_table_data is True and ideal_table_data is True:
        logging.info(f"Preparing table data for reference and ideal traces")
        table_data = [
            {
                "Trace": trace_name,
                # "Reference area": area_reference_only_area,
                # "Reference intersection area": intersection_areas_ref[trace_name],
                "Reference intersection area (%)": overlap_percentage_reference[
                    trace_name
                ],
                # "Reference divergence area": non_intersection_areas_ref[trace_name],
                "Reference difference area (%)": non_overlap_percentage_reference[
                    trace_name
                ],
                "Reference score": overlap_percentage_reference[trace_name]
                - non_overlap_percentage_reference[trace_name],
                # "Ideal area": area_ideal_only_area,
                # "Ideal intersection area": intersection_areas_idl[trace_name],
                "Ideal intersection area (%)": overlap_percentage_ideal[trace_name],
                # "Ideal divergence area": non_intersection_areas_idl[trace_name],
                "Ideal difference area (%)": non_overlap_percentage_ideal[trace_name],
                "Ideal score": overlap_percentage_ideal[trace_name]
                - non_overlap_percentage_ideal[trace_name],
            }
            for trace_name in intersection_areas_ref
        ]
        tab = pd.DataFrame(table_data).to_dict("records")

    elif reference_table_data is True and ideal_table_data is False:
        logging.info(f"Preparing table data for reference traces only")
        table_data = [
            {
                "Trace": trace_name,
                # "Reference area": area_reference_only_area,
                # "Reference intersection area": intersection_areas_ref[trace_name],
                "Reference Intersection area (%)": overlap_percentage_reference[
                    trace_name
                ],
                # "Reference divergence area": non_intersection_areas_ref[trace_name],
                "Reference difference area (%)": non_overlap_percentage_reference[
                    trace_name
                ],
                "Reference score": overlap_percentage_reference[trace_name]
                - non_overlap_percentage_reference[trace_name],
            }
            for trace_name in intersection_areas_ref
        ]
        tab = pd.DataFrame(table_data).to_dict("records")

    elif reference_table_data is False and ideal_table_data is True:
        logging.info(f"Preparing table data for ideal traces only")
        table_data = [
            {
                "Trace": trace_name,
                # "Ideal area": area_ideal_only_area,
                # "Ideal intersection Area": intersection_areas_idl[trace_name],
                "Ideal intersection area (%)": overlap_percentage_ideal[trace_name],
                # "Ideal divergence Area": non_intersection_areas_idl[trace_name],
                "Ideal difference area (%)": non_overlap_percentage_ideal[trace_name],
                "Ideal score": overlap_percentage_ideal[trace_name]
                - non_overlap_percentage_ideal[trace_name],
            }
            for trace_name in intersection_areas_idl
        ]
        tab = pd.DataFrame(table_data).to_dict("records")
    else:
        logging.info(f"No table data to prepare")
        tab = pd.DataFrame([{}]).to_dict("records")

    return (
        fig,
        "\n".join(log_store),
        tab,
        "Radar plot successfully created see below",
        True,
        "success",
    )


@app.callback(
    Output("download-html-radar", "data"),
    Input("download-html-radar-btn", "n_clicks"),
    State("radar-plot", "figure"),
    prevent_initial_call=True,
)
def download_html_radar(
    n_clicks: int,
    figure: dict,
) -> dict:
    """
    Function to download the graph as an interactive HTML file
    Args:
        n_clicks: int: Number of clicks on the download button
        figure: dict: Figure
    Returns:
        dict: Data
    """
    html_str = pio.to_html(figure, full_html=False)
    now = datetime.now()
    string_date = now.strftime("%d-%m-%Y_%H-%M-%S")
    return dict(content=html_str, filename=f"plotly_radar_graph_{string_date}.html")


##### Radar table functions #####


@app.callback(
    Output("radar-data-table", "data"),
    Output("radar-data-table", "columns"),
    Input("radar-data-table-store", "data"),
)
def update_radar_data_table(
    data,
) -> Tuple[List[dict[Hashable, Any]] | List[dict[str, str]]]:
    """
    Function to update the radar data table with the data provided. This will update the data and columns of the table
    with the data provided.
    Args:
        data: The data to be displayed in the table
    Returns:
        data: The data to be displayed in the table
        columns: The columns to be displayed in the table
    """
    if data is None:
        raise PreventUpdate

    df = pd.DataFrame(data)
    columns = [
        dict(
            name=i,
            id=i,
            type="numeric",
            format=Format(precision=2, scheme=Scheme.fixed),
        )
        for i in df.columns
    ]
    data = df.to_dict("records")
    return data, columns


@app.callback(
    Output("download-table-radar", "data"),
    Input("download-table-radar-btn", "n_clicks"),
    State("radar-data-table-store", "data"),
    prevent_initial_call=True,
)
def download_the_table_radar(
    n_clicks: int,
    data_df: dict,
) -> dict:
    """
    Function to download the graph as an interactive HTML file
    Args:
        n_clicks: int: Number of clicks on the download button
        figure: dict: Figure
    Returns:
        dict: Data
    """
    df = pd.DataFrame(data_df)
    now = datetime.now()
    string_date = now.strftime("%d-%m-%Y_%H-%M-%S")
    return dcc.send_data_frame(
        df.to_csv, f"radar_intersection_and_divergence_{string_date}.csv", index=False
    )


##### Utility functions #####


def find_column_name_from_re(cols: List[str], rex="SMILES"):
    """
    Function to find a column name from a list of columns using a regular expression
    ignore case.
    Args:
        cols: List[str]: List of columns
        rex: str: Regular expression
    Returns:
        str: Column name
    """
    for col in cols:
        if re.search(rex, col, re.IGNORECASE):
            return col
    return None


@app.callback(
    Output("row-id", "options"),
    Output("reference-row-id", "options"),
    State("stored-dataframe", "data"),
    Input("id-column", "value"),
)
def update_row_id_options(stored_data, id_column) -> Tuple[List[dict], List[dict]]:
    """
    Function to update the row id options with the unique values from the column selected. This will update the row id
    and reference row id dropdowns with the unique values from the column selected.
    Args:
        stored_data: The data to be used to populate the dropdowns
        id_column: The column to be used to populate the dropdowns
    Returns:
        options: The options to be displayed in the dropdown
        reference_options: The reference options to be displayed in the dropdown
    """
    if stored_data is None:
        raise PreventUpdate

    logging.debug(f"Updating row choices with column {id_column}")
    df = pd.DataFrame(stored_data)
    options = [
        {"label": str(row), "value": row} for row in df[str(id_column).strip()].unique()
    ]

    reference_options = [
        {"label": "None", "value": "None"},
        {"label": "mean", "value": "mean"},
        {"label": "median", "value": "median"},
        {"label": "mean_pareto_best", "value": "mean_pareto_best"},
        {"label": "mean_pareto_worst", "value": "mean_pareto_worst"},
    ] + options
    return options, reference_options


@app.callback(
    Output("directions", "value"),
    Input("directions", "value"),
    prevent_initial_call=True,
)
def update_directions(value) -> str:
    """
    Function to update the directions value to remove any whitespace. This will remove any whitespace from the directions
    value.
    Args:
        value: The value to be cleaned
    Returns:
        value: The cleaned value
    """
    if len(value.strip()) == 0:
        return None
    else:
        return value.strip()


@app.callback(
    Output("x-axis-copy", "content"),
    Input("x-axis-copy", "n_clicks"),
    Input("x-axis-column", "value"),
    prevent_initial_call=True,
)
def x_custom_copy(n_clicks, value) -> str:
    """
    Function to copy the value to the clipboard. This will copy the value to the clipboard.
    Args:
        n_clicks: The number of clicks on the button
        value: The value to be copied
    Returns:
        value: The value to be copied
    """
    logging.debug(f"the number of clicks is {n_clicks}")
    logging.debug(f"Copying {value} to clipboard")
    return value


@app.callback(
    Output("y-axis-copy", "content"),
    Input("y-axis-copy", "n_clicks"),
    Input("y-axis-column", "value"),
    prevent_initial_call=True,
)
def y_custom_copy(n_clicks, value) -> str:
    """
    Function to copy the value to the clipboard. This will copy the value to the clipboard.
    Args:
        n_clicks: The number of clicks on the button
        value: The value to be copied
    Returns:
        value: The value to be copied
    """
    logging.debug(f"the number of clicks is {n_clicks}")
    logging.debug(f"Copying {value} to clipboard")
    return value


@app.callback(
    Output("z-axis-copy", "content"),
    Input("z-axis-copy", "n_clicks"),
    Input("z-axis-column", "value"),
    prevent_initial_call=True,
)
def z_custom_copy(n_clicks, value) -> str:
    """
    Function to copy the value to the clipboard. This will copy the value to the clipboard.
    Args:
        n_clicks: The number of clicks on the button
        value: The value to be copied
    Returns:
        value: The value to be copied
    """
    logging.debug(f"the number of clicks is {n_clicks}")
    logging.debug(f"Copying {value} to clipboard")
    return value


@app.callback(
    Output("colour-copy", "content"),
    Input("colour-copy", "n_clicks"),
    Input("colour-column", "value"),
    prevent_initial_call=True,
)
def colour_custom_copy(n_clicks, value) -> str:
    """
    Function to copy the value to the clipboard. This will copy the value to the clipboard.
    Args:
        n_clicks: The number of clicks on the button
        value: The value to be copied
    Returns:
        value: The value to be copied
    """
    logging.debug(f"the number of clicks is {n_clicks}")
    logging.debug(f"Copying {value} to clipboard")
    return value


@app.callback(
    Output("size-copy", "content"),
    Input("size-copy", "n_clicks"),
    Input("size-column", "value"),
    prevent_initial_call=True,
)
def size_custom_copy(n_clicks, value) -> str:
    """
    Function to copy the value to the clipboard. This will copy the value to the clipboard.
    Args:
        n_clicks: The number of clicks on the button
        value: The value to be copied
    Returns:
        value: The value to be copied
    """
    logging.debug(f"the number of clicks is {n_clicks}")
    logging.debug(f"Copying {value} to clipboard")
    return value


def preprocess_label(
    label: str,
    every_n_spaces: int = 3,
    linebreak: str = "<br>",
) -> str:
    """
    Function to preprocess the label for the plot
    Args:
        label: str: Label
        every_n_spaces: int: Every n spaces
    Returns:
        LiteralString | str: Wrapped label
    """
    labs = label.replace("_", " ").split()
    wrapped_lab = ""
    for ith, ent in enumerate(labs):
        if ith % every_n_spaces == 0 and ith != len(labs) - 1:
            wrapped_lab += f"{ent}{linebreak}"
        elif ith == len(labs) - 1:
            wrapped_lab += f"{ent}"
        else:
            wrapped_lab += f"{ent} "
    return wrapped_lab.strip()


def smiles_to_image(smiles: str) -> str:
    """
    Function to convert a SMILES to jpeg string
    Args:
        smiles: str: SMILES
    Returns:
        str: Image string in jpeg format
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol)
        buffer = BytesIO()
        img.save(buffer, format="jpeg")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64, {img_str}"
    except Exception as e:
        logging.error(f"Error in converting SMILES to image: {smiles} {e}")
        return None


def inchi_to_image(inchi: str) -> str:
    """
    Function to convert an InChI to jpeg string
    Args:
        inchi: str: inchi
    Returns:
        str: Image string in jpeg format
    """
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    img = Draw.MolToImage(mol)
    buffer = BytesIO()
    img.save(buffer, format="jpeg")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64, {img_str}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the analysis app")

    parser.add_argument(
        "-a",
        "--host",
        type=str,
        action="store",
        default="127.0.0.1",
        help="Host for the server",
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        action="store",
        default=8050,
        help="Port for the app",
    )

    parser.add_argument(
        "-d",
        "--debug_off",
        action="store_false",
        default=True,
        help="Run dash in production mode",
    )

    op = parser.parse_args()

    app.run_server(host=op.host, port=op.port, debug=op.debug_off)
