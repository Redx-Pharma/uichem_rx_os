[![Python package](https://github.com/Redx-Pharma/uichem_rx_os/actions/workflows/testing.yaml/badge.svg)](https://github.com/Redx-Pharma/uichem_rx_os/actions/workflows/testing.yaml)

[![Deploy static content to Pages](https://github.com/Redx-Pharma/uichem_rx_os/actions/workflows/static.yaml/badge.svg)](https://github.com/Redx-Pharma/uichem_rx_os/actions/workflows/static.yaml)

# User Interface

## Introduction

This repository contains a Web application built using Dash as a User Interface (UI) for the computational analysis if molecular data. At present the application supports the following:
* Parsing of uploaded CSV files
* Interactive plotting of the data uploaded
* Download of the plots as interactive html and static png

## How to use
1. You will need to have python 3 installed and avaliable
1. You should build a blank virtual environment. For example in conda `conda create -n ui python=3.11`
1. You should then build the environment using pip by running `pip install .` from the top level directory of this repository i.e. the directory containing the pyproject.toml
1. After you have successfully installed this try to run: `python app/app.py` from the same top level directory
    * This will start a local server running the app
1. Navigate to the local URL (by defualt this will be http://127.0.0.1:8050/) on the machine the server is running on
1. Follow the apps instructions

### User Instructions
* Begin by uploading a csv file of molecular data to the app. Ensure you are happy to upload this file to the host where this application is running.
* Once uploaded you will see a data table preview and see success messages for uploading
* If there is a column SMILES or inchi then the code will automatically pre-generate molecule images which are displayed on the scatter graph
* You can also plot radar plots to compare multiple properties between entries. In addition, you can define reference and ideal cases on the radar plots for comparison.

## Q&A
* There is a logging pannel in the top right of the app. For most major forseen errors, if they occur, they should be reported in that pannel. Additional errors are reported on the console.
* The package has automated testsing using pytest
* A token with repository scope can be saved as a secret in the repoository to allow for additional custom packages from proivate repositories to be installed. This should be saved under the secrets section of the repository and named `TOKENFILE`. The token also will need to be authorized for use in the appropiate GitHub organization (go to the [token](https://github.com/settings/tokens) page and under configure SSO press the organization label and follow the steps).
* For the Docker build you should not need any special steps. Simply `docker build -t ui:latest .` should build the applictaion.
* If you need to include additioanl priviate repo packages you will also need a token with repo and package access. Docker can build the image using the token as a secret to avoid it being avaliable in the image that is built. This token needs to be saved in to a file on its own. Assuming you save this in `home/user/token` and you are running from the top level diretcory of this repository, then the Docker build command is:
    ```bash
    DOCKER_BUILDKIT=1 docker build --secret id=TOKENFILE,src=path/to/token -t ui:latest .
    ```
* To run from a docker container on a local machine assuming the image name if `ui:latest` and built using the instructions in the bullet above, type the following:
    ```bash
    docker run --rm -p 8050:8050 ui:latest conda run --no-capture-ouput -n dashapp python /app/app/app.py --host 0.0.0.0 --debug_off
    ```

## Contributions
Authors: J. L. McDonagh
