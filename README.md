# Dynamic Author-Persona Topic Model (DAP) #


## Introduction ##
See /docs/dap_2018_arxiv.pdf for technical information on the dynamic author-persona topic model (DAP).

## Getting Started ##
1. Clone the repo:

   ```bash
   cd ~
   git clone https://github.com/robert-giaquinto/dynamic-author-persona-topic-model.git
   ```

2. Virtual environments.

    It may be easiest to install dependencies into a virtual environment. To create the virtual environment for python 2.7 run:

   ```bash
   cd dynamic-author-persona-topic-model
   virtualenv venv
   ```

   For python 3+ run:

   ```bash
   cd dynamic-author-persona-topic-model
   python -m venv ./venv
   ```

   To activate the virtualenv run:

   ```bash
   source ~/dynamic-author-persona-topic-model/venv/bin/activate
   ```

3. Installing the necessary python packages.

   A requirements.txt file, listing all packages used for this project is included in the repository. To install them first make sure your virtual environment is activated, then run the following line of code:

   ```bash
   pip install --upgrade pip
   ```
   ```bash
   pip install -r ~/dynamic-author-persona-topic-model/requirements.txt
   ```

4. Install dap package.

    This is done to allow for absolute imports, which make it easy to load python files can be spread out in different folders. To do this navigate to the `~/dynamic-author-persona-topic-model` directory and run:

   ```bash
   python setup.py develop
   ```

5. Preparing data for the model

   TODO: build tutorial for easily accessible dataset

6. Running the model

   See /scripts/ for examples of running the model and setting various model parameters.

## Project Structure ##
* `docs/` - Documentation on the model, including derivation and papers related to the model.
* `log/` - Log files from running the programs.
* `scripts/` - Bash scripts for running programs.
* `dap/` - Directory containing various sub-packages of the project and any files shared across sub-packages.
