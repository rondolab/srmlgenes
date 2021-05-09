
# srMLGenes

This repository contains a web tool publised as a companion to [our 2021 paper](https://doi.org/10.1101/2021.05.06.443024) on inference of recessive selection from human population data. It allows users to explore inferences of dominance and selection and gene 
enrichments in different categories. You can also look at the raw data in the `dominance_data` directory to explore for yourself or design your own analyses.

A live server running this tool can be found at https://jordad05.u.hpc.mssm.edu/srmlgenes/.
Alternatively, see the instructions below to run it locally on your own computer.

## Run Locally

Clone the project (takes about 2 GB of storage space)

```bash
  git clone https://github.com/rondolab/srmlgenes
```

Go to the project directory

```bash
  cd srmlgenes
```

Initialize the virtual environment

```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
```

Start the server

```bash
  python index.wsgi
```
Open a web browser, and follow the instructions on the terminal for how to 
point your browser to the page.
  
## Citation
DJ Balick, DM Jordan, SR Sunyaev, and R Do. "Overcoming constraints on the detection of recessive selection in human genes from population frequency data." bioRxiv 2021.05.06.443024 (2021). doi: 10.1101/2021.05.06.443024
