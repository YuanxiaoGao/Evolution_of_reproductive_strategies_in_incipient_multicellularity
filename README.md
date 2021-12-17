# Evolution_of_reproductive_strategies_in_incipient_multicellularity

**Overview**

This repository contains the simulation process of the work "Evolution of reproductive strategies in incipient multicellularity" by Yuanxiao, et al.
The aim is to investigate the growth rate of multicellular organisms under differnt forms of size effects and cell interactional forms.

**Organization**

1. Simulation.
 
      This folder includes the python scripts for calculating the growth rate of organisms under different conditions.

2. Data.

      This folder stores all the data (growth rates) of organisms under different conditions. Since the data are too large, here we show the compressed data rather than the original data.

3. Figure.

      This folder contains the python scripts that read the data in foler data and generate corresponding figures.

**Usage**

Growth_rate_*.py contains the source code for calculating growth rate.

make_*.py are the values of parameters. The file generates a run.sh file, which will be executed on cluster.

**Requirement**

All the codes were developed in Python 3.6.
