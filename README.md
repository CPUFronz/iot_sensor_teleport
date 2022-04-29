# IoT Sensor Localization and Teleport

This is the source code for our paper "To Share or Not to Share: On Location Privacy in IoT Sensor Data" for [IoTDI 2022](https://conferences.computer.org/iotDI/2022/)

To create the environment install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and run 
```
conda env create -f environment.yml
```

Our experiments for the different data sets are in [Ozone.ipynb](Ozone.ipynb), [Cow.ipynb](Cow.ipynb) and [Solar.ipynb](Solar.ipynb).

Unfortunately we can not share the data for cow activity and solar data. But the data for ozone is publicly available. Just download the ozone data from [https://www.bafu.admin.ch/bafu/de/home/themen/luft/zustand/daten/datenabfrage-nabel.html](https://www.bafu.admin.ch/bafu/de/home/themen/luft/zustand/daten/datenabfrage-nabel.html) for 2020 for all stations except Jungfraujoch as CSV files and put them into `./data/ch_data` and run the notebook.
