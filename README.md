# IoT Sensor Localization and Teleport

This is the source code for our paper ["To Share or Not to Share: On Location Privacy in IoT Sensor Data"](http://olgasaukh.com/paper/papst22locationprivacy.pdf) for [IoTDI 2022](https://conferences.computer.org/iotDI/2022/)

To create the environment install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and run 
```
conda env create -f environment.yml
```

Our experiments for the different data sets are in [Ozone.ipynb](Ozone.ipynb), [Cow.ipynb](Cow.ipynb) and [Solar.ipynb](Solar.ipynb).

Unfortunately we can not share the data for cow activity and solar data. But the data for ozone is publicly available. Just download the ozone data from [https://www.bafu.admin.ch/bafu/de/home/themen/luft/zustand/daten/datenabfrage-nabel.html](https://www.bafu.admin.ch/bafu/de/home/themen/luft/zustand/daten/datenabfrage-nabel.html) for 2020 for all stations except Jungfraujoch as CSV files and put them into `./data/ch_data` and run the notebook.

## Bibtex Citation
```
@inproceedings{papst2022locationprivacy,
  author={Franz Papst and Naomi Stricker and Rahim Entezari and Olga Saukh},
  booktitle={2022 IEEE/ACM Seventh International Conference on Internet-of-Things Design and Implementation (IoTDI)}, 
  title={To Share or Not to Share: On Location Privacy in IoT Sensor Data}, 
  year={2022},
  pages={128-140},
  doi={10.1109/IoTDI54339.2022.00015}
}
```
