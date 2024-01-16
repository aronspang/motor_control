### Installation

created with python version 3.10 on linux machine

to create python environment run 
```python -m venv /path/to/new/venv```

to activate run
```source venv/bin/activate```

to install requirenment run
```pip install -r requirements.txt```

### Usage

set reward type, number of training steps and evaluation episodes in train.py

run with

```python train.py```

find results in 'logs' dir

to plot the mean over logs run either 

```python plot_all.py```

or specify log to plot in plot.py and run

```python plot.py```
