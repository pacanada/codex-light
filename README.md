# codex-light

Project to explore how to recreate a (super) light version of codex, only for python programming language. 

WIP

## Quickstart

```cmd
python -m venv .venv
// for windows .venv/Scripts/activate
source .venv/bin/activate
pip install -r requirements.txt
```

## Training and inference

Train a model with (change as you wish the default configuration of the transformer):
```
python train.py
```
Run the tiny version (256 kB) with
```
python inference_v1.py 
```
```
Model has been loaded from True
INPUT
------------------

            "Input values must 

GENERATED
------------------
=* soul
   ""
    febttw fllestp(secthirnct rate is (matep[0] * larst = [1] 4, 0:, "Slinvat(_cu, "+ stageme:
     nreter: Checort ice rod weypals:
         of nod mi Aremny)



pr  imal the fountor preicke imn of ABo vetysfpacorn oul blis itestd clitr ecilopt
    Lof inverurl_alstiate te meclution p
```

## Bigger model

Run a 6MB transformer (trained on kaggle notebooks using `notebooks/train_kaggle.ipynb` on GPU):
```cmd
python inference_v1_kaggle.py 
```

```
Model has been loaded from True
INPUT
------------------

            "Input values must either be float or int: " f"{list(locals().values())}"
        )
    projected_x = ((x * distance) / (z + distance)) * scale
    projected_y = ((y * distance) / (z + di

GENERATED
------------------
ff)))
       with += np.splagocits(int, propes, dict):
        for i i in (
          ass _y] == exValse()


if is_main_eximpleate():
    "Hore Remural the namess for equal a imbe erroriations chautn of mat_positive
    match m:
        curromsitic_value data doctest
    """"

    return st(syprippe
```
almost AGI ah?
## TODO:

  - Log training with `wandb` 
  - Use more advance tokenizer (right now is on character level)