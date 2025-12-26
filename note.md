1. Choose folder path that stores the `saved_models` artifacts (pseudolabels, logs) in `semilearn/algorithms/pet/pet_hook.py`
2. Uncomment lines in `scripts/[clip | dinov2]/run_supervised.sh` to select hyperparam settings + dataset to run.
3. In `gen_config_pet.py`, uncomment the n-shot that is not needed and add if-else to select only (dataset, n-shot) to create configs for
4. In case of resuming training, comment out in `semilearn/core`
```
# self.start_epoch = checkpoint['epoch']
# self.epoch = self.start_epoch
```
and update this line to load model successfully
```
checkpoint = torch.load(load_path, map_location='cpu') # from this
checkpoint = torch.load(load_path, map_location='cpu', weights_only=False) # to this
```