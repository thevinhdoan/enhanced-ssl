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

---
`curl https://rclone.org/install.sh | sudo bash` to install rclone

`rclone config` to configure Google Drive
```bash
n) New remote
name> gdrive
Storage> drive
client_id> (press Enter)
client_secret> (press Enter)
scope> 1   # Full access
root_folder_id> (Enter)
service_account_file> (Enter)
Edit advanced config? n
Use auto config? y
```
**Note**: Browser on local machine will open to prompt logging into Google account.

Compress the data and then upload by running 
```bash
rclone copy dataset.tar.gz gdrive:datasets/ \
  --progress \
  --transfers 8 \
  --checkers 8
```

Verify upload by `rclone ls gdrive:datasets`

Download the data by running
```bash
rclone copy gdrive:datasets/dataset.tar.gz . \
  --progress \
  --transfers 8 \
  --checkers 8
```

**Note**: Remove config at the end of instance life.