
# HARP: Human-Aware Routing Policies for Multi-Stage Decision Pipelines

![harp-diag](images/harp_diagram.jpg)


## Run the data pipeline
please run from root:
```
chmod +x harp/scripts/data.sh

harp/scripts/data.sh harp/config/data_config.yaml
```

## Window Early Exit

### Training
- `python -m harp.scripts.train_window_early_exit`
- Or with custom config: `python -m harp.scripts.train_window_early_exit --config harp/config/window_early_exit_config.yaml`
- Checkpoints saved to `harp/models/checkpoints/window_early_exit/`

### Testing
- `python -m harp.scripts.test_window_early_exit --model <path_to_model.pt>`
- Example: `python -m harp.scripts.test_window_early_exit --model harp/models/checkpoints/window_early_exit/window_early_exit_final.pt`
- Optional: `--tau_junior 0.85` to adjust confidence threshold
