from harp.data.collect import collect
from harp.data.process import clean_df, group_by_reviewer, inject_rejected, encode, compute_and_save_numeric_stats
from harp.data.cutoffs import calculate_cutoffs
from harp.data.reviewer_costs import calculate_c, plot_c_modes

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import argparse
import sys

def generate_data(config):
    print(f"Starting Data Generation Pipeline with Random State: {config['random_state']}")
    
    print("... Collecting and Cleaning Data")
    collect(config['collect_samples'], config['sample_dir'])
    
    raw_df = pd.read_csv(config['raw_input_path'])
    cleaned_df = clean_df(raw_df)



    print("... Splitting Data (Train/Val/Test)")
    df_train_val, df_test = train_test_split(
        cleaned_df, 
        test_size=config['test_size'], 
        random_state=config['random_state'],
        shuffle=True
    )

    remaining_portion = 1.0 - config['test_size']
    relative_val_size = config['val_size'] / remaining_portion

    df_train, df_val = train_test_split(
        df_train_val, 
        test_size=relative_val_size, 
        random_state=config['random_state'],
        shuffle=True
    )



    print(f"... Injecting Rejected Claims (Rate: {config['injection_rate']})")
    df_train = inject_rejected(df_train, config['injection_rate'])
    df_val = inject_rejected(df_val, config['injection_rate'])
    df_test = inject_rejected(df_test, config['injection_rate'])



    print("... Calculating Global Complexity Cutoffs")
    full_dataset = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)
    
    cutoffs = calculate_cutoffs(
        df=full_dataset,
        scores=config['scores'],
        n_reviewers=config['n_reviewers'],
        n_init=config['kmeans_init'],
        random_state=config['random_state'],
        plot=config["plot_cutoffs"]
    )



    print("... Assigning Reviewer Tiers")
    reviewer_rand = {"traditional":config["traditional_random_error"], "out_of_depth":config["reviewer_out_of_depth_error"]}
    df_train = group_by_reviewer(df_train, cutoffs, config['scores'], reviewer_rand)
    df_val = group_by_reviewer(df_val, cutoffs, config['scores'])
    df_test = group_by_reviewer(df_test, cutoffs, config['scores'])


    print(f"... Saving Processed Datasets to {config['output_dir_raw']}")
    os.makedirs(config['output_dir_raw'], exist_ok=True)
    
    df_train.to_csv(os.path.join(config['output_dir_raw'], "train.csv"), index=False)
    df_val.to_csv(os.path.join(config['output_dir_raw'], "val.csv"), index=False)
    df_test.to_csv(os.path.join(config['output_dir_raw'], "test.csv"), index=False)



    print(f"... Encoding and Saving to {config['output_dir_encoded']}")
    os.makedirs(config['output_dir_encoded'], exist_ok=True)
    
    encode(df_train, os.path.join(config['output_dir_encoded'], "train.csv"))
    encode(df_val, os.path.join(config['output_dir_encoded'], "val.csv"))
    encode(df_test, os.path.join(config['output_dir_encoded'], "test.csv"))

    train_csv = os.path.join(config['output_dir_encoded'], "train.csv")
    compute_and_save_numeric_stats(train_csv)

    print(f"... Calculating Reviewer Costs with {config['reviewer_cost_params']['cost_mode']}")

    mrr_costs = calculate_c(df_train, config['n_reviewers'], config['reviewer_cost_params'])
    scarcity_costs = calculate_c(df_train, config['n_reviewers'], {"cost_mode":"scarcity", "scarcity_c_base":config['reviewer_cost_params']["scarcity_c_base"]})

    print(f"Found costs: {mrr_costs if config['reviewer_cost_params']['cost_mode'] == 'mrr' else scarcity_costs}")

    plot_c_modes(mrr_costs, scarcity_costs)

    print("Dataset generation complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate HARP Dataset from CMS Claims")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()


    try:
        with open(args.config, "r") as f:

            config = yaml.safe_load(f)
            print(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        sys.exit(1)


    try:
        generate_data(config=config)
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)