#!/bin/bash

show_help() {
    echo "Usage: ./train.sh [CONFIG_PATH]"
    echo ""
    echo "Description:"
    echo "  Runs the HARP training pipeline using the specified YAML configuration."
    echo "  Wraps the command: python -m harp.models.train"
    echo ""
    echo "Arguments:"
    echo "  CONFIG_PATH    Path to the YAML configuration file (e.g., config.yaml)"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message and exit"
    echo ""
    echo "Example:"
    echo "  ./train.sh harp/config/csc_g_train_config.yaml"
}

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

if [[ -z "$1" ]]; then
    echo "Error: Missing configuration file argument."
    echo "Try './train.sh --help' for more information."
    exit 1
fi

if [[ ! -f "$1" ]]; then
    echo "Error: File '$1' not found."
    exit 1
fi

# Run the Python Training Module
python -m harp.models.train --config "$1"