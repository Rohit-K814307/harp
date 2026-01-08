#!/bin/bash


show_help() {
    echo "Usage: ./data.sh [CONFIG_PATH]"
    echo ""
    echo "Description:"
    echo "  Runs the HARP data generation pipeline using the specified YAML configuration."
    echo "  Wraps the command: python -m harp.data.__init__"
    echo ""
    echo "Arguments:"
    echo "  CONFIG_PATH    Path to the YAML configuration file (e.g., config.yaml)"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message and exit"
    echo ""
    echo "Example:"
    echo "  ./data.sh config.yaml"
}


if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi


if [[ -z "$1" ]]; then
    echo "Error: Missing configuration file argument."
    echo "Try './data.sh --help' for more information."
    exit 1
fi


if [[ ! -f "$1" ]]; then
    echo "Error: File '$1' not found."
    exit 1
fi

# 4. Run the Python Module
echo "Launching HARP Data Pipeline..."
echo "Config: $1"
echo "-----------------------------------"

# We pass the argument to the python script using the --config flag we built earlier
python -m harp.data.__init__ --config "$1"