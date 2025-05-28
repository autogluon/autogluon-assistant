### CLI


```bash
mlzero -i INPUT_DATA_FOLDER [-o OUTPUT_DIR] [-c CONFIG_PATH] [-n MAX_ITERATIONS] [--need-user-input] [-u INITIAL_USER_INPUT] [-e EXTRACT_TO] [-v VERBOSITY_LEVEL]
```


### Required Arguments

- `-i, --input`:  
  Path to the input data folder. This directory should contain training/testing files and optionally a description file.

### Optional Arguments

- `-o, --output`:  
  Path to the output directory. If not specified, a timestamped folder under `runs/` will be automatically generated.

- `-c, --config`:  
  Path to the YAML configuration file. Default: `configs/default.yaml`.

- `-n, --max-iterations`:  
  Maximum number of iterations. Default is `5`.

- `--need-user-input`:  
  Whether to prompt user input at each iteration. Defaults to `False`.

- `-u, --user-input`:  
  Initial user input to use in the first iteration. Optional.

- `-e, --extract-to`:  
  If the input folder contains archive files, unpack them into this directory. If not specified, archives are not unpacked.

- `-v, --verbosity`:  
  Increase logging verbosity level. Use `-v <level>` where level is an integer:
  
  | `-v` value | Logging Level  |
  |------------|----------------|
  | 0          | BRIEF          |
  | 1          | CRITICAL       |
  | 2          | ERROR          |
  | 3          | WARNING        |
  | 4          | BRIEF          |
  | 5          | INFO           |
  | 6          | MODEL_INFO     |
  | 7 or more  | DEBUG          |

### Examples

```bash
# Basic usage
mlzero -i ./data

# Custom output directory and verbosity
mlzero -i ./data -o ./results -v 5

# Use archive extraction and limit iterations
mlzero -i ./data -n 3 -e ./tmp_extract -v 6