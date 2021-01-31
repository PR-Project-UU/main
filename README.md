# Pattern Recognition Project

## Program Arguments
The program uses several arguments to control the flow of the program.
They are described in this section.

Argument         | Description                                                        | Shorthand | Type    | Default         | Notes
:---             | :---                                                               | :---:     | :---:   | :---            | :---
`--delete`       | Delete files from after downloading or preprocessing them.         | `-e`      | N/A     | False           | 1
`--epochs`       | The amount of epochs to train for.                                 | `-c`      | Integer | `150`           | N/A
`--batches`      | Sets the number of batches to train on per epoch.                  | `-b`      | Integer | `100`           | N/A
`--load-path`    | The path to load images from                                       | `-p`      | String  | `"./data/raw"`  | 4
`--log`          | Enables debug, or sets the output level.                           | `-l`      | String  | `"info"`        | 2
`--mode`         | Sets the mode the program is run in.                               | `-m`      | String  | `"generate"`    | 3
`--model`        | Sets the model to use for predicting or further training.          | `-o`      | String  | N/A             | 5
`--no-overwrite` | Prevents overwriting files during preprocessing.                   | `-n`      | N/A     | False           | N/A
`--save-path`    | Where to save files after they've been downloaded or preprocessed. | `-s`      | String  | `"./data/raw"`  | N/A
`--save-pickle`  | Saves predicted images to pickles rather than PNG images.          | None      | N/A     | False           | N/A

### Argument Notes
#### 1. Delete
The delete flag does different things in different modes.  
In the data generation mode (`--mode generate`), this flag ensures files are deleted from the Google Drive account after they have been downloaded.  
In preprocess mode (`--mode preprocess`), this flag deletes the raw images from the disk after they've been preprocessed and saved as a pickle file.  
The flag does nothing in the other two modes.

#### 2. Log
The log flag controls the output level of the console.
Full logs are always written to file.
It knows three different modes:
* If it's not present it defaults the output level to INFO level.
* If present without any value it sets the output level to DEBUG.
* If present with a value it sets the output level to the specified value.

The logging modes are as follows in reverse order of importance: `debug`, `info`, `warning`, `error`, and `critical`.  
If you choose a mode with low importance like INFO, messages of higher importance like ERROR will still be printed.

#### 3. Mode
The mode flag controls the mode the program is run in. Four modes have been implemented.  
The default mode is data generation (`generate`), and uses Google Earth Engine to generate and acquire data files.  
The preprocess mode (`preprocess`) prepares the generated images for use in the neural network.  
The model training mode (`train`) trains a Generative Adversarial Model (GAN) on the acquired data. (not yet implemented)  
The image prediction mode (`predict`) uses a trained model to predict/generate a new image from acquired data. (not yet implemented)

#### 4. Load Path
The load path flag informs the program where to load image files from.
Its behavior is dependent on the mode set (with the `--mode` flag).  
In preprocessing mode (`--mode preprocess`) it selects the path to load raw `.tif` images from and preprocesses them to numpy data-cubes.  
In training mode (`--mode train`) this is the path where the program will look for preprocessed images to use in training and testing.  
In predict mode (`--mode predict`) this is the path where it will try to find the image to run a prediction on. The model to run in set by the `--model` flag.

#### 5. Model
The model flag selects which model to load.
What to do with the loaded model depends on the selected mode (`--mode`).  
In training mode (`--mode train`) the loaded model is trained further, rather than training a new model from scratch.  
In predict mode (`--mode predict`) the loaded model is used to predict images from input(s).  
In the other modes (generate and preprocessing) this flag does nothing.

### Batch Prediction
If the predict mode is used (`--mode predict`), but no load path is provided (`--load-path`),
the program will try to read the files to predict on from the input stream (stdin).  
This means that either on the command-line or through a text file, this mode is able to work through a list of files.
The first line of the input stream should contain just the number _n_ of files that you wish to process.
Then on the following _n_ lines, you can enter one file to predict off per line.
