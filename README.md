# Pattern Recognition Project

## Program Arguments
The program uses several arguments to control the flow of the program.
They are described in this section.

Argument     | Description                                                        | Shorthand | Type   | Default      | Notes
:---         | :---                                                               | :---:     | :---:  | :---         | :---
`--delete`   | Delete files from after downloading or preprocessing them.         | `-e`      | N/A    | False        | 1
`--log`      | Enables debug, or sets the output level.                           | `-l`      | String | `"info"`     | 2
`--mode`     | Sets the mode the program is run in.                               | `-m`      | String | `"generate"` | 3
`--savepath` | Where to save files after they've been downloaded or preprocessed. | `-s`      | String | `"./"`       | N/A

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
