# Pattern Recognition Project

## Program Arguments
The program uses several arguments to control the flow of the program.
They are described in this section.

Argument     | Description                                       | Shorthand | Type   | Default      | Notes
:---         | :---                                              | :---:     | :---:  | :---         | :---
`--debug`    | Enables debug, or sets the output level.          | `-d`      | String | `"info"`     | 1
`--delete`   | Delete files from Google Drive after downloading. | `-e`      | N/A    | False        | N/A
`--mode`     | Sets the mode the program is run in.              | `-m`      | String | `"generate"` | 2
`--savepath` | Where to save downloaded files in download mode.  | `-s`      | String | `"./"`       | N/A

### Argument Notes
#### 1. Debug
The debug flag controls the output level of the console.
Full logs are always written to file.
It knows three different modes:
* If it's not present it defaults the output level to INFO level.
* If present without any value it sets the output level to DEBUG.
* If present with a value it sets the output level to the specified value.

The logging modes are as follows in reverse order of importance: `debug`, `info`, `warning`, `error`, and `critical`.  
If you choose a mode with low importance like INFO, messages of higher importance like ERROR will still be printed.

#### 2. Mode
The mode flag controls the mode the program is run in. Three modes have been implemented.  
The default mode is data generation (`generate`), and uses Google Earth Engine to generate and acquire data files.  
The model training mode (`train`) trains a Generative Adversarial Model (GAN) on the acquired data. (not yet implemented)  
The image prediction mode (`predict`) uses a trained model to predict/generate a new image from acquired data. (not yet implemented)
