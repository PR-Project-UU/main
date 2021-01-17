> ⚠ Conda Required ⚠
>
> You'll need a version of conda for this, like Anaconda or miniconda. After installing it you can confirm it's present by calling `conda --help` on your preferred command line.

# 1. Creating the environment

There's two ways of doing this.

## Generating the Conda Environment from a file

If you have access to the file `ee-environment.yaml` shared in this repository you can generate the environment using that. Use:

```bash
conda env create -f ee-environment.yaml
```

You can activate the environment by using:

```bash
conda activate ee    # To activate the environment
conda deactivate ee  # To deactivate the environment
```

Note that if you have an environment already named 'ee' you will run into trouble. You can either remove the old environment or use the `--clone` option to avoid any issues, do note that this will require you to use a name other than `ee`:

```bash
conda env create -f ee-environment.yaml --clone
```

## Setting up Python for Earth Engine API for yourself

Let's set up a special environment for our Earth Engine API, we'll call it 'ee' here:

```bash
conda create --name ee
```

You can now jump into it using:

```bash
conda activate ee
```

Next up, when in the 'ee' environment we install the API using conda-forge:

```bash
conda install -c conda-forge earthengine-api
```

# 2. Authenticate

To actually make use of the earth engine API you'll need to authenticate your computer with your Google account (the one that is linked to your Google Developer account linked to Earth Engine. If you haven't such an account you can set it up by going to the Earth Engine website and signing up.)

We authenticate by using:

```bash
earthengine authenticate
```

# 3. Test your environment

We can test if we were successful by creating a python file as follows (or running the following code in the interpreter):

```python
import ee

# Initialize the Earth Engine module
ee.Initialize()

# Print metadata for a DEM dataset
print(ee.Image('USGS/SRTMGL1_003').getInfo())
```

This should print a mess of an object showing the metadata for a DEM image.

_That's all folks 😀_