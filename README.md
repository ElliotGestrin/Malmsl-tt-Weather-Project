# Malmslätt Weather Project

## Introduction

This is a project performed in the course TFYA85 - Alternative Energy at Linköping University. 

The project trains a neural network-based combined regressor and classifier for weather data taken from [SMHIs open data](https://www.smhi.se/data/meteorologi/ladda-ner-meteorologiska-observationer). This data is preprocessed and combined into train- and test-files which train the model. 

See below for further details.

## Setup

The below instructions assume you're using windows powershell and might have to be adapted if you're using another system.

The project requires python version 3.9. Install this from: `https://www.python.org/downloads/release/python-3913/`

Furthermore, the project requires several dependencies. To keep these controlled and separate, we recommend using a virtual environment. In the root of the repo create a new one via:

``` bash
> py -3.9 -m venv env
```
Whenever you wish to use this environment you need to activate it. This is done in the following way:

``` bash
> env/Scripts/activate
```

To leave the environment call:
```
> deactivate
```

You then need to install the required packages. This is done with the following call, with the environment active:
```
> pip install -r requirements.txt
```

### Model inference
If you only wish to perform inference on the supplied model, call `Models.py`. 

```
> python Models.py
```
This will perform inference on the model saved to `Models/Weather_c8_reg.pt`. Note that this address can be easily changed in `Models.py`, but the hidden layers of the model infered and saved must of course be the same.

### Model training
Then the separate data files from SMHI have to be combined. Navigate into the `Data` folder and call `MergeData.py` via the following:

```
> cd Data
> python MergeData.py
```

The data should then be split into training and test data and one return to the root.

```
> python SplitData.py
> cd ..
```
Now you can train your own model via calling `Models.py`. Only models which improves on the test-data are saved and these are saved in `Models/...`. To do this:

```
> python TrainModel.py
```

## Data

This will create three separate time-series. To see the details of what's in each, see `MergeData.py`. In short theses are 
- **MalmslättFull**: Covers 1951-2022 and excludes rain amount
- **MalmslättEarly**: Covers 1951-2013 and excludes rain amount (as this wasn't yet measured)
- **MalmslättLate**: Covers 1951-2013 and includes rain amount

Any datapoints with faulty readings are entirely discarded and any gaps during which data wasn't read, for example air moisture is measured irregularly, are linearly interpolated between the closest available measurements.

These are then split in training and testing data with a proportion of 70% and 30% respectively.

## Models
 
The model uses a joint base which then splits into four heads:
- Continuous regression. 3 linear layers.
- Month classification. 1 linear layer.
- Day classification. 1 linear layer.
- Hour classification. 1 linear layer.

The sizes of the base can be specified with the `h_dims` parameter, see the example in `Models.py`. Note that you can't load a model saved with different base.

Note that the continuous head works via predicting the standard deviation from the mean for each continuous parameter, which are normalized to approximately `N(0,1)`. This to make the span more consistent, as some parameters are on the order of 10000 and some order of 1. 

On creation the content of the data should be specified with the `features` parameter, see the example in `Models.py`. This should be in the same order as the input to the model and should have `Month`, `Day` and `Year` as the first three, in some order. 

The input to the model should be a torch tensor with `(batch size, data+flags)` where data is the pure value for each input feature, in the order specified, and flags are binary flags where flag *i* specifies if feature *i* is missing and should be predicted (*true*) or is given (*false*). Any flagged (missing) values should be set to 0. 

The return from the model is a tuple of the predicted values for all parameters in the following order
> cont, month, day, hour

The continous values are in the order specified upon creation and the categorical are in the form of softmaxed-probability vectors. 

Note that the model processes `Month` and `Hour` via embeddings, by default an embedding dimension of 3. `Day` is used to performed a weighted sum of the embeddings of the embeddings from `Month`. This means that the 15th of June gives purely June's embedding, and the 30th of June would give an embedding which is the average of June's and July's.

## Training

The models are trained via minibatch with the `Adam` optimiser. A constant learning rate is used. 

During training each part of the minibatch has a random number of parameters removed and flagged. This should conteract overfitting and be a suitable learning task for the inference task targeted.

Each head has a separate loss-function which is then combined. This joint-loss is then what's optimised. `NLL-loss` is used for the categorical heads and `MSE-loss` is used for the continuous head. 

Each epoch the loss on test-data is computed. If this is an improvement, the current model weights are saved overwriting any previous saves from this iteration.

## Results

The model is far from perfect. The full model involving both categorical and regressor heads does learn something, for example it predicts winter days to be colder than summer days, but it is still lacking.

The model which only trained the regressor performed far better, fascinatingly enough able to even make reasonable predictions of month, day and hour despite these heads not even being trained. Why this is, I'm unsure. 

To test these yourself, follow the instructions in [Model Inference](#model-inference) above.

This work is mostly a proof of concept and a learning experience for me in dealing with mixed and complex data. In this regard, the project was a success.

## Further work

The current parameters and hyperparameters are somewhat arbitrarily chosen. A fair improvement could probably be gained via optimising these. Adding proper schedules for learning rate, batch size and percentage data "lost" could also conceivably have a lot of impact. 

At the start of this project I considered creating a function which expands the model by an additional feature, rain amount, while maintaining the other parameters. This would be relevant for the real world usability of the project.

Testing exclusion of some of the predicted parameters, probably the categorical, or weighting of the loss functions would also be very sensible. Initial experiments excluding the categorical these entirely does allow for a far lower validation loss. This of course depends on the requirements of the real world scenario. This is the model `Weather_c8_pure_reg.pt`. 

Lastly, the data collected could be used for several other forms of weather work. For example:
- Identification of faulty (modified) data
- Filtering of noisy (modified) data

I might pursue these improvements if given time. Should someone else wish to do so, they are free to go for it.