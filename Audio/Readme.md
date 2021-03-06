# To install Comet ML API
$ pip install comet_ml

# To obtain API follow instructions:
https://www.comet.ml/docs/rest-api/getting-started/

# Further documentation on Python & Comet ML
https://www.comet.ml/docs/python-sdk/getting-started/

# Import comet_ml at the top of your file
from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="",
    project_name="",
    workspace="",
)

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": 0.5,
    "steps": 100000,
    "batch_size": 50,
}
experiment.log_parameters(hyper_params)

# Or report single hyperparameters:
hidden_layer_size = 50
experiment.log_parameter("hidden_layer_size", hidden_layer_size)

# Log any time-series metrics:
train_accuracy = 3.14
experiment.log_metric("accuracy", train_accuracy, step=0)

# Run The code example as normal
> <speechrecogniser.py>
