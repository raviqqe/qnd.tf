import tensorflow.contrib.learn.python.learn.learn_runner as learn_runner

from .experiment import def_def_experiment_fn
from .flag import FLAGS, add_flag


def def_run():
    add_flag("output_dir",
             default="output",
             help="Directory where checkpoint and event files are stored")

    def_experiment_fn = def_def_experiment_fn()

    def run(model_fn, input_fn):
        return learn_runner.run(def_experiment_fn(model_fn, input_fn),
                                FLAGS.output_dir)

    return run