module scVIDE

using StatsBase
using Conda
using PyCall
using Random
using Distributed

# Load python packages
random = pyimport("random")
os = pyimport("os")
np = pyimport("numpy")
pd = pyimport("pandas")
scvi = pyimport("scvi")
scvi_dataset = pyimport("scvi.dataset")
scvi_models = pyimport("scvi.models")
scvi_inference = pyimport("scvi.inference")
torch = pyimport("torch")
scvi_inference_autotune = pyimport("scvi.inference.autotune")
hyperopt = pyimport("hyperopt")

export
    alternative_model_scVI,
    null_model_scVI,
    subsample_scVI_cells!,
    subsample_scVI_cells_idxs,
    jackstraw_scVI,


include("scVIDE_algorithm.jl")

end # of module scVIDE