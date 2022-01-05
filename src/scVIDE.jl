module scVIDE

using StatsBase
using Conda
using PyCall
using Random
using Distributed

# Load python packages
const random = PyNULL()
const os = PyNULL()
const np = PyNULL()
const pd = PyNULL()
const scvi = PyNULL()
const scvi_dataset = PyNULL()
const scvi_models = PyNULL()
const scvi_inference = PyNULL()
const torch = PyNULL()
const scvi_inference_autotune = PyNULL()
const hyperopt = PyNULL()

function __init__()
    copy!(random, pyimport("random"))
    copy!(os, pyimport("os"))
    copy!(np, pyimport("np"))
    copy!(pd, pyimport("pd"))
    copy!(scvi, pyimport("scvi"))
    copy!(scvi_dataset, pyimport_conda("scvi.dataset", "scvi"))
    copy!(scvi_models, pyimport_conda("scipy.models", "scvi"))
    copy!(scvi_inference, pyimport_conda("scipy.inference", "scvi"))
    copy!(torch, pyimport("torch"))
    copy!(scvi_inference_autotune, pyimport_conda("scvi.inference.autotune", "scvi"))
    copy!(hyperopt, pyimport("hyperopt"))
end

#random = pyimport("random")
#os = pyimport("os")
#np = pyimport("numpy")
#pd = pyimport("pandas")
#scvi = pyimport("scvi")
#scvi_dataset = pyimport("scvi.dataset")
#scvi_models = pyimport("scvi.models")
#scvi_inference = pyimport("scvi.inference")
#torch = pyimport("torch")
#scvi_inference_autotune = pyimport("scvi.inference.autotune")
#hyperopt = pyimport("hyperopt")

export
    alternative_model_scVI,
    null_model_scVI,
    subsample_scVI_cells!,
    subsample_scVI_cells_idxs,
    jackstraw_scVI


include("scVIDE_algorithm.jl")

end # of module scVIDE