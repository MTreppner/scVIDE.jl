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
    copy!(np, pyimport("numpy"))
    copy!(pd, pyimport("pandas"))
    copy!(scvi, pyimport("scvi"))
    copy!(scvi_dataset, pyimport("scvi.dataset"))
    copy!(scvi_models, pyimport("scvi.models"))
    copy!(scvi_inference, pyimport("scvi.inference"))
    copy!(torch, pyimport("torch"))
    copy!(scvi_inference_autotune, pyimport("scvi.inference.autotune"))
    copy!(hyperopt, pyimport("hyperopt"))
end

export
    alternative_model_scVI,
    null_model_scVI,
    subsample_scVI_cells!,
    subsample_scVI_cells_idxs,
    jackstraw_scVI


include("scVIDE_algorithm.jl")

end # of module scVIDE