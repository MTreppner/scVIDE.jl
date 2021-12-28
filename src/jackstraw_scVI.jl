using Clustering
using CSV
using Cairo
using Fontconfig
using Gadfly
using MultivariateStats
using DataFrames
using UnicodePlots
using PyCall
using Serialization
using Random
using StatsBase
using Statistics
using TSne
using ColorSchemes
using VegaLite
using Distributions
using Distributed

"""
    jackstraw_scVI(n_epochs, lr, n_genes, n_cells)
`n_epochs` number of epochs for training scVI when reference=false.
`n_genes` number of genes to read from dataset.
`n_cells` number of cells to subsample from dataset.
`s` number of samples to use for jackstraw sampling.
`B` number of jackstraw iterations.
`reference` whether to initialize scVI randomly or train the model.
"""
function jackstraw_scVI(;
    data_path::String,
    n_epochs::Int64,
    lr::Float64,
    n_genes::Int64,
    n_cells::Int64,
    n_latent::Int64,
    n_hidden::Int64,
    n_layers::Int64,
    batch_size::Int64,
    s::Int64,
    B::Int64,
    seed::Array{Int64,1},
    pretrained_model::String="",
    subsampling::Bool=false,
    use_autotune::Bool=false,
    search_space)

    # Trained models
    alternative_model_scVI_out = alternative_model_scVI(data_path=data_path,
                                                        n_epochs=n_epochs, 
                                                        lr=lr, 
                                                        n_genes=n_genes, 
                                                        n_cells=n_cells,
                                                        n_latent = n_latent,
                                                        n_hidden = n_hidden,
                                                        n_layers = n_layers,
                                                        batch_size = batch_size,
                                                        pretrained_model=pretrained_model,
                                                        subsampling=subsampling,
                                                        use_autotune=use_autotune,
                                                        search_space=search_space
    );

    vae = nothing
    trainer = nothing
    full = nothing

    if isempty(pretrained_model) == false
        null_model_scVI_out = vcat(pmap(params -> null_model_scVI(data_path=data_path,
                                                                n_epochs=params[1], 
                                                                lr=params[2], 
                                                                n_genes=params[3], 
                                                                n_cells=params[4], 
                                                                n_latent = params[5],
                                                                n_hidden = params[6],
                                                                n_layers = params[7],
                                                                batch_size = params[8],  
                                                                s=params[9], 
                                                                B=params[10],
                                                                seed=params[11],
                                                                pretrained_model=pretrained_model,
                                                                subsampling=subsampling),
                Iterators.product(n_epochs, lr, n_genes, n_cells, n_latent, n_hidden, n_layers, batch_size, s, B, seed))...
        );        
    else
        if use_autotune == true
            null_model_scVI_out = vcat(pmap(params -> null_model_scVI(data_path=data_path,
                                                                    n_epochs=alternative_model_scVI_out.autotune_out["result"]["best_epoch"], 
                                                                    lr=alternative_model_scVI_out.autotune_out["result"]["space"]["train_func_tunable_kwargs"]["lr"], 
                                                                    n_genes=params[1], 
                                                                    n_cells=params[2], 
                                                                    n_latent=alternative_model_scVI_out.autotune_out["result"]["space"]["model_tunable_kwargs"]["n_latent"],
                                                                    n_hidden=alternative_model_scVI_out.autotune_out["result"]["space"]["model_tunable_kwargs"]["n_hidden"],
                                                                    n_layers=alternative_model_scVI_out.autotune_out["result"]["space"]["model_tunable_kwargs"]["n_layers"],
                                                                    batch_size = params[3],    
                                                                    s=params[4], 
                                                                    B=params[5],
                                                                    seed=params[6],
                                                                    pretrained_model=pretrained_model,
                                                                    subsampling=subsampling),
                Iterators.product(n_genes, n_cells, batch_size, s, B, seed))...
            );
        else
            null_model_scVI_out = vcat(pmap(params -> null_model_scVI(data_path=data_path,
                                                                    n_epochs=params[1], 
                                                                    lr=params[2], 
                                                                    n_genes=params[3], 
                                                                    n_cells=params[4], 
                                                                    n_latent = params[5],
                                                                    n_hidden = params[6],
                                                                    n_layers = params[7],
                                                                    batch_size = params[8],    
                                                                    s=params[9], 
                                                                    B=params[10],
                                                                    seed=params[11],
                                                                    pretrained_model=pretrained_model,
                                                                    subsampling=subsampling),
                    Iterators.product(n_epochs, lr, n_genes, n_cells, n_latent, n_hidden, n_layers, batch_size, s, B, seed))...
            );
        end
    end

    interrupt()
    rmprocs(workers())

    (alternative_model_scVI_out=alternative_model_scVI_out,
    null_model_scVI_out=null_model_scVI_out)
end
