
"""
    alternative_model_scVI(n_epochs, lr, n_genes, n_cells)
Trains a scVI model and extracts the contribution to the evidence
lower bound for each cell. If `reference=true` then scVI will be randomly
initialized and not trained to be used as a reference.
`n_epochs` number of epochs for training scVI when reference=false.
`n_genes` number of genes to read from dataset.
`n_cells` number of cells to subsample from dataset.
`reference` whether to initialize scVI randomly or train the model.
"""
function alternative_model_scVI(;
    data_path::String,
    n_epochs::Int64,
    lr::Float64,
    n_genes::Int64,
    n_cells::Int64,
    n_latent::Int64,
    n_hidden::Int64,
    n_layers::Int64,
    batch_size::Int64,
    pretrained_model::String="",
    subsampling::Bool=false,
    use_autotune::Bool=false,
    search_space)

    #using StatsBase
    #using PyCall
    #using Random
    #using Distributed

    # Load python packages
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
    
    if isempty(pretrained_model) == false
        countmatrix = scvi_dataset.CsvDataset(data_path, save_path = "", new_n_genes = n_genes);
        
        if subsampling == true
            Random.seed!(111)
            subsample_scVI_cells!(countmatrix, n_cells);
        end

        # Load pre-trained model
        trainer = torch.load(pretrained_model)

        full = trainer.create_posterior(trainer.model, countmatrix);
        latent, batch_indices, labels = full.sequential().get_latent(give_mean=true);
            
        elbo_per_cell_null = full.elbo_sample()
        elbo_null = full.elbo()
        
        random.seed(111)
        torch.manual_seed(111)
        trainer.train(n_epochs=n_epochs,lr=lr)

        full = trainer.create_posterior(trainer.model, countmatrix);
        latent, batch_indices, labels = full.sequential().get_latent(give_mean=true);
            
        elbo_per_cell = full.elbo_sample()
        elbo = full.elbo()       
    else

        if use_autotune == true
            countmatrix = scvi_dataset.CsvDataset(data_path, save_path = "", new_n_genes = n_genes);

            if subsampling == true
                Random.seed!(111)
                subsample_scVI_cells!(countmatrix, n_cells);
            end

            best_trainer, trials = scvi_inference_autotune.auto_tune_scvi_model(
                    gene_dataset=countmatrix,
                    space=search_space,
                    parallel=false,
                    pickle_result=false,
                    exp_key=string("autotune_out",n_cells),
                    max_evals=30,
            )
            
            # Set hyperparameters
            use_batches = false
            use_cuda = true

            # Train the model and output model likelihood every epoch
            vae = scvi_models.VAE(countmatrix.nb_genes, 
                n_batch=countmatrix.n_batches * use_batches, 
                n_latent=trials.best_trial["result"]["space"]["model_tunable_kwargs"]["n_latent"], 
                n_hidden=trials.best_trial["result"]["space"]["model_tunable_kwargs"]["n_hidden"],
                n_layers=trials.best_trial["result"]["space"]["model_tunable_kwargs"]["n_layers"], 
                reconstruction_loss=trials.best_trial["result"]["space"]["model_tunable_kwargs"]["reconstruction_loss"], 
                dispersion = "gene-cell",
                latent_distribution = "ln",
                log_variational = false
            );
                
            trainer = scvi_inference.UnsupervisedTrainer(vae,
                countmatrix,
                use_cuda=use_cuda,
                frequency=1,
                batch_size=batch_size,
                n_epochs_kl_warmup= 30
            );

            full = trainer.create_posterior(trainer.model, countmatrix);
            latent, batch_indices, labels = full.sequential().get_latent(give_mean=true);
                
            elbo_per_cell_null = full.elbo_sample()
            elbo_null = full.elbo()
    
            full = best_trainer.create_posterior(best_trainer.model, countmatrix);
            latent, batch_indices, labels = full.sequential().get_latent(give_mean=true);
                
            elbo_per_cell = full.elbo_sample()
            elbo = full.elbo()           
        else
            countmatrix = scvi_dataset.CsvDataset(data_path, save_path = "", new_n_genes = n_genes);

            if subsampling == true
                Random.seed!(111)
                subsample_scVI_cells!(countmatrix, n_cells);
            end
    
            # Set hyperparameters
            use_batches = false
            use_cuda = true
    
            # Set seeds
            random.seed(111)
            torch.manual_seed(111)
        
            # Train the model and output model likelihood every epoch
            vae = scvi_models.VAE(countmatrix.nb_genes, 
                n_batch=countmatrix.n_batches * use_batches, 
                n_latent=n_latent, 
                n_hidden=n_hidden,
                n_layers=n_layers,  
                dispersion = "gene-cell",
                latent_distribution = "ln",
                log_variational = false
            );
                
            trainer = scvi_inference.UnsupervisedTrainer(vae,
                countmatrix,
                train_size=0.7,
                use_cuda=use_cuda,
                frequency=1,
                batch_size=batch_size,
                n_epochs_kl_warmup= 30
            );
    
            full = trainer.create_posterior(trainer.model, countmatrix);
            latent, batch_indices, labels = full.sequential().get_latent(give_mean=true);
                
            elbo_per_cell_null = full.elbo_sample()
            elbo_null = full.elbo()
            
            trainer.train(n_epochs=n_epochs,lr=lr)
    
            full = trainer.create_posterior(trainer.model, countmatrix);
            latent, batch_indices, labels = full.sequential().get_latent(give_mean=true);
                
            elbo_per_cell = full.elbo_sample()
            elbo = full.elbo()           
    
            best = []
            best_trainer = trainer
        end
    end

    if use_autotune == true
        (elbo_per_cell=elbo_per_cell, elbo=elbo, elbo_per_cell_null=elbo_per_cell_null, elbo_null=elbo_null , trainer=best_trainer, model=best_trainer.model, countmatrix=countmatrix, autotune_out=trials.best_trial)
    else
        (elbo_per_cell=elbo_per_cell, elbo=elbo, elbo_per_cell_null=elbo_per_cell_null, elbo_null=elbo_null , trainer=best_trainer, countmatrix=countmatrix)
    end
end

"""
    null_model_scVI(n_epochs, lr, n_genes, n_cells)
Trains a scVI model and extracts the contribution to the evidence
lower bound for each cell by using the jackstraw approach. A prespecified
number of cells `s` is extracted and the columns are sampled with replacement,
such that for these cells a null distribution is generated.
`n_epochs` number of epochs for training scVI when reference=false.
`n_genes` number of genes to read from dataset.
`n_cells` number of cells to subsample from dataset.
`s` number of samples to use for jackstraw sampling.
`B` number of jackstraw iterations.
`reference` whether to initialize scVI randomly or train the model.
"""
function null_model_scVI(;
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
    seed::Int64,
    pretrained_model::String="",
    subsampling::Bool=false)

    #using StatsBase
    #using PyCall
    #using Random
    #using Distributed

    # Load python packages
    #random = pyimport("random")
    #os = pyimport("os")
    #np = pyimport("numpy")
    #pd = pyimport("pandas")
    #scvi = pyimport("scvi")
    #scvi_dataset = pyimport("scvi.dataset")
    #scvi_models = pyimport("scvi.models")
    #scvi_inference = pyimport("scvi.inference")
    #torch = pyimport("torch")
    
    # TODO: remove random seed and use repetitions instead.
    #Random.seed!(seed);
    elbo_per_cell_append = Array{Float64,1}()
    arr_sample_append = Array{Int64,1}()
    elbo_per_cell_append_null = Array{Float64,1}()
    for i in 1:B   

        if isempty(pretrained_model) == false

            # Read data
            countmatrix = scvi_dataset.CsvDataset(data_path, save_path = "", new_n_genes = n_genes);

            if subsampling == true
                Random.seed!(111)
                subsample_scVI_cells!(countmatrix, n_cells);
            end
    
            # Make copy of original countmatrix
            countmatrix_tmp = deepcopy(countmatrix.X);

            # Load pre-trained model
            trainer = torch.load(pretrained_model)

            # Subsample s jackstraw sample
            s_cells = s
            arr = collect(1:size(countmatrix.X,1))
            arr_sample = StatsBase.sample(arr, s_cells; replace=false)

            # Sample variables of s jackstraw observations with replacement
            jackstraw_variables = Array{Float64,2}(undef, size(countmatrix.X[arr_sample,:]))
            for y = 1:size(arr_sample,1)
                jackstraw_variables[y,:] .= StatsBase.sample(countmatrix.X[arr_sample[y],:], size(countmatrix.X,2), replace=true)
            end
            countmatrix_tmp[arr_sample,:] .= jackstraw_variables

            # Override original countmatrix
            countmatrix.X = countmatrix_tmp

            # Override countmatrix in pre-trained model
            trainer.gene_dataset = countmatrix

            full = trainer.create_posterior(trainer.model, countmatrix);
                
            elbo_per_cell_null = full.elbo_sample()
        
            elbo_per_cell_tmp_null = elbo_per_cell_null[arr_sample]
            append!(elbo_per_cell_append_null, elbo_per_cell_tmp_null)

            # Training based on pre-trained model
            random.seed(111)
            torch.manual_seed(111)
            trainer.train(n_epochs=n_epochs,lr=lr)
                
            full = trainer.create_posterior(trainer.model, countmatrix);
            global posterior_object = full
                
            elbo_per_cell = full.elbo_sample();
            elbo_per_cell_tmp = elbo_per_cell[arr_sample];
            append!(elbo_per_cell_append, elbo_per_cell_tmp);
                
            append!(arr_sample_append, arr_sample);

            vae = nothing;
            trainer = nothing;
            full = nothing;
        else 
    
            # Read data
            countmatrix = scvi_dataset.CsvDataset(data_path, save_path = "", new_n_genes = n_genes);

            if subsampling == true
                Random.seed!(111)
                subsample_scVI_cells!(countmatrix, n_cells);
            end
    
            # Make copy of original countmatrix
            countmatrix_tmp = deepcopy(countmatrix.X);

            # Subsample s jackstraw sample
            s_cells = s
            arr = collect(1:size(countmatrix.X,1))
            arr_sample = StatsBase.sample(arr, s_cells; replace=false)

            # Sample variables of s jackstraw observations with replacement
            jackstraw_variables = Array{Float64,2}(undef, size(countmatrix.X[arr_sample,:]))
            for y = 1:size(arr_sample,1)
                jackstraw_variables[y,:] .= StatsBase.sample(countmatrix.X[arr_sample[y],:], size(countmatrix.X,2), replace=true)
            end
            countmatrix_tmp[arr_sample,:] .= jackstraw_variables

            # Override original countmatrix
            countmatrix.X = countmatrix_tmp

            # Set hyperparameters
            use_batches = false
            use_cuda = true

            # Set seeds
            random.seed(111)
            torch.manual_seed(111)
            
            # Train the model and output model likelihood every epoch
            vae = scvi_models.VAE(countmatrix.nb_genes, 
                n_batch=countmatrix.n_batches * use_batches, 
                n_latent=n_latent, 
                n_hidden=n_hidden,
                n_layers=n_layers, 
                dispersion = "gene-cell",
                latent_distribution = "ln",
                log_variational = false
            );

            trainer = scvi_inference.UnsupervisedTrainer(vae,
                countmatrix,
                train_size=0.7,
                use_cuda=use_cuda,
                frequency=1,
                batch_size=batch_size,
                n_epochs_kl_warmup= 30
            );

            full = trainer.create_posterior(trainer.model, countmatrix);
                
            elbo_per_cell_null = full.elbo_sample()
        
            elbo_per_cell_tmp_null = elbo_per_cell_null[arr_sample]
            append!(elbo_per_cell_append_null, elbo_per_cell_tmp_null)
                
            trainer.train(n_epochs=n_epochs,lr=lr);
                
            full = trainer.create_posterior(trainer.model, countmatrix);
            global posterior_object = full
                
            elbo_per_cell = full.elbo_sample();
            elbo_per_cell_tmp = elbo_per_cell[arr_sample];
            append!(elbo_per_cell_append, elbo_per_cell_tmp);
                
            append!(arr_sample_append, arr_sample);

            vae = nothing;
            trainer = nothing;
            full = nothing;
        end
    end
    (elbo_per_cell_append=elbo_per_cell_append, jackstraw_indices=arr_sample_append, elbo_per_cell_append_null=elbo_per_cell_append_null, posterior_object = posterior_object)
end


function subsample_scVI_cells!(data ,n_cells::Int64)

    # Random cell indices
    arrayy = collect(1:size(data.X,1))
    array_sample = StatsBase.sample(arrayy, n_cells; replace=false)
    array_sample .= array_sample .- 1

    # Override original countmatrix
    data.update_cells(array_sample)
end

function subsample_scVI_cells_idxs(data ,n_cells::Int64)

    # Random cell indices
    arrayy = collect(1:size(data.X,1))
    array_sample = StatsBase.sample(arrayy, n_cells; replace=false)
    array_sample .= array_sample .- 1

    # Override original countmatrix
    data.update_cells(array_sample)

    array_sample
end

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
