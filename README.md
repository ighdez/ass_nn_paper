# Replication code of the ASS-NN paper

This is the replication code of the article titled "An economically-consistent discrete choice model with flexible utility specification based on artificial neural networks". We include a series of notebooks to replicate the results presented in our paper.

If you are interested only in the results, please run the files `gen_tables_synth_linear.ipynb` (linear synthetic data), `gen_tables_synth_loglinear.ipynb` (log-linear synthetic data)and `gen_tables_synth_swissmetro.ipynb` (empirical data). These files take the `.pickle` files that contain the predicted marginal utilities (stored in the `/results` folder) and create the tables and figures presented in the manuscript.

If you are interested on re-training the models, you can run the files `train_synth_linear.ipynb` (linear synthetic data), `train_synth_loglinear.ipynb` (log-linear synthetic data) and `train_synth_swissmetro.ipynb` (empirical data). Please, proceed with caution, as the `.picke` files will be overwritten.

The rest of the files correspond to the data generation files of pseudo-synthetic datasets, the preprocessing of the empirical dataset, the estimation of the multinomial Logit (MNL) models and the topology search process for the ASS-NN under each dataset.

These codes are written by José Ignacio Hernández during his PhD at the Delft University of Technology.