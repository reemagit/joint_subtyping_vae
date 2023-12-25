import click
from mofapy2.run.entry_point import entry_point
import pandas as pd

from pathlib import Path

from utils import loader_funcs
from utils.coupled_dataset_module import CoupledDatasetModule

@click.command()
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path())
def main(model_dir, out_dir):
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)

    data_dir = Path('data/coupled')

    data_dir = Path(data_dir)

    splits_df = loader_funcs.load_splits(model_dir / "splits_df.tsv")

    cdm = CoupledDatasetModule(data_dir, bs=-1)
    cdm.setup()
    cdm.split(splits_df=splits_df, scaler='minmax')

    expr, pheno = cdm.get_full_data(cond=False, concat=False)

    # MOFA init
    # initialise the entry point
    ent = entry_point()

    mofa_data = [[expr.values, ], [pheno.values, ]]

    ent.set_data_options(
        scale_groups=False,
        scale_views=False
    )

    # option 2: nested matrix format (faster)
    ent.set_data_matrix(mofa_data, likelihoods=["gaussian", "gaussian"], views_names=['expr', 'pheno'])

    ent.set_model_options(
        factors=30,
        spikeslab_weights=True,
        ard_factors=False,
        ard_weights=True
    )

    ent.set_train_options(
        iter=1000,
        convergence_mode="fast",
        startELBO=1,
        freqELBO=1,
        dropR2=None,
        gpu_mode=False,
        verbose=True,
        seed=1
    )

    ent.build()
    ent.run()

    Z_fact = ent.model.nodes["Z"].getExpectation()
    Z_fact_df = pd.DataFrame(Z_fact, index=expr.index, columns=[f'Z{i}' for i in range(Z_fact.shape[1])])

    out_dir.mkdir(parents=True, exist_ok=True)

    Z_fact_df.to_csv(out_dir / 'Z_fact.tsv', sep='\t')

    #ent.save(outfile=out_dir / 'mofa.h5')


if __name__ == '__main__':
    main()
