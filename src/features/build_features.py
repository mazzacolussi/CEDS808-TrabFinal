import pandas as pd
import click
import os

import logging
from utils.transformers import BuildFeatures

import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--input_dataset_name', default='processed_dataset.csv', help='Nome do dataset processado', type=str)
@click.option('--output_dataset_name', default='interim_dataset.csv', help='Nome do dataset interim', type=str)
def main(input_dataset_name, output_dataset_name):
    logger = logging.getLogger(__name__)

    logger.info('Iniciando a criação das features.')
    df = pd.read_csv(os.path.join('data', 'processed', input_dataset_name))
    logger.info(f'Shape da base: {df.shape}.')

    criador_features = BuildFeatures()

    logger.info('Iniciando processamento.')
    df = criador_features.transform(df)
    logger.info(f'Sucesso! Processamento finalizado. Shape da tabela processada: {df.shape}')

    logger.info(f"Salvando a base interim.")
    os.makedirs(os.path.join('data', 'interim'), exist_ok=True)
    path_output = os.path.join('data', 'interim', output_dataset_name)
    df.to_csv(path_output, index=False)
    logger.info(f"Sucesso! Base interim salva!")


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()