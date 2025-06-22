import pandas as pd
import numpy as np
import click
import os

import logging

import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--input_dataset_name', default='raw_dataset.csv', help='Nome do dataset raw', type=str)
@click.option('--output_dataset_name', default='processed_dataset.csv', help='Nome do dataset processado', type=str)
def main(input_dataset_name, output_dataset_name):
    logger = logging.getLogger(__name__)

    logger.info('Iniciando a criação da base processada.')

    logger.info('Leitura da base raw.')
    df = pd.read_csv(os.path.join('data', 'raw', input_dataset_name), sep=";")
    logger.info(f'Sucesso! Base lida. Shape da base: {df.shape}')

    logger.info('Tratando dados duplicado')
    df = df.drop_duplicates()
    logger.info(f'Linhas duplicadas removidas. Shape: {df.shape}.')

    logger.info("Ajustando target para booleano.")
    df["y"] = (df["y"]=="yes").astype(int).fillna(0)

    logger.info("Ajustando valores da coluna 'default'.")
    if 'default' in df.columns:
        df['default'] = df['default'].replace({'yes': 'unknown'})
        logger.info("Coluna 'default' ajustada com sucesso: 'yes' -> 'unknown'.")
    else:
        logger.warning("Coluna 'default' não encontrada na base.")

    logger.info("Salvando a base processada.")
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    path_output = os.path.join('data', 'processed', output_dataset_name)
    df.to_csv(path_output, index=False)
    logger.info("Sucesso! Base processada salva!")


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()