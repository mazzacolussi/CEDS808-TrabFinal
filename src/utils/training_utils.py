import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def find_specific_variables(features_dict, specific_key, specific_value=None):

    keys_with_specific_key=[]

    for k, sub_dict in features_dict.items():
        if isinstance(sub_dict, dict):
            if specific_key in sub_dict:
                if specific_value:
                    if sub_dict[specific_key] == specific_value:
                        keys_with_specific_key.append(k)
                else:
                    keys_with_specific_key.append(k)
    return keys_with_specific_key


def get_features_attribute(features, attribute):
    
    features_to_group={}

    for k, v in features.items():
        if attribute in v:
            features_to_group[k] = v[attribute]
    return features_to_group


def plot_variable_distribution(df: pd.DataFrame, var: str, bins=range(0, 101, 10)):
    """
    Gera um gráfico de distribuição de uma variável (input) do dataframe (input).

    Parâmetros:
    -----------
    df : pd.DataFrame

    var : str

    bins : lista com os ranges, opcional (default=range(0, 101, 10))
        Intervalo usado para um agrupamento da variável numérica. Ignorado se a variável fornecida for categórica.
    """

    if var == 'month':
        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        month_counts = df[var].value_counts().reindex(month_order)

        plt.figure(figsize=(8, 4))
        ax = sns.lineplot(x=month_counts.index, y=month_counts.values, marker='o')

        for x, y in zip(month_counts.index, month_counts.values):
            ax.text(x, y + 50, f'{y:,.0f}'.replace(',', '.'), ha='center', va='bottom')

        ax.set_xlabel('Month')
        ax.set_ylabel('Sample Volume')
        ax.set_title('Sample Distribution by Contact Month')
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        plt.tight_layout()
        plt.show()
        return

    if pd.api.types.is_numeric_dtype(df[var]):
        labels = [f"{i}-{i+9}" for i in bins[:-1]]
        group_var = f'{var}_group'
        df[group_var] = pd.cut(df[var], bins=bins, labels=labels, right=False)
        group_counts = df[group_var].value_counts().sort_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})
        sns.barplot(x=group_counts.index, y=group_counts.values, color='lightblue', ax=ax1)

        for container in ax1.containers:
            labels = [f"{val.get_height():,.0f}".replace(',', '.') for val in container]
            ax1.bar_label(container, labels=labels)

        ax1.set_xlabel('Range')
        ax1.set_ylabel('Sample Volume')
        ax1.set_title(f'Sample Volume by {var} Range')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

        sns.boxplot(y=df[var], color='lightblue', ax=ax2)
        ax2.set_title(f'{var.capitalize()} Distribution')
        ax2.set_ylabel(var.capitalize())
        ax2.set_xlabel('')
        ax2.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

        plt.tight_layout()
        plt.show()

    else:
        value_counts = df[var].value_counts().sort_index()
        plt.figure(figsize=(max(8, len(value_counts) * 0.6), 5))
        ax = sns.barplot(x=value_counts.index, y=value_counts.values, color='lightblue')

        for patch in ax.patches:
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + 5,
                f'{height:,.0f}'.replace(',', '.'),
                ha='center',
                va='bottom'
            )

        ax.set_xlabel(var.capitalize())
        ax.set_ylabel('Sample Volume')
        ax.set_title(f'Sample Distribution by {var}')
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

        if len(value_counts) > 5:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()


def plot_boxplot(df: pd.DataFrame, cat_var: str, num_var: str, figsize=(8, 4), palette='pastel'):
    """
    Plota um boxplot para a variável numérica agrupada pela variável categórica fornecida.

    Parametros:
    -----------
    df : pd.DataFrame

    cat_var : str

    num_var : str

    figsize : tuple (default=(8, 4))

    palette : str (default='pastel')
    """
    
    plt.figure(figsize=figsize)
    sns.boxplot(x=cat_var, y=num_var, data=df, palette=palette)

    plt.xlabel(cat_var.capitalize())
    plt.ylabel(num_var.capitalize())
    plt.title(f'{num_var} Distribution by {cat_var}')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    plt.tight_layout()
    plt.show()
