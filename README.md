# Trabalho final GA020

Este repositório contém o notebook final da disciplina GA020, com estudo numérico das equações de Euler 1D aplicadas ao problema de despressurização de CO2 em uma tubulação.

O trabalho apresenta:

- interpretação das equações conservativas de massa, quantidade de movimento e energia;
- discussão de consistência, estabilidade e convergência;
- comparação entre diferença centrada e esquema upwind para um problema advectivo;
- avanço conjunto das três equações conservativas usando fluxo de Rusanov;
- comparação qualitativa com o Teste 8 de Metallinou Log et al. (2024);
- indicação dos próximos passos para reproduzir os experimentos com volumes finitos.

## Arquivos principais

- `notebooks/apresentacao_evolucao_pesquisa.ipynb`: notebook principal do trabalho, incluindo o texto, figuras e células com os códigos principais.
- `apresentacao_evolucao_pesquisa.pdf`: versão em PDF do material.
- `resources/figures/`: figuras usadas no notebook.
- `scripts/euler1d/`: scripts auxiliares usados para gerar a comparação escalar e os resultados do sistema completo.

## Como abrir

Com o ambiente Pixi instalado, execute:

```sh
pixi install --frozen
pixi shell
jupyter lab
```

Depois, abra:

```text
notebooks/apresentacao_evolucao_pesquisa.ipynb
```

## Scripts auxiliares

Os principais resultados numéricos estão incluídos como células de código no notebook e também foram mantidos nos scripts:

```sh
python scripts/euler1d/passo_2_pypde_comparacao.py
python scripts/euler1d/passo_4_sistema_completo.py
python scripts/euler1d/passo_4_pressao_008m_refinado.py
```

Esses scripts usam CoolProp para calcular propriedades termodinâmicas do CO2. A comparação escalar também usa `py-pde`. Caso o ambiente Pixi usado na correção não tenha essas bibliotecas instaladas, instale essas dependências no ambiente Python antes de executar os scripts.
