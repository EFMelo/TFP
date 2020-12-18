# Total Factor Productivity (TFP)

> Previsão dos últimos 10 anos do TFP de 3 países: USA, CAN e MEX.

### Dataset

A base de dados é composta pelo TFP dos 3 países entre 1950 e 2011. A imagem abaixo ilustra o comportamento do TFP ao longo desse período.

<p align="center">
  <img width="390" height="226" src="https://i.imgur.com/yu998EB.png">
</p>

Foi considerado que essa base de dados contém 3 séries temporais, uma para cada país. Os resultados das previsões nos últimos 10 anos são mostrados abaixo.

### Resultados

Para informar o país que se deseja fazer a previsão, só precisa modificar a variável ``country`` no arquivo ``main.py``:

```python
country = 'MEX'  # <<<<<<<<<<  Options = 'USA', 'CAN' or 'MEX'. Just modify that line
```

Previsões da LSTM:

- USA

<p align="center">
  <img width="390" height="226" src="https://i.imgur.com/nZqAstR.png">
</p

- CAN

<p align="center">
  <img width="390" height="226" src="https://i.imgur.com/mA6kF1k.png">
</p

- MEX

<p align="center">
  <img width="390" height="226" src="https://i.imgur.com/jaJYDJt.png">
</p

**OBS. 1:** Após a linha vertical (cor azul), tem-se a comparação da previsão da rede neural com os valores reais durante os 10 anos.

**OBS. 2:** Ao executar outras vezes, pode-se ter resultados diferentes, devido a inicialização dos pesos ser aleatória (inicialização normalizada de Glorot e Bengio, 2010). No entanto, nota-se que, a curva prevista pela LSTM segue a tendência dos valores de TFP reais.