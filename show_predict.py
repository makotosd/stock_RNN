#!/usr/bin/python
# -*- coding: Shift_JIS -*-

#######################################################################
# jupyter-notebook上で動かす。真の値と、予測値をグラフ化します。
# このスクリプトの実行の前に、predict.pyを実行すること。

#######################################################################
## list 16
# インポート＆実行済みの場合、以下の3行はなくてもよい
import pandas as pd
import cufflinks as cf
cf.go_offline()

# 正解データと予測データ
correct_data = dataset[dataset.series_data.index >= '2007'].series_data
predict_data = predict_air_quality

for feature in air_quality.columns:
    plot_data = pd.DataFrame({
        'correct': correct_data[feature],
        'predicted': predict_data[feature]
    }).iplot(
      asFigure = True,
      title = feature
    )
    plot_data['layout']['paper_bgcolor'] = '#FFFFFF'
    plot_data.iplot()
    
    
    
    
    
    
    
    
    