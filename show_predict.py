#!/usr/bin/python
# -*- coding: Shift_JIS -*-

#######################################################################
# jupyter-notebook上で動かす。真の値と、予測値をグラフ化します。
#######################################################################
## list 16
# インポート＆実行済みの場合、以下の3行はなくてもよい
import sys
import pandas as pd
import cufflinks as cf
import predict
import merge_companies
cf.go_offline()


#################################################
#  予測を実行
#################################################
ccs = sys.argv
ccs.pop(0)  # 先頭(script名)を削除
input_dataset = merge_companies.merge_companies(ccs)
predict_dataset = predict.predict(input_dataset)


# 正解データと予測データ
correct_data = input_dataset[input_dataset.index >= '2007']
predict_data = predict_dataset

for feature in predict_dataset.columns:
    plot_data = pd.DataFrame({
        'correct': correct_data[feature],
        'predicted': predict_data[feature]
    }).iplot(
      asFigure = True,
      title = feature
    )
    plot_data['layout']['paper_bgcolor'] = '#FFFFFF'
    plot_data.iplot()
    
    
    
    
    
    
    
    
    