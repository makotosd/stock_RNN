#!/usr/bin/python
# -*- coding: Shift_JIS -*-

#######################################################################
# jupyter-notebook��œ������B�^�̒l�ƁA�\���l���O���t�����܂��B
#######################################################################
## list 16
# �C���|�[�g�����s�ς݂̏ꍇ�A�ȉ���3�s�͂Ȃ��Ă��悢
import sys
import pandas as pd
import cufflinks as cf
import predict
import merge_companies
cf.go_offline()


#################################################
#  �\�������s
#################################################
ccs = sys.argv
ccs.pop(0)  # �擪(script��)���폜
input_dataset = merge_companies.merge_companies(ccs)
predict_dataset = predict.predict(input_dataset)


# �����f�[�^�Ɨ\���f�[�^
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
    
    
    
    
    
    
    
    
    