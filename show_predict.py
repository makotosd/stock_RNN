#!/usr/bin/python
# -*- coding: Shift_JIS -*-

#######################################################################
# jupyter-notebook��œ������B�^�̒l�ƁA�\���l���O���t�����܂��B
# ���̃X�N���v�g�̎��s�̑O�ɁApredict.py�����s���邱�ƁB

#######################################################################
## list 16
# �C���|�[�g�����s�ς݂̏ꍇ�A�ȉ���3�s�͂Ȃ��Ă��悢
import pandas as pd
import cufflinks as cf
cf.go_offline()

# �����f�[�^�Ɨ\���f�[�^
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
    
    
    
    
    
    
    
    
    