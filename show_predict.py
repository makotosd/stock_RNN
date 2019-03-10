#!/usr/bin/python
# -*- coding: Shift_JIS -*-

#######################################################################
#  jupyter notebook��œ������B�^�̒l�ƁA�\���l���O���t�����܂��B
#######################################################################
# �C���|�[�g�����s�ς݂̏ꍇ�A�ȉ���3�s�͂Ȃ��Ă��悢
import pandas as pd
import cufflinks as cf
import argparse
import predict
import merge_companies

# arg �p�[�T�̐���
parser = argparse.ArgumentParser(description='�\���l�Ɛ^�l�̔�r�A�ۑ��A����')

# �I�v�V�����Q�̐ݒ�
parser.add_argument('--cc', nargs='*', help='company code')
parser.add_argument('--output', help='�\���l�Ɛ^�l�̌���(csv)�̏o�́B�����͍s��Ȃ��B')
parser.add_argument('--input', help='�\���l�Ɛ^�l�̌���(csv)�̓��́B�\���͍s��Ȃ��B')

args = parser.parse_args()  # �����̉�͂����s


# �^�l�̓ǂݍ���
input_dataset = merge_companies.merge_companies(args.cc)

#  �\�������s
if args.input is None:  # �ǂݍ��݃t�@�C���̎w�肪�Ȃ��@���@�\���̎��{
    predict_dataset = predict.predict(input_dataset)

    if args.output is not None:  # �������݃t�@�C���̎w�肪���� �� �t�@�C��������
        predict_dataset.to_csv(str(args.output))

else:
    predict_dataset = pd.read_csv(args.input, index_col=0)


# �����f�[�^�Ɨ\���f�[�^
correct_data = input_dataset[input_dataset.index >= '2007']
predict_data = predict_dataset

cf.go_offline()  # plotly���I�t���C���Ŏ��s�B
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
