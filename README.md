### フォルダ構成
[ディレクトリ構成]

|
|----data #pytorchでダウンロードしたデータ。pytorchにより作成される
|----mlflow #mlflowで記録した実験結果が入っている。mlflowにより作成される
|----outputs #実験を実行毎にその時のconfig.yml等が保存される・hydraにより作成される
|----b3_session_cnn.py   #実験を行うpyファイル
|----config.yaml #実験設定。b3_session_cnn.pyでhydraを使って読みこみ

### コマンド
* python b3_session_cnn.py 
* python b3_session_cnn.py --multirun epochs=3,5,7
* bash b3_session.sh

変更するもの
1.transform
2.dataloader.batchsize
3.model.encdim
4.optimizer.lr
