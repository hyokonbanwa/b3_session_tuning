### フォルダ構成
[ディレクトリ構成]

|
|----data #pytorchでダウンロードしたデータ。pytorchにより作成される
|----mlflow #mlflowで記録した実験結果が入っている。mlflowにより作成される
|----outputs #実験を実行毎にその時のconfig.yml等が保存される・hydraにより作成される
|----b3_session_cnn.py   #実験を行うpyファイル
|----config.yaml #実験設定。b3_session_cnn.pyでhydraを使って読みこみ
