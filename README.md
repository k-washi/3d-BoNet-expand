# 3d-BoNet-expand

## 環境の確認 on google colab

```bash
!nvcc --version

!ls /usr/local
# cuda-10.1
```

## google driveからデータセットの読み込み

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

```bash
!tar -zxf "/content/drive/My Drive/data/dataset_name.tar.gz"
```

## tensorflowをアップグレード

```bash
!pip uninstall tensorflow
!pip install tensorflow==2.1
```