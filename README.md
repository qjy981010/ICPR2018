# ICPR2018
### 更新 4.30
- 增加attention.py, 使用self-attention机制
- LSTM 修改为 GRU
- 保持一通道
- 修改CRNN.py

### 更新
- lr默认为延续之前优化方法里的lr或最开始的0.00005；若指定了lr，则修改之前保存的optim里的lr
- 增加了check_epoch，为保存并测试的checkpoint
- 如果想一直训练，就把epoch_num调大_

- 通过命令行参数控制参数，`python main.py -h`，data_size一定记得设，别的参数自己调
- test时也测试训练集准确率
- 模型储存时也存入了优化方法和epoch，以前存的可以自己写个小脚本转化成这种格式

### 目录结构

```
ICPR2018  
  |—— data: croped images  
  |—— models: trained models  
  |—— letters.txt: 存放所有字符的txt  
```

### 提示

如果有上传不了的文件，注意查看.gitignore文件


