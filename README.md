# ICPR2018

### 更新

5.2:
- 改用GRU
- 加了许多dropout
- 改了输出到result.txt的bug
- 67%，基本没有过拟合，但loss下降很慢

4.30:
- 在test中，将label与得到的结果输出到result.txt
- 默认优化方法换成了adadelta，lr默认0.01
- 加了weight-decay，默认5e-4，减轻过拟合
- 加了colorjitter，减轻过拟合
- 训练集从100000加到了110000
- 增加了对不正常图片的try-except
- 默认增加了bn
- 对LSTM增加了0.2的dropout
- 初始化方法换为xavier normal

before:
- lr默认为延续之前优化方法里的lr或最开始的0.01；若指定了lr，则修改之前保存的optim里的lr
- 增加了checkepoch，为保存并测试的checkpoint
- 如果想一直训练，就把epochnum调大

- 通过命令行参数控制参数，`python main.py -h`，datasize一定记得设，别的参数自己调
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


