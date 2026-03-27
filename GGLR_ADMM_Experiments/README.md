# Graph-Guided Regularized Logistic Regression (GGLR) ADMM Experiments

### 论文数值实验代码：PKM-ADMM 对比实验 

包含几种随机ADMM算法：随机ADMM、SAG-ADMM、SAGA-ADMM、SVRG-ADMM、SPIDER-ADMM、PKM-ADMM 

### 运行环境 

Python 3.10 

安装依赖：pip install -r requirements.txt 

### 运行方式

直接运行主程序：python main.py 

### 输出结果

1. 控制台打印所有算法收敛信息 

2. Results/ 文件夹下自动保存3张对比曲线（png格式）：   

- 目标函数间隙收敛曲线   

- 原始残差收敛曲线   

- 对偶残差收敛曲线 

### 代码说明 

1. algorithms/ 下每个文件对应一种独立算法，严格按照论文公式实现 

2. utils/ 提供数据生成、指标计算、绘图工具 

3. main.py 统一调度所有算法，自动对比绘图