# highway_exp

env
- 原HighWayEnv 基础上，装openai

    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openai



exp1
- main3.py是主函数
- two_llm.py是llm函数页面
- 车数量更改在 merge_env_v1.py 235行

        %        
        if self.num_CAV==None:
            self.num_CAV=4
        if self.num_HDV==None:
            self.num_HDV=4