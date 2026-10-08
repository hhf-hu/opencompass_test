#opemcompass_deepseek_test.py
from opencompass.models import OpenAI
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from ..opencompass.configs.datasets.lawbench.lawbench_one_shot_gen_002588 import lawbench_datasets as lawbench_one_shot_datasets
    from ..opencompass.configs.datasets.lawbench.lawbench_zero_shot_gen_002588 import lawbench_datasets as lawbench_zero_shot_datasets
    #Reasoning
    from ..opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import cmmlu_datasets
    from ..opencompass.configs.datasets.mmlu.mmlu_openai_simple_evals_gen_b618ea import mmlu_datasets
    from ..opencompass.configs.datasets.gsm8k.gsm8k_0shot_gen_a58960 import gsm8k_datasets
    #Summarizer
    from ..opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups

    #Math
    from ..opencompass.configs.datasets.aime2024.aime2024_0shot_nocot_genericllmeval_academic_gen import aime2024_datasets
    from ..opencompass.configs.datasets.bbh.bbh_0shot_nocot_academic_gen import bbh_datasets
    #General Reasoning
    from ..opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets
    from ..opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_dcae0e import humaneval_datasets
    #Instruction Following
    from ..opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets
    from ..opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import LCBCodeGeneration_dataset
    #from ..opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen import math_datasets
    #from ..opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets
    #Summary Groups
    #from ..opencompass.configs.summarizers.groups.bbh import bbh_summary_groups
    #from ..opencompass.configs.summarizers.groups.mmlu_pro import mmlu_pro_summary_groups

datasets = [
    *lawbench_one_shot_datasets,
    *lawbench_zero_shot_datasets,
    *cmmlu_datasets,
    *mmlu_datasets,
    *gsm8k_datasets,
    *mmlu_summary_groups,
    *aime2024_datasets,
    *bbh_datasets,
    *gpqa_datasets,
    *aime2024_datasets,
    *humaneval_datasets,
    *ifeval_datasets,
    *LCBCodeGeneration_dataset,
    # *math_datasets,
    # *mmlu_pro_datasets,
    # *bbh_summary_groups,
    # *mmlu_pro_summary_groups,
    # *iwslt2017_datasets,
]
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')),
               []) + [LCBCodeGeneration_dataset]


api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

#ip = '14.103.168.91'
#port = 50000
ip = '0.0.0.0'
port = 30000
models = [
    dict(
        type=OpenAI,                       # 使用 OpenAI 模型
        # 以下为 `OpenAI` 初始化参数
        abbr='DeepSeek-V3-0324',   # 模型简称
        path='deepseek-ai/DeepSeek-V3-0324',            # 指定模型类型
        openai_api_base = f"http://{ip}:{port}/v1/chat/completions",
        key='-',                   # OpenAI API Key
        max_seq_len=4096,          # 最大输入长度
        # mode = 'rear',
        retry = 3,
        # 以下参数为各类模型都有的参数，非 `OpenAI` 的初始化参数
        tokenizer_path='deepseek-ai/DeepSeek-V3-0324', # 请求服务时的 tokenizer name 或 path, 为None时使用默认tokenizer gpt-4
        run_cfg=dict(num_gpus=0),                # 资源需求（不需要 GPU）
        max_out_len=4096,                         # 最长生成长度
        batch_size=20,                            # 批次大小
        temperature = 0.1,
        meta_template = api_meta_template
    ),
]

work_dir = "outputs/"