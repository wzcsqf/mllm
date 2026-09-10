"""
MiniMind 中文量化评估框架
支持指标：
  1. PPL（困惑度）—— 语言建模基础指标
  2. ROUGE / BLEU（jieba分词）—— 生成质量参考指标
  3. CLUE（TNEWS/AFQMC）—— 中文语言理解准确率（loglikelihood方式）

用法：
  python eval_metrics.py --weight full_sft --metrics ppl rouge clue
  python eval_metrics.py --weight full_sft --metrics all --clue_data_path eval_data/eval_clue.jsonl
  python eval_metrics.py --weight_file dpo_768_beta0.15.pth --metrics ppl clue
"""
import os, sys, json, time, math, argparse, warnings
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

# ============================================================
# 内置中文评测数据（无需下载，开箱即用）
# ============================================================

# PPL 评测：中文常识文本语料
PPL_CORPUS = [
    "太阳从东方升起，西方落下。这是地球自转的结果。",
    "水在零度以下会结冰，在一百度时会沸腾。这是水的基本物理性质。",
    "中国的首都是北京，上海是中国最大的城市之一。",
    "春天万物复苏，夏天骄阳似火，秋天硕果累累，冬天白雪皑皑。",
    "光合作用的公式是：二氧化碳加水在光照条件下生成有机物和氧气。",
    "地球绕太阳公转一周大约需要365天，这就是一年的时间。",
    "牛顿第一定律指出，物体在不受外力作用时保持静止或匀速直线运动。",
    "中国有56个民族，其中汉族人口最多，各民族和谐共处。",
    "长城是中国古代的军事防御工程，全长超过两万公里。",
    "人类的DNA由四种碱基组成：腺嘌呤、鸟嘌呤、胞嘧啶和胸腺嘧啶。",
    "唐诗宋词是中国文学的瑰宝，李白和杜甫被称为李杜。",
    "地球的大气层由氮气、氧气、氩气等组成，其中氮气占比最高。",
    "计算机的基本运算包括加减乘除，二进制是计算机内部的数据表示方式。",
    "黄河是中国的母亲河，长江是中国最长的河流。",
    "人体有206块骨头，骨骼系统支撑和保护着人体的各个器官。",
    "二十四节气是中国古代用来指导农事的补充历法。",
    "声音在空气中的传播速度约为每秒340米，光速约为每秒30万公里。",
    "中国的四大发明是造纸术、印刷术、火药和指南针。",
    "细胞是生物体的基本结构和功能单位，由细胞膜、细胞质和细胞核组成。",
    "中华人民共和国成立于1949年10月1日，首都是北京。",
]

# ROUGE/BLEU 评测：问答对（生成质量评估）
QA_PAIRS = [
    {"question": "为什么天空是蓝色的？", "reference": "天空呈现蓝色是因为瑞利散射。太阳光中的蓝色光波长较短，被大气中的分子散射得更多，所以我们看到的天空是蓝色的。"},
    {"question": "什么是光合作用？", "reference": "光合作用是植物利用阳光、二氧化碳和水，在叶绿体中合成有机物并释放氧气的过程。这是地球上几乎所有生命能量的最终来源。"},
    {"question": "中国的首都是哪里？", "reference": "中国的首都是北京。北京位于华北平原北部，是中国的政治、文化、科技和国际交往中心。"},
    {"question": "水的化学式是什么？", "reference": "水的化学式是H2O，表示一个水分子由两个氢原子和一个氧原子组成。"},
    {"question": "地球有多大？", "reference": "地球的赤道半径约为6371公里，表面积约5.1亿平方公里，体积约1.08万亿立方公里，是太阳系中第五大行星。"},
    {"question": "什么是人工智能？", "reference": "人工智能是研究如何让计算机模拟人类智能行为的学科，包括机器学习、深度学习、自然语言处理、计算机视觉等方向。"},
    {"question": "为什么会有四季？", "reference": "四季的变化是因为地球绕太阳公转时，地轴倾斜约23.5度，导致不同时期太阳直射点在南北回归线之间移动，各地接收到的太阳辐射量不同。"},
    {"question": "长城有多长？", "reference": "中国长城总长度超过21196公里，是世界上最长的人工建筑。其中明长城是最为完好的部分，全长约8851公里。"},
    {"question": "什么是牛顿第一定律？", "reference": "牛顿第一定律也叫惯性定律，指出一切物体在不受外力作用时，总保持静止状态或匀速直线运动状态，直到有外力迫使它改变这种状态。"},
    {"question": "中国的母亲河是哪条河？", "reference": "黄河被称为中国的母亲河。黄河流域是中华文明的发祥地，中国古代的大部分朝代都在这里建都。"},
]


# ============================================================
# CLUE 评测标签映射
# ============================================================

# TNEWS 新闻分类：标准 CLUE 标签 (100-109) → 中文
TNEWS_LABELS_STANDARD = {
    100: "故事", 101: "文化", 102: "娱乐", 103: "体育", 104: "财经",
    105: "房产", 106: "汽车", 107: "教育", 108: "科技", 109: "军事",
}

# AFQMC 语义相似度
AFQMC_LABELS = {0: "不相似", 1: "相似"}


# ============================================================
# 模型加载
# ============================================================

def init_model(args):
    """加载模型和tokenizer（复用 eval_llm.py 逻辑）"""
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        # 优先使用 --weight_file 直接指定完整文件名
        if hasattr(args, 'weight_file') and args.weight_file:
            ckp = f'./{args.save_dir}/{args.weight_file}'
        else:
            moe_suffix = '_moe' if args.use_moe else ''
            ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        print(f"  加载权重: {ckp}")
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def generate_response(model, tokenizer, question, device, max_new_tokens=256):
    """生成模型回答（贪心解码，保证可复现）"""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response


# ============================================================
# 指标1：困惑度 (Perplexity)
# ============================================================

def compute_ppl(model, tokenizer, corpus, device, stride=128, max_length=512):
    """
    计算中文语料上的困惑度
    使用滑动窗口避免显存溢出
    """
    print("\n" + "=" * 60)
    print("📊 指标1：困惑度 (Perplexity)")
    print("=" * 60)

    full_text = "\n".join(corpus)
    encodings = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = encodings["input_ids"].to(device)
    seq_len = input_ids.size(1)

    nlls_all = []
    prev_end_index = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        input_chunk = input_ids[:, begin_loc:end_loc]
        trg_len = end_loc - prev_end_index
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls_all.append(neg_log_likelihood)
        prev_end_index = end_loc
        if end_loc == seq_len:
            break

    avg_nll = torch.stack(nlls_all).mean()
    ppl = torch.exp(avg_nll).item()

    print(f"  语料长度: {seq_len} tokens")
    print(f"  平均 NLL: {avg_nll.item():.4f}")
    print(f"  困惑度 PPL: {ppl:.2f}")
    print(f"  ✅ PPL 越低越好，一般 < 30 为合理范围")

    return {"ppl": round(ppl, 2), "nll": round(avg_nll.item(), 4)}


# ============================================================
# 指标2：ROUGE / BLEU（基于 jieba 分词）
# ============================================================

def _lcs_length(a, b):
    """最长公共子序列长度（空间优化版）"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _compute_bleu(reference, hypothesis, max_n=4):
    """简化版 BLEU-n 计算"""
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = [tuple(reference[i:i + n]) for i in range(len(reference) - n + 1)]
        hyp_ngrams = [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1)]
        if len(hyp_ngrams) == 0:
            precisions.append(0)
            continue
        ref_counts = defaultdict(int)
        for ng in ref_ngrams:
            ref_counts[ng] += 1
        clipped = sum(
            min(hyp_ngrams.count(ng), ref_counts.get(ng, 0))
            for ng in set(hyp_ngrams)
        )
        precisions.append(clipped / len(hyp_ngrams))

    if min(precisions) == 0:
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = min(1.0, math.exp(1 - len(reference) / max(len(hypothesis), 1)))
    return round(bp * math.exp(log_avg), 4)


def compute_rouge_bleu(model, tokenizer, qa_pairs, device):
    """计算生成文本与参考答案的 ROUGE-1/2/L 和 BLEU-4 分数"""
    print("\n" + "=" * 60)
    print("📊 指标2：ROUGE / BLEU（jieba 分词）")
    print("=" * 60)

    try:
        import jieba
    except ImportError:
        print("  ⚠️ 需要安装 jieba: pip install jieba")
        return None

    scores = {"rouge1": [], "rouge2": [], "rouge_l": [], "bleu4": []}

    for i, pair in enumerate(qa_pairs):
        response = generate_response(model, tokenizer, pair["question"], device)
        ref = pair["reference"]

        ref_tokens = list(jieba.cut(ref))
        gen_tokens = list(jieba.cut(response))

        # ROUGE-1 (unigram F1)
        ref_set, gen_set = set(ref_tokens), set(gen_tokens)
        overlap_1 = ref_set & gen_set
        p1 = len(overlap_1) / max(len(gen_set), 1)
        r1 = len(overlap_1) / max(len(ref_set), 1)
        f1_1 = 2 * p1 * r1 / max(p1 + r1, 1e-8)
        scores["rouge1"].append(f1_1)

        # ROUGE-2 (bigram F1)
        ref_bi = set(tuple(ref_tokens[j:j + 2]) for j in range(len(ref_tokens) - 1))
        gen_bi = set(tuple(gen_tokens[j:j + 2]) for j in range(len(gen_tokens) - 1))
        overlap_2 = ref_bi & gen_bi
        p2 = len(overlap_2) / max(len(gen_bi), 1)
        r2 = len(overlap_2) / max(len(ref_bi), 1)
        f1_2 = 2 * p2 * r2 / max(p2 + r2, 1e-8)
        scores["rouge2"].append(f1_2)

        # ROUGE-L (LCS F1)
        lcs_len = _lcs_length(ref_tokens, gen_tokens)
        pl = lcs_len / max(len(gen_tokens), 1)
        rl = lcs_len / max(len(ref_tokens), 1)
        f1_l = 2 * pl * rl / max(pl + rl, 1e-8)
        scores["rouge_l"].append(f1_l)

        # BLEU-4
        bleu_score = _compute_bleu(ref_tokens, gen_tokens, max_n=4)
        scores["bleu4"].append(bleu_score)

        print(f"  [{i + 1}/{len(qa_pairs)}] Q: {pair['question'][:25]}...")
        print(f"    ROUGE-L: {f1_l:.4f} | BLEU-4: {bleu_score:.4f}")
        print(f"    生成: {response[:60]}...")

    results = {}
    for key in scores:
        avg = float(np.mean(scores[key]))
        results[key] = round(avg, 4)
        print(f"\n  📈 {key.upper()} 平均分: {avg:.4f}")

    print(f"\n  ✅ ROUGE-L 和 BLEU 越高越好（范围 0~1）")
    return results


# ============================================================
# 指标3：CLUE 基准测试（loglikelihood 方式）
# ============================================================

def _get_loglikelihood(model, tokenizer, prompt_text, label_text, device):
    """计算模型对 prompt+label 中 label 部分的平均对数概率"""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt_text + label_text, add_special_tokens=True)["input_ids"]
    label_ids = full_ids[len(prompt_ids):]

    if len(label_ids) == 0:
        return -1e9

    input_tensor = torch.tensor([full_ids]).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits[0]

    # logits[t] 预测 token[t+1]，label 第 i 个 token 对应 logits[prompt_len + i - 1]
    prompt_len = len(prompt_ids)
    log_probs = []
    for i, tok_id in enumerate(label_ids):
        pos = prompt_len + i - 1
        if pos < 0:
            continue
        token_log_probs = F.log_softmax(logits[pos].float(), dim=-1)
        log_probs.append(token_log_probs[tok_id].item())

    return sum(log_probs) / len(log_probs) if log_probs else -1e9


def _load_clue_data(data_path):
    """加载 CLUE JSONL 数据"""
    if not os.path.exists(data_path):
        print(f"  ⚠️ CLUE 数据文件不存在: {data_path}")
        print(f"  请先下载 CLUE 数据并生成 JSONL 文件")
        return []
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('{'):
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def compute_clue(model, tokenizer, data_path, device, max_samples=None):
    """
    CLUE 基准测试（loglikelihood 方式）
    支持 TNEWS（新闻分类）和 AFQMC（语义相似度）
    不需要模型输出结构化答案，通过比较各选项的对数概率来判定
    """
    print("\n" + "=" * 60)
    print(f"📊 指标3：CLUE 基准测试（loglikelihood）")
    print("=" * 60)

    samples = _load_clue_data(data_path)
    if not samples:
        return None

    if max_samples and max_samples < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, max_samples)

    # --- 动态构建 TNEWS 标签映射 ---
    # 数据中 label 可能是整数(100-109 或 0-9 或其他)，需要映射到中文
    tnews_label_set = set()
    for s in samples:
        if s.get("task") == "tnews":
            tnews_label_set.add(s["label"])

    if tnews_label_set:
        # 判断是标准 CLUE 编码 (100-109) 还是其他整数编码
        if all(isinstance(lb, int) and 100 <= lb <= 109 for lb in tnews_label_set):
            # 标准 CLUE: 100=故事, 101=文化, ...
            tnews_labels = {lb: TNEWS_LABELS_STANDARD[lb] for lb in tnews_label_set if lb in TNEWS_LABELS_STANDARD}
        elif all(isinstance(lb, int) for lb in tnews_label_set):
            # 非标准整数编码
            _category_names = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事"]
            min_lb = min(tnews_label_set)
            # 检测是 0-indexed 还是 1-indexed
            offset = 1 if min_lb >= 1 else 0
            tnews_labels = {}
            for lb in sorted(tnews_label_set):
                idx = lb - offset
                if 0 <= idx < len(_category_names):
                    tnews_labels[lb] = _category_names[idx]
                else:
                    tnews_labels[lb] = f"类别{lb}"
            print(f"  TNEWS 检测到 {offset}-indexed 标签 (offset={offset})")
        else:
            # 字符串标签（如 "news_story"）
            _str_map = {"news_story": "故事", "news_culture": "文化", "news_entertainment": "娱乐",
                        "news_sports": "体育", "news_finance": "财经", "news_house": "房产",
                        "news_car": "汽车", "news_edu": "教育", "news_tech": "科技", "news_military": "军事"}
            tnews_labels = {lb: _str_map.get(str(lb), str(lb)) for lb in tnews_label_set}
        print(f"  TNEWS 标签映射: {tnews_labels}")

    # --- 逐条评测 ---
    task_correct = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct, total_count = 0, 0

    for idx, sample in enumerate(samples):
        task = sample.get("task", sample.get("type", ""))
        label = sample.get("label", sample.get("answer", ""))

        # 构造 prompt 和候选标签
        if task == "tnews" or "headline" in sample or "keywords" in sample:
            task = "tnews"
            text = sample.get("sentence", sample.get("headline", sample.get("text", "")))
            prompt = f"请对以下新闻标题进行分类：\n{text}\n分类结果："
            labels = tnews_labels
            # label 保持原始类型（int 或 str），与 labels 的 key 类型一致
        elif task == "afqmc" or ("sentence1" in sample and "sentence2" in sample):
            task = "afqmc"
            s1 = sample.get("sentence1", "")
            s2 = sample.get("sentence2", "")
            prompt = f"请判断以下两个句子是否语义相似：\n句子1：{s1}\n句子2：{s2}\n判断结果："
            labels = AFQMC_LABELS
            # label 可能是 int(0/1)，确保类型匹配
            if isinstance(label, str):
                label = int(label)
        else:
            continue

        # 计算每个候选标签的 loglikelihood
        best_label, best_score = None, -1e9
        for label_key, label_text in labels.items():
            score = _get_loglikelihood(model, tokenizer, prompt, label_text, device)
            if score > best_score:
                best_score = score
                best_label = label_key

        # 判断是否正确（类型已统一，直接比较）
        is_correct = (best_label == label)
        task_correct[task]["total"] += 1
        task_correct[task]["correct"] += int(is_correct)
        total_count += 1
        total_correct += int(is_correct)

        if (idx + 1) % 50 == 0 or idx == 0:
            acc = total_correct / total_count * 100
            print(f"  [{idx + 1}/{len(samples)}] 当前准确率: {acc:.1f}%")

    # 汇总结果
    results = {}
    print(f"\n  📈 各任务准确率:")
    for task, stats in task_correct.items():
        acc = stats["correct"] / max(stats["total"], 1) * 100
        results[task] = {"accuracy": round(acc, 2), "correct": stats["correct"], "total": stats["total"]}
        print(f"    {task.upper()}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

    overall_acc = total_correct / max(total_count, 1) * 100
    results["overall"] = round(overall_acc, 2)
    print(f"\n  🏆 CLUE 综合准确率: {overall_acc:.1f}%")
    print(f"  ✅ 准确率越高越好")
    print(f"  📌 参考：随机基线 TNEWS≈{100/max(len(tnews_label_set),1):.0f}%, AFQMC=50%")

    return results


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MiniMind 中文量化评估")
    parser.add_argument('--metrics', nargs='+', default=['ppl'],
                        choices=['ppl', 'rouge', 'clue', 'all'],
                        help='选择评估指标（可多选，all=全部）')
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str,
                        help="权重名称（pretrain, full_sft, rlhf, ppo_actor, grpo 等）")
    parser.add_argument('--weight_file', default=None, type=str,
                        help="直接指定权重完整文件名（优先于 --weight），如 dpo_768_beta0.15.pth")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称")
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true')
    parser.add_argument('--clue_data_path', default='eval_data/eval_clue.jsonl', type=str,
                        help='CLUE 评测数据路径（JSONL格式）')
    parser.add_argument('--clue_max_samples', default=None, type=int,
                        help='CLUE 最大评测样本数（None=全部）')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--output', default=None, type=str,
                        help='结果保存路径（JSON格式），默认打印到终端')
    args = parser.parse_args()

    # 展开 all
    if 'all' in args.metrics:
        args.metrics = ['ppl', 'rouge', 'clue']

    # 确定权重标识
    weight_name = args.weight_file if args.weight_file else args.weight

    print("=" * 60)
    print(f"🚀 MiniMind 中文量化评估")
    print(f"   权重: {weight_name} | 指标: {args.metrics}")
    print(f"   设备: {args.device}")
    print("=" * 60)

    # 加载模型
    model, tokenizer = init_model(args)

    # 逐项计算指标
    all_results = {"weight": weight_name, "metrics": {}}

    if 'ppl' in args.metrics:
        ppl_result = compute_ppl(model, tokenizer, PPL_CORPUS, args.device)
        if ppl_result:
            all_results["metrics"]["ppl"] = ppl_result

    if 'rouge' in args.metrics:
        rouge_result = compute_rouge_bleu(model, tokenizer, QA_PAIRS, args.device)
        if rouge_result:
            all_results["metrics"]["rouge_bleu"] = rouge_result

    if 'clue' in args.metrics:
        clue_result = compute_clue(
            model, tokenizer, args.clue_data_path, args.device, args.clue_max_samples
        )
        if clue_result:
            all_results["metrics"]["clue"] = clue_result

    # 汇总输出
    print("\n" + "=" * 60)
    print("📋 评估结果汇总")
    print("=" * 60)

    if "ppl" in all_results["metrics"]:
        print(f"  困惑度 PPL: {all_results['metrics']['ppl']['ppl']}")
    if "rouge_bleu" in all_results["metrics"]:
        rb = all_results["metrics"]["rouge_bleu"]
        print(f"  ROUGE-1: {rb.get('rouge1', '-')} | ROUGE-2: {rb.get('rouge2', '-')} | "
              f"ROUGE-L: {rb.get('rouge_l', '-')} | BLEU-4: {rb.get('bleu4', '-')}")
    if "clue" in all_results["metrics"]:
        clue = all_results["metrics"]["clue"]
        for task, stats in clue.items():
            if task == "overall":
                continue
            print(f"  CLUE-{task.upper()}: {stats['accuracy']}%")
        print(f"  CLUE 综合: {clue.get('overall', '-')}%")

    # 保存结果
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 结果已保存到: {args.output}")

    print()


if __name__ == "__main__":
    main()
